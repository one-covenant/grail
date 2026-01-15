"""GPU integration tests with actual vLLM server.

Tests the complete offline GRPO pipeline with real server backends.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Ensure repo root and src on sys.path BEFORE any imports from these paths
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_SRC_DIR = _THIS_FILE.parents[1] / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from grail_offline.data.offline_rollouts import (  # noqa: E402
    OfflineRolloutGenerator,
    RolloutGenConfig,
)

from grail.model.provider import get_model, get_tokenizer  # noqa: E402
from grail.shared.chat_templates import build_qwen_chat_template  # noqa: E402
from grail.shared.prompt_constants import SYSTEM_PROMPT  # noqa: E402
from grail.trainer.algorithms.grpo import GRPOAlgorithm  # noqa: E402
from grail.trainer.config import EvalConfig, TrainingConfig  # noqa: E402
from grail.trainer.eval_planner import EvaluationPlan  # noqa: E402
from grail.trainer.evaluator import EvaluatorService  # noqa: E402


class VLLMServerManager:
    """Manages vLLM server lifecycle for testing."""

    def __init__(self, model_id: str, port: int, gpu_id: int = 0) -> None:
        self.model_id = model_id
        self.port = port
        # Clamp to a valid device index for the current machine
        self.gpu_id = max(0, min(int(gpu_id), max(0, torch.cuda.device_count() - 1)))
        self.process: subprocess.Popen[bytes] | None = None
        self.base_url = f"http://127.0.0.1:{port}"

    def start(self, timeout: float = 180.0) -> None:
        """Start vLLM server and wait for readiness."""
        print(
            f"Starting vLLM server on port {self.port} with {self.model_id} on GPU {self.gpu_id}..."
        )

        # Set environment to use specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # Start vLLM server
        cmd = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "1024",
            "--gpu-memory-utilization",
            "0.4",
            "--disable-log-requests",
            "--trust-remote-code",
        ]

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        # Wait for server to be ready
        start_time = time.time()
        ready = False
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                raise RuntimeError("vLLM server process terminated unexpectedly")

            try:
                # Check if server is responding
                import requests

                response = requests.get(f"{self.base_url}/v1/models", timeout=3.0)
                if response.status_code == 200:
                    print(f"✅ vLLM server ready on {self.base_url}")
                    ready = True
                    # Give it a bit more time to fully stabilize
                    time.sleep(3.0)
                    return
            except Exception:
                pass

            time.sleep(3.0)

        if not ready:
            raise TimeoutError(f"vLLM server did not become ready within {timeout}s")

    def shutdown(self) -> None:
        """Shutdown vLLM server."""
        if self.process is not None:
            print("Shutting down vLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                print("Force killing vLLM server...")
                self.process.kill()
                self.process.wait()
            self.process = None
            print("✅ vLLM server shutdown complete")

    def __enter__(self) -> VLLMServerManager:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()


async def test_vllm_server_rollout_generation() -> None:
    """Test rollout generation with actual vLLM server."""
    if not torch.cuda.is_available():
        print("SKIP: GPU not available")
        return

    model_id = "Qwen/Qwen2.5-0.5B"  # Small model for fast testing
    port = 30100

    with VLLMServerManager(model_id, port, gpu_id=0):
        # Create rollout generator pointing to vLLM server
        tokenizer = get_tokenizer(model_id)

        cfg = RolloutGenConfig(
            backend="vllm_server",
            base_url=f"http://127.0.0.1:{port}",
            batch_size=2,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            rollouts_per_problem=4,
        )

        generator = OfflineRolloutGenerator(tokenizer=tokenizer, config=cfg)

        # Generate groups
        seeds = [3001, 3002]
        groups = await generator.generate_groups(seeds)

        # Validate
        assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"
        for group in groups:
            assert len(group.rollouts) == 4, f"Expected 4 rollouts, got {len(group.rollouts)}"
            assert group.is_valid(advantage_tolerance=1e-5, rollouts_per_problem=4)

            # Check that we actually got completions
            for rollout in group.rollouts:
                assert rollout.completion_length > 0, "Completion should not be empty"
                assert len(rollout.tokens) > rollout.prompt_length
                print(
                    f"  Rollout: prompt_len={rollout.prompt_length}, comp_len={rollout.completion_length}, reward={rollout.reward:.3f}, adv={rollout.advantage:.3f}"
                )

    print("✅ vLLM server rollout generation test passed")


async def test_vllm_server_training_epoch() -> None:
    """Test full GRPO training epoch with vLLM server generation."""
    if not torch.cuda.is_available():
        print("SKIP: GPU not available")
        return

    model_id = "Qwen/Qwen2.5-0.5B"
    port = 30101

    with VLLMServerManager(model_id, port, gpu_id=1):
        # Setup accelerator first to know target device
        from accelerate import Accelerator

        accelerator = Accelerator(mixed_precision="no")
        device = accelerator.device

        # Load models directly to accelerator device
        chat_template = build_qwen_chat_template(SYSTEM_PROMPT)
        tokenizer = get_tokenizer(model_id, chat_template=chat_template)
        train_model = get_model(model_id, device=str(device), eval_mode=False)
        ref_model = get_model(model_id, device=str(device), eval_mode=True)

        print(f"  Models loaded to {device}")

        # Generate rollouts via server
        cfg = RolloutGenConfig(
            backend="vllm_server",
            base_url=f"http://127.0.0.1:{port}",
            batch_size=4,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            rollouts_per_problem=4,
        )

        generator = OfflineRolloutGenerator(tokenizer=tokenizer, config=cfg)
        seeds = [4001, 4002, 4003, 4004]
        groups = await generator.generate_groups(seeds)

        print(
            f"Generated {len(groups)} groups with {sum(len(g.rollouts) for g in groups)} rollouts"
        )

        # Train epoch
        optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-4)

        algo = GRPOAlgorithm()
        train_cfg = TrainingConfig(lr=1e-4, batch_size=8)

        metrics = await algo.train_epoch(
            model=train_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            groups=groups,
            optimizer=optimizer,
            accelerator=accelerator,
            monitor=None,
            window=0,
            config=train_cfg,
        )

        # Validate metrics
        assert "loss_total" in metrics
        assert "loss_kl" in metrics
        assert "reward_mean" in metrics
        assert isinstance(metrics["loss_total"], float)
        print(
            f"✅ Training metrics: loss_total={metrics['loss_total']:.4f}, kl={metrics['kl_divergence']:.4f}, reward_mean={metrics['reward_mean']:.3f}"
        )

    print("✅ vLLM server training epoch test passed")


async def test_vllm_server_evaluator() -> None:
    """Test evaluator service with vLLM server backend."""
    if not torch.cuda.is_available():
        print("SKIP: GPU not available")
        return

    model_id = "Qwen/Qwen2.5-0.5B"
    port = 30102
    device = "cuda:0"

    with VLLMServerManager(model_id, port, gpu_id=3):
        # Load model and tokenizer
        chat_template = build_qwen_chat_template(SYSTEM_PROMPT)
        tokenizer = get_tokenizer(model_id, chat_template=chat_template)
        model = get_model(model_id, device=device, eval_mode=True)

        def env_factory() -> Any:
            from grail.environments.sat_env import SATEnv

            return SATEnv()

        # Create evaluator with vllm_server backend
        eval_cfg = EvalConfig(batch_size=4, replicates=3, do_sample=True, max_new_tokens=64)
        eval_cfg.backend = "vllm_server"
        eval_cfg.server_host = "127.0.0.1"
        eval_cfg.server_port = port

        evaluator = EvaluatorService(
            model=model,
            tokenizer=tokenizer,
            env_factory=env_factory,
            config=eval_cfg,
            monitor=None,
            device=device,
        )

        # Run evaluation
        plan = EvaluationPlan(ids=["10", "11", "12"], replicates=3, cycle_index=0, seed_base=9999)
        metrics = await evaluator.run_cycle(plan)

        # Validate
        assert "pass@1" in metrics
        assert "mean@1" in metrics
        assert isinstance(metrics["pass@1"], float)
        print(
            f"✅ Evaluation metrics: pass@1={metrics['pass@1']:.2f}, mean@1={metrics['mean@1']:.3f}"
        )

    print("✅ vLLM server evaluator test passed")


async def test_end_to_end_iteration() -> None:
    """Test complete iteration: generate → train → evaluate with vLLM server."""
    if not torch.cuda.is_available():
        print("SKIP: GPU not available")
        return

    model_id = "Qwen/Qwen2.5-0.5B"
    port = 30103

    with VLLMServerManager(model_id, port, gpu_id=5):
        # Setup accelerator first
        from accelerate import Accelerator

        accelerator = Accelerator(mixed_precision="no")
        device = accelerator.device

        # Load models to accelerator device
        chat_template = build_qwen_chat_template(SYSTEM_PROMPT)
        tokenizer = get_tokenizer(model_id, chat_template=chat_template)
        train_model = get_model(model_id, device=str(device), eval_mode=False)
        ref_model = get_model(model_id, device=str(device), eval_mode=True)

        print(f"🔄 Phase 1: Generate rollouts (server on GPU 5, training on {device})")
        cfg = RolloutGenConfig(
            backend="vllm_server",
            base_url=f"http://127.0.0.1:{port}",
            batch_size=4,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            rollouts_per_problem=4,
        )

        generator = OfflineRolloutGenerator(tokenizer=tokenizer, config=cfg)
        train_seeds = [5001, 5002, 5003, 5004]
        groups = await generator.generate_groups(train_seeds)
        print(f"  Generated {len(groups)} groups")

        print("🔄 Phase 2: Train epoch")
        optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-4)

        algo = GRPOAlgorithm()
        train_cfg = TrainingConfig(lr=1e-4, batch_size=8)

        train_metrics = await algo.train_epoch(
            model=train_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            groups=groups,
            optimizer=optimizer,
            accelerator=accelerator,
            monitor=None,
            window=0,
            config=train_cfg,
        )
        print(f"  Training loss: {train_metrics['loss_total']:.4f}")

        print("🔄 Phase 3: Evaluate")

        def env_factory() -> Any:
            from grail.environments.sat_env import SATEnv

            return SATEnv()

        eval_cfg = EvalConfig(batch_size=4, replicates=2, do_sample=True, max_new_tokens=64)
        eval_cfg.backend = "vllm_server"
        eval_cfg.server_host = "127.0.0.1"
        eval_cfg.server_port = port

        evaluator = EvaluatorService(
            model=train_model,
            tokenizer=tokenizer,
            env_factory=env_factory,
            config=eval_cfg,
            monitor=None,
            device=str(device),
        )

        eval_ids = ["20", "21", "22", "23"]
        plan = EvaluationPlan(ids=eval_ids, replicates=2, cycle_index=0, seed_base=7777)
        eval_metrics = await evaluator.run_cycle(plan)

        print(f"  Eval pass@1: {eval_metrics['pass@1']:.2f}, mean@1: {eval_metrics['mean@1']:.3f}")

        print("✅ End-to-end iteration test passed")


async def main() -> None:
    """Run all GPU integration tests."""
    if not torch.cuda.is_available():
        print("❌ GPU not available, skipping GPU integration tests")
        return

    print(f"\n{'=' * 80}")
    print("GPU Integration Tests with vLLM Server")
    print(f"{'=' * 80}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"vLLM version: {__import__('vllm').__version__}")
    print(f"{'=' * 80}\n")

    tests = [
        ("vLLM Server Rollout Generation", test_vllm_server_rollout_generation),
        ("vLLM Server Training Epoch", test_vllm_server_training_epoch),
        ("vLLM Server Evaluator", test_vllm_server_evaluator),
        ("End-to-End Iteration", test_end_to_end_iteration),
    ]

    passed = 0
    for name, test_func in tests:
        try:
            print(f"\n{'=' * 60}")
            print(f"Running: {name}")
            print(f"{'=' * 60}")
            await test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"GPU Integration Test Results: {passed}/{len(tests)} passed")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
