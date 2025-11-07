#!/usr/bin/env python3
"""Simple training runner that starts vLLM and runs offline GRPO training.

This script:
1. Loads GPU configuration from conf/offline_grpo.yaml
2. Starts a vLLM inference server on the configured GPU
3. Runs the offline GRPO trainer which uses the vLLM server for rollout generation
4. Cleans up the vLLM server on exit

Both the vLLM server and training script run in the same virtual environment.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parents[1]

# Use the offline trainer's own .venv which includes vLLM
VENV_PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"

# Default configuration (overridden by conf/offline_grpo.yaml)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
VLLM_PORT = 30001
VLLM_HOST = "127.0.0.1"
MAX_MODEL_LEN = 2048


def check_vllm_ready(host: str, port: int, timeout: float = 180.0) -> bool:
    """Poll vLLM server until ready or timeout."""
    url = f"http://{host}:{port}/v1/models"
    start = time.time()
    
    logger.info("Checking vLLM server readiness", extra={"url": url, "timeout": timeout})
    
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=3.0)
            if resp.status_code == 200:
                logger.info("vLLM server is ready", extra={"url": url})
                return True
        except requests.exceptions.RequestException:
            pass
        
        elapsed = int(time.time() - start)
        if elapsed % 10 == 0 and elapsed > 0:
            logger.info("Waiting for vLLM server", extra={"elapsed_seconds": elapsed})
        
        time.sleep(2)
    
    logger.error("vLLM server not ready after timeout", extra={"timeout": timeout})
    return False


def _load_gpu_config() -> tuple[str, int | None, float]:
    """Load GPU configuration from offline_grpo.yaml.
    
    Returns:
        Tuple of (strategy, vllm_gpu_id, vllm_memory_util)
    """
    try:
        import yaml
        config_path = SCRIPT_DIR / "conf" / "offline_grpo.yaml"
        logger.info("Loading GPU configuration", extra={"config_path": str(config_path)})
        
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        strategy = cfg.get("gpu", {}).get("strategy", "multi").lower()
        if strategy == "multi":
            vllm_gpu = cfg.get("gpu", {}).get("vllm_gpu", 1)
            vllm_mem = cfg.get("gpu", {}).get("vllm_gpu_memory_utilization", 0.85)
        else:
            vllm_gpu = None
            vllm_mem = cfg.get("gpu", {}).get("vllm_gpu_memory_utilization_single", 0.25)
        
        logger.info(
            "GPU configuration loaded",
            extra={"strategy": strategy, "vllm_gpu": vllm_gpu, "vllm_memory_util": vllm_mem},
        )
        return strategy, vllm_gpu, vllm_mem
    except Exception as e:
        logger.warning("Failed to load GPU config, using defaults", extra={"error": str(e)})
        return "multi", 1, 0.85


def main() -> int:
    """Main entry point."""
    # Load GPU configuration
    gpu_strategy, vllm_gpu_id, vllm_mem_util = _load_gpu_config()
    
    logger.info("=" * 80)
    logger.info("GRAIL Offline Trainer - GSM8K Environment")
    logger.info("=" * 80)
    logger.info(
        "Configuration loaded",
        extra={
            "gpu_strategy": gpu_strategy,
            "vllm_gpu": vllm_gpu_id,
            "model": MODEL_ID,
            "vllm_server": f"http://{VLLM_HOST}:{VLLM_PORT}",
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": vllm_mem_util,
        },
    )
    
    # Check if venv Python exists
    if not VENV_PYTHON.exists():
        logger.error(
            "Virtual environment not found",
            extra={"venv_python": str(VENV_PYTHON), "hint": "Run: cd scripts/offline_trainer && uv sync"},
        )
        return 1
    
    # Start vLLM server using the offline trainer's venv
    logger.info("Starting vLLM server")
    vllm_log = SCRIPT_DIR / "vllm_server.log"
    
    # Set CUDA_VISIBLE_DEVICES for vLLM server if using multi-GPU strategy
    vllm_env = os.environ.copy()
    if gpu_strategy == "multi" and vllm_gpu_id is not None:
        vllm_env["CUDA_VISIBLE_DEVICES"] = str(vllm_gpu_id)
        logger.info("vLLM using dedicated GPU", extra={"gpu_id": vllm_gpu_id})
    
    vllm_cmd = [
        str(VENV_PYTHON),
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", str(vllm_mem_util),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", "32",
        "--trust-remote-code",
    ]
    
    with open(vllm_log, "w") as f:
        vllm_proc = subprocess.Popen(
            vllm_cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(SCRIPT_DIR),
            env=vllm_env,
        )
    
    logger.info("vLLM server started", extra={"pid": vllm_proc.pid, "log_file": str(vllm_log)})
    
    # Wait for server to be ready
    if not check_vllm_ready(VLLM_HOST, VLLM_PORT):
        logger.error("Failed to start vLLM server", extra={"log_file": str(vllm_log)})
        vllm_proc.kill()
        return 1
    
    # Run offline trainer
    logger.info("=" * 80)
    logger.info("Starting Offline Training")
    logger.info("=" * 80)
    
    try:
        # Run training script using uv from the offline_trainer directory
        # This ensures we use the local .venv with vLLM and grail dependencies
        result = subprocess.run(
            ["uv", "run", "python", "run_offline_grpo.py"],
            cwd=str(SCRIPT_DIR),
            check=False,
        )
        exit_code = result.returncode
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        exit_code = 130
    finally:
        # Cleanup
        logger.info("Stopping vLLM server")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=10)
            logger.info("vLLM server stopped gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Force killing vLLM server")
            vllm_proc.kill()
            vllm_proc.wait()
            logger.info("vLLM server killed")
    
    if exit_code == 0:
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
    else:
        logger.error("Training failed", extra={"exit_code": exit_code})
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

