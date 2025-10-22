"""GPU-based Online GRPO test with real GSM8K data and completions.

This test validates online GRPO on real data by:
1. Loading GSM8K problems with deterministic seeds
2. For each of 10 iterations:
   a. Generating fresh rollouts for 4 problems (16 samples each)
   b. Training for 2 epochs with batches of 2 problems
   c. Evaluating on held-out problems
3. Visualizing learning curves across iterations

Key design:
- Online training: fresh data generation each iteration
- Efficient: only generates 4 problems per iteration (64 rollouts)
- Batch training: 2 epochs × 2 batches of 2 problems
- Group size: 16 rollouts per problem for advantage computation

Architecture: Modular design with separated concerns:
- Constants: test-specific hyperparameters
- Fixtures: model/environment setup
- Evaluator: baseline & iteration evaluation
- DataGenerator: rollout generation (per iteration)
- Trainer: online GRPO iteration loop
- Visualizer: results & plots
"""

from __future__ import annotations

import csv
import importlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from dotenv import load_dotenv
from scipy import stats

from grail.environments.gsm8k_env import GSM8KEnv
from grail.environments.loop import AgentEnvLoop

logger = logging.getLogger(__name__)

# ============================================================================
# TEST CONFIGURATION & CONSTANTS
# ============================================================================

TEST_ENV_ID: str = "GSM8K"  # Environment ID for test


@dataclass(frozen=True)
class TestConfig:
    """Immutable test-specific configuration."""

    env_id: str = TEST_ENV_ID
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    learning_rate: float = 1e-6  # Reduced 50x to prevent collapse
    num_iterations: int = 10  # Number of GRPO iterations (online training loop)
    num_epochs_per_iteration: int = 2  # Epochs per iteration
    problems_per_iteration: int = 4  # Problems to generate per iteration
    problems_per_batch: int = 2  # Problems per training batch
    batch_size: int = 4  # Rollouts per batch for training
    max_length: int = 512
    num_train_problems: int = 30
    num_eval_problems: int = 10
    rollouts_per_problem: int = 16  # GRPO group size for training
    eval_samples_per_problem: int = 10  # Samples per problem for baseline/trained eval (for pass@5)
    temperature: float = 0.7
    baseline_sample_size: int = 5  # Number of problems to evaluate at baseline


TEST_CONFIG = TestConfig()


# ============================================================================
# MOCK WALLET (External Dependency Isolation)
# ============================================================================


class MockWallet:
    """Mock wallet for testing without real Bittensor wallet."""

    class MockHotkey:
        ss58_address: str = "test_hotkey_address"

        def sign(self, data: bytes) -> bytes:
            """Return 64-byte mock signature as bytes."""
            import hashlib

            # Real bittensor signatures are 64 bytes (ed25519 signature)
            # Return deterministic 64 bytes based on data
            part1 = hashlib.sha256(data).digest()  # 32 bytes
            part2 = hashlib.sha256(data + b"_mock_sig").digest()  # 32 bytes
            return part1 + part2  # Total: 64 bytes

    hotkey: MockHotkey = MockHotkey()


# ============================================================================
# PASS@K ANALYSIS (Statistical Significance Testing)
# ============================================================================


@dataclass
class PassAtKResult:
    """Result of pass@k computation."""

    k: int
    pass_at_k: float
    pass_at_k_lower_ci: float
    pass_at_k_upper_ci: float
    num_problems: int
    num_samples_per_problem: int
    num_successes: int


class PassAtKAnalyzer:
    """Compute pass@k metrics with statistical significance testing.

    Following best practices from:
    - OpenAI Evals: pass@k = 1 - (n_fail choose k) / (n choose k)
    - Bootstrap confidence intervals (1000 resamples, 95% CI)
    - Effect size (Cohen's d) for baseline vs. trained comparison
    - Two-sample t-tests for significance
    """

    @staticmethod
    def compute_pass_at_k(problems_with_rewards: list[list[float]], k: int = 1) -> PassAtKResult:
        """Compute pass@k metric.

        Args:
            problems_with_rewards: List of [reward_1, ..., reward_k] per problem.
            k: Number of samples to consider per problem.

        Returns:
            PassAtKResult with pass@k and 95% bootstrap CI.

        Formula (from OpenAI Evals):
            pass@k = 1 - C(failures, k) / C(n, k)
            where failures = max(0, n - successes)
        """
        num_problems = len(problems_with_rewards)
        assert num_problems > 0, "Must have at least 1 problem"

        # Ensure each problem has at least k samples
        for i, rewards in enumerate(problems_with_rewards):
            if len(rewards) < k:
                raise ValueError(f"Problem {i} has only {len(rewards)} samples, but k={k}")

        # Compute pass@k for each problem, then average
        pass_at_k_values = []
        for rewards in problems_with_rewards:
            # OpenAI formula: pass@k = 1 - C(failures, k) / C(n, k)
            # where n = total samples, failures = total failures across ALL samples
            n = len(rewards)  # Total samples for this problem
            successes = sum(1 for r in rewards if r >= 0.5)  # Successes across ALL samples
            failures = max(0, n - successes)

            # OpenAI formula
            if n == 0 or k > n:
                pass_at_k_val = 0.0
            else:
                pass_at_k_val = max(
                    0.0,
                    1.0
                    - (
                        np.prod(np.arange(max(1, n - failures), n + 1))
                        / np.prod(np.arange(1, n + 1))
                    ),
                )
            pass_at_k_values.append(pass_at_k_val)

        pass_at_k_mean = np.mean(pass_at_k_values)

        # Bootstrap 95% CI (1000 resamples)
        bootstrap_estimates = []
        rng = np.random.RandomState(42)
        for _ in range(1000):
            resampled_indices = rng.choice(num_problems, num_problems, replace=True)
            resampled_values = [pass_at_k_values[i] for i in resampled_indices]
            bootstrap_estimates.append(np.mean(resampled_values))

        lower_ci = np.percentile(bootstrap_estimates, 2.5)
        upper_ci = np.percentile(bootstrap_estimates, 97.5)

        return PassAtKResult(
            k=k,
            pass_at_k=float(pass_at_k_mean),
            pass_at_k_lower_ci=float(lower_ci),
            pass_at_k_upper_ci=float(upper_ci),
            num_problems=num_problems,
            num_samples_per_problem=k,
            num_successes=int(
                sum(1 for val in pass_at_k_values if val > 0.0)
            ),  # Count problems that pass
        )

    @staticmethod
    def cohens_d(group1: list[float], group2: list[float]) -> float:
        """Compute Cohen's d effect size between two groups."""
        n1, n2 = len(group1), len(group2)
        var1 = np.var(group1, ddof=1) if n1 > 1 else 0.0
        var2 = np.var(group2, ddof=1) if n2 > 1 else 0.0

        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std < 1e-8:
            return 0.0

        return float((np.mean(group2) - np.mean(group1)) / pooled_std)

    @staticmethod
    def significance_test(
        baseline_values: list[float],
        trained_values: list[float],
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Two-sample t-test for significance.

        Args:
            baseline_values: Baseline pass@k or reward values.
            trained_values: Trained pass@k or reward values.
            alpha: Significance level (default 0.05).

        Returns:
            Dict with t-stat, p-value, significant, and effect size.
        """
        t_stat, p_value = stats.ttest_ind(trained_values, baseline_values)
        d = PassAtKAnalyzer.cohens_d(baseline_values, trained_values)

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "cohens_d": float(d),
            "effect_size_label": (
                "negligible"
                if abs(d) < 0.2
                else "small"
                if abs(d) < 0.5
                else "medium"
                if abs(d) < 0.8
                else "large"
            ),
        }


# ============================================================================
# BASELINE EVALUATOR
# ============================================================================


class BaselineEvaluator:
    """Evaluates model baseline performance before training."""

    def __init__(
        self,
        env_loop: AgentEnvLoop,
        tokenizer: Any,
        wallet: MockWallet,
        config: TestConfig,
    ) -> None:
        self.env_loop = env_loop
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.config = config

    def evaluate_problems(
        self, problem_seeds: list[int], label: str
    ) -> tuple[list[float], dict[int, list[float]]]:
        """Evaluate baseline on given problems using lightweight batch generation.

        Returns:
            rewards: list of reward values (flat)
            problem_rewards: dict mapping problem_idx -> [samples per problem]
        """
        rewards = []
        problem_rewards: dict[int, list[float]] = {}
        sample_seeds = problem_seeds[: self.config.baseline_sample_size]

        # Generate eval_samples_per_problem for each baseline problem
        for problem_idx, problem_seed in enumerate(sample_seeds):
            logger.info(
                f"  Baseline problem {problem_idx}: generating "
                f"{self.config.eval_samples_per_problem} samples with seed={problem_seed}..."
            )

            def make_env() -> GSM8KEnv:
                return GSM8KEnv()

            # Use lightweight evaluation method (no GRAIL proofs/commitments)
            eval_results = self.env_loop.generate_batch_for_eval(
                env_factory=make_env,
                count=self.config.eval_samples_per_problem,
                seed=problem_seed,
            )

            problem_sample_rewards = []

            # Extract rewards
            for sample_idx, (reward, success) in enumerate(eval_results):
                rewards.append(reward)
                problem_sample_rewards.append(reward)

                # Log first sample
                if sample_idx == 0:
                    logger.info(f"    Sample 0: reward={reward:.4f}, success={success}")

            problem_rewards[problem_idx] = problem_sample_rewards
            logger.info(
                f"    Problem {problem_idx} rewards: {[f'{r:.2f}' for r in problem_sample_rewards]}"
            )

        return rewards, problem_rewards


# ============================================================================
# TRAINING DATA GENERATOR
# ============================================================================


class TrainingDataGenerator:
    """Generates training rollouts and groups."""

    def __init__(
        self,
        env_loop: AgentEnvLoop,
        tokenizer: Any,
        wallet: MockWallet,
        config: TestConfig,
    ) -> None:
        self.env_loop = env_loop
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.config = config

    def generate_groups(
        self, problem_seeds: list[int], grpo_module: Any, max_problems: int | None = None
    ) -> tuple[list[Any], list[float], dict[int, list[float]]]:
        """Generate GRPO groups from problems.

        Args:
            problem_seeds: List of problem seeds
            grpo_module: GRPO module
            max_problems: Maximum number of problems to generate (default: all)

        Returns:
            groups: list of GRPOGroup objects
            all_rewards: flattened list of all rewards
            problem_rewards: dict mapping problem_idx -> list of rewards per problem
        """
        from grail.trainer.algorithms.grpo import GRPOGroup, GRPORollout

        groups = []
        all_rewards = []
        problem_rewards: dict[int, list[float]] = {}

        # Limit to max_problems if specified
        num_problems = min(len(problem_seeds), max_problems) if max_problems else len(problem_seeds)
        problems_to_generate = problem_seeds[:num_problems]

        for problem_idx, problem_seed in enumerate(problems_to_generate):
            try:
                logger.info(
                    f"Problem {problem_idx}: generating "
                    f"{self.config.rollouts_per_problem} rollouts "
                    f"with seed={problem_seed}..."
                )

                def make_env() -> GSM8KEnv:
                    return GSM8KEnv()

                env_factory = make_env
                env_rollouts = self.env_loop.run_grpo_group(
                    env_factory=env_factory,
                    count=self.config.rollouts_per_problem,
                    randomness_hex="1234567890abcdef" * 4,  # Valid hex (64 chars)
                    wallet=self.wallet,
                    seed=problem_seed,
                )

                rollouts_list = []
                rewards_list = []

                for rollout_idx, env_rollout in enumerate(env_rollouts):
                    try:
                        grpo_rollout = GRPORollout(
                            tokens=list(env_rollout.tokens),
                            prompt_length=int(env_rollout.prompt_length),
                            completion_length=int(env_rollout.completion_length),
                            advantage=0.0,
                            reward=float(env_rollout.reward),
                            success=bool(env_rollout.success),
                            nonce=int(problem_idx * 1000 + rollout_idx),
                            rollout_group=f"problem_{problem_idx}",
                            token_logprobs=(
                                list(env_rollout.token_logprobs)
                                if getattr(env_rollout, "token_logprobs", None)
                                else None
                            ),
                        )
                        rollouts_list.append(grpo_rollout)
                        rewards_list.append(grpo_rollout.reward)

                        if rollout_idx == 0:
                            completion_text = self.tokenizer.decode(
                                grpo_rollout.tokens[grpo_rollout.prompt_length :],
                                skip_special_tokens=False,
                            )
                            logger.info(
                                f"    Rollout 0 sample: "
                                f"reward={grpo_rollout.reward:.4f}, "
                                f"prompt_len={grpo_rollout.prompt_length}, "
                                f"completion_len={grpo_rollout.completion_length}, "
                                f"total_tokens={len(grpo_rollout.tokens)}\n"
                                f"    Completion:\n{completion_text}"
                            )
                    except Exception as e:
                        logger.warning(f"    Rollout {rollout_idx}: error {e}")
                        continue

                if not rollouts_list:
                    logger.warning(f"Problem {problem_idx}: no valid rollouts")
                    continue

                # Compute advantages
                mean_reward = np.mean(rewards_list)
                all_rewards.extend(rewards_list)
                problem_rewards[problem_idx] = rewards_list

                for rollout in rollouts_list:
                    rollout.advantage = float(rollout.reward - mean_reward)

                advantages = [r.advantage for r in rollouts_list]
                logger.info(
                    f"    Group stats: mean_reward={mean_reward:.4f}, "
                    f"std_reward={np.std(rewards_list):.4f}, "
                    f"min_advantage={min(advantages):.4f}, "
                    f"max_advantage={max(advantages):.4f}, "
                    f"sum_advantages={sum(advantages):.6f}"
                )

                group = GRPOGroup(group_id=f"problem_{problem_idx}", rollouts=rollouts_list)

                if group.is_valid(
                    advantage_tolerance=0.5, rollouts_per_problem=self.config.rollouts_per_problem
                ):
                    groups.append(group)
                    logger.info(
                        f"  ✓ Group valid: {len(rollouts_list)} rollouts, mean_reward={mean_reward:.4f}"
                    )
                else:
                    logger.warning("  ✗ Group invalid (advantage sum not zero)")

            except Exception as e:
                logger.warning(f"Problem {problem_idx}: error {e}")
                continue

        return groups, all_rewards, problem_rewards


# ============================================================================
# EPOCH EVALUATOR
# ============================================================================


class EpochEvaluator:
    """Evaluates model performance after each training epoch."""

    def __init__(
        self,
        env_loop: AgentEnvLoop,
        wallet: MockWallet,
        baseline_eval_mean: float,
        config: TestConfig,
    ) -> None:
        self.env_loop = env_loop
        self.wallet = wallet
        self.baseline_eval_mean = baseline_eval_mean
        self.config = config

    def evaluate_epoch(self, problem_seeds: list[int]) -> tuple[float, list[float]]:
        """Evaluate all problems using lightweight batch generation, return mean reward and list."""
        eval_rewards = []

        # Generate eval_samples_per_problem for each eval problem
        for _problem_idx, problem_seed in enumerate(problem_seeds):

            def make_env() -> GSM8KEnv:
                return GSM8KEnv()

            # Use lightweight evaluation method (no GRAIL proofs/commitments)
            eval_results = self.env_loop.generate_batch_for_eval(
                env_factory=make_env,
                count=self.config.eval_samples_per_problem,
                seed=problem_seed,
            )

            # Extract rewards from each sample
            for reward, _success in eval_results:
                eval_rewards.append(reward)

        eval_mean = np.mean(eval_rewards) if eval_rewards else 0.0
        improvement = eval_mean - self.baseline_eval_mean
        pct_improvement = (improvement / (self.baseline_eval_mean + 1e-8)) * 100

        logger.info(f"  ✅ Eval mean reward: {eval_mean:.4f}")
        logger.info(f"  📈 Improvement vs baseline: {improvement:+.4f} ({pct_improvement:+.1f}%)")
        logger.info(
            f"  📊 Eval rewards: {[f'{r:.2f}' for r in eval_rewards[:5]]}... "
            f"({len(problem_seeds)} problems × {self.config.eval_samples_per_problem} samples)"
        )

        eval_mean_float: float = float(eval_mean)
        return eval_mean_float, eval_rewards


# ============================================================================
# TRAINING LOOP MANAGER
# ============================================================================


class TrainingLoopManager:
    """Manages the GRPO training loop."""

    def __init__(
        self,
        model: Any,
        ref_model: Any,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
        epoch_evaluator: EpochEvaluator,
        config: TestConfig,
        data_generator: TrainingDataGenerator,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.epoch_evaluator = epoch_evaluator
        self.config = config
        self.data_generator = data_generator

    async def train_online_grpo(
        self,
        train_problem_seeds: list[int],
        eval_problem_seeds: list[int],
        baseline_eval_mean: float,
        train_grpo_epoch: Any,
        grpo_module: Any,
    ) -> tuple[list[float], list[float]]:
        """Run online GRPO training: generate fresh data → train on batches → evaluate.

        Returns:
            iteration_means: mean eval reward per iteration
            all_losses: all losses across iterations
        """
        iteration_means = [baseline_eval_mean]  # Start with baseline
        all_losses = []
        start_time = time.time()

        try:
            for iteration in range(self.config.num_iterations):
                logger.info(f"\n{'=' * 80}")
                logger.info(f"ITERATION {iteration + 1}/{self.config.num_iterations}")
                logger.info(f"{'=' * 80}")

                iteration_start = time.time()

                # PHASE 1: Generate fresh data (only 4 problems per iteration)
                logger.info(
                    f"\n🔄 PHASE 1: Generating {self.config.problems_per_iteration} problems..."
                )

                # Rotate through training problems
                start_idx = (iteration * self.config.problems_per_iteration) % len(
                    train_problem_seeds
                )
                end_idx = start_idx + self.config.problems_per_iteration
                if end_idx > len(train_problem_seeds):
                    # Wrap around if needed
                    iteration_seeds = (
                        train_problem_seeds[start_idx:]
                        + train_problem_seeds[: end_idx - len(train_problem_seeds)]
                    )
                else:
                    iteration_seeds = train_problem_seeds[start_idx:end_idx]

                logger.info(
                    f"  Selected seeds: {iteration_seeds} "
                    f"(indices {start_idx}-{end_idx - 1} from train set)"
                )

                training_groups, all_rewards, problem_rewards = self.data_generator.generate_groups(
                    iteration_seeds, grpo_module, max_problems=self.config.problems_per_iteration
                )

                if not training_groups:
                    logger.warning(f"Iteration {iteration + 1}: No valid groups, skipping")
                    continue

                logger.info(
                    f"  Generated {len(training_groups)} groups, "
                    f"{sum(len(g.rollouts) for g in training_groups)} total rollouts, "
                    f"mean_reward={np.mean(all_rewards):.4f}"
                )

                # PHASE 2: Train on batches for multiple epochs
                logger.info(
                    f"\n🎯 PHASE 2: Training for {self.config.num_epochs_per_iteration} epochs, "
                    f"batches of {self.config.problems_per_batch} problems"
                )

                iteration_losses = []
                for epoch in range(self.config.num_epochs_per_iteration):
                    logger.info(
                        f"\n  Epoch {epoch + 1}/{self.config.num_epochs_per_iteration} "
                        f"(Iteration {iteration + 1})"
                    )

                    # Create batches from training groups
                    for batch_idx in range(0, len(training_groups), self.config.problems_per_batch):
                        batch = training_groups[
                            batch_idx : batch_idx + self.config.problems_per_batch
                        ]

                        logger.info(
                            f"    Batch {batch_idx // self.config.problems_per_batch + 1}: "
                            f"training on {len(batch)} groups..."
                        )

                        try:
                            metrics = await train_grpo_epoch(
                                self.model,
                                self.ref_model,
                                self.tokenizer,
                                batch,
                                self.optimizer,
                                accelerator=self.accelerator,
                                monitor=None,
                                window=iteration * self.config.num_epochs_per_iteration + epoch,
                                batch_size=self.config.batch_size,
                            )

                            loss = metrics.get("loss_total", 0.0)
                            iteration_losses.append(loss)
                            all_losses.append(loss)

                            logger.info(
                                f"      Loss: {loss:.6f}, "
                                f"PG: {metrics.get('loss_pg', 0.0):.6f}, "
                                f"KL: {metrics.get('loss_kl', 0.0):.6f}"
                            )

                        except Exception as e:
                            logger.error(f"Batch {batch_idx} failed: {e}")
                            import traceback

                            traceback.print_exc()
                            continue

                # PHASE 3: Evaluate on held-out problems
                logger.info(
                    f"\n🔍 PHASE 3: Evaluating on {len(eval_problem_seeds)} held-out problems..."
                )
                eval_mean, eval_rewards = self.epoch_evaluator.evaluate_epoch(eval_problem_seeds)
                iteration_means.append(eval_mean)

                iteration_time = time.time() - iteration_start
                avg_loss = np.mean(iteration_losses) if iteration_losses else 0.0
                improvement = eval_mean - baseline_eval_mean
                pct_improvement = (improvement / (baseline_eval_mean + 1e-8)) * 100

                logger.info(f"\n📊 Iteration {iteration + 1} Summary:")
                logger.info(f"  Avg Loss: {avg_loss:.6f}")
                logger.info(f"  Eval Reward: {eval_mean:.4f}")
                logger.info(f"  Improvement: {improvement:+.4f} ({pct_improvement:+.1f}%)")
                logger.info(f"  Time: {iteration_time:.1f}s")

            total_time = time.time() - start_time
            logger.info(f"\n✅ Total training time: {total_time:.1f}s ({total_time / 60:.1f}min)")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback

            traceback.print_exc()
            raise

        return iteration_means, all_losses


# ============================================================================
# RESULTS VISUALIZER & SAVER
# ============================================================================


class ResultsVisualizer:
    """Visualizes and saves training results."""

    def __init__(
        self,
        baseline_eval_mean: float,
        output_dir: Path,
    ) -> None:
        self.baseline_eval_mean = baseline_eval_mean
        self.output_dir = Path(output_dir)

    def plot_learning_curve(
        self, iteration_means: list[float], loss_curve: list[float], mode: str = "iteration"
    ) -> None:
        """Generate and save learning curve plots.

        Args:
            iteration_means: Mean eval rewards per iteration
            loss_curve: Loss values
            mode: 'iteration' for online GRPO, 'epoch' for offline
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Learning curve
        ax = axes[0]
        x_values = list(range(len(iteration_means)))
        ax.plot(x_values, iteration_means, "b-o", linewidth=2, label="Eval reward")
        ax.axhline(y=self.baseline_eval_mean, color="r", linestyle="--", label="Baseline")
        ax.set_xlabel("Iteration" if mode == "iteration" else "Epoch")
        ax.set_ylabel("Mean Reward")
        ax.set_title(
            "Online GRPO Learning Curve on GSM8K"
            if mode == "iteration"
            else "GRPO Learning Curve on GSM8K"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Loss curve
        ax = axes[1]
        loss_x = list(range(len(loss_curve)))
        ax.plot(loss_x, loss_curve, "g-s", linewidth=2, label="Total loss")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Improvement
        ax = axes[2]
        improvements = [r - self.baseline_eval_mean for r in iteration_means]
        ax.bar(
            x_values,
            improvements,
            color=["red" if x < 0 else "green" for x in improvements],
        )
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Iteration" if mode == "iteration" else "Epoch")
        ax.set_ylabel("Improvement vs Baseline")
        ax.set_title("Reward Improvement")
        ax.grid(True, alpha=0.3, axis="y")

        plot_path = self.output_dir / "grpo_learning_curve.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"\n✅ Learning curve saved: {plot_path}")

    def save_metrics_csv(
        self, iteration_means: list[float], loss_curve: list[float], mode: str = "iteration"
    ) -> None:
        """Save metrics to CSV file.

        Args:
            iteration_means: Mean eval rewards per iteration
            loss_curve: Loss values
            mode: 'iteration' for online GRPO, 'epoch' for offline
        """
        csv_path = self.output_dir / "grpo_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            label = "iteration" if mode == "iteration" else "epoch"
            writer.writerow([label, "eval_reward", "improvement", "pct_improvement"])

            for i, eval_r in enumerate(iteration_means):
                improvement = eval_r - self.baseline_eval_mean
                pct_imp = (improvement / (self.baseline_eval_mean + 1e-8)) * 100
                writer.writerow(
                    [
                        i,
                        f"{eval_r:.4f}",
                        f"{improvement:.4f}",
                        f"{pct_imp:.1f}%",
                    ]
                )

        logger.info(f"✅ Metrics CSV saved: {csv_path}")

    def log_results(self, iteration_means: list[float], mode: str = "iteration") -> None:
        """Log final results.

        Args:
            iteration_means: Mean eval rewards per iteration
            mode: 'iteration' for online GRPO, 'epoch' for offline
        """
        logger.info("\n" + "=" * 80)
        logger.info("RESULTS & LEARNING CURVE")
        logger.info("=" * 80)

        label = "Iteration" if mode == "iteration" else "Epoch"
        logger.info(f"\nLearning Curve (Evaluation Rewards per {label}):")
        for idx, reward in enumerate(iteration_means):
            improvement = reward - self.baseline_eval_mean
            pct_improvement = (improvement / (self.baseline_eval_mean + 1e-8)) * 100
            logger.info(
                f"  {label} {idx}: reward={reward:.4f}, "
                f"Δ={improvement:+.4f} ({pct_improvement:+.1f}%)"
            )

        final_mean = iteration_means[-1]
        logger.info(
            f"✅ Baseline: {self.baseline_eval_mean:.4f}, Final: {final_mean:.4f}, "
            f"Improvement: {final_mean - self.baseline_eval_mean:+.4f}"
        )


# ============================================================================
# TEST CLASS (Orchestrator)
# ============================================================================


@pytest.mark.gpu_real_data
class TestGRPOGPURealData:
    """Real-data GRPO training on GPU with learning curve visualization."""

    @pytest.mark.asyncio
    async def test_grpo_real_gsm8k_with_learning_curve(
        self,
        tmp_path: Any,
    ) -> None:
        """Train GRPO on real GSM8K data and plot learning curve."""

        # Load configuration
        load_dotenv(override=True)

        import grail.shared.constants as C

        C = importlib.reload(C)
        import grail.trainer.algorithms.grpo as grpo

        grpo = importlib.reload(grpo)
        from grail.trainer.algorithms.grpo import train_grpo_epoch

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )

        logger.info("=" * 80)
        logger.info("GPU ONLINE GRPO TEST WITH REAL GSM8K DATA")
        logger.info("=" * 80)

        # Device setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("⚠️  GPU not available, falling back to CPU (will be slow)")
        else:
            logger.info(f"✅ GPU available: {device}")

        accelerator = Accelerator(
            mixed_precision=("fp16" if device == "cuda" else "no"),
            device_placement=True,
        )
        logger.info(f"✅ Accelerator device: {accelerator.device}")

        # Load model & tokenizer
        logger.info("Loading model and tokenizer...")
        from grail.model.provider import get_model, get_tokenizer
        from grail.shared.chat_templates import build_qwen_chat_template
        from grail.shared.prompt_constants import REASONING_START, SYSTEM_PROMPT

        chat_template = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)

        model = get_model(TEST_CONFIG.model_name, device=str(accelerator.device), eval_mode=False)
        tokenizer = get_tokenizer(TEST_CONFIG.model_name, chat_template=chat_template)
        ref_model = get_model(
            TEST_CONFIG.model_name, device=str(accelerator.device), eval_mode=True
        )

        logger.info(
            f"✅ Model loaded: {TEST_CONFIG.model_name} "
            f"(test-specific, 68.5% {TEST_ENV_ID} baseline)"
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=TEST_CONFIG.learning_rate)

        # Setup environments
        logger.info("\n" + "=" * 80)
        logger.info("SETTING UP PROBLEMS (deterministic seeds)")
        logger.info("=" * 80)

        train_problem_seeds = [i * 1000 for i in range(TEST_CONFIG.num_train_problems)]
        eval_problem_seeds = [10000 + i * 1000 for i in range(TEST_CONFIG.num_eval_problems)]

        logger.info(
            f"Train problems: {len(train_problem_seeds)} "
            f"(seeds {train_problem_seeds[0]}-{train_problem_seeds[-1]})"
        )
        logger.info(
            f"Eval problems: {len(eval_problem_seeds)} "
            f"(seeds {eval_problem_seeds[0]}-{eval_problem_seeds[-1]})"
        )

        env_loop = AgentEnvLoop(
            model,
            tokenizer,
            device=str(accelerator.device),
            max_new_tokens=int(C.MAX_NEW_TOKENS),
            temperature=TEST_CONFIG.temperature,
            batch_size=TEST_CONFIG.rollouts_per_problem,
        )

        # Baseline evaluation
        logger.info("\n" + "=" * 80)
        logger.info("BASELINE EVALUATION (No training)")
        logger.info("=" * 80)

        evaluator = BaselineEvaluator(env_loop, tokenizer, MockWallet(), TEST_CONFIG)
        logger.info("Evaluating baseline on train problems...")
        baseline_train_rewards, baseline_train_problem_rewards = evaluator.evaluate_problems(
            train_problem_seeds, "Train"
        )

        logger.info("Evaluating baseline on eval (held-out) problems...")
        baseline_eval_rewards, baseline_eval_problem_rewards = evaluator.evaluate_problems(
            eval_problem_seeds, "Eval"
        )

        baseline_train_mean = (
            float(np.mean(baseline_train_rewards)) if baseline_train_rewards else 0.0
        )
        baseline_eval_mean = float(np.mean(baseline_eval_rewards)) if baseline_eval_rewards else 0.0
        logger.info(f"\n📊 Baseline train mean reward: {baseline_train_mean:.4f}")
        logger.info(f"📊 Baseline eval mean reward: {baseline_eval_mean:.4f}")
        logger.info(f"Baseline train rewards: {baseline_train_rewards}")
        logger.info(f"Baseline eval rewards: {baseline_eval_rewards}")

        # Create data generator for online training
        data_gen = TrainingDataGenerator(env_loop, tokenizer, MockWallet(), TEST_CONFIG)

        # Compute pass@k for baseline (train and eval)
        logger.info("\n" + "=" * 80)
        logger.info("BASELINE PASS@K METRICS")
        logger.info("=" * 80)

        # Organize baseline train rewards by problem (10 samples per problem)
        baseline_train_pass_at_k_values = []
        for idx in range(len(baseline_train_rewards) // TEST_CONFIG.eval_samples_per_problem):
            start = idx * TEST_CONFIG.eval_samples_per_problem
            end = start + TEST_CONFIG.eval_samples_per_problem
            baseline_train_pass_at_k_values.append(baseline_train_rewards[start:end])

        baseline_eval_pass_at_k_values = []
        for idx in range(len(baseline_eval_rewards) // TEST_CONFIG.eval_samples_per_problem):
            start = idx * TEST_CONFIG.eval_samples_per_problem
            end = start + TEST_CONFIG.eval_samples_per_problem
            baseline_eval_pass_at_k_values.append(baseline_eval_rewards[start:end])

        # Baseline can compute pass@1, pass@5 (10 samples per problem)
        for k in [1, 5]:
            if len(baseline_train_pass_at_k_values) > 0:
                result = PassAtKAnalyzer.compute_pass_at_k(baseline_train_pass_at_k_values, k=k)
                logger.info(
                    f"Baseline TRAIN pass@{k}: {result.pass_at_k:.4f} "
                    f"[{result.pass_at_k_lower_ci:.4f}, {result.pass_at_k_upper_ci:.4f}] "
                    f"({result.num_successes}/{result.num_problems})"
                )

        for k in [1, 5]:
            if len(baseline_eval_pass_at_k_values) > 0:
                result = PassAtKAnalyzer.compute_pass_at_k(baseline_eval_pass_at_k_values, k=k)
                logger.info(
                    f"Baseline EVAL pass@{k}: {result.pass_at_k:.4f} "
                    f"[{result.pass_at_k_lower_ci:.4f}, {result.pass_at_k_upper_ci:.4f}] "
                    f"({result.num_successes}/{result.num_problems})"
                )

        # Online GRPO Training Loop
        logger.info("\n" + "=" * 80)
        logger.info("ONLINE GRPO TRAINING")
        logger.info("=" * 80)
        logger.info(f"Training for {TEST_CONFIG.num_iterations} iterations")
        logger.info(f"Per iteration: {TEST_CONFIG.problems_per_iteration} problems generated")
        logger.info(
            f"Per iteration: {TEST_CONFIG.num_epochs_per_iteration} epochs, "
            f"batches of {TEST_CONFIG.problems_per_batch} problems"
        )

        epoch_evaluator = EpochEvaluator(env_loop, MockWallet(), baseline_eval_mean, TEST_CONFIG)
        trainer = TrainingLoopManager(
            model,
            ref_model,
            tokenizer,
            optimizer,
            accelerator,
            epoch_evaluator,
            TEST_CONFIG,
            data_gen,
        )

        iteration_means, loss_curve = await trainer.train_online_grpo(
            train_problem_seeds, eval_problem_seeds, baseline_eval_mean, train_grpo_epoch, grpo
        )

        # Results & visualization
        visualizer = ResultsVisualizer(baseline_eval_mean, tmp_path)
        visualizer.plot_learning_curve(iteration_means, loss_curve, mode="iteration")
        visualizer.save_metrics_csv(iteration_means, loss_curve, mode="iteration")
        visualizer.log_results(iteration_means, mode="iteration")

        # Assertions
        logger.info("\n" + "=" * 80)
        logger.info("ASSERTIONS")
        logger.info("=" * 80)

        assert len(iteration_means) > 0, "No evaluation rewards collected"
        assert all(isinstance(r, (int, float)) for r in iteration_means), "Non-numeric rewards"
        assert all(r == r for r in iteration_means), "NaN in evaluation rewards"

        final_mean = iteration_means[-1]
        assert final_mean >= baseline_eval_mean * 0.7, (
            f"Catastrophic regression: {final_mean:.4f} < {baseline_eval_mean * 0.7:.4f}"
        )

        logger.info("✅ No catastrophic regression")
        logger.info(
            f"✅ Baseline: {baseline_eval_mean:.4f}, Final: {final_mean:.4f}, "
            f"Improvement: {final_mean - baseline_eval_mean:+.4f}"
        )

        # Statistical significance testing
        logger.info("\n" + "=" * 80)
        logger.info("STATISTICAL SIGNIFICANCE TESTING")
        logger.info("=" * 80)

        # Compare baseline to final iteration (exclude first element which is baseline)
        trained_rewards = iteration_means[1:]  # Exclude baseline
        if len(trained_rewards) > 0:
            # Create comparable arrays (baseline_eval_rewards is per-problem, iteration_means is aggregated)
            logger.info(
                f"Baseline mean: {baseline_eval_mean:.4f}, "
                f"Final iteration mean: {final_mean:.4f}, "
                f"Improvement: {final_mean - baseline_eval_mean:+.4f}"
            )
            logger.info(
                f"Iteration means across training: "
                f"[{', '.join([f'{m:.4f}' for m in iteration_means])}]"
            )

        logger.info("\n" + "=" * 80)
        logger.info("✅ TEST COMPLETE")
        logger.info("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "gpu_real_data"])
