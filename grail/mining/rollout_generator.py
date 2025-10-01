"""Generic rollout generation for GRAIL RL environments - GRPO Version."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# Imports for type hints only - actual types are Any in method signatures
import bittensor as bt
import torch

from ..shared.chat_templates import build_qwen_chat_template
from ..shared.constants import MAX_NEW_TOKENS
from ..shared.hf_compat import resolve_hidden_size

logger = logging.getLogger(__name__)


# Qwen-style reasoning/solution tagging and system prompt
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

ADVANTAGE_STD_MIN = 1e-8

SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
    # TODO: replace shorting logic with a better approach later
    "Keep the reasoning succinct (≤25 steps, ≤500 tokens)."
)


@dataclass
class GRPORollout:
    """Single rollout for GRPO training with GRAIL proof support."""

    # Token data for GRAIL proof
    tokens: list[int]  # All tokens (prompt + completion)
    token_logprobs: list[float]  # Logprobs for all tokens

    # Training masks
    prompt_length: int  # Where prompt ends and completion begins
    completion_length: int

    # Rewards and advantages
    reward: float
    advantage: float  # Computed after all rollouts collected

    # Trajectory for analysis
    trajectory: list[tuple[Any, Any, float]]
    success: bool

    # GRAIL proof fields
    commitments: list[dict]  # Activation commitments per position
    signature: bytes
    beacon: dict
    proof_version: str  # Proof version identifier


class RolloutGenerator(ABC):
    """
    Updated base class for GRPO rollout generation.
    Generates multiple rollouts per problem for proper GRPO training.
    """

    def __init__(
        self,
        model: Any,  # AutoModelForCausalLM instance
        tokenizer: Any,  # AutoTokenizer instance
        device: str = "cuda",
        rollouts_per_problem: int = 4,
    ):
        """
        Initialize with support for multiple rollouts.

        Args:
            model: The language model to use for decisions
            tokenizer: The tokenizer for the model
            device: Device to run on (cuda/cpu)
            rollouts_per_problem: Number of rollouts to generate per problem
                (default 4 for GRPO)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.rollouts_per_problem = rollouts_per_problem

        # Inject Qwen-style chat template with system prompt and reasoning
        # start token
        try:
            tpl = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)
            # Set only if not already matching to avoid unnecessary churn
            if getattr(self.tokenizer, "chat_template", None) != tpl:
                self.tokenizer.chat_template = tpl
        except Exception:
            # Non-fatal: fallback to tokenizer default template
            logger.debug("Unable to set custom chat_template; using default.")

    def generate_grpo_rollouts(
        self, problem: Any, randomness_hex: str, wallet: bt.wallet
    ) -> list[GRPORollout]:
        """
        Generate multiple rollouts for GRPO training with GRAIL proofs.

        This is the main entry point that:
        1. Generates multiple rollouts per problem
        2. Collects token-level logprobs
        3. Computes GRPO advantages
        4. Integrates with GRAIL proof system

        Args:
            problem: The problem instance (environment-specific)
            randomness_hex: Hex string for GRAIL proof (from drand/block hash)
            wallet: Bittensor wallet object (bt.wallet) for
                cryptographic signatures

        Returns:
            List of GRPORollout objects with GRPO advantages computed
        """
        rollouts = []

        # TODO: super inefficient! add dynamic batching; vllm-support; etc soon
        for _ in range(self.rollouts_per_problem):
            # Initialize environment
            env = self.create_environment(problem)
            state = self.reset_environment(env)

            # Generate rollout with logprobs tracking
            rollout = self._generate_single_rollout(problem, env, state, randomness_hex, wallet)
            rollouts.append(rollout)

        # Compute GRPO advantages across the group
        rewards = [r.reward for r in rollouts]
        advantages = self._compute_grpo_advantages(rewards)

        # Update advantages in rollouts
        for rollout, advantage in zip(rollouts, advantages):
            rollout.advantage = advantage

        return rollouts

    def _generate_single_rollout(
        self,
        problem: Any,
        env: Any,
        state: Any,
        randomness_hex: str,
        wallet: bt.wallet,
    ) -> GRPORollout:
        """Generate a single rollout with logprob tracking and GRAIL proof."""
        # Create prompt
        prompt = self.create_prompt(problem, env, state, [])

        # Apply chat template if available
        messages = [{"role": "user", "content": prompt}]
        prompt_with_template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokenized = self.tokenizer(
            prompt_with_template,
            return_tensors="pt",
            return_attention_mask=True,
        )

        prompt_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        prompt_length = prompt_ids.shape[1]

        # Generate with logprobs
        with torch.inference_mode():
            outputs = self.model.generate(
                prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.get_max_tokens(),
                temperature=self.get_temperature(),
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Extract generated tokens and logprobs
        all_token_ids = outputs.sequences[0].tolist()
        completion_ids = all_token_ids[prompt_length:]

        # Extract logprobs for generated tokens
        logprobs = self._extract_logprobs(outputs.scores, completion_ids)

        # Full token logprobs (0s for prompt, actual for completion)
        all_logprobs = [0.0] * prompt_length + logprobs

        # Parse and execute actions from generated text
        generated_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        action = self.parse_action(generated_text, env, state)

        # Execute trajectory in environment
        trajectory = []
        total_reward = 0
        done = False

        # Step through environment with parsed action
        next_state, reward, done, info = self.step_environment(env, action)
        trajectory_entry = self.create_trajectory_entry(state, action, reward, info)
        trajectory.append(trajectory_entry)
        total_reward += reward

        if "success" not in info:
            logger.warning("step_environment did not return 'success' in info; defaulting to False")
        success = bool(info.get("success", False))

        # Continue stepping if needed (for multi-step environments)
        # state = next_state
        # while not done and len(trajectory) < self.get_max_tokens():
        #     # For subsequent steps, we might need to generate more actions
        #     # This depends on the environment - for SAT, we usually make
        #     # all assignments at once
        #     break

        # Generate GRAIL proof components
        from ..protocol.grail_verifier import GRAILVerifier
        from ..shared.constants import GRAIL_PROOF_VERSION, LAYER_INDEX

        # Initialize MRS verifier
        hidden_dim = resolve_hidden_size(self.model)
        verifier = GRAILVerifier(hidden_dim=hidden_dim)

        # Generate coefficient vector from randomness
        r_vec = verifier.generate_r_vec(randomness_hex)

        # Compute proof commitments
        commitments = []

        with torch.inference_mode():
            token_tensor = torch.tensor([all_token_ids], dtype=torch.long).to(self.device)
            model_outputs = self.model(token_tensor, output_hidden_states=True)
            # Get hidden states at target layer
            h_layer = model_outputs.hidden_states[LAYER_INDEX][0]

            for pos in range(len(all_token_ids)):
                if pos < h_layer.size(0):
                    commitment = verifier.create_commitment(h_layer[pos], r_vec, pos)
                    commitments.append(commitment)

        # Sign commit binding (not used in payload, just stored in rollout for reference)
        # The actual signature used for validation is created in mine.py package_rollout_data
        import hashlib
        import json

        commitment_data = json.dumps(commitments, sort_keys=True)
        commitment_hash = hashlib.sha256(commitment_data.encode()).digest()
        signature = wallet.hotkey.sign(commitment_hash)  # Placeholder, real sig in mine.py

        return GRPORollout(
            tokens=all_token_ids,
            token_logprobs=all_logprobs,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            reward=total_reward,
            advantage=0.0,  # Will be set after all rollouts generated
            trajectory=trajectory,
            success=success,
            commitments=commitments,
            signature=signature,
            beacon={"randomness": randomness_hex},
            proof_version=GRAIL_PROOF_VERSION,
        )

    def _extract_logprobs(self, scores: list[torch.Tensor], token_ids: list[int]) -> list[float]:
        """Extract log probabilities for generated tokens."""
        logprobs = []
        for i, token_id in enumerate(token_ids):
            if i < len(scores):
                score_dist = torch.softmax(scores[i][0], dim=-1)
                token_logprob = torch.log(score_dist[token_id]).item()
                logprobs.append(token_logprob)
            else:
                logprobs.append(0.0)
        return logprobs

    # TODO: optimize with torch tensors for efficiency; support off-policy
    # variants later
    def _compute_grpo_advantages(self, rewards: list[float]) -> list[float]:
        """
        GRPO advantages with per-group baseline and variance normalization.
        Ensures zero-mean within group; scales by std for stability.
        """
        n = len(rewards)
        if n == 0:
            return []
        mean_reward = sum(rewards) / n
        centered = [r - mean_reward for r in rewards]
        std = (sum(a * a for a in centered) / n) ** 0.5
        denom = max(std, ADVANTAGE_STD_MIN)
        return [a / denom for a in centered]

    @abstractmethod
    def create_environment(self, problem: Any) -> Any:
        """Create and return the environment for this problem."""
        pass

    @abstractmethod
    def reset_environment(self, env: Any) -> Any:
        """Reset the environment and return initial state."""
        pass

    @abstractmethod
    def create_prompt(self, problem: Any, env: Any, state: Any, trajectory: list) -> str:
        """Create a prompt for the model based on current state."""
        pass

    @abstractmethod
    def parse_action(self, text: str, env: Any, state: Any) -> Any:
        """Parse the model's text output into an environment action."""
        pass

    @abstractmethod
    def step_environment(self, env: Any, action: Any) -> tuple:
        """Take a step in the environment and return
        (next_state, reward, done, info)."""
        pass

    def create_trajectory_entry(self, state: Any, action: Any, reward: float, info: dict) -> Any:
        """Create an entry for the trajectory list.

        Default implementation returns a minimal tuple:
        (step_index, action, reward).

        For single-step environments, the step_index is 0. Subclasses can
        override to include richer information if needed.
        """
        # NOTE step_index is 0 because we're focusing on single-turn RL
        return (0, action, reward)

    @abstractmethod
    def get_final_info(self, env: Any, trajectory: list, total_reward: float) -> dict:
        """Get final information to include in the rollout result."""
        pass

    # Configurable parameters (can be overridden by subclasses)

    def get_max_tokens(self) -> int:
        """Maximum new tokens to generate per decision."""
        return int(MAX_NEW_TOKENS)

    def get_temperature(self) -> float:
        """Temperature for sampling."""
        return 0.7
