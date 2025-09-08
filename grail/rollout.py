"""Generic rollout generation for GRAIL RL environments - GRPO Version."""

import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
# Imports for type hints only - actual types are Any in method signatures
import bittensor as bt

logger = logging.getLogger(__name__)


@dataclass
class GRPORollout:
    """Single rollout for GRPO training with GRAIL proof support."""
    # Token data for GRAIL proof
    tokens: List[int]  # All tokens (prompt + completion)
    token_logprobs: List[float]  # Logprobs for all tokens

    # Training masks
    prompt_length: int  # Where prompt ends and completion begins
    completion_length: int

    # Rewards and advantages
    reward: float
    advantage: float  # Computed after all rollouts collected

    # Trajectory for analysis
    trajectory: List[Tuple[Any, Any, float]]
    success: bool

    # GRAIL proof fields (from existing grail.py)
    s_vals: List[int]
    signature: bytes
    beacon: Dict


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
        rollouts_per_problem: int = 4
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

    def generate_grpo_rollouts(
        self,
        problem: Any,
        randomness_hex: str,
        wallet: bt.wallet
    ) -> List[GRPORollout]:
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

        for i in range(self.rollouts_per_problem):
            # Initialize environment
            env = self.create_environment(problem)
            state = self.reset_environment(env)

            # Generate rollout with logprobs tracking
            rollout = self._generate_single_rollout(
                problem, env, state, randomness_hex, wallet
            )
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
        wallet: bt.wallet
    ) -> GRPORollout:
        """Generate a single rollout with logprob tracking and GRAIL proof."""
        # Create prompt
        prompt = self.create_prompt(problem, env, state, [])
        # Tokenize with explicit attention mask to ensure proper
        # distinction between content and padding tokens
        tokenized = self.tokenizer(
            prompt, return_tensors="pt", return_attention_mask=True
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
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Extract generated tokens and logprobs
        all_token_ids = outputs.sequences[0].tolist()
        completion_ids = all_token_ids[prompt_length:]

        # Extract logprobs for generated tokens
        logprobs = self._extract_logprobs(outputs.scores, completion_ids)

        # Full token logprobs (0s for prompt, actual for completion)
        all_logprobs = [0.0] * prompt_length + logprobs

        # Parse and execute actions from generated text
        generated_text = self.tokenizer.decode(
            completion_ids, skip_special_tokens=True
        )
        action = self.parse_action(generated_text, env, state)

        # Execute trajectory in environment
        trajectory = []
        total_reward = 0
        done = False

        # Step through environment with parsed action
        next_state, reward, done, info = self.step_environment(env, action)
        trajectory_entry = self.create_trajectory_entry(
            state, action, reward, info
        )
        trajectory.append(trajectory_entry)
        total_reward += reward

        # Continue stepping if needed (for multi-step environments)
        state = next_state
        while not done and len(trajectory) < self.get_max_tokens():
            # For subsequent steps, we might need to generate more actions
            # This depends on the environment - for SAT, we usually make
            # all assignments at once
            break

        # Check final success
        final_info = self.get_final_info(env, trajectory, total_reward)
        success = final_info.get('success', False)

        # Generate GRAIL proof components (using existing grail.py logic)
        from .grail import r_vec_from_randomness, dot_mod_q, sign_commit_binding, LAYER_INDEX

        # Compute s_vals for GRAIL proof
        r_vec = r_vec_from_randomness(
            randomness_hex, self.model.config.hidden_size
        )
        s_vals = []

        with torch.inference_mode():
            token_tensor = torch.tensor(
                [all_token_ids], dtype=torch.long
            ).to(self.device)
            model_outputs = self.model(token_tensor, output_hidden_states=True)
            # Last layer hidden states
            h_layer = model_outputs.hidden_states[-1][0]

            for pos in range(len(all_token_ids)):
                if pos < h_layer.size(0):
                    s_val = dot_mod_q(h_layer[pos], r_vec)
                    s_vals.append(s_val)

        # Sign commit binding using wallet
        model_id = getattr(self.model, "name_or_path", "unknown")
        signature = sign_commit_binding(all_token_ids, randomness_hex, model_id, LAYER_INDEX, s_vals, wallet)

        return GRPORollout(
            tokens=all_token_ids,
            token_logprobs=all_logprobs,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            reward=total_reward,
            advantage=0.0,  # Will be set after all rollouts generated
            trajectory=trajectory,
            success=success,
            s_vals=s_vals,
            signature=signature,
            beacon={"randomness": randomness_hex}
        )

    def _extract_logprobs(
        self, scores: List[torch.Tensor], token_ids: List[int]
    ) -> List[float]:
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

    def _compute_grpo_advantages(self, rewards: List[float]) -> List[float]:
        """
        Compute GRPO advantages: reward - mean(group_rewards).
        This is the core of GRPO - advantages sum to zero within each group.
        """
        if not rewards:
            return []
        mean_reward = sum(rewards) / len(rewards)
        return [r - mean_reward for r in rewards]

    def _get_model_decision(
        self, prompt: str, env: Any, state: Any
    ) -> Tuple[List[int], Any]:
        """
        Get model's decision for the current state.

        Args:
            prompt: The prompt to send to the model
            env: The environment instance
            state: Current environment state

        Returns:
            Tuple of (tokens generated, action to take)
        """
        # Tokenize with explicit attention mask to ensure proper
        # distinction between content and padding tokens
        tokenized = self.tokenizer(
            prompt, return_tensors="pt", return_attention_mask=True
        )
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        with torch.inference_mode():
            try:
                gen = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.get_max_tokens(),
                    temperature=self.get_temperature(),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            except RuntimeError as e:
                if "inf" in str(e) or "nan" in str(e):
                    # Fallback to greedy decoding if sampling fails
                    logger.debug(
                        f"Sampling failed, using greedy decoding: {e}"
                    )
                    gen = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.get_max_tokens(),
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                else:
                    raise

        # Extract tokens and parse action
        tokens = gen[0].tolist()

        # Validate tokens are within vocabulary range
        vocab_size = len(self.tokenizer)
        invalid_tokens = [t for t in tokens if t < 0 or t >= vocab_size]
        if invalid_tokens:
            logger.warning(
                f"Found invalid tokens: {invalid_tokens[:5]}... "
                "Clamping to valid range"
            )
            tokens = [max(0, min(t, vocab_size - 1)) for t in tokens]

        generated_text = self.tokenizer.decode(
            gen[0][len(input_ids[0]):], skip_special_tokens=True
        )
        action = self.parse_action(generated_text, env, state)

        return tokens, action

    # Abstract methods that subclasses must implement

    @abstractmethod
    def create_environment(self, problem: Any) -> Any:
        """Create and return the environment for this problem."""
        pass

    @abstractmethod
    def reset_environment(self, env: Any) -> Any:
        """Reset the environment and return initial state."""
        pass

    @abstractmethod
    def create_prompt(
        self, problem: Any, env: Any, state: Any, trajectory: List
    ) -> str:
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

    @abstractmethod
    def create_trajectory_entry(
        self, state: Any, action: Any, reward: float, info: Dict
    ) -> Any:
        """Create an entry for the trajectory list."""
        pass

    @abstractmethod
    def get_final_info(
        self, env: Any, trajectory: List, total_reward: float
    ) -> Dict:
        """Get final information to include in the rollout result."""
        pass

    # Configurable parameters (can be overridden by subclasses)

    def get_max_tokens(self) -> int:
        """Maximum new tokens to generate per decision."""
        return 20

    def get_temperature(self) -> float:
        """Temperature for sampling."""
        return 0.7
