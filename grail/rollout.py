"""Generic rollout generation for GRAIL RL environments."""

import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class RolloutGenerator(ABC):
    """
    Abstract base class for generating rollouts in any environment.
    
    This provides the common structure for:
    1. Interacting with an LLM model
    2. Collecting tokens for GRAIL proofs
    3. Tracking trajectories and rewards
    
    Subclasses implement environment-specific logic.
    """
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str = "cuda"):
        """
        Initialize with a model and tokenizer.
        
        Args:
            model: The language model to use for decisions
            tokenizer: The tokenizer for the model
            device: Device to run on (cuda/cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_rollout(self, problem: Any) -> Dict:
        """
        Generate a complete rollout for the given problem.
        
        This is the main entry point that:
        1. Initializes the environment
        2. Collects model decisions
        3. Returns tokens and trajectory
        
        Args:
            problem: The problem instance (environment-specific)
        
        Returns:
            Dictionary with at minimum:
            - tokens: List of all token IDs generated
            - trajectory: List of (state, action, reward) tuples
            - total_reward: Sum of all rewards
            - success: Whether the problem was solved
        """
        # Initialize environment
        env = self.create_environment(problem)
        state = self.reset_environment(env)
        
        # Execute rollout
        trajectory = []
        total_reward = 0
        done = False
        all_tokens = []
        
        while not done:
            # Get model's decision
            prompt = self.create_prompt(problem, env, state, trajectory)
            tokens, action = self._get_model_decision(prompt, env, state)
            
            # Store tokens for GRAIL proof
            all_tokens.extend(tokens)
            
            # Take action in environment
            next_state, reward, done, info = self.step_environment(env, action)
            
            # Record trajectory
            trajectory_entry = self.create_trajectory_entry(state, action, reward, info)
            trajectory.append(trajectory_entry)
            
            total_reward += reward
            state = next_state
        
        # Get final results from environment
        final_info = self.get_final_info(env, trajectory, total_reward)
        
        return {
            "tokens": all_tokens,
            "trajectory": trajectory,
            "total_reward": total_reward,
            **final_info
        }
    
    def _get_model_decision(self, prompt: str, env: Any, state: Any) -> tuple[List[int], Any]:
        """
        Get model's decision for the current state.
        
        Args:
            prompt: The prompt to send to the model
            env: The environment instance
            state: Current environment state
        
        Returns:
            Tuple of (tokens generated, action to take)
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            try:
                gen = self.model.generate(
                    input_ids,
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
                    logger.debug(f"Sampling failed, using greedy decoding: {e}")
                    gen = self.model.generate(
                        input_ids,
                        max_new_tokens=self.get_max_tokens(),
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                else:
                    raise
        
        # Extract tokens and parse action
        tokens = gen[0].tolist()
        generated_text = self.tokenizer.decode(gen[0][len(input_ids[0]):], skip_special_tokens=True)
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
    def create_prompt(self, problem: Any, env: Any, state: Any, trajectory: List) -> str:
        """Create a prompt for the model based on current state."""
        pass
    
    @abstractmethod
    def parse_action(self, text: str, env: Any, state: Any) -> Any:
        """Parse the model's text output into an environment action."""
        pass
    
    @abstractmethod
    def step_environment(self, env: Any, action: Any) -> tuple:
        """Take a step in the environment and return (next_state, reward, done, info)."""
        pass
    
    @abstractmethod
    def create_trajectory_entry(self, state: Any, action: Any, reward: float, info: Dict) -> Any:
        """Create an entry for the trajectory list."""
        pass
    
    @abstractmethod
    def get_final_info(self, env: Any, trajectory: List, total_reward: float) -> Dict:
        """Get final information to include in the rollout result."""
        pass
    
    # Configurable parameters (can be overridden by subclasses)
    
    def get_max_tokens(self) -> int:
        """Maximum new tokens to generate per decision."""
        return 20
    
    def get_temperature(self) -> float:
        """Temperature for sampling."""
        return 0.7