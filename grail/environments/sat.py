"""SAT Problem Solver for GRAIL RL System."""

from typing import List, Dict, Tuple, Any, Callable, Optional, cast
import re
import random
import hashlib

from .base import Parser, RewardVector
from ..mining.rollout_generator import RolloutGenerator

# SAT Problem Configuration
MIN_VARS = 3
MAX_VARS = 10
MIN_CLAUSES = 5
MAX_CLAUSES = 20
CLAUSE_LENGTH = 3  # 3-SAT


class SATProblem:
    """Represents a SAT problem instance."""

    def __init__(self, num_vars: int, clauses: List[List[int]], seed: str):
        self.num_vars = num_vars
        self.clauses = clauses
        self.seed = seed
        self.solution = None

    def to_text(self) -> str:
        """Convert SAT problem to text format for LLM processing."""
        text = f"SAT Problem (seed: {self.seed[:8]}...):\n"
        text += f"Variables: {self.num_vars}\n"
        text += "Clauses:\n"
        for i, clause in enumerate(self.clauses):
            clause_str = " OR ".join([f"{'NOT ' if lit < 0 else ''}x{abs(lit)}" for lit in clause])
            text += f"  ({clause_str})\n"
        return text

    def check_solution(self, assignment: List[bool]) -> bool:
        """Check if assignment satisfies all clauses."""
        if len(assignment) != self.num_vars:
            return False

        for clause in self.clauses:
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if lit > 0 and assignment[var_idx]:
                    satisfied = True
                    break
                elif lit < 0 and not assignment[var_idx]:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True


def generate_sat_problem(seed: str, difficulty: float = 0.5) -> SATProblem:
    """Generate a SAT problem from seed with controlled difficulty."""
    # Use seed for deterministic generation
    rng = random.Random(hashlib.sha256(seed.encode()).digest())

    # Scale problem size based on difficulty
    num_vars = rng.randint(
        MIN_VARS + int((MAX_VARS - MIN_VARS) * difficulty * 0.5),
        MIN_VARS + int((MAX_VARS - MIN_VARS) * difficulty),
    )
    num_clauses = rng.randint(
        MIN_CLAUSES + int((MAX_CLAUSES - MIN_CLAUSES) * difficulty * 0.5),
        MIN_CLAUSES + int((MAX_CLAUSES - MIN_CLAUSES) * difficulty),
    )

    clauses = []
    for _ in range(num_clauses):
        clause = []
        vars_in_clause = rng.sample(range(1, num_vars + 1), min(CLAUSE_LENGTH, num_vars))
        for var in vars_in_clause:
            # Randomly negate
            if rng.random() < 0.5:
                clause.append(-var)
            else:
                clause.append(var)
        clauses.append(clause)

    return SATProblem(num_vars, clauses, seed)


class SATParser(Parser):
    """Parser for extracting boolean assignments from SAT completion text."""

    def parse(self, completion: str, problem: SATProblem) -> List[bool]:
        """Extract boolean assignment from completion text.

        Args:
            completion: Raw completion text from model
            problem: SAT problem instance for context

        Returns:
            List of boolean values representing variable assignments
        """
        text = completion.strip()
        actions = []

        # Look for sequences of 0s and 1s
        for char in text:
            if char in "01":
                actions.append(int(char))
                if len(actions) >= problem.num_vars:
                    break

        # If we didn't find enough, look for patterns like "x1=0"
        if len(actions) < problem.num_vars:
            pattern = r"x\d+=([01])|variable\s+\d+\s*[:=]\s*([01])"
            matches = re.findall(pattern, text.lower())
            for match in matches:
                value = match[0] if match[0] else match[1]
                if value:
                    actions.append(int(value))

        # Pad with 0s if not enough assignments
        while len(actions) < problem.num_vars:
            actions.append(0)

        # Convert to boolean and return only required number
        return [bool(x) for x in actions[: problem.num_vars]]


def sat_correctness_reward(assignment: List[bool], problem: SATProblem) -> float:
    """Reward based on whether assignment satisfies all clauses.

    Args:
        assignment: Boolean assignment for variables
        problem: SAT problem instance

    Returns:
        10.0 if solution is correct, -1.0 otherwise
    """
    if problem.check_solution(assignment):
        return 10.0
    else:
        return -1.0


def create_sat_reward_vector(
    correctness_weight: float = 1.0, partial_weight: float = 0.0, efficiency_weight: float = 0.0
) -> RewardVector:
    """Create a standard SAT reward vector with common reward functions.

    Args:
        correctness_weight: Weight for correctness reward (default: 1.0)
        partial_weight: Unused (kept for compatibility)
        efficiency_weight: Unused (kept for compatibility)

    Returns:
        RewardVector configured for SAT solving
    """
    from typing import cast

    reward_functions = cast(List[Callable[[Any, Any], float]], [sat_correctness_reward])
    weights = [correctness_weight]
    parser = SATParser()

    # Declarative per-function bounds
    SAT_CORRECTNESS_BOUNDS = (-1.0, 10.0)

    return RewardVector(
        reward_functions,
        weights,
        parser,
        bounds=[SAT_CORRECTNESS_BOUNDS],
    )


class SATRolloutGenerator(RolloutGenerator):
    """SAT-specific rollout generator using RewardVector instead of environment."""
    _current_problem: Optional[SATProblem]

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        rollouts_per_problem: int = 4,
        reward_vector: Optional[RewardVector] = None,
    ) -> None:
        """Initialize SAT rollout generator.

        Args:
            model: Language model for generation
            tokenizer: Tokenizer for the model
            device: Device to run on
            rollouts_per_problem: Number of rollouts per problem for GRPO
            reward_vector: Custom reward vector, creates default if None
        """
        super().__init__(model, tokenizer, device, rollouts_per_problem)
        self.reward_vector = reward_vector or create_sat_reward_vector()
        # Create a dummy environment for compatibility
        self._current_problem = None

    # Required abstract methods from RolloutGenerator
    def create_environment(self, problem: SATProblem) -> SATProblem:
        """Return the problem itself as 'environment' - no separate env needed."""
        self._current_problem = problem
        return problem

    def reset_environment(self, env: SATProblem) -> Dict[str, Any]:
        """Return initial state - just the problem description."""
        return {"problem": env.to_text(), "num_vars": env.num_vars, "clauses": env.clauses}

    def create_prompt(
        self, problem: SATProblem, env: SATProblem, state: Dict[str, Any], trajectory: List
    ) -> str:
        """Create prompt for SAT solving."""
        prompt = f"SAT Problem:\n{problem.to_text()}\n"
        prompt += "Assign each variable as 0 (false) or 1 (true).\n"
        prompt += (
            "Provide all assignments in order. For example: "
            "0 1 0 1 means x1=0, x2=1, x3=0, x4=1\n"
        )
        prompt += "Solution: "
        return prompt

    def parse_action(self, text: str, env: SATProblem, state: Dict[str, Any]) -> List[bool]:
        """Parse completion text into boolean assignment using the reward vector's parser."""
        if self.reward_vector.parser:
            parsed = self.reward_vector.parser.parse(text, env)
            # Ensure we return a concrete List[bool]
            try:
                assignment: List[bool] = [bool(x) for x in cast(List[Any], parsed)]
            except Exception:
                parser = SATParser()
                assignment = parser.parse(text, env)
            return assignment
        else:
            # Fallback parsing
            parser = SATParser()
            return parser.parse(text, env)

    def step_environment(
        self, env: SATProblem, action: List[bool]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Single-shot evaluation using reward vector."""
        # Reconstruct completion text for reward computation
        completion = " ".join("1" if val else "0" for val in action)

        # Compute reward using reward vector
        reward = self.reward_vector.compute_reward(completion, env)
        # Optional: warn if reward outside declarative bounds
        try:
            if hasattr(self.reward_vector, "has_bounds") and self.reward_vector.has_bounds():
                low, high = self.reward_vector.reward_bounds()
                if reward < low or reward > high:
                    # Lazy import to avoid circular logger config
                    import logging

                    logging.getLogger(__name__).warning(
                        "SAT reward %.4f outside composed bounds [%.4f, %.4f]",
                        reward,
                        low,
                        high,
                    )
        except Exception:
            # Non-fatal: continue without bounds warning
            pass

        # Check success
        success = env.check_solution(action)

        # Count satisfied clauses
        satisfied_clauses = 0
        if len(action) == env.num_vars:
            for clause in env.clauses:
                for lit in clause:
                    var_idx = abs(lit) - 1
                    if (lit > 0 and action[var_idx]) or (lit < 0 and not action[var_idx]):
                        satisfied_clauses += 1
                        break

        state = {"problem": env.to_text(), "assignment": action, "success": success}

        info = {
            "success": success,
            "satisfied_clauses": satisfied_clauses,
            "total_clauses": len(env.clauses),
            "assignment": action,
        }

        return state, reward, True, info  # Always done after one step

    def create_trajectory_entry(
        self, state: Dict[str, Any], action: List[bool], reward: float, info: Dict[str, Any]
    ) -> Tuple[int, List[bool], float]:
        """Create trajectory entry."""
        return (0, action, reward)  # Simple: step 0, assignment, reward

    def get_final_info(
        self, env: SATProblem, trajectory: List, total_reward: float
    ) -> Dict[str, Any]:
        """Get final SAT-specific information."""
        if trajectory:
            _, assignment, _ = trajectory[-1]
            success = env.check_solution(assignment)
            satisfied_clauses = 0
            if len(assignment) == env.num_vars:
                for clause in env.clauses:
                    for lit in clause:
                        var_idx = abs(lit) - 1
                        if (lit > 0 and assignment[var_idx]) or (
                            lit < 0 and not assignment[var_idx]
                        ):
                            satisfied_clauses += 1
                            break
        else:
            assignment = [False] * env.num_vars
            success = False
            satisfied_clauses = 0

        return {
            "success": success,
            "satisfied_clauses": satisfied_clauses,
            "assignment": assignment,
            "sat_problem": {"seed": env.seed, "num_vars": env.num_vars, "clauses": env.clauses},
        }
