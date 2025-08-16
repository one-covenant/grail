"""SAT Problem Environment for GRAIL RL System."""

import random
import hashlib
from ..rollout import RolloutGenerator
from typing import List, Dict, Tuple, Any

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
        MIN_VARS + int((MAX_VARS - MIN_VARS) * difficulty)
    )
    num_clauses = rng.randint(
        MIN_CLAUSES + int((MAX_CLAUSES - MIN_CLAUSES) * difficulty * 0.5),
        MIN_CLAUSES + int((MAX_CLAUSES - MIN_CLAUSES) * difficulty)
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


class SATEnvironment:
    """RL environment for solving SAT problems."""
    
    def __init__(self, problem: SATProblem):
        self.problem = problem
        self.assignment = [False] * problem.num_vars
        self.current_var = 0
        self.steps = 0
        self.max_steps = problem.num_vars * 2
        self.trajectory = []
        
    def reset(self) -> Dict:
        """Reset environment to initial state."""
        self.assignment = [False] * self.problem.num_vars
        self.current_var = 0
        self.steps = 0
        self.trajectory = []
        return self.get_state()
    
    def get_state(self) -> Dict:
        """Get current state as dict for LLM processing."""
        return {
            "problem": self.problem.to_text(),
            "current_assignment": self.assignment.copy(),
            "current_var": self.current_var,
            "steps": self.steps,
            "satisfied_clauses": self.count_satisfied_clauses()
        }
    
    def count_satisfied_clauses(self) -> int:
        """Count how many clauses are currently satisfied."""
        count = 0
        for clause in self.problem.clauses:
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx < len(self.assignment):
                    if (lit > 0 and self.assignment[var_idx]) or \
                       (lit < 0 and not self.assignment[var_idx]):
                        count += 1
                        break
        return count
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Take action (0=false, 1=true for current variable)."""
        if self.current_var >= self.problem.num_vars:
            return self.get_state(), 0, True, {"success": False}
        
        # Record action in trajectory
        self.trajectory.append((self.current_var, action))
        
        # Apply action
        self.assignment[self.current_var] = bool(action)
        self.current_var += 1
        self.steps += 1
        
        # Calculate reward
        satisfied = self.count_satisfied_clauses()
        total_clauses = len(self.problem.clauses)
        
        # Check if done
        done = False
        success = False
        
        if self.current_var >= self.problem.num_vars:
            # All variables assigned
            done = True
            success = self.problem.check_solution(self.assignment)
            if success:
                reward = 10.0  # Big reward for solving
            else:
                reward = satisfied / total_clauses - 1.0  # Partial credit
        elif self.steps >= self.max_steps:
            done = True
            reward = -1.0  # Penalty for timeout
        else:
            # Intermediate reward based on progress
            reward = (satisfied / total_clauses) * 0.1
        
        info = {
            "success": success,
            "satisfied_clauses": satisfied,
            "total_clauses": total_clauses
        }
        
        return self.get_state(), reward, done, info
    
    def render_trajectory(self) -> str:
        """Render trajectory as text for LLM."""
        text = "Solution trajectory:\n"
        for var, val in self.trajectory:
            text += f"  x{var+1} = {val}\n"
        return text


class SATRolloutGenerator(RolloutGenerator):
    """Generate SAT rollouts using an LLM model."""
    
    def create_environment(self, problem: SATProblem) -> SATEnvironment:
        """Create a SAT environment for the given problem."""
        return SATEnvironment(problem)
    
    def reset_environment(self, env: SATEnvironment) -> Dict:
        """Reset the SAT environment."""
        return env.reset()
    
    def create_prompt(self, problem: SATProblem, env: SATEnvironment, state: Dict, trajectory: List) -> str:
        """Create a prompt for the LLM to decide the next variable assignment."""
        prompt = f"SAT Problem:\n{problem.to_text()}\n"
        
        if trajectory:
            prompt += "\nAssignments so far:\n"
            for var, action, _ in trajectory:
                prompt += f"  x{var+1} = {action}\n"
        
        prompt += f"\nCurrent state: {env.count_satisfied_clauses()}/{len(problem.clauses)} clauses satisfied\n"
        prompt += f"Next variable: x{env.current_var+1}\n"
        prompt += "Should x{} be 0 (false) or 1 (true)? Consider which value satisfies more clauses.\n".format(env.current_var+1)
        prompt += "Answer with just '0' or '1':"
        
        return prompt
    
    def parse_action(self, text: str, env: SATEnvironment, state: Dict) -> int:
        """Parse the action from LLM output."""
        text = text.strip().lower()
        
        # Look for explicit 0 or 1
        if '1' in text or 'true' in text:
            return 1
        elif '0' in text or 'false' in text:
            return 0
        
        # Check for yes/no style answers
        if 'yes' in text:
            return 1
        elif 'no' in text:
            return 0
        
        # Default to trying true first (can be randomized)
        return 1 if env.current_var % 2 == 0 else 0
    
    def step_environment(self, env: SATEnvironment, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Take a step in the SAT environment."""
        return env.step(action)
    
    def create_trajectory_entry(self, state: Dict, action: int, reward: float, info: Dict) -> Tuple[int, int, float]:
        """Create a trajectory entry for SAT (variable index, action, reward)."""
        # Get the variable index from the state (before the action was taken)
        var_idx = state["current_var"]
        return (var_idx, action, reward)
    
    def get_final_info(self, env: SATEnvironment, trajectory: List, total_reward: float) -> Dict:
        """Get final SAT-specific information."""
        # Check if the final assignment satisfies all clauses
        success = env.problem.check_solution(env.assignment)
        
        return {
            "success": success,
            "satisfied_clauses": env.count_satisfied_clauses(),
            "assignment": env.assignment,
            "sat_problem": {
                "seed": env.problem.seed,
                "num_vars": env.problem.num_vars,
                "clauses": env.problem.clauses
            }
        }