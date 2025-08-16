"""SAT Problem Environment for GRAIL RL System."""

import random
import hashlib
from typing import List, Dict, Tuple

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