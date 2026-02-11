"""Example usage of the Python code generation environment.

This demonstrates how to use the MBPP and HumanEval environments for
training and evaluating code generation models with GRPO.
"""

from grail.environments import create_env
from grail.environments.core import ChatMessage


def mbpp_example():
    """Train on MBPP dataset."""
    print("=" * 80)
    print("MBPP Training Example")
    print("=" * 80)

    # Create MBPP training environment
    env = create_env("mbpp", split="train")

    # Reset to get a problem
    obs = env.reset(seed=42)

    print(f"\nProblem:\n{obs.messages[0].content}\n")

    # Example completion (you would generate this with your model)
    completion = """<start_working_out>
I need to implement a function that finds the longest chain from pairs.
This is a dynamic programming problem similar to longest increasing subsequence.
For each pair (a, b), I can chain it with another pair (c, d) if b < c.
I'll sort pairs by first element and use DP to find the maximum chain length.
</end_working_out>
<SOLUTION>
class Pair:
    def __init__(self, a, b):
        self.a = a
        self.b = b

def max_chain_length(arr, n):
    max_len = 1

    # Sort by first element
    arr.sort(key=lambda x: x.a)

    # dp[i] stores max chain length ending at i
    dp = [1] * n

    for i in range(1, n):
        for j in range(0, i):
            if arr[j].b < arr[i].a and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1

    return max(dp) if dp else 0
</SOLUTION>"""

    # Execute step and get reward
    obs, reward, terminated, truncated, info = env.step(
        ChatMessage(role="assistant", content=completion)
    )

    print("Results:")
    print(f"  Reward: {reward:.4f}")
    print(f"  Success: {info['success']}")
    print(f"  Tests passed: {info['tests_passed']}/{info['tests_total']}")
    print(f"  Reward breakdown: {info['reward_components']}")


def humaneval_example():
    """Evaluate on HumanEval benchmark."""
    print("\n" + "=" * 80)
    print("HumanEval Evaluation Example")
    print("=" * 80)

    # Create HumanEval environment
    env = create_env("humaneval")

    # Reset to get first problem
    obs = env.reset(seed=0)

    print(f"\nProblem:\n{obs.messages[0].content}\n")

    # Example completion
    completion = """<start_working_out>
This function should check if any two numbers in the list are closer
than the given threshold. I'll use nested loops to compare all pairs.
</end_working_out>
<SOLUTION>
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    \"\"\"
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
</SOLUTION>"""

    obs, reward, terminated, truncated, info = env.step(
        ChatMessage(role="assistant", content=completion)
    )

    print("Results:")
    print(f"  Reward: {reward:.4f}")
    print(f"  Success: {info['success']}")
    print(f"  Tests passed: {info['tests_passed']}/{info['tests_total']}")


def dataset_splits_example():
    """Show different dataset splits."""
    print("\n" + "=" * 80)
    print("Dataset Splits")
    print("=" * 80)

    # MBPP has train/validation/test splits
    _train_env = create_env("mbpp", split="train")  # 374 examples
    _val_env = create_env("mbpp", split="validation")  # 90 examples
    _test_env = create_env("mbpp", split="test")  # 500 examples

    print("\nMBPP splits:")
    print("  Train: 374 examples")
    print("  Validation: 90 examples")
    print("  Test: 500 examples")

    # HumanEval only has test split
    _humaneval_env = create_env("humaneval")  # 164 examples

    print("\nHumanEval:")
    print("  Test only: 164 examples")


if __name__ == "__main__":
    mbpp_example()
    humaneval_example()
    dataset_splits_example()
