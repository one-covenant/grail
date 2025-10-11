# #!/usr/bin/env python3
# """
# Unit test: exploit passes sampling shape check with realistic tokens.

# Uses real model/tokenizer from constants.py to show boundary manipulation
# exploit with properly spaced SAT solution tokens.
# """

# from typing import Any

# import pytest
# import torch

# from grail.grail import Verifier
# from grail.mining.rollout_generator import SOLUTION_END, SOLUTION_START
# from grail.shared.constants import MODEL_NAME


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for real model inference")
# def test_solution_tokens_pass_sampling_check_via_boundaries() -> None:
#     """Exploit: include <SOLUTION>1 1 1 1 1</SOLUTION> manipulating boundaries.

#     Strategy: Set prompt_length to exclude solution tokens from sampling
#     analysis, leaving only high-probability tokens in the region.
#     """

#     # Use real model and tokenizer from constants
#     verifier = Verifier(model_name=MODEL_NAME)

#     # Create realistic SAT solution text with proper spacing
#     solution_bits = "1 1 1 1 1"  # 5-variable SAT solution with spaces
#     solution_text = f"{SOLUTION_START}{solution_bits}{SOLUTION_END}"

#     # Add high-probability suffix tokens (newline + space)
#     suffix_text = "\n "

#     # Build full text: minimal prompt + solution + high-prob suffix
#     prompt_text = "Answer: "
#     full_text = prompt_text + solution_text + suffix_text

#     # Tokenize with real tokenizer
#     prompt_tokens = verifier.tokenizer.encode(prompt_text, add_special_tokens=False)
#     full_tokens = verifier.tokenizer.encode(full_text, add_special_tokens=False)
#     solution_tokens = verifier.tokenizer.encode(solution_text, add_special_tokens=False)
#     suffix_tokens = verifier.tokenizer.encode(suffix_text, add_special_tokens=False)

#     print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
#     print(f"Solution tokens ({len(solution_tokens)}): {solution_tokens}")
#     print(f"Suffix tokens ({len(suffix_tokens)}): {suffix_tokens}")
#     print(f"Full sequence ({len(full_tokens)}): {full_tokens}")

#     # Key exploit: manipulate boundaries to exclude solution from analysis
#     commit: dict[str, Any] = {
#         "tokens": full_tokens,
#         "rollout": {
#             # Set prompt_length to include solution tokens as "prompt"
#             # Only analyze the high-probability suffix tokens
#             "prompt_length": len(full_tokens) - len(suffix_tokens),
#             "completion_length": len(suffix_tokens),  # Only suffix analyzed
#         },
#     }

#     # Test the sampling shape check directly
#     passes, metrics = verifier._token_sampling_shape_check(commit)

#     print(f"Sampling check result: {'PASS' if passes else 'FAIL'}")
#     print(f"Metrics: {metrics}")

#     # Collect probabilities to see what was actually analyzed
#     probs = verifier._collect_chosen_token_probs(commit)
#     if probs:
#         prob_strs = [f"{p:.4f}" for p in probs]
#         print(f"Analyzed {len(probs)} token probabilities: {prob_strs}")

#     # The exploit should pass because:
#     # 1. Only suffix tokens analyzed - these have high probability
#     # 2. Solution tokens excluded via boundary manipulation
#     assert passes is True

#     # Verify the exploit worked as expected
#     # Should trigger "insufficient steps" or high-prob bypass
#     assert len(suffix_tokens) <= 8

#     # Verify solution tokens are present in the sequence
#     assert len(full_tokens) > len(prompt_tokens) + len(suffix_tokens)
#     print("Full sequence contains solution tokens")

#     # Verify the key exploit: analysis region only covers suffix tokens
#     analysis_start = commit["rollout"]["prompt_length"]
#     analysis_end = analysis_start + commit["rollout"]["completion_length"]

#     # The analysis region should be just the suffix (last 2 tokens)
#     expected_analysis_start = len(full_tokens) - len(suffix_tokens)
#     assert analysis_start == expected_analysis_start
#     assert analysis_end == len(full_tokens)

#     print(f"Analysis region: tokens[{analysis_start}:{analysis_end}] = suffix")
