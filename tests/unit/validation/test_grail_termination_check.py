# #!/usr/bin/env python3
# """
# Unit test: show the exploit passes by fudging rollout metadata.

# Notes:
# - Keep test fast by monkeypatching heavy model calls.
# - Use shared constants where applicable.
# - Focus on termination/sampling checks not bound to the signature.
# """

# from typing import Any, cast

# import grail.grail as grail_mod
# from grail.environments import generate_sat_problem
# from grail.grail import Verifier
# from grail.shared.constants import CHALLENGE_K, MAX_NEW_TOKENS


# def test_exploit_passes_with_fudged_rollout_metadata(
#     monkeypatch: Any,
# ) -> None:
#     """Exploit: claim max-length termination and minimize sampling region.

#     We stub out crypto/model-heavy checks to isolate vulnerable paths:
#     - Signature/model proof forced to pass.
#     - Token validity uses generous limits.
#     - Sampling collection returns insufficient steps.
#     """

#     # 1) Make Verifier lightweight and deterministic
#     def fake_init(
#         self: Verifier,
#         model_name: str = grail_mod.MODEL_NAME,
#     ) -> None:
#         self.device = cast(Any, None)
#         self.tokenizer = type(
#             "Tok",
#             (),
#             {"eos_token_id": 0, "eos_token": ""},
#         )()
#         # Minimal stub for model/config fields accessed by verifier utilities
#         self.model = type(
#             "ModelStub",
#             (),
#             {"config": object(), "name_or_path": model_name},
#         )()
#         self._last_tokens_hash = None
#         self._last_step_logits = None
#         self._current_wallet = None

#     monkeypatch.setattr(grail_mod.Verifier, "__init__", fake_init)

#     # 2) Force GRAIL proof to succeed without running the model
#     monkeypatch.setattr(
#         grail_mod.Verifier,
#         "verify",
#         lambda self, *a, **k: True,
#     )

#     # 3) Loosen token validity constraints to avoid loading real configs/models
#     monkeypatch.setattr(
#         grail_mod,
#         "resolve_vocab_size",
#         lambda _cfg: 1_000_000,
#     )
#     monkeypatch.setattr(
#         grail_mod,
#         "resolve_max_context_length",
#         lambda _cfg: 10_000,
#     )

#     # 4) Starve the sampling-shape check so it passes via "insufficient" path
#     monkeypatch.setattr(
#         grail_mod.Verifier,
#         "_collect_chosen_token_probs",
#         lambda self, commit: [],
#     )

#     # 5) Build minimal tokens and a SAT problem matching regeneration
#     tokens = [999, 123, 456]
#     seed = "unit_test_seed"
#     difficulty = 0.5
#     problem = generate_sat_problem(seed, difficulty)
#     sat_data: dict[str, Any] = {
#         "seed": seed,
#         "difficulty": difficulty,
#         "num_vars": problem.num_vars,
#         "clauses": problem.clauses,
#     }

#     # Fudged rollout metadata: declare max-length termination
#     # and tiny completion slice
#     commit: dict[str, Any] = {
#         "tokens": tokens,
#         "model": {
#             "name": "ignored",
#             "layer_index": grail_mod.LAYER_INDEX,
#         },
#         "sat_problem": sat_data,
#         "rollout": {
#             # completion is 1 token
#             "prompt_length": len(tokens) - 1,
#             # trigger max-length path
#             "completion_length": MAX_NEW_TOKENS,
#             # skip solution check
#             "success": False,
#         },
#     }
#     proof_pkg: dict[str, Any] = {"indices": []}

#     v = Verifier(model_name="stub")

#     ok, checks = v.verify_rollout(
#         commit,
#         proof_pkg,
#         prover_address=("5FictitiousAddressForUnitTest"),
#         challenge_randomness="deadbeef",
#         min_k=CHALLENGE_K,
#     )

#     # The exploit should (incorrectly) pass under current implementation
#     assert ok is True
#     assert checks["tokens_valid"] is True
#     assert checks["proof_valid"] is True
#     assert checks["sat_problem_valid"] is True
#     assert checks["termination_valid"] is True
#     # Passes due to insufficient steps
#     assert checks["token_distribution_valid"] is True
#     # Vacuously true when success == False
#     assert checks["solution_valid"] is True
