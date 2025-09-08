# Experiments: GRAIL Proof Vulnerabilities

This folder contains self-contained experiments demonstrating protocol weaknesses identified in `docs/GRAIL_PROOF_SECURITY_REVIEW.md` using the real model and verifier. Signature checks are bypassed to isolate protocol behavior; no other stubs are used.

Real-model variants (no stubs)
- `real_challenge_k.py`
- `real_k_equals_one.py`
- `real_tolerance_slack.py`
- `real_input_validation_dos.py`
- `real_model_layer_ambiguity.py`

Requirements: install real deps and allow model download/caching.
- python -m venv .venv && source .venv/bin/activate
- pip install -e .[dev]
- Then run any script, e.g.: `python experiments/real_challenge_k.py`

Run all experiments
- `python experiments/run_all.py`
