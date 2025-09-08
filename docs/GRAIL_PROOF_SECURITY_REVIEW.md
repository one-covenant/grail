# GRAIL Proof Protocol Security Review

## Summary
This review evaluates the security of the GRAIL proof flow in `grail/grail.py` (Prover/Verifier). The key risks were prover‑controlled challenges, small `k` checks, weak signature binding, numeric non‑determinism, and insufficient input validation. Several fixes are recommended (and some already implemented) to improve soundness and robustness.

## Critical Findings
- Prover‑controlled challenge: Prover supplied the open seed, enabling seed grinding to cherry‑pick indices. The verifier must supply/derive the challenge (e.g., drand + chain context) and ignore prover indices.
- Variable `k` controlled by prover: Using `len(indices)` lets a prover set `k=1`. Enforce a verifier‑side minimum `k` and cap maximum for runtime.
- Insufficient signature scope: Signing only `s_vals` is ambiguous. Sign a structured payload: `H(tokens) || beacon.R || model_id || layer_index || H(s_vals)` with a domain tag.

## Determinism & Numeric Stability
- Float accumulation error: `dot_mod_q` used float32 dot, which can differ across devices/kernels. Prefer exact integer accumulation (`int32 × int32 → int64`, then mod `PRIME_Q`). If floats remain, increase tolerance and pin deterministic settings.
- Tolerance fragility: Small `TOLERANCE` causes spurious mismatches across environments. Calibrate or move to integer math.

## Input Validation & DoS
- Validate before compute: non‑empty tokens/s_vals, equal lengths, token range within vocab, `len(tokens) ≥ min_k`. Treat errors as `False`, not exceptions.

## Challenge/Index Derivation
- Public, unpredictable seed: Derive challenge from drand (future round relative to commit anchor) and optionally mix with chain/window hash. All verifiers must compute the same seed deterministically.
- Ignore prover indices: Verifier computes indices from `H(tokens)` and the challenge; prover‑provided indices are not trusted.

## Model/Layer Binding
- Bind model identity and layer index in the signed payload to avoid ambiguity/downgrade. Consider including a weights/revision hash for unambiguous identity across verifiers.

## Additional Notes
- PRF setup with domain separation is sound; keep separate labels for sketch vs. open.
- Avoid logging large/sensitive arrays; current logging is mostly acceptable.

## Recommended Remediations (prioritized)
1) Verifier‑supplied, drand‑anchored challenge; ignore prover indices/randomness.
2) Enforce minimum `k` and cleanly reject short sequences.
3) Strengthen signature binding to include tokens hash, beacon randomness, model id, layer index, and `H(s_vals)`.
4) Switch `dot_mod_q` to integer accumulation (or calibrate tolerance and deterministic flags if floats remain).
5) Add strict input validation and bounds checks; handle errors without crashing.
6) Optionally bind a model weights/revision hash for exact identity.

