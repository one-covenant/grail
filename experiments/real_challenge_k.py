"""
Real-model experiment: Prover-controlled challenge + small k

Uses the actual model and verifier implementation (no module stubs).
We only bypass signature verification to focus on the protocol flaw.
"""

import logging


def main() -> None:
    import grail.grail as gg

    logging.getLogger("grail").setLevel(logging.INFO)
    # Bypass signature verification to isolate protocol behavior
    gg.verify_commit_signature = lambda commit, wallet_address: True

    verifier = gg.Verifier(model_name=gg.MODEL_NAME)

    # Prover chooses tokens and commit randomness (for r_vec)
    # Ensure sequence length >= CHALLENGE_K so verifier can open min_k indices
    tokens = list(range(1, 1 + max(gg.CHALLENGE_K, 24)))
    commit_rand = "deadbeefcafebabe"

    # Compute true s_vals
    r_vec = gg.r_vec_from_randomness(commit_rand, verifier.model.config.hidden_size)
    full_ids = gg.torch.tensor(tokens, dtype=gg.torch.long, device=verifier.device).unsqueeze(0)
    with gg.torch.no_grad():
        outs = verifier.model(full_ids, output_hidden_states=True)
    h_layer = outs.hidden_states[gg.LAYER_INDEX][0]
    s_true = [gg.dot_mod_q(h_layer[i], r_vec) for i in range(len(tokens))]

    # Prover chooses small k and sets open randomness accordingly
    k = 3
    open_rand = "abad1dea"  # prover-chosen (cheating)
    indices = gg.indices_from_root(tokens, open_rand, len(tokens), k)

    # Cheat: correct only at opened indices
    s_cheat = [999999 for _ in tokens]
    for i in indices:
        s_cheat[i] = s_true[i]

    model_name = getattr(verifier.model, "name_or_path", gg.MODEL_NAME)
    commit = {
        "beacon": {"round": 1, "randomness": commit_rand},
        "model": {"name": model_name, "layer_index": gg.LAYER_INDEX},
        "tokens": tokens,
        "s_vals": s_cheat,
        "signature": "00",
    }
    proof_pkg = {"round_R1": {"randomness": open_rand}, "indices": indices}

    # Verifier enforces its own challenge randomness and min k
    verifier_rand = "feedbead"  # verifier-chosen
    ok = verifier.verify(
        commit,
        proof_pkg,
        prover_address="stub",
        challenge_randomness=verifier_rand,
        min_k=gg.CHALLENGE_K,
    )
    print("Exploit: challenge control + small k ->", "PASS" if ok else "FAIL")
    print(f"Indices: {indices}; Correct only at those positions (k={k}, n={len(tokens)})")


if __name__ == "__main__":
    main()
