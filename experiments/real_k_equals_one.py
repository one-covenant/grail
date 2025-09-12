"""
Real-model experiment: k=1 trivializes verification.

Uses the actual model/verifier; bypasses signature verification only.
"""


def main() -> None:
    import grail.grail as gg

    gg.verify_commit_signature = lambda commit, wallet_address: True
    verifier = gg.Verifier(model_name=gg.MODEL_NAME)

    tokens = list(range(1, 1 + max(gg.CHALLENGE_K, 24)))
    commit_rand = "a1b2c3d4"

    r_vec = gg.r_vec_from_randomness(commit_rand, verifier.model.config.hidden_size)
    full_ids = gg.torch.tensor(tokens, dtype=gg.torch.long, device=verifier.device).unsqueeze(0)
    with gg.torch.no_grad():
        outs = verifier.model(full_ids, output_hidden_states=True)
    h_layer = outs.hidden_states[gg.LAYER_INDEX][0]
    s_true = [gg.dot_mod_q(h_layer[i], r_vec) for i in range(len(tokens))]

    k = 1
    open_rand = "1337"  # prover-chosen (cheating)
    indices = gg.indices_from_root(tokens, open_rand, len(tokens), k)
    idx = indices[0]

    s_cheat = [424242 for _ in tokens]
    s_cheat[idx] = s_true[idx]

    model_name = getattr(verifier.model, "name_or_path", gg.MODEL_NAME)
    commit = {
        "beacon": {"round": 1, "randomness": commit_rand},
        "model": {"name": model_name, "layer_index": gg.LAYER_INDEX},
        "tokens": tokens,
        "s_vals": s_cheat,
        "signature": "00",
    }
    proof_pkg = {"round_R1": {"randomness": open_rand}, "indices": indices}

    # Verifier-enforced randomness and minimum k
    verifier_rand = "feedbead"
    ok = verifier.verify(
        commit,
        proof_pkg,
        prover_address="stub",
        challenge_randomness=verifier_rand,
        min_k=gg.CHALLENGE_K,
    )
    print("Exploit: k=1 ->", "PASS" if ok else "FAIL")
    print(f"Opened index {idx}; only that position is correct")


if __name__ == "__main__":
    main()
