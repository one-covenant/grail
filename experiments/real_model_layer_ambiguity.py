"""
Real-model experiment: Model/layer ambiguity.

Shows that the same commit passes at one layer index and fails at another,
underscoring the need to bind LAYER_INDEX in the signature.
"""


def main() -> None:
    import grail.grail as gg

    gg.verify_commit_signature = lambda commit, wallet_address: True
    verifier = gg.Verifier(model_name=gg.MODEL_NAME)

    tokens = [10, 20, 30, 40]
    commit_rand = "b16b00b5"

    r_vec = gg.r_vec_from_randomness(commit_rand, verifier.model.config.hidden_size)
    full_ids = gg.torch.tensor(tokens, dtype=gg.torch.long, device=verifier.device).unsqueeze(0)
    with gg.torch.no_grad():
        outs = verifier.model(full_ids, output_hidden_states=True)
    h_layer = outs.hidden_states[gg.LAYER_INDEX][0]
    s_vals = [gg.dot_mod_q(h_layer[i], r_vec) for i in range(len(tokens))]

    model_name = getattr(verifier.model, "name_or_path", gg.MODEL_NAME)
    commit = {
        "beacon": {"round": 1, "randomness": commit_rand},
        "model": {"name": model_name, "layer_index": gg.LAYER_INDEX},
        "tokens": tokens,
        "s_vals": s_vals,
        "signature": "00",
    }
    open_rand = "aa"
    k = 3
    indices = gg.indices_from_root(tokens, open_rand, len(tokens), k)
    proof_pkg = {"round_R1": {"randomness": open_rand}, "indices": indices}

    ok_default = verifier.verify(
        commit,
        proof_pkg,
        prover_address="stub",
        challenge_randomness=open_rand,
        min_k=k,
    )

    old = gg.LAYER_INDEX
    try:
        gg.LAYER_INDEX = -2
        ok_other = verifier.verify(
            commit,
            proof_pkg,
            prover_address="stub",
            challenge_randomness=open_rand,
            min_k=k,
        )
    finally:
        gg.LAYER_INDEX = old

    # With proper binding, commit should verify only on the bound layer
    # Treat that as exploit blocked -> FAIL
    print(
        "Exploit: model/layer ambiguity ->",
        "FAIL" if (ok_default and not ok_other) else "PASS",
    )
    print(
        "Observed: passes only on bound layer (-1) and fails on -2"
        if (ok_default and not ok_other)
        else f"Observed: ok_default={ok_default}, ok_other={ok_other}"
    )


if __name__ == "__main__":
    main()
