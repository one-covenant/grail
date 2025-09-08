"""
Real-model experiment: Tolerance admits incorrect s_vals within ±TOLERANCE.
"""


def run_case(delta: int):
    import grail.grail as gg

    gg.verify_s_vals_signature = lambda s, sig, addr: True
    verifier = gg.Verifier(model_name=gg.MODEL_NAME)

    tokens = list(range(1, 1 + max(gg.CHALLENGE_K, 24)))
    commit_rand = "feedface"

    r_vec = gg.r_vec_from_randomness(commit_rand, verifier.model.config.hidden_size)
    full_ids = gg.torch.tensor(tokens, dtype=gg.torch.long, device=verifier.device).unsqueeze(0)
    with gg.torch.no_grad():
        outs = verifier.model(full_ids, output_hidden_states=True)
    h_layer = outs.hidden_states[gg.LAYER_INDEX][0]
    s_true = [gg.dot_mod_q(h_layer[i], r_vec) for i in range(len(tokens))]

    s_cheat = [((v + delta) % gg.PRIME_Q) for v in s_true]

    verifier_rand = "feedbead"
    k = 3
    indices = gg.indices_from_root(tokens, verifier_rand, len(tokens), k)

    commit = {"beacon": {"round": 1, "randomness": commit_rand}, "tokens": tokens, "s_vals": s_cheat, "signature": "00"}
    proof_pkg = {"round_R1": {"randomness": verifier_rand}, "indices": indices}
    ok = verifier.verify(
        commit,
        proof_pkg,
        prover_address="stub",
        challenge_randomness=verifier_rand,
        min_k=gg.CHALLENGE_K,
    )
    return ok, gg.TOLERANCE


def main():
    ok2, tol = run_case(delta=2)
    print(f"Exploit: tolerance +2 (<= {tol}) ->", "PASS" if ok2 else "FAIL")
    ok4, tol = run_case(delta=4)
    print(f"Control: tolerance +4 (> {tol}) ->", "PASS" if ok4 else "FAIL")


if __name__ == "__main__":
    main()
