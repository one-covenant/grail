"""
Real-model experiment: Input validation / DoS by malformed commit.
"""


def main():
    import grail.grail as gg

    gg.verify_s_vals_signature = lambda s, sig, addr: True
    verifier = gg.Verifier(model_name=gg.MODEL_NAME)

    tokens = [1, 2, 3, 4, 5]
    commit_rand = "aa55"
    s_vals = []  # malformed: empty s_vals triggers IndexError on any index

    proof_rand = "bb66"
    # Use k=1 so the first access triggers immediately
    indices = gg.indices_from_root(tokens, proof_rand, len(tokens), 1)

    commit = {"beacon": {"round": 1, "randomness": commit_rand}, "tokens": tokens, "s_vals": s_vals, "signature": "00"}
    proof_pkg = {"round_R1": {"randomness": proof_rand}, "indices": indices}

    try:
        ok = verifier.verify(
            commit,
            proof_pkg,
            prover_address="stub",
            challenge_randomness=proof_rand,
            min_k=1,
        )
        print("Exploit: malformed commit ->", "PASS (unexpected)" if ok else "FAIL (cleanly rejected)")
    except Exception as e:
        print("Exploit: malformed commit -> CRASH (DoS)")
        print(f"Exception: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
