"""
Run all real-model experiments and summarize results.

Usage:
  python experiments/run_all.py
"""

import importlib
import io
import sys
from contextlib import redirect_stdout

EXPERIMENTS = [
    ("real_challenge_k", "Prover-controlled challenge + small k"),
    ("real_k_equals_one", "k=1 trivializes verification"),
    ("real_tolerance_slack", "Tolerance admits small errors"),
    ("real_input_validation_dos", "Malformed commit causes crash (DoS)"),
    ("real_model_layer_ambiguity", "Layer not bound in commit"),
]


def run_one(mod_name: str) -> tuple[str, str]:
    buf = io.StringIO()
    status = "UNKNOWN"
    with redirect_stdout(buf):
        try:
            mod = importlib.import_module(f"experiments.{mod_name}")
            if hasattr(mod, "main"):
                mod.main()
            else:
                print(f"{mod_name}: missing main()")
        except Exception as e:
            print(f"{mod_name}: CRASH ({type(e).__name__}: {e})")
    out = buf.getvalue().strip()
    # Infer status from output
    if "-> PASS" in out:
        status = "PASS"
    elif "-> FAIL" in out:
        status = "FAIL"
    elif "CRASH" in out or "Exception" in out:
        status = "CRASH"
    return out, status


def main() -> None:
    print("Running GRAIL proof security experiments...\n")
    results = []
    for mod, title in EXPERIMENTS:
        out, status = run_one(mod)
        print(f"=== {title} ({mod}) ===")
        print(out)
        print()
        results.append((mod, status))

    print("Summary:")
    for mod, status in results:
        print(f"- {mod}: {status}")
    # Non-zero exit on unexpected FAIL when exploit is expected to pass
    # Here we always exit 0; adjust if you want CI enforcement.
    sys.exit(0)


if __name__ == "__main__":
    main()
