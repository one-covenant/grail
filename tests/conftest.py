import pathlib
import sys

import pytest

from .proof_test_utils import generate_realistic_sat_prompt


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide safe default environment for tests.

    Keeps tests deterministic and avoids accidental network/monitoring.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("GRAIL_MONITORING_BACKEND", "null")
    monkeypatch.setenv("BT_NETWORK", "test")
    monkeypatch.setenv("NETUID", "1")
    monkeypatch.setenv("GRAIL_ROLLOUTS_PER_PROBLEM", "4")
    # Ensure project root is on path when running from different CWDs
    root = str(pathlib.Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)


@pytest.fixture(scope="session")
def sat_prompts() -> list[str]:
    """Pregenerated realistic SAT prompts for proof tests.

    Uses deterministic seeds and production chat templates to ensure
    prompts match actual mining scenarios.
    """
    from transformers import AutoTokenizer

    from grail.shared.constants import MODEL_NAME

    # Load tokenizer for chat template application
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    seeds = [
        "test_seed_easy_01",
        "test_seed_medium_02",
        "test_seed_hard_03",
        "test_seed_compact_04",
        "test_seed_large_05",
    ]
    difficulties = [0.3, 0.5, 0.7, 0.5, 0.8]
    return [
        generate_realistic_sat_prompt(seed, diff, tokenizer)
        for seed, diff in zip(seeds, difficulties)
    ]
