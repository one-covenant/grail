"""
Shared fixtures and builders for validator tests.

We provide:
- wallet_stub: deterministic validator identity (hotkey) for sampling/stability tests.
- chain_manager_stub: minimal stub used by _process_wallet_window (bucket lookup only).
- make_verifier_stub: factory for a deterministic Verifier double that records calls.
- build_inference: helper to construct realistic rollout entries with required fields.
"""

# This file provides pytest fixtures; keep types simple to reduce static analysis friction.

import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest


@pytest.fixture(autouse=True, name="_set_test_env")
def _set_test_env_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide safe default environment for tests.

    Keeps tests deterministic and avoids accidental network/monitoring.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("GRAIL_MONITORING_BACKEND", "null")
    monkeypatch.setenv("BT_NETWORK", "test")
    monkeypatch.setenv("NETUID", "1")
    monkeypatch.setenv("GRAIL_ROLLOUTS_PER_PROBLEM", "4")
    # Ensure project root is on path when running from different CWDs
    root_path = pathlib.Path(__file__).resolve().parents[1]
    root = str(root_path)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_inference(
    wallet: str,
    window: int,
    block_hash: str,
    nonce: int,
    rollout_group: Optional[str] = None,
    total_reward: float = 0.5,
    success: bool = False,
    assignment: Optional[list[int]] = None,
    challenge: str = "C",
    sig: str = "S",
) -> dict[str, Any]:
    """
    Build a single inference/rollout with all required fields for the validator.

    Why important:
    - Ensures tests feed structurally valid data into the validator, focusing assertions
      on business logic (gating, sampling, GRPO checks) rather than schema errors.
    """
    tokens: list[int] = []
    commit: dict[str, Any] = {
        "rollout": {
            "total_reward": total_reward,
            "success": success,
            "assignment": assignment if assignment is not None else [nonce],
            "prompt_length": 0,
            "completion_length": 0,
        },
        "sat_problem": {},
        "tokens": tokens,
    }
    return {
        "window_start": window,
        "nonce": nonce,
        "sat_seed": f"{wallet}-{block_hash}-{nonce}",
        "block_hash": block_hash,
        "commit": commit,
        "proof": {"p": 1},
        "challenge": challenge,
        "hotkey": wallet,
        "signature": sig,
        "rollout_group": rollout_group,
    }


@pytest.fixture(name="build_inference")
def build_inference_fixture() -> Callable[..., dict[str, Any]]:
    return build_inference


@pytest.fixture
def wallet_stub() -> Any:
    """
    Minimal validator wallet stub exposing .hotkey.ss58_address.

    Why important:
    - Sampling seed uses validator hotkey; determinism per-validator hinges on this.
    """
    return SimpleNamespace(hotkey=SimpleNamespace(ss58_address="VAL_HOTKEY_123"))


@pytest.fixture
def chain_manager_stub() -> Any:
    """
    Minimal chain manager stub satisfying get_bucket_for_hotkey for storage routing.

    Why important:
    - Keeps tests hermetic by avoiding any real storage / credentials.
    """

    class CM:
        def get_bucket_for_hotkey(self, _: str) -> None:
            return None

    return CM()


class _TokenizerStub:
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return ""


class _VerifierStub:
    def __init__(self, result_fn: Callable[[dict[str, Any], dict[str, Any], str], Any]) -> None:
        self.tokenizer = _TokenizerStub()
        self._result_fn = result_fn

    def verify_rollout(
        self,
        commit: dict[str, Any],
        proof: dict[str, Any],
        wallet_addr: str,
        challenge_randomness: str,
        log_identity: str,
    ) -> Any:
        return self._result_fn(commit, proof, wallet_addr)


@pytest.fixture
def make_verifier_stub() -> Callable[[Callable[[dict[str, Any], dict[str, Any], str], Any]], Any]:
    """
    Factory for a deterministic Verifier stub.

    result_fn(commit, proof, wallet_addr) -> (is_valid: bool, checks: dict)
    """

    def factory(result_fn: Callable[[dict[str, Any], dict[str, Any], str], Any]) -> Any:
        return _VerifierStub(result_fn)

    return factory
