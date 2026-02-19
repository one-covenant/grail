"""Private loader for affinetes environment modules from vendor submodule.

Auto-clones the affinetes repo on first use if the vendor submodule is missing.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_DIR = _REPO_ROOT / "vendor" / "affinetes"
_ENVS_DIR = _VENDOR_DIR / "environments"
_REPO_URL = "https://github.com/AffineFoundation/affinetes.git"

_loaded: dict[str, bool] = {}
_clone_attempted = False


def _ensure_repo() -> None:
    """Clone affinetes repo into vendor/ if not already present (one-time)."""
    global _clone_attempted
    if _ENVS_DIR.exists():
        return
    if _clone_attempted:
        raise ImportError(
            f"Affinetes repo not available at {_VENDOR_DIR} and auto-clone already failed. "
            "Try manually: git clone https://github.com/AffineFoundation/affinetes.git vendor/affinetes"
        )
    _clone_attempted = True

    logger.info("Affinetes not found at %s — auto-cloning from %s", _VENDOR_DIR, _REPO_URL)
    _VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", _REPO_URL, str(_VENDOR_DIR)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error("git clone failed (exit %d): %s", result.returncode, result.stderr.strip())
            raise ImportError(
                f"Failed to auto-clone affinetes (exit {result.returncode}): {result.stderr.strip()}"
            )
        logger.info("Affinetes cloned successfully to %s", _VENDOR_DIR)
    except FileNotFoundError:
        logger.error("git not found on PATH — cannot auto-clone affinetes")
        raise ImportError(
            "git is not installed or not on PATH. "
            "Install git or manually clone: git clone https://github.com/AffineFoundation/affinetes.git vendor/affinetes"
        ) from None
    except subprocess.TimeoutExpired:
        logger.error("git clone timed out after 120s")
        raise ImportError(
            "git clone timed out. Check your network or manually clone: "
            "git clone https://github.com/AffineFoundation/affinetes.git vendor/affinetes"
        ) from None


def _ensure_env_path(env_subdir: str) -> None:
    """Add affinetes environment directory to sys.path (idempotent, lazy)."""
    if env_subdir in _loaded:
        return
    _ensure_repo()
    path = str(_ENVS_DIR / env_subdir)
    if not Path(path).exists():
        raise ImportError(
            f"Affinetes environment directory not found: {path}. "
            "The repo may be an incompatible version."
        )
    if path not in sys.path:
        sys.path.insert(0, path)
    _loaded[env_subdir] = True
    logger.debug("Added affinetes env to sys.path: %s", path)


def load_trace_task():
    """Lazy-load TraceTask from affinetes trace environment."""
    _ensure_env_path("trace")
    from trace_task import TraceTask  # type: ignore[import-not-found]

    return TraceTask


def load_logic_task():
    """Lazy-load LogicTaskV2 from affinetes lgc-v2 environment."""
    _ensure_env_path("primeintellect/lgc-v2")
    from logic_task_v2 import LogicTaskV2  # type: ignore[import-not-found]

    return LogicTaskV2


def load_logic_verifiers():
    """Lazy-load verifier classes from affinetes lgc-v2 environment."""
    _ensure_env_path("primeintellect/lgc-v2")
    from games.verifiers import verifier_classes  # type: ignore[import-not-found]
    from base.data import Data  # type: ignore[import-not-found]

    return verifier_classes, Data
