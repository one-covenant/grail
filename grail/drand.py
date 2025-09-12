#!/usr/bin/env python3
"""
Drand distributed randomness beacon integration for GRAIL.

Features:
- v2-first HTTP API with v1 fallbacks
- Correct chain hashes (quicknet=3s, default=30s)
- Programmatic fetch of chain info (genesis_time, period) with caching
- Robust networking (Session, retries, shuffled relays, sensible timeouts)
- Uniform schema for real & mock beacons
"""

from __future__ import annotations

import json
import logging
import os
import random
from threading import Lock
from typing import Any

import requests
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ──────────────────────────  RELAYS / NETWORKING  ──────────────────────────

DRAND_URLS = [
    "https://api.drand.sh",
    "https://api2.drand.sh",
    "https://api3.drand.sh",
    "https://drand.cloudflare.com",
    "https://api.drand.secureweb3.com:6875",
]

_RETRY = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=0.25,
    status_forcelist=(429, 502, 503, 504),
    allowed_methods={"GET"},
    raise_on_status=False,
)
_SESSION = requests.Session()
_SESSION.mount("https://", HTTPAdapter(max_retries=_RETRY))
_HEADERS = {"User-Agent": "GRAIL-drand/0.2"}

# ─────────────────────────────  CHAINS / STATE  ─────────────────────────────

# NOTE: hashes are authoritative identifiers of chains (v2 uses them in the path).
# quicknet: 3s, unchained; default: 30s, chained.
DRAND_CHAINS: dict[str, dict[str, Any]] = {
    "quicknet": {
        "hash": "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971",
        "description": "Fast 3-second randomness (unchained, recommended)",
        # Reasonable defaults; will be refreshed via /info on first use:
        "period": 3,
        "genesis_time": None,
    },
    "default": {
        "hash": "8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce",
        "description": "Original 30-second chain (chained)",
        "period": 30,
        "genesis_time": None,
    },
}

DEFAULT_CHAIN = os.getenv("DRAND_CHAIN", "quicknet").strip() or "quicknet"

_current_chain = DEFAULT_CHAIN
# Cached, resolved parameters (populated lazily)
_DRAND_CHAIN_HASH: str | None = None
_DRAND_GENESIS_TIME: int | None = None
_DRAND_PERIOD: int | None = None

# Cache chain-info lookups by chain-hash to avoid repeated /info calls
_CHAIN_INFO_CACHE: dict[str, dict[str, Any]] = {}

# Thread-safety for globals and mock counter
_LOCK = Lock()
_BEACON_COUNTER = 0

# ───────────────────────────────  UTILITIES  ────────────────────────────────


def _shuffle_urls() -> list[str]:
    urls = DRAND_URLS[:]
    random.shuffle(urls)
    return urls


def _get_chain_record(name: str) -> dict[str, Any]:
    if name not in DRAND_CHAINS:
        raise ValueError(f"Unknown drand chain '{name}'. Available: {list(DRAND_CHAINS.keys())}")
    return DRAND_CHAINS[name]


def _parse_chain_info_payload(
    payload: dict[str, Any],
) -> tuple[int | None, int | None]:
    """
    Extract (genesis_time, period) from v2 or v1 /info responses.
    Be tolerant to possible key naming variants.
    """
    # Common/expected v2 keys
    gt = (
        payload.get("genesis_time") or payload.get("genesisTime") or payload.get("genesis")
    )  # some docs use 'genesis'
    pd = payload.get("period") or payload.get("round_time") or payload.get("roundTime")

    # Many responses encode period as seconds int; sometimes strings—normalize:
    try:
        if isinstance(pd, str):
            pd = int(pd)
    except Exception:
        pd = None

    try:
        if isinstance(gt, str):
            gt = int(gt)
    except Exception:
        gt = None

    return gt, pd


def _http_get_json(paths: list[str]) -> dict[str, Any] | None:
    """
    Try all relays × all paths (v2-first, then v1) and return first JSON payload.
    """
    for base in _shuffle_urls():
        for path in paths:
            if not path:
                continue
            url = f"{base}{path}"
            try:
                r = _SESSION.get(url, timeout=(1.5, 3.5), headers=_HEADERS)
                if r.status_code == 200:
                    try:
                        return r.json()  # type: ignore[no-any-return]
                    except Exception as e:
                        logger.debug(
                            f"[Drand] JSON parse error for {url}: {e}; body[:160]={r.text[:160]!r}"
                        )
                        continue
                else:
                    logger.debug(f"[Drand] GET {url} -> HTTP {r.status_code} {r.text[:160]!r}")
            except Exception as e:
                logger.debug(f"[Drand] GET {url} error: {e}")
    return None


def _fetch_chain_info(chain_hash: str) -> dict[str, Any] | None:
    """
    Programmatically fetch chain info (genesis_time, period, pubkey…) from relays.
    Tries v2 then v1 shapes; caches on success.
    """
    if chain_hash in _CHAIN_INFO_CACHE:
        return _CHAIN_INFO_CACHE[chain_hash]

    v2_info = f"/v2/chains/{chain_hash}/info"
    v1_info = f"/{chain_hash}/info"
    # Additional v1 fallback for 'default' (some relays allow /info at root):
    root_info = "/info" if chain_hash == DRAND_CHAINS["default"]["hash"] else None

    payload = _http_get_json([v2_info, v1_info, root_info])  # type: ignore[list-item]
    if not payload:
        logger.debug(f"[Drand] chain info fetch failed for {chain_hash}")
        return None

    # Normalize core fields we care about; keep full payload for callers.
    gt, pd = _parse_chain_info_payload(payload)
    if gt is not None:
        payload["genesis_time"] = gt
    if pd is not None:
        payload["period"] = pd

    _CHAIN_INFO_CACHE[chain_hash] = payload
    return payload


def _ensure_params(refresh: bool = False) -> None:
    """
    Ensure _DRAND_CHAIN_HASH / _DRAND_GENESIS_TIME / _DRAND_PERIOD are populated.
    If refresh=True, re-fetch chain info and overwrite cached values.
    """
    global _DRAND_CHAIN_HASH, _DRAND_GENESIS_TIME, _DRAND_PERIOD

    rec = _get_chain_record(_current_chain)
    chain_hash = rec["hash"]
    if (
        refresh
        or _DRAND_CHAIN_HASH != chain_hash
        or _DRAND_GENESIS_TIME is None
        or _DRAND_PERIOD is None
    ):
        # Start with configured defaults:
        gt = rec.get("genesis_time")
        pd = rec.get("period")

        # Try to fetch live chain info to override:
        info = _fetch_chain_info(chain_hash)
        if info:
            gt_live = info.get("genesis_time")
            pd_live = info.get("period")
            if isinstance(gt_live, int) and gt_live > 0:
                gt = gt_live
            if isinstance(pd_live, int) and pd_live > 0:
                pd = pd_live

        _DRAND_CHAIN_HASH = chain_hash
        _DRAND_GENESIS_TIME = int(gt) if gt is not None else None
        _DRAND_PERIOD = int(pd) if pd is not None else None

        msg = (
            f"[Drand] chain={_current_chain} hash={_DRAND_CHAIN_HASH} "
            f"period={_DRAND_PERIOD} genesis_time={_DRAND_GENESIS_TIME}"
        )
        if info:
            logger.info(msg + " (refreshed via /info)")
        else:
            logger.info(msg + " (using configured defaults)")


# ─────────────────────────────  PUBLIC API  ─────────────────────────────


def set_chain(chain_name: str, refresh_info: bool = True) -> None:
    """
    Switch the active drand chain. Optionally refresh chain info from relays.

    Args:
        chain_name: 'quicknet' or 'default'
        refresh_info: if True, programmatically fetch /info and update period/genesis_time
    """
    global _current_chain
    _get_chain_record(chain_name)  # validate early
    _current_chain = chain_name
    with _LOCK:
        _ensure_params(refresh=refresh_info)
    logger.info(
        f"Switched to drand chain '{chain_name}': {DRAND_CHAINS[chain_name]['description']}"
    )


def get_current_chain() -> dict[str, Any]:
    """
    Return details of the active chain, including resolved period/genesis_time if known.
    """
    _ensure_params(refresh=False)
    rec = _get_chain_record(_current_chain).copy()
    rec.update(
        {
            "name": _current_chain,
            "hash": _DRAND_CHAIN_HASH,
            "period": _DRAND_PERIOD,
            "genesis_time": _DRAND_GENESIS_TIME,
        }
    )
    return rec


def get_drand_beacon(round_id: int | None = None, use_fallback: bool = True) -> dict[str, Any]:
    """
    Fetch randomness from the drand network (v2-first, v1 fallback).

    Returns:
        {
            "source": "drand",
            "chain": <name>,
            "chain_hash": <hex>,
            "period": <int seconds>,
            "round": <int>,
            "randomness": <hex str>,
            "signature": <hex str | None>,
            "previous_signature": <hex str | None>,
        }
    """
    _ensure_params(refresh=False)
    if _DRAND_CHAIN_HASH is None or _DRAND_PERIOD is None:
        raise RuntimeError("Drand chain parameters not initialized")

    rid = "latest" if round_id is None else str(int(round_id))
    v2_path = f"/v2/chains/{_DRAND_CHAIN_HASH}/rounds/{rid}"
    v1_path = f"/{_DRAND_CHAIN_HASH}/public/{rid}"
    # Default chain extra fallback (root v1 path without chain hash)
    root_v1 = f"/public/{rid}" if _DRAND_CHAIN_HASH == DRAND_CHAINS["default"]["hash"] else None

    data = _http_get_json([v2_path, v1_path, root_v1])  # type: ignore[list-item]
    if not data:
        logger.warning("[Drand] All relays/paths failed to fetch beacon")
        if use_fallback:
            return get_mock_beacon()
        raise RuntimeError("drand fetch failed (all relays/paths)")

    rnd = data.get("randomness")
    rno = data.get("round")
    if rnd is None or rno is None:
        logger.debug(f"[Drand] Missing fields in response: {json.dumps(data)[:200]}")
        if use_fallback:
            return get_mock_beacon()
        raise RuntimeError("drand response missing required fields")

    logger.debug(f"[Drand-{_current_chain}] ok round={rno} rand={str(rnd)[:8]}…")
    return {
        "source": "drand",
        "chain": _current_chain,
        "chain_hash": _DRAND_CHAIN_HASH,
        "period": _DRAND_PERIOD,
        "round": int(rno),
        "randomness": str(rnd),
        "signature": data.get("signature"),
        "previous_signature": data.get("previous_signature"),
    }


def get_mock_beacon() -> dict[str, Any]:
    """
    Fallback mock beacon for testing/development; uniform schema.
    """
    global _BEACON_COUNTER
    _ensure_params(refresh=False)
    with _LOCK:
        _BEACON_COUNTER += 1
        rno = _BEACON_COUNTER
    rnd = os.urandom(32).hex()
    logger.debug(f"[MockBeacon] round={rno} randomness={rnd[:8]}…")
    return {
        "source": "mock",
        "chain": _current_chain,
        "chain_hash": _DRAND_CHAIN_HASH,
        "period": _DRAND_PERIOD,
        "round": rno,
        "randomness": rnd,
        "signature": None,
        "previous_signature": None,
    }


def get_beacon(
    round_id: str = "latest", use_drand: bool = True, use_fallback: bool = True
) -> dict[str, Any]:
    """
    Convenience wrapper:
      - round_id: "latest" or round number as string/int
      - use_drand=False forces mock
      - use_fallback=False raises on network errors (useful in tests)
    """
    if not use_drand:
        return get_mock_beacon()
    try:
        rid = None if str(round_id) == "latest" else int(round_id)
        return get_drand_beacon(rid, use_fallback=use_fallback)
    except Exception as e:
        logger.warning(f"[Drand] get_beacon error: {e}")
        if use_fallback:
            return get_mock_beacon()
        raise


def get_round_at_time(timestamp: int) -> int:
    """
    Compute the drand round number for a given UNIX timestamp.

    Spec nuance:
      * At t == genesis_time, the round is 1.
      * For t < genesis_time, there is no round yet -> return 0.
      * For t > genesis_time, round = 1 + floor((t - genesis_time)/period).
    """
    _ensure_params(refresh=False)
    if _DRAND_GENESIS_TIME is None or _DRAND_PERIOD is None:
        raise RuntimeError("Drand chain parameters not initialized")

    if timestamp < _DRAND_GENESIS_TIME:
        return 0
    return 1 + (timestamp - _DRAND_GENESIS_TIME) // _DRAND_PERIOD


def get_expected_round() -> int | None:
    """
    Query the chain health to get the current/expected round (v2).
    Useful to clamp future-round requests.
    """
    _ensure_params(refresh=False)
    path = f"/v2/chains/{_DRAND_CHAIN_HASH}/health"
    payload = _http_get_json([path])
    if not payload:
        return None
    # health payload typically includes fields like "expected_round" or similar
    for key in ("expected_round", "expectedRound", "expected"):
        if key in payload:
            try:
                return int(payload[key])
            except Exception:
                pass
    # Some relays nest the fields
    try:
        return int(payload.get("round", {}).get("expected"))  # be liberal
    except Exception:
        return None


# ───────────────────────────────  BOOTSTRAP  ───────────────────────────────

# Honor DRAND_CHAIN env var at import time and warm up parameters.
try:
    _ensure_params(refresh=True)
except Exception as _e:
    # Don't crash imports; we will retry on first call.
    logger.debug(f"[Drand] initial chain-info refresh failed: {_e}")
