#!/usr/bin/env python3
"""
Rollout similarity detection over rolling windows.

Maintains per-miner rolling digest counts, an inverted index from digest to miners,
and computes efficient pairwise overlap to flag miners with excessive identical
rollouts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class SimilarityViolation:
    miner_a: str
    miner_b: str
    shared_pairs: int
    total_a: int
    total_b: int
    similarity: float


class SimilarityDetector:
    """
    Tracks rollout digest overlaps over a fixed rolling horizon.

    - add_window(miner, window, digest_counts, total_rollouts): updates state
    - compute_violations(): returns miner pairs above similarity threshold
    """

    def __init__(self, horizon_windows: int = 12, threshold: float = 0.5) -> None:
        self.horizon: int = max(1, int(horizon_windows))
        self.threshold: float = float(threshold)

        # Per-miner windowed digest counts
        self._window_counts: Dict[str, Dict[int, Counter[str]]] = defaultdict(dict)
        # Per-miner rolling digest counts (sum over last `horizon` windows)
        self._rolling_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        # Per-miner total digested rollouts (sum over last `horizon` windows)
        self._totals: Dict[str, int] = defaultdict(int)
        # Inverted index: digest -> miner -> count (rolling)
        self._inv: Dict[str, Dict[str, int]] = defaultdict(dict)

    def add_window(
        self, miner: str, window: int, digest_counts: Dict[str, int], total_rollouts: int
    ) -> None:
        """Add or replace a miner's digest counts for a specific window, updating rolling state."""
        miner = str(miner)
        window = int(window)

        # Remove old snapshot for this window if replacing
        if window in self._window_counts[miner]:
            self._remove_window(miner, window)

        snap = Counter({d: int(c) for d, c in digest_counts.items() if int(c) > 0})
        self._window_counts[miner][window] = snap

        # Apply to rolling counts and inverted index
        if snap:
            self._rolling_counts[miner].update(snap)
            for dig, c in snap.items():
                self._inv[dig][miner] = int(self._inv[dig].get(miner, 0)) + int(c)

        self._totals[miner] = int(self._totals.get(miner, 0)) + int(total_rollouts)

        # Evict beyond horizon if needed
        self._evict_old(miner)

    def _remove_window(self, miner: str, window: int) -> None:
        """Remove an existing window snapshot from rolling structures (internal)."""
        snap = self._window_counts[miner].pop(window, None)
        if not snap:
            return

        # Update rolling counts and inverted index
        self._rolling_counts[miner].subtract(snap)
        # Clean zeros in rolling counts
        self._rolling_counts[miner] += Counter()
        for dig, c in snap.items():
            prev = int(self._inv[dig].get(miner, 0)) - int(c)
            if prev > 0:
                self._inv[dig][miner] = prev
            else:
                self._inv[dig].pop(miner, None)
                if not self._inv[dig]:
                    self._inv.pop(dig, None)

        # Adjust totals conservatively by the sum of snapshot
        self._totals[miner] = max(0, int(self._totals.get(miner, 0)) - int(sum(snap.values())))

    def _evict_old(self, miner: str) -> None:
        """Keep only the most recent `horizon` windows for a miner."""
        windows = sorted(self._window_counts[miner].keys())
        if len(windows) <= self.horizon:
            return
        to_remove = windows[: len(windows) - self.horizon]
        for w in to_remove:
            self._remove_window(miner, w)

    def compute_violations(self) -> List[SimilarityViolation]:
        """Compute pairwise overlaps only where digests are shared across miners."""
        overlaps: Dict[Tuple[str, str], int] = defaultdict(int)

        for digest, miner_counts in self._inv.items():
            miners = list(miner_counts.keys())
            if len(miners) < 2:
                continue
            # Quadratic in number of miners sharing this digest, typically tiny
            for i in range(len(miners)):
                mi = miners[i]
                ci = int(miner_counts[mi])
                if ci <= 0:
                    continue
                for j in range(i + 1, len(miners)):
                    mj = miners[j]
                    cj = int(miner_counts[mj])
                    if cj <= 0:
                        continue
                    a, b = (mi, mj) if mi <= mj else (mj, mi)
                    overlaps[(a, b)] += ci * cj

        violations: List[SimilarityViolation] = []
        for (a, b), shared in overlaps.items():
            ta = int(self._totals.get(a, 0))
            tb = int(self._totals.get(b, 0))
            denom = ta * tb
            if denom <= 0:
                continue
            sim = float(shared) / float(denom)
            if sim >= self.threshold:
                violations.append(
                    SimilarityViolation(
                        miner_a=a,
                        miner_b=b,
                        shared_pairs=int(shared),
                        total_a=ta,
                        total_b=tb,
                        similarity=sim,
                    )
                )

        return violations

