"""Parse SYNC_METRIC lines from S4 miner logs (goal.md Phase 2 S4c).

Usage:
    python3 parse_sync_metrics.py <miner_log> [--json-output out.json]

Reads:
  SYNC_METRIC {"metric": "SYNC_METRIC", "backend": "sglang",
               "event": "sync_weights", "update_weights_from_disk_s": 1.234, ...}
plus the miner's structured checkpoint events:
  {"event": "checkpoint", "window": N, "method": "full", "load_sec": 19.07, ...}

Reports steady-state stats (first sync per boot = cold, skipped) in the
verl/slime/bcp parser style.
"""

import argparse
import json
import re
import statistics
from typing import Any

SYNC_RE = re.compile(r"^SYNC_METRIC (\{.*\})\s*$")
CKPT_RE = re.compile(r"(\{\"event\": \"checkpoint\".*?\})")


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    d = {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    if len(values) >= 2:
        d["stdev"] = statistics.stdev(values)
    return d


def fmt(d: dict[str, Any]) -> str:
    if d.get("n", 0) == 0:
        return "(no data)"
    parts = [f"n={d['n']}", f"mean={d['mean']:.3f}", f"median={d['median']:.3f}"]
    if "stdev" in d:
        parts.append(f"stdev={d['stdev']:.3f}")
    parts += [f"min={d['min']:.3f}", f"max={d['max']:.3f}"]
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    syncs: list[dict[str, Any]] = []
    starts: list[dict[str, Any]] = []
    ckpts: list[dict[str, Any]] = []
    with open(args.log, errors="replace") as f:
        for line in f:
            m = SYNC_RE.match(line)
            if m:
                rec = json.loads(m.group(1))
                (starts if rec.get("event") == "server_start" else syncs).append(rec)
                continue
            m = CKPT_RE.search(line)
            if m:
                try:
                    ckpts.append(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    pass

    print(f"### SYNC_METRIC events\n  server_start: {len(starts)}, sync_weights: {len(syncs)}")
    for rec in starts:
        print(f"  server_start (cold pipeline boot): {rec.get('server_start_s')}s")

    reload_all = [r["update_weights_from_disk_s"] for r in syncs if "update_weights_from_disk_s" in r]
    print("\n### SGLang /update_weights_from_disk (all syncs are steady: server already up)")
    for i, v in enumerate(reload_all):
        print(f"  #{i}  {v:.3f}s  ({syncs[i].get('checkpoint')})")
    reload_stats = stats(reload_all)
    print(f"  {fmt(reload_stats)}")

    dl = [c["load_sec"] for c in ckpts if c.get("method") == "full" and c.get("success")]
    dl_steady = dl[1:] if len(dl) > 1 else dl  # first = cold (page cache, connection setup)
    print("\n### checkpoint download+verify (miner load_or_update_model, method=full)")
    print(f"  all: {[round(v, 2) for v in dl]}")
    dl_stats = stats(dl_steady)
    print(f"  steady (skip first): {fmt(dl_stats)}")

    if reload_stats.get("n", 0) and dl_stats.get("n", 0):
        total = reload_stats["mean"] + dl_stats["mean"]
        print("\n### T_miner_sync ~= download + SGLang reload (proof-model reload excluded)")
        print(f"  mean: {dl_stats['mean']:.2f} + {reload_stats['mean']:.2f} = {total:.2f}s "
              f"(payload 3.1 GB, Qwen2.5-1.5B)")

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(
                {
                    "server_start": starts,
                    "sync_weights": syncs,
                    "checkpoint_events": ckpts,
                    "reload_stats": reload_stats,
                    "download_stats_steady": dl_stats,
                },
                f,
                indent=2,
            )
        print(f"\nJSON written to {args.json_output}")


if __name__ == "__main__":
    main()
