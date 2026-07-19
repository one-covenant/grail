"""
S13 preflight — probe moto S3 mock throughput ceiling.

We plan to use moto to simulate R2 in same-cluster tests. Since moto is a
Python-single-process in-memory S3 mock (Flask + Werkzeug), its throughput
may be CPU-bound (~200-500 MB/s) rather than network-bound. This preflight
puts/gets dummy payloads of 3 sizes and reports MB/s, so we can decide
whether moto is fast enough for the transfer-segment bench (S13a/S13b)
or whether we need to source a real MinIO binary.

Usage (inside grail-miner.sif, moto server already up on --port):
    python s13_preflight_moto.py --port 19555 --sizes 1,3,5
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import boto3
from botocore.config import Config


def now_iso() -> str:
    # Simple UTC timestamp
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_bytes(size_gb: float) -> bytes:
    # Use os.urandom to avoid page-cache-friendly zero-fill compression illusions;
    # boto3 will send raw bytes regardless. Building in-memory keeps the source
    # off any file system cache and makes the timing pure network+moto.
    n = int(size_gb * (1024**3))
    print(f"  building {size_gb} GiB payload ({n} bytes) via os.urandom...", flush=True)
    t0 = time.perf_counter()
    buf = os.urandom(n)
    print(f"  built in {time.perf_counter()-t0:.2f}s", flush=True)
    return buf


def new_client(port: int) -> "boto3.client":
    return boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{port}",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            # boto3 default multipart threshold is 8MB; keep default so numbers
            # reflect the real client behavior.
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def run_one(client, bucket: str, key: str, payload: bytes) -> dict:
    size_bytes = len(payload)
    size_mb = size_bytes / (1024**2)

    # PUT
    t_put_start = time.perf_counter()
    client.put_object(Bucket=bucket, Key=key, Body=payload)
    t_put = time.perf_counter() - t_put_start

    # GET
    t_get_start = time.perf_counter()
    resp = client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    t_get = time.perf_counter() - t_get_start

    ok = len(body) == size_bytes and body[:64] == payload[:64] and body[-64:] == payload[-64:]

    return {
        "size_mb": size_mb,
        "size_gb": size_bytes / (1024**3),
        "t_put_s": t_put,
        "t_get_s": t_get,
        "put_mbps": size_mb / t_put if t_put > 0 else 0.0,
        "get_mbps": size_mb / t_get if t_get > 0 else 0.0,
        "roundtrip_ok": ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=19555)
    ap.add_argument("--sizes", type=str, default="0.5,1,3",
                    help="Comma-separated payload sizes in GiB")
    ap.add_argument("--rounds", type=int, default=3, help="repeats per size")
    ap.add_argument("--bucket", type=str, default="grail")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]

    print(f"[S13 preflight] port={args.port} sizes={sizes} rounds={args.rounds}", flush=True)
    client = new_client(args.port)

    # Sanity: bucket present, empty
    try:
        objs = client.list_objects_v2(Bucket=args.bucket).get("KeyCount", 0)
        print(f"  bucket '{args.bucket}' reachable ({objs} objects)", flush=True)
    except Exception as e:
        print(f"FAIL: cannot reach bucket: {e}", flush=True)
        return 2

    results = []
    for size in sizes:
        print(f"\n=== size {size} GiB ({args.rounds} rounds) ===", flush=True)
        payload = make_bytes(size)
        for r in range(args.rounds):
            key = f"s13-preflight/round-{r}-{int(size*1024)}mb.bin"
            rec = run_one(client, args.bucket, key, payload)
            rec["round"] = r
            results.append(rec)
            print(
                f"  r={r} size={rec['size_gb']:.2f}GiB "
                f"PUT={rec['t_put_s']:.3f}s ({rec['put_mbps']:.1f} MB/s) "
                f"GET={rec['t_get_s']:.3f}s ({rec['get_mbps']:.1f} MB/s) "
                f"ok={rec['roundtrip_ok']}",
                flush=True,
            )
            # cleanup to keep moto memory manageable
            client.delete_object(Bucket=args.bucket, Key=key)
        del payload

    # Steady-state = drop first per size, mean rest
    steady = {}
    for size in sizes:
        recs = [r for r in results if abs(r["size_gb"] - size) < 1e-6][1:]
        if not recs:
            continue
        n = len(recs)
        steady[f"{size}GiB"] = {
            "n_steady": n,
            "put_mbps_mean": sum(r["put_mbps"] for r in recs) / n,
            "get_mbps_mean": sum(r["get_mbps"] for r in recs) / n,
            "t_put_s_mean": sum(r["t_put_s"] for r in recs) / n,
            "t_get_s_mean": sum(r["t_get_s"] for r in recs) / n,
        }

    payload_out = {
        "ts": now_iso(),
        "moto_port": args.port,
        "sizes_gb": sizes,
        "rounds_per_size": args.rounds,
        "results": results,
        "steady": steady,
    }
    print("\n=== steady ===", flush=True)
    print(json.dumps(steady, indent=2), flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload_out, f, indent=2)
        print(f"\nJSON written: {args.out}", flush=True)

    # decision hint
    max_get = max((v["get_mbps_mean"] for v in steady.values()), default=0)
    max_put = max((v["put_mbps_mean"] for v in steady.values()), default=0)
    print(f"\n[decision hint] put_max={max_put:.0f} MB/s  get_max={max_get:.0f} MB/s", flush=True)
    if min(max_put, max_get) < 100:
        print("  → moto is CPU-bound. Consider real MinIO for S13 bench.", flush=True)
    else:
        print("  → moto throughput acceptable. Proceed with S13a/S13b bench.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
