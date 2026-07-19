"""
S13 transfer segment bench — client side.

Sends a fixed payload N rounds to an S3 endpoint (moto or MinIO), records:
- t_put_s / t_get_s (wall clock, boto3 put_object / get_object)
- put_mbps / get_mbps
- mem_cpu_baseline_gib / mem_cpu_hwm_gib / mem_cpu_delta_gib (VmRSS / VmHWM)

Steady-state = drop first round, mean the rest.

Usage (inside grail-miner.sif with boto3):
    python s13_transfer_bench.py \
        --endpoint http://<host>:<port> \
        --size-gb 5 --rounds 4 \
        --out /path/to/result.json

Payload is generated once via os.urandom (5 GiB stays resident in this
process's heap for the whole run so PUT/GET reflect boto3 + network only,
not payload generation).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time

import boto3
from botocore.config import Config


GIB = 1024**3
MIB = 1024**2


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_vm(field: str) -> float:
    """Return the requested /proc/self/status VmXXX value in GiB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(f"{field}:"):
                    m = re.search(r"(\d+)\s*kB", line)
                    if m:
                        return int(m.group(1)) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def make_payload(size_gb: float) -> bytes:
    n = int(size_gb * GIB)
    print(f"[build] {size_gb} GiB ({n} bytes) via os.urandom...", flush=True)
    t0 = time.perf_counter()
    buf = os.urandom(n)
    print(f"[build] done in {time.perf_counter()-t0:.2f}s", flush=True)
    return buf


def new_client(endpoint: str) -> "boto3.client":
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def wait_endpoint(endpoint: str, timeout_s: int = 300) -> bool:
    """Poll the endpoint until it answers, or timeout."""
    import urllib.request
    import urllib.error

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(endpoint, timeout=3) as r:
                r.read(1)
            return True
        except urllib.error.HTTPError:
            # 404/403 still means server is up
            return True
        except Exception:
            time.sleep(2)
    return False


def run_round(client, bucket: str, key: str, payload: bytes) -> dict:
    size_bytes = len(payload)
    size_mb = size_bytes / MIB

    mem_baseline_rss = read_vm("VmRSS")
    mem_baseline_hwm = read_vm("VmHWM")

    t_put_start = time.perf_counter()
    client.put_object(Bucket=bucket, Key=key, Body=payload)
    t_put = time.perf_counter() - t_put_start

    t_get_start = time.perf_counter()
    resp = client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    t_get = time.perf_counter() - t_get_start

    ok = (
        len(body) == size_bytes
        and body[:64] == payload[:64]
        and body[-64:] == payload[-64:]
    )

    mem_hwm = read_vm("VmHWM")
    mem_rss_after = read_vm("VmRSS")

    return {
        "size_gb": size_bytes / GIB,
        "size_mb": size_mb,
        "t_put_s": t_put,
        "t_get_s": t_get,
        "put_mbps": size_mb / t_put if t_put > 0 else 0.0,
        "get_mbps": size_mb / t_get if t_get > 0 else 0.0,
        "mem_cpu_rss_baseline_gib": mem_baseline_rss,
        "mem_cpu_hwm_baseline_gib": mem_baseline_hwm,
        "mem_cpu_hwm_peak_gib": mem_hwm,
        "mem_cpu_rss_after_gib": mem_rss_after,
        "mem_cpu_delta_gib": mem_hwm - mem_baseline_hwm,
        "roundtrip_ok": ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True,
                    help="S3 endpoint URL, e.g. http://172.16.0.5:19555")
    ap.add_argument("--bucket", default="grail")
    ap.add_argument("--size-gb", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--role", default="unlabeled",
                    help="baseline_loopback / cross_worker (for JSON tagging)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait-timeout-s", type=int, default=300)
    args = ap.parse_args()

    print(f"[S13-bench] role={args.role} endpoint={args.endpoint} "
          f"size={args.size_gb}GiB rounds={args.rounds}", flush=True)

    print(f"[S13-bench] hostname={socket.gethostname()}", flush=True)

    if not wait_endpoint(args.endpoint, args.wait_timeout_s):
        print(f"FAIL: endpoint {args.endpoint} not reachable in "
              f"{args.wait_timeout_s}s", flush=True)
        return 2

    client = new_client(args.endpoint)

    # Ensure bucket exists (idempotent). Worker may hit an empty bucket first.
    try:
        client.head_bucket(Bucket=args.bucket)
    except Exception:
        try:
            client.create_bucket(Bucket=args.bucket)
            print(f"[S13-bench] created bucket {args.bucket}", flush=True)
        except Exception as e:
            # Race with master, ignore
            print(f"[S13-bench] bucket creation race (ignored): {e}", flush=True)

    payload = make_payload(args.size_gb)

    results = []
    for r in range(args.rounds):
        key = f"s13/{args.role}/round-{r}-{int(args.size_gb*1024)}mb.bin"
        rec = run_round(client, args.bucket, key, payload)
        rec["round"] = r
        rec["role"] = args.role
        results.append(rec)
        print(
            f"  r={r} PUT={rec['t_put_s']:.3f}s ({rec['put_mbps']:.1f} MB/s) "
            f"GET={rec['t_get_s']:.3f}s ({rec['get_mbps']:.1f} MB/s) "
            f"mem_hwm={rec['mem_cpu_hwm_peak_gib']:.2f} GiB "
            f"(Δ={rec['mem_cpu_delta_gib']:+.2f}) ok={rec['roundtrip_ok']}",
            flush=True,
        )
        # Delete to keep moto memory bounded across rounds.
        client.delete_object(Bucket=args.bucket, Key=key)

    del payload

    # Steady = drop first, mean rest
    steady_recs = results[1:]
    if steady_recs:
        n = len(steady_recs)
        steady = {
            "n_steady": n,
            "put_mbps_mean": sum(r["put_mbps"] for r in steady_recs) / n,
            "get_mbps_mean": sum(r["get_mbps"] for r in steady_recs) / n,
            "t_put_s_mean": sum(r["t_put_s"] for r in steady_recs) / n,
            "t_get_s_mean": sum(r["t_get_s"] for r in steady_recs) / n,
            "mem_cpu_hwm_peak_gib_mean": sum(r["mem_cpu_hwm_peak_gib"] for r in steady_recs) / n,
            "mem_cpu_delta_gib_mean": sum(r["mem_cpu_delta_gib"] for r in steady_recs) / n,
        }
    else:
        steady = {}

    out_obj = {
        "ts": now_iso(),
        "hostname": socket.gethostname(),
        "role": args.role,
        "endpoint": args.endpoint,
        "size_gb": args.size_gb,
        "rounds": args.rounds,
        "results": results,
        "steady": steady,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"\n[S13-bench] steady = {json.dumps(steady, indent=2)}", flush=True)
    print(f"[S13-bench] JSON written: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
