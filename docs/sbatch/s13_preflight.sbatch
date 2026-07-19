#!/bin/bash
#SBATCH -J pyq-grail-s13-preflight
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --reservation=swe-rl
#SBATCH -w slurmd-109
#SBATCH -o /storage/openpsi/users/pengzai.pyq/grail-runs/logs/s13/s13-preflight-%j.out
#SBATCH -e /storage/openpsi/users/pengzai.pyq/grail-runs/logs/s13/s13-preflight-%j.err
#
# S13 preflight: probe moto S3 mock throughput ceiling (loopback).
# Sizes 0.5/1/3 GiB × 3 rounds. Steady = drop first, mean rest.
# If moto < 100 MB/s → need real MinIO. Otherwise proceed with S13a/S13b.

set -euxo pipefail

WORKDIR=/storage/openpsi/users/pengzai.pyq
GRAIL_SRC=$WORKDIR/grail
SIF=/storage/openpsi/images/grail-miner.sif
PORT=19555
LOG_ROOT=$WORKDIR/grail-runs/logs/s13
JSON_OUT=$LOG_ROOT/s13-preflight-$SLURM_JOB_ID.json

mkdir -p "$LOG_ROOT"

# ---- source moto helper (setsid + process-group teardown, from S1-S4) ----
source $GRAIL_SRC/docs/scripts/moto_helper.sh

# Teardown even if the bench fails
cleanup() { moto_stop || true; }
trap cleanup EXIT

echo "===== S13 preflight start: $(date -u +%FT%TZ) node=$(hostname) job=$SLURM_JOB_ID ====="
moto_start "$PORT" || exit 1

# Run the actual bench inside the same sif for python 3.12 + boto3 parity
singularity exec --no-home -B /storage:/storage \
  --env PYTHONPATH=$WORKDIR/tools/moto_env312 \
  "$SIF" /app/.venv/bin/python \
    "$GRAIL_SRC/docs/scripts/s13_preflight_moto.py" \
    --port "$PORT" \
    --sizes 0.5,1,3 \
    --rounds 3 \
    --out "$JSON_OUT"
RC=$?

echo "===== S13 preflight done rc=$RC JSON=$JSON_OUT ====="
exit $RC
