#!/bin/bash
#SBATCH -J pyq-grail-s6-disk-segments
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --reservation=app_algo
#SBATCH -w slurmd-44
#SBATCH -o /storage/openpsi/users/pengzai.pyq/logs/grail-%j.out
#SBATCH -e /storage/openpsi/users/pengzai.pyq/logs/grail-%j.err
#
# S6: per-segment disk-backend sync bench, Qwen3-30B (57 GB).
# CPFS budget: staging + store + cache = ~171 GB, cleaned at end.
# --mem=400G: 57 GB state dict in RAM + page cache headroom.

set -uxo pipefail

WORKDIR=/storage/openpsi/users/pengzai.pyq
GRAIL_SRC=$WORKDIR/grail
SIF=/storage/openpsi/images/grail-miner.sif
JSON_OUT=$WORKDIR/logs/grail_s6_segments_$SLURM_JOB_ID.json

mkdir -p "$WORKDIR/.cache/home" "$WORKDIR/logs"
rm -rf "$WORKDIR/grail_bench/s6"
df -h /storage | tail -1

# forensics for the silent-death at 939692: catch signals to the batch shell,
# and write python output straight to CPFS (bypasses slurm stdout capture)
trap 'echo "TRAP: batch shell got SIGTERM at $(date +%T)"' TERM
trap 'echo "TRAP: batch shell got SIGINT at $(date +%T)"' INT
INNER_LOG=$WORKDIR/logs/grail-s6-inner-$SLURM_JOB_ID.log

singularity exec --nv --no-home -B /storage:/storage \
  -B $WORKDIR/tools/libs/libnuma.so.1:/usr/lib/x86_64-linux-gnu/libnuma.so.1 \
  --env S6_JSON_OUT=$JSON_OUT,S6_ITERS=3 \
  "$SIF" bash -c "export HOME=$WORKDIR/.cache/home CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/app/.venv/bin:/usr/local/cuda/bin:\$PATH PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && exec /app/.venv/bin/python $GRAIL_SRC/docs/scripts/s6_disk_segments_bench.py > $INNER_LOG 2>&1"
RC=$?

df -h /storage | tail -1
echo "===== S6 done rc=$RC ; JSON: $JSON_OUT ; inner log: $INNER_LOG ====="
