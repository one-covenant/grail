#!/bin/bash
#SBATCH -J pyq-grail-s8-nccl
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=00:45:00
#SBATCH --reservation=app_algo
#SBATCH -w slurmd-44
#SBATCH -o /storage/openpsi/users/pengzai.pyq/logs/grail-%j.out
#SBATCH -e /storage/openpsi/users/pengzai.pyq/logs/grail-%j.err
#
# S8: grail-over-NCCL bench (goal.md Phase 2 S8). MODE=dense|delta via export.
# Two torchrun ranks on 2 GPUs, Qwen3-30B on each.

set -uxo pipefail

WORKDIR=/storage/openpsi/users/pengzai.pyq
GRAIL_SRC=$WORKDIR/grail
SIF=/storage/openpsi/images/grail-miner.sif
MODE=${MODE:-dense}
JSON_OUT=$WORKDIR/logs/grail_s8_${MODE}_$SLURM_JOB_ID.json

mkdir -p "$WORKDIR/.cache/home" "$WORKDIR/logs"

# GPU-empty preflight (singularity children escape slurm teardown; see memory)
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "FAIL: GPUs not empty:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi

singularity exec --nv --no-home -B /storage:/storage \
  -B $WORKDIR/tools/libs/libnuma.so.1:/usr/lib/x86_64-linux-gnu/libnuma.so.1 \
  --env S8_MODE=$MODE,S8_ROUNDS=6,S8_DENSITY=0.01,S8_JSON_OUT=$JSON_OUT,OMP_NUM_THREADS=60,NCCL_DEBUG=WARN \
  "$SIF" bash -c "export HOME=$WORKDIR/.cache/home CUDA_HOME=/usr/local/cuda PATH=/app/.venv/bin:\$PATH PYTHONUNBUFFERED=1 && exec /app/.venv/bin/torchrun --standalone --nproc_per_node=2 $GRAIL_SRC/docs/scripts/s8_nccl_bench.py"
RC=$?

echo "===== S8 $MODE done rc=$RC ; JSON: $JSON_OUT ====="
