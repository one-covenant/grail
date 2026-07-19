#!/bin/bash
#SBATCH -J pyq-grail-s5-30b-reload
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --reservation=app_algo
#SBATCH -w slurmd-44
#SBATCH -o /storage/openpsi/users/pengzai.pyq/logs/grail-%j.out
#SBATCH -e /storage/openpsi/users/pengzai.pyq/logs/grail-%j.err
#
# S5a: SGLang /update_weights_from_disk microbench on Qwen3-30B-A3B (57 GB).
# Same server flags / env fixes as the grail miner (libnuma bind, CUDA_HOME
# override, writable HOME, PYTHONUNBUFFERED). No miner, no moto.

set -uxo pipefail

WORKDIR=/storage/openpsi/users/pengzai.pyq
GRAIL_SRC=$WORKDIR/grail
SIF=/storage/openpsi/images/grail-miner.sif
JSON_OUT=$WORKDIR/logs/grail_s5_reload_$SLURM_JOB_ID.json

mkdir -p "$WORKDIR/.cache/home" "$WORKDIR/logs"
rm -rf "$WORKDIR/.cache/home/.cache/tvm-ffi"

singularity exec --nv --no-home -B /storage:/storage \
  -B $WORKDIR/tools/libs/libnuma.so.1:/usr/lib/x86_64-linux-gnu/libnuma.so.1 \
  --env S5_JSON_OUT=$JSON_OUT,S5_COLD_ITERS=11,S5_WARM_ITERS=3 \
  "$SIF" bash -c "export HOME=$WORKDIR/.cache/home CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/app/.venv/bin:/usr/local/cuda/bin:\$PATH PYTHONUNBUFFERED=1 && exec /app/.venv/bin/python $GRAIL_SRC/docs/scripts/s5_sglang_reload_bench.py"
RC=$?

echo "===== S5a done rc=$RC ; JSON: $JSON_OUT ====="
