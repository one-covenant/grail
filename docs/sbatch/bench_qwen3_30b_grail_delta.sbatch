#!/bin/bash
#SBATCH -J pyq-grail-qwen3-30b-delta-bench
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=04:00:00
#SBATCH --reservation=app_algo
#SBATCH -w slurmd-44
#SBATCH -o /storage/openpsi/users/pengzai.pyq/logs/grail-%j.out
#SBATCH -e /storage/openpsi/users/pengzai.pyq/logs/grail-%j.err
#
# grail sparse delta checkpoint bench on Qwen3-30B-A3B (goal.md §5.2).
#
# CPU-bound single process: encode/apply run on CPU state dicts; GPUs are
# requested only to hold the whole node (reserved anyway) and avoid CPU/mem
# bandwidth interference. --mem=800G because density 0.60 peaks ~400 GB RSS
# (base 60 + current 60 + encoded sparse 180 + loaded sparse ... sequential,
# see script header; slurmd-22 probed at 1.3T available).
#
# CPFS budget: at most one delta file on disk at a time (<= ~185 GB at
# density 0.60), deleted after each iter.

set -euxo pipefail

WORKDIR=/storage/openpsi/users/pengzai.pyq
GRAIL_SRC=$WORKDIR/grail
CONTAINER=/storage/openpsi/images/verl-vllm.sif
MODEL_PATH=/storage/openpsi/models/Qwen__Qwen3-30B-A3B
BENCH_DIR=$WORKDIR/grail_bench/bench_$SLURM_JOB_ID
JSON_OUT=$WORKDIR/logs/grail_bench_$SLURM_JOB_ID.json

PLAN=${PLAN:-0.01:11,0.05:11,0.30:4,0.60:3}

mkdir -p "$BENCH_DIR" "$WORKDIR/logs" "$WORKDIR/.cache"/{home,hf,xdg}

[ -e "$CONTAINER" ]  || { echo "FAIL: container missing"; exit 1; }
[ -d "$GRAIL_SRC" ]  || { echo "FAIL: grail src missing"; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "FAIL: model missing"; exit 1; }

df -h /storage | tail -2
echo "===== deltas write to: $BENCH_DIR ====="
echo "===== JSON output: $JSON_OUT ====="

srun --mpi=pmi2 --ntasks=1 singularity exec --nv --no-home --writable-tmpfs \
    -B /storage:/storage \
    "$CONTAINER" bash -lc '
      set -euxo pipefail
      export HOME='"$WORKDIR"'/.cache/home
      export HF_HOME='"$WORKDIR"'/.cache/hf
      export XDG_CACHE_HOME='"$WORKDIR"'/.cache/xdg
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
      export PYTHONPATH='"$GRAIL_SRC"':${PYTHONPATH:-}
      export PYTHONUNBUFFERED=1
      export PYTHONFAULTHANDLER=1
      export OMP_NUM_THREADS=64

      python3 '"$GRAIL_SRC"'/docs/scripts/bench_qwen3_30b_grail_delta.py \
        --model-path '"$MODEL_PATH"' \
        --work-dir '"$BENCH_DIR"' \
        --plan '"$PLAN"' \
        --json-output '"$JSON_OUT"'
    '

# ---- post-run cleanup ----
echo "===== POST-RUN cleanup ====="
if [ -d "$BENCH_DIR" ]; then
  du -sh "$BENCH_DIR" || true
  rm -rf "$BENCH_DIR"
fi
df -h /storage | tail -2
echo "JSON output: $JSON_OUT"
