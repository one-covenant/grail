#!/bin/bash
#SBATCH -J pyq-grail-preflight
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --reservation=swe-rl
#SBATCH -w slurmd-22
#SBATCH -o /storage/openpsi/users/pengzai.pyq/logs/grail-%j.out
#SBATCH -e /storage/openpsi/users/pengzai.pyq/logs/grail-%j.err
#
# grail delta checkpoint preflight + smoke (goal.md §4 + §5.1).
# CPU-only: delta encode/apply runs on CPU state dicts, no GPU needed.

set -euxo pipefail

WORKDIR=/storage/openpsi/users/pengzai.pyq
GRAIL_SRC=$WORKDIR/grail
CONTAINER=/storage/openpsi/images/verl-vllm.sif

mkdir -p "$WORKDIR/logs" "$WORKDIR/grail_bench/smoke" "$WORKDIR/.cache"/{home,hf,xdg}

[ -e "$CONTAINER" ] || { echo "FAIL: container missing"; exit 1; }
[ -d "$GRAIL_SRC" ] || { echo "FAIL: grail src missing"; exit 1; }

srun --mpi=pmi2 --ntasks=1 singularity exec --no-home --writable-tmpfs \
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
      export OMP_NUM_THREADS=16

      python3 '"$GRAIL_SRC"'/docs/scripts/probe_grail.py
    '
