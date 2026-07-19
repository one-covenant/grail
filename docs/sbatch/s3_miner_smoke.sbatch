#!/bin/bash
#SBATCH -J pyq-grail-s3-miner-smoke
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:50:00
#SBATCH --reservation=app_algo
#SBATCH -w slurmd-44
#SBATCH -o /storage/openpsi/users/pengzai.pyq/logs/grail-%j.out
#SBATCH -e /storage/openpsi/users/pengzai.pyq/logs/grail-%j.err
#
# S3 / Phase A: minimal end-to-end grail miner (goal.md Phase 2 S3).
#   GPU 0 = SGLang generation server, GPU 1 = HF proof model.
#   Chain fully mocked; R2 = moto; checkpoint = Qwen2.5-1.5B published at the
#   current mock window with env_id=sat (synthetic, no dataset downloads).
# Acceptance: checkpoint loaded, "Pipeline engine initialized successfully",
# miner survives >= 2 windows without SystemExit.
#
# Mock chain pacing: 6 s/block => 3 min/window (model load + SGLang startup
# need headroom; 2 s/block was fine for S2's no-GPU loop but too fast here).

set -uxo pipefail

WORKDIR=/storage/openpsi/users/pengzai.pyq
GRAIL_SRC=$WORKDIR/grail
SIF=/storage/openpsi/images/grail-miner.sif
LOG=$WORKDIR/logs/grail-s3-miner-$SLURM_JOB_ID.log

R2ENV=R2_ENDPOINT_URL=http://127.0.0.1:19555,R2_FORCE_PATH_STYLE=true,R2_BUCKET_ID=grail,R2_ACCOUNT_ID=testing,R2_READ_ACCESS_KEY_ID=testing,R2_READ_SECRET_ACCESS_KEY=testing,R2_WRITE_ACCESS_KEY_ID=testing,R2_WRITE_SECRET_ACCESS_KEY=testing
MOCKENV=GRAIL_MOCK_BLOCK_TIME_S=6,GRAIL_MOCK_GENESIS_TS=1780000000

mkdir -p "$WORKDIR/.cache/home" "$WORKDIR/logs"
rm -rf "$WORKDIR/.cache/home/.cache/grail"   # cross-run checkpoint cache (S2 lesson)
rm -rf "$WORKDIR/.cache/home/.cache/tvm-ffi" # stale failed JIT builds (S3 iter 2 lesson)

source $GRAIL_SRC/docs/scripts/moto_helper.sh
moto_start 19555 || exit 1

echo "===== publish checkpoint at current mock window (env_id=sat) ====="
singularity exec --no-home -B /storage:/storage \
  --env $R2ENV,$MOCKENV,GRAIL_PUBLISH_WINDOW=auto,GRAIL_PUBLISH_ONLY=1,GRAIL_ENV_ID=sat \
  "$SIF" bash -c "export HOME=$WORKDIR/.cache/home && exec /app/.venv/bin/python $GRAIL_SRC/docs/scripts/s1_publish_consume.py" \
  || { moto_stop; exit 1; }

echo "===== start mocked miner (2 GPUs, 25 min) ====="
# libnuma bind: grail-miner.sif lacks libnuma.so.1 which sgl_kernel's
# common_ops.abi3.so links against (root cause of S3 iter 1-2 SGLang exit 1).
# Extracted from verl-vllm.sif to tools/libs/.
singularity exec --nv --no-home -B /storage:/storage \
  -B $WORKDIR/tools/libs/libnuma.so.1:/usr/lib/x86_64-linux-gnu/libnuma.so.1 \
  --env $R2ENV,$MOCKENV,BT_WALLET_COLD=grailbench,BT_WALLET_HOT=default,NETUID=200,GRAIL_MONITORING_BACKEND=null,GRAIL_S2_RUNTIME_S=1500,GRAIL_PIPELINE_BACKEND=sglang,GRAIL_PIPELINE_VLLM_GPU=0,GRAIL_PIPELINE_PROOF_GPU=1,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
  "$SIF" bash -c "export HOME=$WORKDIR/.cache/home CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/app/.venv/bin:/usr/local/cuda/bin:\$PATH && exec /app/.venv/bin/python $GRAIL_SRC/docs/scripts/s2_run_miner_mocked.py" \
  > "$LOG" 2>&1
MINER_RC=$?
# CUDA_HOME/CUDA_PATH override: the host leaks CUDA_HOME=/usr/local/PPU_SDK/...
# into the container (singularity passes host env), so SGLang's JIT kernel
# compile calls a non-existent nvcc and the launcher SIGKILLs the tree
# (S3 iter 2 root cause). /usr/local/cuda is the container's real toolkit.

moto_stop

echo "===== S3 ACCEPTANCE CHECK (miner rc=$MINER_RC, 124/timeout-style exit is fine) ====="
echo "--- chainmock markers ---"
grep -c "chainmock" "$LOG" || true
echo "--- checkpoint loads ---"
grep -E "Loading checkpoint for window|Model updated in-place" "$LOG" | head -5
echo "--- pipeline init ---"
grep -c "Pipeline engine initialized successfully" "$LOG" || true
echo "--- windows touched ---"
grep -oE "window [0-9]+" "$LOG" | sort -u | wc -l
echo "--- crash check (want 0) ---"
grep -cE "SystemExit|Pipeline init failed" "$LOG" || true
echo "full log: $LOG"
