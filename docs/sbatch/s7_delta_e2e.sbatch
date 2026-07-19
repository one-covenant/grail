#!/bin/bash
#SBATCH -J pyq-grail-s7-delta-e2e
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=03:00:00
#SBATCH --reservation=app_algo
#SBATCH -w slurmd-44
#SBATCH -o /storage/openpsi/users/pengzai.pyq/logs/grail-%j.out
#SBATCH -e /storage/openpsi/users/pengzai.pyq/logs/grail-%j.err
#
# S7 run #3: grail DELTA-chained end-to-end, SEQUENCED (goal.md Phase 2 S7).
# Lessons from 939734/939775: (a) moto multipart completion needs
# GRAIL_S3_READ_TIMEOUT_S; (b) 57GB miner download via moto is ~34MB/s (~28min),
# so rotations must NOT start until the miner is up -> three phases:
#   1. rotator bootstrap-only: publish FULL at window W
#   2. pre-seed miner local cache with the staged checkpoint (skips the 28-min
#      moto download; consumer verifies local manifest and uses it directly)
#   3. start miner; once pipeline is up, rotator deltas-only (fast path per window)

set -uxo pipefail

WORKDIR=/storage/openpsi/users/pengzai.pyq
GRAIL_SRC=$WORKDIR/grail
SIF=/storage/openpsi/images/grail-miner.sif
MINER_LOG=$WORKDIR/logs/grail-s7-miner-$SLURM_JOB_ID.log
ROTATOR_LOG=$WORKDIR/logs/grail-s7-rotator-$SLURM_JOB_ID.log

ROTATIONS=${ROTATIONS:-4}
RUNTIME_S=${RUNTIME_S:-9000}

R2ENV=R2_ENDPOINT_URL=http://127.0.0.1:19555,R2_FORCE_PATH_STYLE=true,R2_BUCKET_ID=grail,R2_ACCOUNT_ID=testing,R2_READ_ACCESS_KEY_ID=testing,R2_READ_SECRET_ACCESS_KEY=testing,R2_WRITE_ACCESS_KEY_ID=testing,R2_WRITE_SECRET_ACCESS_KEY=testing,GRAIL_S3_READ_TIMEOUT_S=900
MOCKENV=GRAIL_MOCK_BLOCK_TIME_S=12,GRAIL_MOCK_GENESIS_TS=1780000000

mkdir -p "$WORKDIR/.cache/home" "$WORKDIR/logs"
rm -rf "$WORKDIR/.cache/home/.cache/grail" "$WORKDIR/.cache/home/.cache/tvm-ffi" "$WORKDIR/grail_bench/s7"

source $GRAIL_SRC/docs/scripts/moto_helper.sh

# GPU-empty preflight: singularity children can ESCAPE slurm job teardown on
# this cluster (S7 run #3 root cause: run #2's scancel'd miner survived, its
# SGLang held 126.9GB on GPU0 and the new server OOM'd). Fail loudly if dirty.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "FAIL: GPUs not empty — escaped processes from a previous run:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi

moto_start 19555 || exit 1

echo "===== phase 1: bootstrap FULL (foreground) ====="
singularity exec --no-home -B /storage:/storage \
  --env $R2ENV,$MOCKENV,S7_MODE=bootstrap,BT_WALLET_COLD=grailbench,GRAIL_ENV_ID=sat,OMP_NUM_THREADS=64 \
  "$SIF" bash -c "export HOME=$WORKDIR/.cache/home PYTHONUNBUFFERED=1 && exec /app/.venv/bin/python $GRAIL_SRC/docs/scripts/s7_rotate_delta_publisher.py" \
  > "$ROTATOR_LOG" 2>&1 || { echo "bootstrap failed"; tail -20 "$ROTATOR_LOG"; moto_stop; exit 1; }

ANCHOR=$(grep -m1 '"event": "bootstrap_full"' "$ROTATOR_LOG" | grep -oE '"window": [0-9]+' | grep -oE '[0-9]+')
echo "anchor window: $ANCHOR"
[ -n "$ANCHOR" ] || { echo "no anchor parsed"; moto_stop; exit 1; }

echo "===== phase 2: pre-seed miner local cache (skip 28-min moto download) ====="
CACHE_DIR=$WORKDIR/.cache/home/.cache/grail/checkpoints/checkpoint-$ANCHOR
mkdir -p "$CACHE_DIR"
cp $WORKDIR/grail_bench/s7/staging/* "$CACHE_DIR"/
ls -la "$CACHE_DIR"

echo "===== phase 3: start miner, wait for pipeline, then deltas ====="
setsid singularity exec --nv --no-home -B /storage:/storage \
  -B $WORKDIR/tools/libs/libnuma.so.1:/usr/lib/x86_64-linux-gnu/libnuma.so.1 \
  --env $R2ENV,$MOCKENV,BT_WALLET_COLD=grailbench,BT_WALLET_HOT=default,NETUID=200,GRAIL_MONITORING_BACKEND=null,GRAIL_S2_RUNTIME_S=$RUNTIME_S,GRAIL_PIPELINE_BACKEND=sglang,GRAIL_PIPELINE_VLLM_GPU=0,GRAIL_PIPELINE_PROOF_GPU=1,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
  "$SIF" bash -c "export HOME=$WORKDIR/.cache/home CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/app/.venv/bin:/usr/local/cuda/bin:\$PATH PYTHONUNBUFFERED=1 && exec /app/.venv/bin/python $GRAIL_SRC/docs/scripts/s2_run_miner_mocked.py" \
  > "$MINER_LOG" 2>&1 &
MINER_PID=$!

for i in $(seq 1 90); do
  sleep 10
  grep -q "Pipeline engine initialized successfully" "$MINER_LOG" && break
  kill -0 $MINER_PID 2>/dev/null || { echo "miner died before pipeline init"; tail -25 "$MINER_LOG"; moto_stop; exit 1; }
done
grep -q "Pipeline engine initialized successfully" "$MINER_LOG" || { echo "pipeline never initialized"; tail -25 "$MINER_LOG"; moto_stop; exit 1; }

singularity exec --no-home -B /storage:/storage \
  --env $R2ENV,$MOCKENV,S7_MODE=deltas,S7_ANCHOR_WINDOW=$ANCHOR,S7_ROTATIONS=$ROTATIONS,S7_DENSITY=0.01,BT_WALLET_COLD=grailbench,GRAIL_ENV_ID=sat,OMP_NUM_THREADS=64 \
  "$SIF" bash -c "export HOME=$WORKDIR/.cache/home PYTHONUNBUFFERED=1 && exec /app/.venv/bin/python $GRAIL_SRC/docs/scripts/s7_rotate_delta_publisher.py" \
  >> "$ROTATOR_LOG" 2>&1
echo "rotator deltas rc=$?"

echo "===== give miner 2 windows to consume last delta, then stop ====="
sleep 800
# kill the whole process group: TERM to the singularity launcher alone does
# not reach the in-container python (moto lesson, now confirmed for miner too)
kill -TERM -- -$MINER_PID 2>/dev/null
sleep 20
kill -KILL -- -$MINER_PID 2>/dev/null
wait $MINER_PID
echo "miner rc=$?"
moto_stop
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true

echo "===== S7 ACCEPTANCE CHECK ====="
echo "--- rotator metrics ---"
grep "^ROTATOR_METRIC" "$ROTATOR_LOG"
echo "--- fast path events ---"
grep -cE "Fast path: applying delta in-place" "$MINER_LOG" || true
grep -E "Fast path complete|Model updated in-place" "$MINER_LOG" | tail -6
echo "--- checkpoint events ---"
grep -oE '\{"event": "checkpoint".*' "$MINER_LOG" | tail -8
echo "--- SYNC_METRIC ---"
grep "^SYNC_METRIC" "$MINER_LOG" | tail -8
echo "--- crash check ---"
grep -cE "SystemExit|Pipeline init failed" "$MINER_LOG" || true
echo "miner log: $MINER_LOG ; rotator log: $ROTATOR_LOG"
