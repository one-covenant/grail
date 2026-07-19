#!/bin/bash
#SBATCH -J pyq-grail-s4-sync-bench
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --reservation=app_algo
#SBATCH -w slurmd-44
#SBATCH -o /storage/openpsi/users/pengzai.pyq/logs/grail-%j.out
#SBATCH -e /storage/openpsi/users/pengzai.pyq/logs/grail-%j.err
#
# S4 / Phase B: miner weight-sync timing bench (goal.md Phase 2 S4).
# Same harness as S3, plus a rotator process that publishes a fresh FULL
# checkpoint every mock window -> miner slow-path reload -> SYNC_METRIC lines.
#
# Tunables (sbatch --export or env):
#   ROTATIONS   how many checkpoints to publish after the initial one (default 2 = S4a shakeout)
#   RUNTIME_S   miner lifetime (default 1200; S4b: ~1200 + ROTATIONS*180 + buffer)

set -uxo pipefail

WORKDIR=/storage/openpsi/users/pengzai.pyq
GRAIL_SRC=$WORKDIR/grail
SIF=/storage/openpsi/images/grail-miner.sif
MINER_LOG=$WORKDIR/logs/grail-s4-miner-$SLURM_JOB_ID.log
ROTATOR_LOG=$WORKDIR/logs/grail-s4-rotator-$SLURM_JOB_ID.log

ROTATIONS=${ROTATIONS:-2}
RUNTIME_S=${RUNTIME_S:-1200}

R2ENV=R2_ENDPOINT_URL=http://127.0.0.1:19555,R2_FORCE_PATH_STYLE=true,R2_BUCKET_ID=grail,R2_ACCOUNT_ID=testing,R2_READ_ACCESS_KEY_ID=testing,R2_READ_SECRET_ACCESS_KEY=testing,R2_WRITE_ACCESS_KEY_ID=testing,R2_WRITE_SECRET_ACCESS_KEY=testing
MOCKENV=GRAIL_MOCK_BLOCK_TIME_S=6,GRAIL_MOCK_GENESIS_TS=1780000000

mkdir -p "$WORKDIR/.cache/home" "$WORKDIR/logs"
rm -rf "$WORKDIR/.cache/home/.cache/grail"   # cross-run checkpoint cache (S2 lesson)
rm -rf "$WORKDIR/.cache/home/.cache/tvm-ffi" # stale failed JIT builds (S3 lesson)

source $GRAIL_SRC/docs/scripts/moto_helper.sh
moto_start 19555 || exit 1

echo "===== publish initial checkpoint at current mock window ====="
singularity exec --no-home -B /storage:/storage \
  --env $R2ENV,$MOCKENV,GRAIL_PUBLISH_WINDOW=auto,GRAIL_PUBLISH_ONLY=1,GRAIL_ENV_ID=sat \
  "$SIF" bash -c "export HOME=$WORKDIR/.cache/home && exec /app/.venv/bin/python $GRAIL_SRC/docs/scripts/s1_publish_consume.py" \
  || { moto_stop; exit 1; }

echo "===== start mocked miner (background, ${RUNTIME_S}s) ====="
singularity exec --nv --no-home -B /storage:/storage \
  -B $WORKDIR/tools/libs/libnuma.so.1:/usr/lib/x86_64-linux-gnu/libnuma.so.1 \
  --env $R2ENV,$MOCKENV,BT_WALLET_COLD=grailbench,BT_WALLET_HOT=default,NETUID=200,GRAIL_MONITORING_BACKEND=null,GRAIL_S2_RUNTIME_S=$RUNTIME_S,GRAIL_PIPELINE_BACKEND=sglang,GRAIL_PIPELINE_VLLM_GPU=0,GRAIL_PIPELINE_PROOF_GPU=1,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
  "$SIF" bash -c "export HOME=$WORKDIR/.cache/home CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/app/.venv/bin:/usr/local/cuda/bin:\$PATH PYTHONUNBUFFERED=1 && exec /app/.venv/bin/python $GRAIL_SRC/docs/scripts/s2_run_miner_mocked.py" \
  > "$MINER_LOG" 2>&1 &
MINER_PID=$!

echo "===== wait for pipeline up, then rotate $ROTATIONS checkpoints ====="
for i in $(seq 1 60); do
  sleep 10
  grep -q "Pipeline engine initialized successfully" "$MINER_LOG" && break
  kill -0 $MINER_PID 2>/dev/null || { echo "miner died before pipeline init"; break; }
done

singularity exec --no-home -B /storage:/storage \
  --env $R2ENV,$MOCKENV,GRAIL_ROTATIONS=$ROTATIONS,BT_WALLET_COLD=grailbench \
  "$SIF" bash -c "export HOME=$WORKDIR/.cache/home && exec /app/.venv/bin/python $GRAIL_SRC/docs/scripts/s4_rotate_publisher.py" \
  > "$ROTATOR_LOG" 2>&1
echo "rotator rc=$?"

echo "===== wait miner to finish ====="
wait $MINER_PID
echo "miner rc=$?"
moto_stop

echo "===== S4 ACCEPTANCE CHECK ====="
echo "--- rotator ---"
tail -5 "$ROTATOR_LOG"
echo "--- SYNC_METRIC lines ---"
grep -c "^SYNC_METRIC" "$MINER_LOG" || true
grep "^SYNC_METRIC" "$MINER_LOG" | tail -15
echo "--- checkpoint events ---"
grep -oE '\{"event": "checkpoint".*' "$MINER_LOG" | tail -8
echo "--- crash check (want 0) ---"
grep -cE "SystemExit|Pipeline init failed" "$MINER_LOG" || true
echo "miner log: $MINER_LOG ; rotator log: $ROTATOR_LOG"
