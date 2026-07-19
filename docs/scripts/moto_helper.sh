#!/bin/bash
# Reusable moto S3-mock helper for grail Phase 2 (goal.md S0-S4).
#
# Usage:  source moto_helper.sh
#         moto_start <port>     # start moto server (inside grail-miner.sif) + create bucket
#         moto_env <port>       # export R2_* env vars for grail comms
#         moto_stop             # kill the server started by moto_start
#
# Lesson (S0 iter 1): compute-node HOST python is 3.6 (login node is 3.10), so
# moto must run inside the container (python 3.12, identical on every node).
# Wheels are cp312 at $WORKDIR/tools/moto_env312; the moto PYTHONPATH is scoped
# to the server process only, so it never leaks into grail's own process.
# Data is in-memory: bucket contents die with the server (fine for S0-S3;
# revisit for S4 60GB payloads — see goal.md risk note).

MOTO_WORKDIR=/storage/openpsi/users/pengzai.pyq
MOTO_SIF=/storage/openpsi/images/grail-miner.sif
MOTO_PY=/app/.venv/bin/python
MOTO_PID=""

moto_start() {
  local port=${1:-19555}
  # Refuse to start if the port is already answering: a stale moto from a
  # previous run means STALE BUCKET STATE (S2 iter 1 burned us: S1's server
  # survived moto_stop because SIGTERM to the singularity launcher does not
  # reach the inner python; the next run then saw S1's checkpoint-30).
  if curl -s -m 2 "http://127.0.0.1:$port/" >/dev/null 2>&1; then
    echo "FAIL: port $port already in use — stale moto from a previous run? Kill it first."
    return 1
  fi
  # setsid => child leads its own process group; moto_stop kills the whole
  # group so the in-container python dies too.
  setsid singularity exec --no-home -B /storage:/storage \
    --env PYTHONPATH=$MOTO_WORKDIR/tools/moto_env312 \
    "$MOTO_SIF" $MOTO_PY -m moto.server -p "$port" \
    > /tmp/moto_$port.log 2>&1 &
  MOTO_PID=$!
  echo "moto server pgid=$MOTO_PID port=$port (containerized, py3.12)"
  for i in $(seq 1 30); do
    sleep 2
    if singularity exec --no-home -B /storage:/storage \
         --env PYTHONPATH=$MOTO_WORKDIR/tools/moto_env312 \
         "$MOTO_SIF" $MOTO_PY - "$port" <<'EOF'
import sys
import boto3
port = sys.argv[1]
s3 = boto3.client("s3", endpoint_url=f"http://127.0.0.1:{port}",
                  aws_access_key_id="testing", aws_secret_access_key="testing",
                  region_name="us-east-1")
s3.create_bucket(Bucket="grail")
objs = s3.list_objects_v2(Bucket="grail").get("KeyCount", 0)
print(f"bucket 'grail' ready ({objs} existing objects)")
assert objs == 0, f"bucket NOT empty ({objs} objects) — stale server state!"
EOF
    then return 0; fi
  done
  echo "FAIL: moto server did not come up"; cat /tmp/moto_$port.log; return 1
}

moto_env() {
  local port=${1:-19555}
  export R2_ENDPOINT_URL="http://127.0.0.1:$port"
  export R2_FORCE_PATH_STYLE=true
  export R2_BUCKET_ID=grail
  export R2_ACCOUNT_ID=testing
  export R2_READ_ACCESS_KEY_ID=testing
  export R2_READ_SECRET_ACCESS_KEY=testing
  export R2_WRITE_ACCESS_KEY_ID=testing
  export R2_WRITE_SECRET_ACCESS_KEY=testing
}

moto_stop() {
  if [ -n "$MOTO_PID" ]; then
    kill -TERM -- "-$MOTO_PID" 2>/dev/null
    sleep 2
    kill -KILL -- "-$MOTO_PID" 2>/dev/null
    wait "$MOTO_PID" 2>/dev/null
  fi
  echo "moto stopped (process group)"
}
