"""S0 acceptance: grail comms.py <-> moto S3 server round-trip (goal.md Phase 2 S0).

Runs INSIDE grail-miner.sif with PYTHONPATH pointing at the host grail checkout.
The moto server runs on the node host (started by the wrapper script); this
script talks to it over HTTP exactly like grail talks to R2.

Required env (set by wrapper):
  R2_ENDPOINT_URL=http://127.0.0.1:<port>
  R2_FORCE_PATH_STYLE=true
  R2_BUCKET_ID / R2_ACCOUNT_ID / R2_{READ,WRITE}_ACCESS_KEY_ID / R2_{READ,WRITE}_SECRET_ACCESS_KEY
"""

import asyncio
import os
import sys

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")

from grail.infrastructure import comms  # noqa: E402


async def main() -> None:
    print(f"endpoint: {os.environ['R2_ENDPOINT_URL']}, bucket: {os.environ['R2_BUCKET_ID']}")
    payload = os.urandom(1024 * 1024)  # 1 MiB random
    key = "grail/s0_probe/roundtrip.bin"

    ok = await comms.upload_file_chunked(key, payload, use_write=True)
    print("upload_file_chunked:", ok)
    assert ok, "upload failed"

    exists = await comms.file_exists(key)
    print("file_exists:", exists)
    assert exists, "uploaded key not found"

    got = await comms.download_file_chunked(key)
    print("download bytes:", None if got is None else len(got))
    assert got == payload, "download mismatch"

    print("S0 ACCEPTANCE: PASS")


if __name__ == "__main__":
    asyncio.run(main())
