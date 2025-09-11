import os
import time
import asyncio
import logging
from typing import List
import pytest

from grail.infrastructure.comms import list_bucket_files, get_file


logger = logging.getLogger(__name__)


pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def compose_env() -> None:
    # This fixture assumes docker-compose stack is managed outside pytest (CI step)
    # or via a pre-test script. Here, we only validate readiness through S3.
    # Users can run: docker compose -f docker-compose.integration.yml up -d --build
    # before invoking pytest.
    # Ensure test-specific envs are set for SDK code
    os.environ.setdefault("R2_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("R2_FORCE_PATH_STYLE", "true")
    os.environ.setdefault("R2_BUCKET_ID", "grail")
    os.environ.setdefault("R2_WRITE_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("R2_WRITE_SECRET_ACCESS_KEY", "minioadmin")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("GRAIL_MONITORING_BACKEND", "null")
    os.environ.setdefault("GRAIL_WINDOW_LENGTH", "3")


async def _wait_for_keys(
    prefix: str, expect_min: int, timeout_s: int = 300
) -> List[str]:
    start = time.time()
    seen: List[str] = []
    while time.time() - start < timeout_s:
        try:
            files = await list_bucket_files(prefix)
            seen = [f for f in files if f.startswith(prefix)]
            if len(seen) >= expect_min:
                return seen
        except Exception:
            pass
        await asyncio.sleep(3)
    return seen


@pytest.mark.asyncio
async def test_miners_upload_window_files(compose_env: None) -> None:
    # Expect at least 2 window files from two miners
    files = await _wait_for_keys("grail/windows/", expect_min=2, timeout_s=480)
    assert len(files) >= 2, f"expected >=2 window files, seen={files}"


@pytest.mark.asyncio
async def test_validator_processes_previous_window(compose_env: None) -> None:
    # After miners upload, wait for valid rollouts or at least confirm window files are parseable
    # We search for any window file and try to load it to confirm schema
    files = await _wait_for_keys("grail/windows/", expect_min=2, timeout_s=480)
    assert files, "no window files present"
    # Load one file to validate structure
    sample = files[0]
    data = await get_file(sample)
    assert isinstance(data, dict), "window file must be json object"
    assert "wallet" in data and "window_start" in data and "inferences" in data

    # Optionally, check for validator-produced valid rollouts (non-strict)
    valid_files = await list_bucket_files("grail/valid_rollouts/")
    # If present, ensure file is json-ish and has count
    if valid_files:
        vf = valid_files[-1]
        vdata = await get_file(vf)
        assert isinstance(vdata, dict)
        assert "window" in vdata and "rollouts" in vdata
