#!/usr/bin/env python3
"""Simple wrapper to run the offline GRPO trainer.

The *offline pipeline* (`grail_offline.pipelines.offline_grpo`) already manages the vLLM
server lifecycle (start, reload, shutdown). This script exists purely as a convenience
entrypoint so users can run:

  uv run python run_training_simple.py

and get the same behavior as running `run_offline_grpo.py` directly.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the offline GRPO trainer via `uv run`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    script_dir = Path(__file__).parent.resolve()
    logger.info("Starting offline GRPO training", extra={"workdir": str(script_dir)})

    result = subprocess.run(
        ["uv", "run", "python", "run_offline_grpo.py"],
        cwd=str(script_dir),
        check=False,
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
