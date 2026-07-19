"""S2 acceptance: run the real MinerNeuron with the chain mock, no GPU.

Expected (goal.md S2 acceptance):
  - passes ensure_registered (mock uid=0)
  - enters main loop, computes window numbers from fake blocks
  - loops on "No checkpoint for window N" (checkpoint fetch comes in S3)
The wrapper kills the miner after GRAIL_S2_RUNTIME_S seconds; the srun
wrapper then greps the log for the acceptance markers.

Bypasses grail.cli entirely (monitoring stays uninitialized -> miner's
`if monitor:` guards handle None).
"""

import asyncio
import logging
import os
import sys

# We bypass grail.cli, so no handlers exist on the "grail.*" logger tree and
# Python's lastResort only passes WARNING+ — INFO logs (incl. streamed SGLang
# server output) would vanish. basicConfig gives everything a real handler.
logging.basicConfig(
    level=os.getenv("GRAIL_LOG_LEVEL", "INFO"),
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
)

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")
sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail/docs/scripts")

import grail_chain_mock  # noqa: E402

grail_chain_mock.install()

from grail.neurons import MinerNeuron  # noqa: E402


async def run_with_deadline() -> None:
    runtime = float(os.getenv("GRAIL_S2_RUNTIME_S", "180"))
    neuron = MinerNeuron(use_drand=False)
    task = asyncio.create_task(neuron.main())
    try:
        await asyncio.wait_for(task, timeout=runtime)
    except asyncio.TimeoutError:
        print(f"\nS2 wrapper: {runtime:.0f}s elapsed, stopping miner (expected)", flush=True)
        neuron.stop_event.set()
        try:
            await asyncio.wait_for(task, timeout=30)
        except (asyncio.TimeoutError, Exception):
            task.cancel()


if __name__ == "__main__":
    asyncio.run(run_with_deadline())
