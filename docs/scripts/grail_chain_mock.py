"""Chain mock for grail miner bench (goal.md Phase 2 S2).

Replaces every bittensor-chain touchpoint the miner path uses, WITHOUT
modifying grail source or the sif image. Import and call install() before
constructing MinerNeuron.

Touchpoints covered (source-verified, miner path only):
  1. BaseNeuron.get_subtensor        -> FakeSubtensor (time-driven blocks)
  2. BaseNeuron.ensure_registered    -> returns uid 0 (base.py:112 SystemExit path)
  3. GrailChainManager.initialize    -> no-op (skips worker process + commitments)
     GrailChainManager.get_bucket    -> None => miner falls back to LOCAL env
                                        credentials (miner.py:147-151), i.e. moto
     GrailChainManager.stop          -> no-op
  4. drand is NOT mocked: pass use_drand=False (official --no-drand path,
     cli/mine.py:242 falls back to block hash randomness)

Block pacing: fake block = (now - GENESIS) / GRAIL_MOCK_BLOCK_TIME_S.
Real chain is 12 s/block, WINDOW_LENGTH=30 => 6 min per window. For bench
iteration speed set GRAIL_MOCK_BLOCK_TIME_S=2 (window every 60 s).
GENESIS is a fixed epoch so separate processes agree on block numbers.
"""

import hashlib
import logging
import os
import time

logger = logging.getLogger("grail.chainmock")

BLOCK_TIME_S = float(os.getenv("GRAIL_MOCK_BLOCK_TIME_S", "12"))
GENESIS_TS = float(os.getenv("GRAIL_MOCK_GENESIS_TS", "1780000000"))


class FakeMetagraph:
    def __init__(self, netuid: int, hotkey: str) -> None:
        self.netuid = netuid
        self.hotkeys = [hotkey]
        self.uids = [0]
        self.n = 1


class FakeSubtensor:
    """Minimal async-subtensor stand-in for the miner code path."""

    def __init__(self, hotkey: str) -> None:
        self._hotkey = hotkey

    async def get_current_block(self) -> int:
        return int((time.time() - GENESIS_TS) / BLOCK_TIME_S)

    async def metagraph(self, netuid: int) -> FakeMetagraph:
        return FakeMetagraph(netuid, self._hotkey)

    async def get_block_hash(self, block: int) -> str:
        return "0x" + hashlib.sha256(f"grail-mock-block-{block}".encode()).hexdigest()

    async def close(self) -> None:
        pass


def _say(msg: str) -> None:
    # print AND log: when grail.cli is bypassed the "grail.*" logger tree has
    # no handlers, so logger.warning alone can vanish (S2 iter 1 evidence)
    print(f"[chainmock] {msg}", flush=True)
    logger.warning(msg)


def install() -> None:
    """Patch grail chain touchpoints in-place. Idempotent."""
    import bittensor as bt

    from grail.infrastructure.chain import GrailChainManager
    from grail.neurons import base as neurons_base

    async def fake_get_subtensor(self):  # noqa: ANN001, ANN202
        if not isinstance(getattr(self, "_subtensor", None), FakeSubtensor):
            wallet = bt.wallet(
                name=os.getenv("BT_WALLET_COLD", "default"),
                hotkey=os.getenv("BT_WALLET_HOT", "default"),
            )
            self._subtensor = FakeSubtensor(wallet.hotkey.ss58_address)
            _say(
                f"MOCK: FakeSubtensor installed (block_time={BLOCK_TIME_S}s, "
                f"current_block={int((time.time() - GENESIS_TS) / BLOCK_TIME_S)})"
            )
        return self._subtensor

    async def fake_ensure_registered(self, wallet, netuid, role="neuron"):  # noqa: ANN001, ANN202
        _say(f"MOCK: ensure_registered bypassed -> uid=0 (role={role})")
        return 0

    async def fake_initialize(self):  # noqa: ANN001, ANN202
        _say("MOCK: GrailChainManager.initialize no-op (no chain commitments)")

    neurons_base.BaseNeuron.get_subtensor = fake_get_subtensor
    neurons_base.BaseNeuron.reset_subtensor = lambda self: None
    neurons_base.BaseNeuron.ensure_registered = fake_ensure_registered
    GrailChainManager.initialize = fake_initialize
    # None => miner.py:147-151 falls back to local env credentials (our moto)
    GrailChainManager.get_bucket = lambda self, uid: None
    GrailChainManager.get_bucket_for_hotkey = lambda self, hotkey: None
    GrailChainManager.stop = lambda self: None
    _say("MOCK: chain mock installed (4 touchpoints)")
