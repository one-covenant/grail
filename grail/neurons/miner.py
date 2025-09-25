"""Miner neuron wrapper.

Stage 1 delegates to the existing CLI entry to avoid behavior changes.
In later stages, this neuron will orchestrate the mining service directly.
"""

from __future__ import annotations

import logging

from .base import BaseNeuron

logger = logging.getLogger(__name__)


class MinerNeuron(BaseNeuron):
    """Runs the mining loop under a unified neuron lifecycle."""

    def __init__(self, use_drand: bool = True) -> None:
        super().__init__()
        self.use_drand = use_drand

    async def run(self) -> None:
        # Delegate to existing CLI entry for now (no behavior change in Stage 1)
        from grail.cli.mine import mine

        mine(use_drand=self.use_drand)
