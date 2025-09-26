"""Trainer neuron wrapper.

Stage 1 delegates to the existing CLI entry to avoid behavior changes.
As training matures, this will host the training service and optional
profiler hooks under the unified neuron lifecycle.
"""

from __future__ import annotations

import logging

from .base import BaseNeuron

logger = logging.getLogger(__name__)


class TrainerNeuron(BaseNeuron):
    """Runs the training loop under a unified neuron lifecycle."""

    def __init__(self) -> None:
        super().__init__()

    async def run(self) -> None:
        # Delegate to existing CLI entry for now (no behavior change in Stage 1)
        from grail.cli.train import train

        train()
