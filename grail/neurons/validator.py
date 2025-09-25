"""Validator neuron wrapper.

Stage 1 delegates to the existing CLI entry to avoid behavior changes.
In later stages, this neuron will orchestrate the validation service directly.
"""

from __future__ import annotations

import logging

from .base import BaseNeuron

logger = logging.getLogger(__name__)


class ValidatorNeuron(BaseNeuron):
    """Runs the validation loop under a unified neuron lifecycle."""

    def __init__(self, use_drand: bool = True, test_mode: bool = False) -> None:
        super().__init__()
        self.use_drand = use_drand
        self.test_mode = test_mode

    async def run(self) -> None:
        # Delegate to existing CLI entry for now (no behavior change in Stage 1)
        from grail.cli.validate import validate

        validate(use_drand=self.use_drand, test_mode=self.test_mode)
