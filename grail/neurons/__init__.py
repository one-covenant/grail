"""Neuron lifecycle abstractions for miner, validator, and trainer.

Stage 1 introduces a minimal neuron layer without changing runtime behavior.
Neurons delegate to existing CLI entry points; later stages will extract
orchestration into services and flip CLI to construct and run neurons.
"""

from .base import BaseNeuron
from .miner import MinerNeuron
from .trainer import TrainerNeuron
from .validator import ValidatorNeuron

__all__ = ["BaseNeuron", "MinerNeuron", "ValidatorNeuron", "TrainerNeuron"]
