"""Proof computation worker for pipelined mining.

Manages an HF model on a dedicated proof GPU (GPU 1) and computes
GRAIL commitments + logprobs for completed rollouts.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class ProofWorker:
    """Manages the HF model on a dedicated proof GPU.

    Loads the model to ``cuda:{config.proof_gpu}`` and exposes a method
    that mirrors ``AgentEnvLoop._batch_compute_commitments_and_logprobs``
    but delegates to the shared ``compute_proofs()`` function.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._device = f"cuda:{config.proof_gpu}"
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._hidden_dim: int | None = None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("ProofWorker model not loaded — call load_model() first")
        return self._tokenizer

    @property
    def device(self) -> str:
        return self._device

    def load_model(self, checkpoint_path: str) -> None:
        """Load (or reload) the HF model onto the proof GPU.

        Args:
            checkpoint_path: Path to the checkpoint directory.
        """
        from ..model.provider import clear_model_and_tokenizer, get_model, get_tokenizer
        from ..shared.hf_compat import resolve_hidden_size

        # Release previous model if any
        if self._model is not None:
            self._model, self._tokenizer = clear_model_and_tokenizer(self._model, self._tokenizer)

        logger.info("ProofWorker: loading model from %s to %s", checkpoint_path, self._device)
        self._model = get_model(str(checkpoint_path), device=self._device, eval_mode=True)
        self._tokenizer = get_tokenizer(str(checkpoint_path))
        self._hidden_dim = resolve_hidden_size(self._model)
        logger.info(
            "ProofWorker: model loaded (hidden_dim=%d, device=%s)",
            self._hidden_dim,
            self._device,
        )

    def set_model(self, model: Any, tokenizer: Any) -> None:
        """Use an already-loaded model instead of loading a duplicate.

        This avoids loading a second copy of the model when the caller
        already has one on the correct device.

        Args:
            model: Pre-loaded HF model (already on the proof GPU).
            tokenizer: Pre-loaded tokenizer.
        """
        from ..shared.hf_compat import resolve_hidden_size

        self._model = model
        self._tokenizer = tokenizer
        self._hidden_dim = resolve_hidden_size(model)
        logger.info(
            "ProofWorker: using shared model (hidden_dim=%d, device=%s)",
            self._hidden_dim,
            self._device,
        )

    def update_model_in_place(self, model: Any) -> None:
        """Update internal references when model weights have been applied in-place.

        Used after fast-path delta updates where the miner's HF model is
        already the same object as the proof worker's model.
        """
        self._model = model
        if self._hidden_dim is None:
            from ..shared.hf_compat import resolve_hidden_size

            self._hidden_dim = resolve_hidden_size(model)

    def compute_commitments_and_logprobs(
        self,
        all_ids_batch: list[list[int]],
        prompt_lens: list[int],
        randomness_hex: str,
        wallet: Any,
    ) -> list[tuple[list[dict], list[float], bytes, dict, str]]:
        """Compute GRAIL proofs on the proof GPU.

        Args:
            all_ids_batch: Full token sequences (prompt + completion) per rollout
            prompt_lens: Prompt length for each sequence
            randomness_hex: Hex randomness string
            wallet: Bittensor wallet for signing

        Returns:
            List of (commitments, logprobs, signature, beacon, proof_version) tuples
        """
        if self._model is None or self._hidden_dim is None:
            raise RuntimeError("ProofWorker model not loaded — call load_model() first")

        from ..environments.loop import compute_proofs

        return compute_proofs(
            self._model,
            self._device,
            self._hidden_dim,
            all_ids_batch,
            prompt_lens,
            randomness_hex,
            wallet,
        )

    def shutdown(self) -> None:
        """Release model and free GPU memory."""
        if self._model is not None:
            from ..model.provider import clear_model_and_tokenizer

            self._model, self._tokenizer = clear_model_and_tokenizer(self._model, self._tokenizer)
        self._hidden_dim = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ProofWorker: shutdown complete")
