"""Tests for PipelinedMiningEngine lifecycle and proof submission."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from grail.environments.backends import GenerationParams
from grail.mining.config import PipelineConfig
from grail.mining.engine import PipelinedMiningEngine


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(enabled=True)


@pytest.fixture
def mock_weight_sync() -> MagicMock:
    ws = MagicMock()
    ws.get_backend.return_value = MagicMock()
    return ws


@pytest.fixture
def mock_proof_worker() -> MagicMock:
    pw = MagicMock()
    pw.tokenizer = MagicMock()
    pw.tokenizer.pad_token_id = 0
    pw.tokenizer.eos_token_id = 1
    pw.tokenizer.decode = MagicMock(return_value="test output")
    return pw


def _make_engine(
    config: PipelineConfig,
    mock_weight_sync: MagicMock,
    mock_proof_worker: MagicMock,
    gen_params: GenerationParams | None = None,
) -> PipelinedMiningEngine:
    return PipelinedMiningEngine(config, mock_weight_sync, mock_proof_worker, gen_params=gen_params)


class TestPipelinedMiningEngine:
    """Engine lifecycle and proof dispatch."""

    def test_default_gen_params(
        self,
        config: PipelineConfig,
        mock_weight_sync: MagicMock,
        mock_proof_worker: MagicMock,
    ) -> None:
        """Engine uses default GenerationParams when none provided."""
        engine = _make_engine(config, mock_weight_sync, mock_proof_worker)
        assert isinstance(engine._gen_params, GenerationParams)
        engine.shutdown()

    def test_custom_gen_params_propagated(
        self,
        config: PipelineConfig,
        mock_weight_sync: MagicMock,
        mock_proof_worker: MagicMock,
    ) -> None:
        """Custom gen_params override the default."""
        params = GenerationParams(temperature=0.5, max_new_tokens=128)
        engine = _make_engine(config, mock_weight_sync, mock_proof_worker, gen_params=params)
        assert engine._gen_params.temperature == 0.5
        assert engine._gen_params.max_new_tokens == 128
        engine.shutdown()

    def test_shutdown_stops_proof_worker(
        self,
        config: PipelineConfig,
        mock_weight_sync: MagicMock,
        mock_proof_worker: MagicMock,
    ) -> None:
        """Shutdown delegates to proof worker and is idempotent."""
        engine = _make_engine(config, mock_weight_sync, mock_proof_worker)
        engine.shutdown()
        mock_proof_worker.shutdown.assert_called_once()

        # Second shutdown should not raise
        engine.shutdown()

    def test_submit_proofs_dispatches_to_worker(
        self,
        config: PipelineConfig,
        mock_weight_sync: MagicMock,
        mock_proof_worker: MagicMock,
    ) -> None:
        """submit_proofs dispatches to proof_worker and returns valid results."""
        engine = _make_engine(config, mock_weight_sync, mock_proof_worker)
        batch_data = [([1, 2, 3, 4, 5], 2, 1.0, {"success": True})]
        mock_proof_worker.compute_commitments_and_logprobs.return_value = [
            ([{"sketch_hash": "abc"}], [0.1, 0.2, 0.3], b"sig", {"randomness": "hex"}, "v1")
        ]

        future = engine.submit_proofs(batch_data, "deadbeef", MagicMock())
        result = future.result(timeout=5)

        assert len(result) == 1
        commitments, logprobs, _sig, _beacon, version = result[0]
        assert commitments == [{"sketch_hash": "abc"}]
        assert logprobs == [0.1, 0.2, 0.3]
        assert version == "v1"
        engine.shutdown()
