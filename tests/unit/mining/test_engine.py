"""Tests for PipelinedMiningEngine pipeline overlap, drain, and exceptions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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


class TestPipelinedMiningEngine:
    def test_init(
        self,
        config: PipelineConfig,
        mock_weight_sync: MagicMock,
        mock_proof_worker: MagicMock,
    ) -> None:
        engine = PipelinedMiningEngine(config, mock_weight_sync, mock_proof_worker)
        assert engine._config is config

    def test_shutdown(
        self,
        config: PipelineConfig,
        mock_weight_sync: MagicMock,
        mock_proof_worker: MagicMock,
    ) -> None:
        engine = PipelinedMiningEngine(config, mock_weight_sync, mock_proof_worker)
        engine.shutdown()
        mock_proof_worker.shutdown.assert_called_once()

    def test_submit_proofs_returns_future(
        self,
        config: PipelineConfig,
        mock_weight_sync: MagicMock,
        mock_proof_worker: MagicMock,
    ) -> None:
        engine = PipelinedMiningEngine(config, mock_weight_sync, mock_proof_worker)
        batch_data = [([1, 2, 3, 4, 5], 2, 1.0, {"success": True})]
        mock_proof_worker.compute_commitments_and_logprobs.return_value = [
            ([{"sketch_hash": "abc"}], [0.1, 0.2, 0.3], b"sig", {"randomness": "hex"}, "v1")
        ]

        future = engine.submit_proofs(batch_data, "deadbeef", MagicMock())
        result = future.result(timeout=5)

        assert len(result) == 1
        assert result[0][0] == [{"sketch_hash": "abc"}]
        engine.shutdown()
