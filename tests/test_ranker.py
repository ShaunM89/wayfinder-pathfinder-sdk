"""Tests for BiEncoderRanker.

Note: These tests mock the model to avoid downloading weights in CI.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pathfinder_sdk.models import CandidateRecommendation, ModelLoadError, ModelNotFoundError
from pathfinder_sdk.ranker import BiEncoderRanker, _MODEL_REGISTRY, _download_with_retry


class TestModelRegistry:
    def test_known_tiers(self):
        assert "default" in _MODEL_REGISTRY
        assert "high" in _MODEL_REGISTRY
        assert "ultra" in _MODEL_REGISTRY


class TestDownloadWithRetry:
    @patch("pathfinder_sdk.ranker.snapshot_download")
    def test_success_on_first_attempt(self, mock_snap):
        mock_snap.return_value = "/fake/path"
        result = _download_with_retry("BAAI/bge-small", "/cache", max_retries=3)
        assert result == "/fake/path"
        mock_snap.assert_called_once()

    @patch("pathfinder_sdk.ranker.snapshot_download")
    def test_success_after_retries(self, mock_snap):
        mock_snap.side_effect = [ConnectionError("broken"), ConnectionError("broken"), "/fake/path"]
        result = _download_with_retry("BAAI/bge-small", "/cache", max_retries=3, base_delay=0.01)
        assert result == "/fake/path"
        assert mock_snap.call_count == 3

    @patch("pathfinder_sdk.ranker.snapshot_download")
    def test_all_retries_exhausted(self, mock_snap):
        mock_snap.side_effect = ConnectionError("broken")
        with pytest.raises(ModelLoadError, match="after 2 attempts"):
            _download_with_retry("BAAI/bge-small", "/cache", max_retries=2, base_delay=0.01)


class TestBiEncoderRankerInit:
    def test_invalid_tier_raises(self):
        with pytest.raises(ModelNotFoundError):
            BiEncoderRanker(model_tier="nonexistent")

    def test_valid_tier(self):
        ranker = BiEncoderRanker(model_tier="default")
        assert ranker.model_tier == "default"


class TestBiEncoderRankerLoadModel:
    @patch("pathfinder_sdk.ranker.SentenceTransformer")
    @patch("pathfinder_sdk.ranker._download_with_retry")
    def test_onnx_success(self, mock_download, mock_st):
        mock_download.return_value = "/fake/model"
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        ranker = BiEncoderRanker(model_tier="default")
        _ = ranker.model  # trigger load

        mock_st.assert_called_once()
        call_kwargs = mock_st.call_args[1]
        assert call_kwargs.get("backend") == "onnx"
        assert ranker.backend == "onnx"

    @patch("pathfinder_sdk.ranker.SentenceTransformer")
    @patch("pathfinder_sdk.ranker._download_with_retry")
    def test_onnx_falls_back_to_pytorch(self, mock_download, mock_st):
        mock_download.return_value = "/fake/model"
        mock_pt_model = MagicMock()
        # First call (ONNX) fails, second call (PyTorch) succeeds
        mock_st.side_effect = [RuntimeError("ONNX not supported"), mock_pt_model]

        ranker = BiEncoderRanker(model_tier="default")
        _ = ranker.model

        assert mock_st.call_count == 2
        assert ranker.backend == "pytorch"

    @patch("pathfinder_sdk.ranker.SentenceTransformer")
    @patch("pathfinder_sdk.ranker._download_with_retry")
    def test_local_path_bypasses_download(self, mock_download, mock_st):
        mock_st.return_value = MagicMock()

        ranker = BiEncoderRanker(local_model_path="/my/local/model")
        _ = ranker.model

        mock_download.assert_not_called()
        mock_st.assert_called()

    @patch("pathfinder_sdk.ranker.SentenceTransformer")
    @patch("pathfinder_sdk.ranker._download_with_retry")
    def test_both_backends_fail_raises(self, mock_download, mock_st):
        mock_download.return_value = "/fake/model"
        mock_st.side_effect = [
            RuntimeError("ONNX fail"),
            RuntimeError("PyTorch fail"),
        ]

        ranker = BiEncoderRanker(model_tier="default")
        with pytest.raises(ModelLoadError, match="ONNX error"):
            _ = ranker.model


class TestBiEncoderRankerRank:
    def test_empty_candidates(self):
        ranker = BiEncoderRanker(model_tier="default")
        result = ranker.rank(task="Find privacy policy", candidates=[], top_n=5)
        assert result == []

    @patch("pathfinder_sdk.ranker.SentenceTransformer")
    @patch("pathfinder_sdk.ranker._download_with_retry")
    def test_ranking_order(self, mock_download, mock_st_class):
        mock_download.return_value = "/fake/model"
        # Mock model.encode to return deterministic embeddings
        mock_model = MagicMock()
        # 3 candidates; task embedding + 3 candidate embeddings
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],   # task
            [0.9, 0.1, 0.0],   # cand 0 — most similar
            [0.5, 0.5, 0.0],   # cand 1
            [0.1, 0.1, 0.1],   # cand 2 — least similar
        ])
        mock_st_class.return_value = mock_model

        ranker = BiEncoderRanker(model_tier="default")
        # Force model load
        _ = ranker.model

        candidates = [
            {"href": "/a", "text": "A"},
            {"href": "/b", "text": "B"},
            {"href": "/c", "text": "C"},
        ]
        result = ranker.rank(task="Find A", candidates=candidates, top_n=2)

        assert len(result) == 2
        assert result[0].rank == 1
        assert result[0].href == "/a"
        assert result[1].rank == 2
        assert result[1].href == "/b"

        # Verify batch encoding was called (critical invariant)
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        texts = call_args[0][0]
        assert len(texts) == 4  # 1 task + 3 candidates

    @patch("pathfinder_sdk.ranker.SentenceTransformer")
    @patch("pathfinder_sdk.ranker._download_with_retry")
    def test_unload_resets_backend(self, mock_download, mock_st_class):
        mock_download.return_value = "/fake/model"
        mock_st_class.return_value = MagicMock()

        ranker = BiEncoderRanker(model_tier="default")
        _ = ranker.model
        assert ranker.backend in ("onnx", "pytorch")

        ranker.unload()
        assert ranker.backend == "pytorch"
        assert ranker._model is None
