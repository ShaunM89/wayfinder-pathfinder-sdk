"""Tests for BiEncoderRanker.

Note: These tests mock the model to avoid downloading weights in CI.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pathfinder_sdk.models import CandidateRecommendation, ModelNotFoundError
from pathfinder_sdk.ranker import BiEncoderRanker, _MODEL_REGISTRY


class TestModelRegistry:
    def test_known_tiers(self):
        assert "default" in _MODEL_REGISTRY
        assert "high" in _MODEL_REGISTRY
        assert "ultra" in _MODEL_REGISTRY


class TestBiEncoderRankerInit:
    def test_invalid_tier_raises(self):
        with pytest.raises(ModelNotFoundError):
            BiEncoderRanker(model_tier="nonexistent")

    def test_valid_tier(self):
        ranker = BiEncoderRanker(model_tier="default")
        assert ranker.model_tier == "default"


class TestBiEncoderRankerRank:
    def test_empty_candidates(self):
        ranker = BiEncoderRanker(model_tier="default")
        result = ranker.rank(task="Find privacy policy", candidates=[], top_n=5)
        assert result == []

    @patch("pathfinder_sdk.ranker.SentenceTransformer")
    def test_ranking_order(self, mock_st_class):
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
