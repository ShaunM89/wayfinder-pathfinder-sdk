"""Tests for Pathfinder core class."""

from unittest.mock import MagicMock, patch

import pytest

from pathfinder_sdk.core import Pathfinder
from pathfinder_sdk.models import RankingResult


class TestPathfinderInit:
    def test_default_init(self):
        pf = Pathfinder()
        assert pf.model_tier == "default"
        assert pf.top_n == 20

    def test_custom_init(self):
        pf = Pathfinder(model="high", top_n=10)
        assert pf.model_tier == "high"
        assert pf.top_n == 10


class TestRankCandidates:
    @patch("pathfinder_sdk.core.BiEncoderRanker")
    @patch("pathfinder_sdk.core.Fetcher")
    def test_with_pre_extracted_candidates(self, mock_fetcher_class, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = []
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher=None)
        candidates = [
            {"href": "/a", "text": "A"},
            {"href": "/b", "text": "B"},
        ]
        result = pf.rank_candidates(
            url="https://example.com",
            task_description="Find A",
            candidates=candidates,
        )

        assert isinstance(result, RankingResult)
        assert result.task_description == "Find A"
        assert result.source_url == "https://example.com"
        assert result.total_links_analyzed == 2
        assert result.model_tier == "default"
        mock_ranker.rank.assert_called_once()

        # Stage latencies populated (fetch should be ~0 since no fetch happened)
        assert "stage_latencies" in result.metadata
        sl = result.metadata["stage_latencies"]
        assert "fetch_ms" in sl
        assert "filter_ms" in sl
        assert "rank_ms" in sl
        assert sl["fetch_ms"] >= 0
        assert sl["filter_ms"] >= 0
        assert sl["rank_ms"] >= 0

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    @patch("pathfinder_sdk.core.Fetcher")
    def test_fetch_when_no_candidates(self, mock_fetcher_class, mock_ranker_class):
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = [
            {"href": "/a", "text": "A"},
        ]
        mock_fetcher_class.return_value = mock_fetcher

        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = []
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher="auto")
        result = pf.rank_candidates(
            url="https://example.com",
            task_description="Find A",
            candidates=None,
        )

        assert isinstance(result, RankingResult)
        mock_fetcher.fetch.assert_called_once_with("https://example.com")

    def test_no_fetcher_and_no_candidates_raises(self):
        pf = Pathfinder(fetcher=None)
        with pytest.raises(ValueError, match="no fetcher is configured"):
            pf.rank_candidates(
                url="https://example.com",
                task_description="Find A",
                candidates=None,
            )

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    @patch("pathfinder_sdk.core.Fetcher")
    def test_top_n_override(self, mock_fetcher_class, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = []
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(top_n=20)
        pf.rank_candidates(
            url="https://example.com",
            task_description="Find A",
            candidates=[{"href": "/a", "text": "A"}],
            top_n=5,
        )

        call_kwargs = mock_ranker.rank.call_args[1]
        assert call_kwargs["top_n"] == 5
