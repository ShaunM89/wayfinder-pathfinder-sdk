"""Tests for batch API: rank_multiple()."""

from unittest.mock import MagicMock, patch

from pathfinder_sdk.core import Pathfinder
from pathfinder_sdk.models import CandidateRecommendation, RankingResult


class TestRankMultiple:
    @patch("pathfinder_sdk.core.BiEncoderRanker")
    @patch("pathfinder_sdk.core.Fetcher")
    def test_rank_multiple_basic(self, mock_fetcher_class, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.side_effect = [
            [CandidateRecommendation(rank=1, href="/a", text="A", score=0.9)],
            [CandidateRecommendation(rank=1, href="/b", text="B", score=0.8)],
        ]
        mock_ranker_class.return_value = mock_ranker

        mock_fetcher = MagicMock()
        mock_fetcher.fetch.side_effect = [
            [{"href": "/a", "text": "A"}],
            [{"href": "/b", "text": "B"}],
        ]
        mock_fetcher_class.return_value = mock_fetcher

        pf = Pathfinder(fetcher="auto")
        results = pf.rank_multiple(
            [
                ("https://example.com/1", "Find A"),
                ("https://example.com/2", "Find B"),
            ]
        )

        assert len(results) == 2
        assert isinstance(results[0], RankingResult)
        assert results[0].source_url == "https://example.com/1"
        assert results[1].source_url == "https://example.com/2"

        # Model should only be loaded once (one ranker instance)
        mock_ranker_class.assert_called_once()

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_rank_multiple_with_pre_extracted(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.side_effect = [
            [CandidateRecommendation(rank=1, href="/a", text="A", score=0.9)],
            [CandidateRecommendation(rank=1, href="/b", text="B", score=0.8)],
        ]
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher=None)
        results = pf.rank_multiple(
            [
                ("https://example.com/1", "Find A", [{"href": "/a", "text": "A"}]),
                ("https://example.com/2", "Find B", [{"href": "/b", "text": "B"}]),
            ]
        )

        assert len(results) == 2
        assert results[0].total_links_analyzed == 1
        assert results[1].total_links_analyzed == 1

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_rank_multiple_top_n_override(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = []
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher=None, top_n=20)
        pf.rank_multiple(
            [("https://example.com", "Find A", [{"href": "/a", "text": "A"}])],
            top_n=5,
        )

        call_kwargs = mock_ranker.rank.call_args[1]
        assert call_kwargs["top_n"] == 5

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_rank_multiple_empty_input(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher=None)
        results = pf.rank_multiple([])

        assert results == []
        mock_ranker.rank.assert_not_called()

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_rank_multiple_preserves_order(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.side_effect = [
            [CandidateRecommendation(rank=1, href="/second", text="Second", score=0.8)],
            [CandidateRecommendation(rank=1, href="/first", text="First", score=0.9)],
        ]
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher=None)
        results = pf.rank_multiple(
            [
                (
                    "https://example.com/1",
                    "First",
                    [{"href": "/first", "text": "First"}],
                ),
                (
                    "https://example.com/2",
                    "Second",
                    [{"href": "/second", "text": "Second"}],
                ),
            ]
        )

        assert results[0].source_url == "https://example.com/1"
        assert results[1].source_url == "https://example.com/2"

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_rank_multiple_task_description_in_result(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = []
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher=None)
        results = pf.rank_multiple(
            [
                ("https://example.com", "Find privacy policy", []),
            ]
        )

        assert results[0].task_description == "Find privacy policy"
