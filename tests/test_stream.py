"""Tests for streaming API: rank_stream()."""

from unittest.mock import MagicMock, patch

from pathfinder_sdk.core import Pathfinder
from pathfinder_sdk.models import CandidateRecommendation


class TestRankStream:
    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_stream_yields_in_order(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = [
            CandidateRecommendation(rank=1, href="/a", text="A", score=0.95),
            CandidateRecommendation(rank=2, href="/b", text="B", score=0.85),
            CandidateRecommendation(rank=3, href="/c", text="C", score=0.75),
        ]
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher=None)
        results = list(
            pf.rank_stream(
                url="https://example.com",
                task_description="Find A",
                candidates=[
                    {"href": "/a", "text": "A"},
                    {"href": "/b", "text": "B"},
                    {"href": "/c", "text": "C"},
                ],
            )
        )

        assert len(results) == 3
        assert results[0].score == 0.95
        assert results[1].score == 0.85
        assert results[2].score == 0.75

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_stream_respects_top_n(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = [
            CandidateRecommendation(rank=1, href="/a", text="A", score=0.95),
            CandidateRecommendation(rank=2, href="/b", text="B", score=0.85),
        ]
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher=None)
        results = list(
            pf.rank_stream(
                url="https://example.com",
                task_description="Find A",
                candidates=[
                    {"href": "/a", "text": "A"},
                    {"href": "/b", "text": "B"},
                    {"href": "/c", "text": "C"},
                ],
                top_n=2,
            )
        )

        assert len(results) == 2

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_stream_early_termination_min_score(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = [
            CandidateRecommendation(rank=1, href="/a", text="A", score=0.95),
            CandidateRecommendation(rank=2, href="/b", text="B", score=0.85),
            CandidateRecommendation(rank=3, href="/c", text="C", score=0.75),
        ]
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher=None)
        results = list(
            pf.rank_stream(
                url="https://example.com",
                task_description="Find A",
                candidates=[
                    {"href": "/a", "text": "A"},
                    {"href": "/b", "text": "B"},
                    {"href": "/c", "text": "C"},
                ],
                min_score=0.80,
            )
        )

        assert len(results) == 2
        assert results[-1].score >= 0.80

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_stream_empty_candidates(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = []
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher=None)
        results = list(
            pf.rank_stream(
                url="https://example.com",
                task_description="Find A",
                candidates=[],
            )
        )

        assert results == []

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_stream_no_candidates_below_min_score(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = [
            CandidateRecommendation(rank=1, href="/a", text="A", score=0.50),
            CandidateRecommendation(rank=2, href="/b", text="B", score=0.40),
        ]
        mock_ranker_class.return_value = mock_ranker

        pf = Pathfinder(fetcher=None)
        results = list(
            pf.rank_stream(
                url="https://example.com",
                task_description="Find A",
                candidates=[{"href": "/a", "text": "A"}],
                min_score=0.80,
            )
        )

        assert results == []
