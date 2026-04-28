"""Tests for CLI entry point."""

import json
from unittest.mock import MagicMock, patch

from pathfinder_sdk.cli import _format_table, main
from pathfinder_sdk.models import (
    CandidateRecommendation,
    FetchError,
    ModelLoadError,
    RankingResult,
)


class TestFormatTable:
    def test_empty_candidates(self):
        result = RankingResult(
            task_description="test",
            source_url="https://example.com",
            candidates=[],
            latency_ms=100.0,
            total_links_analyzed=0,
            total_links_after_filter=0,
            model_tier="default",
        )
        text = _format_table(result)
        assert "No candidates found" in text

    def test_single_candidate(self):
        result = RankingResult(
            task_description="test",
            source_url="https://example.com",
            candidates=[
                CandidateRecommendation(rank=1, href="/a", text="Link A", score=0.95),
            ],
            latency_ms=100.0,
            total_links_analyzed=5,
            total_links_after_filter=3,
            model_tier="default",
        )
        text = _format_table(result)
        assert "Link A" in text
        assert "0.950" in text or "0.95" in text
        assert "1" in text

    def test_multiple_candidates(self):
        result = RankingResult(
            task_description="test",
            source_url="https://example.com",
            candidates=[
                CandidateRecommendation(rank=1, href="/a", text="Link A", score=0.95),
                CandidateRecommendation(rank=2, href="/b", text="Link B", score=0.85),
            ],
            latency_ms=100.0,
            total_links_analyzed=5,
            total_links_after_filter=3,
            model_tier="default",
        )
        text = _format_table(result)
        assert "Link A" in text
        assert "Link B" in text


class TestMain:
    @patch("pathfinder_sdk.cli.Pathfinder")
    def test_rank_json_output(self, mock_pathfinder_class):
        mock_pf = MagicMock()
        mock_pf.rank_candidates.return_value = RankingResult(
            task_description="Find tutorial",
            source_url="https://example.com",
            candidates=[
                CandidateRecommendation(
                    rank=1, href="/tutorial", text="Tutorial", score=0.95
                ),
            ],
            latency_ms=150.0,
            total_links_analyzed=10,
            total_links_after_filter=5,
            model_tier="default",
        )
        mock_pathfinder_class.return_value = mock_pf

        code = main(
            [
                "rank",
                "https://example.com",
                "Find tutorial",
                "--output",
                "json",
                "--model",
                "high",
                "--top-n",
                "10",
                "--cache-dir",
                "/tmp/pf",
                "--fetcher",
                "curl",
            ]
        )

        assert code == 0
        mock_pathfinder_class.assert_called_once_with(
            model="high",
            top_n=10,
            cache_dir="/tmp/pf",
            fetcher="curl",
            device=None,
            quiet=False,
        )
        mock_pf.rank_candidates.assert_called_once_with(
            url="https://example.com", task_description="Find tutorial", top_n=10
        )

    @patch("pathfinder_sdk.cli.Pathfinder")
    def test_rank_table_output(self, mock_pathfinder_class, capsys):
        mock_pf = MagicMock()
        mock_pf.rank_candidates.return_value = RankingResult(
            task_description="Find tutorial",
            source_url="https://example.com",
            candidates=[
                CandidateRecommendation(
                    rank=1, href="/tutorial", text="Tutorial", score=0.95
                ),
            ],
            latency_ms=150.0,
            total_links_analyzed=10,
            total_links_after_filter=5,
            model_tier="default",
        )
        mock_pathfinder_class.return_value = mock_pf

        code = main(
            ["rank", "https://example.com", "Find tutorial", "--output", "table"]
        )

        assert code == 0
        captured = capsys.readouterr()
        assert "Tutorial" in captured.out

    @patch("pathfinder_sdk.cli.Pathfinder")
    def test_fetch_error_exit_code(self, mock_pathfinder_class):
        mock_pf = MagicMock()
        mock_pf.rank_candidates.side_effect = FetchError("blocked")
        mock_pathfinder_class.return_value = mock_pf

        code = main(["rank", "https://example.com", "Find tutorial"])

        assert code == 1

    @patch("pathfinder_sdk.cli.Pathfinder")
    def test_model_load_error_exit_code(self, mock_pathfinder_class):
        mock_pf = MagicMock()
        mock_pf.rank_candidates.side_effect = ModelLoadError("onnx failed")
        mock_pathfinder_class.return_value = mock_pf

        code = main(["rank", "https://example.com", "Find tutorial"])

        assert code == 2

    @patch("pathfinder_sdk.cli.Pathfinder")
    def test_no_candidates_exit_code(self, mock_pathfinder_class):
        mock_pf = MagicMock()
        mock_pf.rank_candidates.return_value = RankingResult(
            task_description="Find tutorial",
            source_url="https://example.com",
            candidates=[],
            latency_ms=50.0,
            total_links_analyzed=0,
            total_links_after_filter=0,
            model_tier="default",
        )
        mock_pathfinder_class.return_value = mock_pf

        code = main(["rank", "https://example.com", "Find tutorial"])

        assert code == 3

    @patch("pathfinder_sdk.cli.Pathfinder")
    def test_json_output_is_valid(self, mock_pathfinder_class, capsys):
        mock_pf = MagicMock()
        mock_pf.rank_candidates.return_value = RankingResult(
            task_description="Find tutorial",
            source_url="https://example.com",
            candidates=[
                CandidateRecommendation(
                    rank=1, href="/tutorial", text="Tutorial", score=0.95
                ),
            ],
            latency_ms=150.0,
            total_links_analyzed=10,
            total_links_after_filter=5,
            model_tier="default",
        )
        mock_pathfinder_class.return_value = mock_pf

        code = main(
            ["rank", "https://example.com", "Find tutorial", "--output", "json"]
        )

        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["source_url"] == "https://example.com"
        assert len(data["candidates"]) == 1
        assert data["candidates"][0]["href"] == "/tutorial"

    @patch("pathfinder_sdk.cli.Pathfinder")
    def test_unload_called_on_success(self, mock_pathfinder_class):
        mock_pf = MagicMock()
        mock_pf.rank_candidates.return_value = RankingResult(
            task_description="Find tutorial",
            source_url="https://example.com",
            candidates=[],
            latency_ms=50.0,
            total_links_analyzed=0,
            total_links_after_filter=0,
            model_tier="default",
        )
        mock_pathfinder_class.return_value = mock_pf

        main(["rank", "https://example.com", "Find tutorial"])

        mock_pf.unload.assert_called_once()

    @patch("pathfinder_sdk.cli.Pathfinder")
    def test_unload_called_on_error(self, mock_pathfinder_class):
        mock_pf = MagicMock()
        mock_pf.rank_candidates.side_effect = FetchError("blocked")
        mock_pathfinder_class.return_value = mock_pf

        main(["rank", "https://example.com", "Find tutorial"])

        mock_pf.unload.assert_called_once()
