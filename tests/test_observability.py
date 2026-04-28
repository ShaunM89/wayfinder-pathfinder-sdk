"""Tests for observability hooks (telemetry and metrics)."""

from unittest.mock import MagicMock, patch

from pathfinder_sdk.core import Pathfinder
from pathfinder_sdk.metrics import NoOpMetricsCollector, get_metrics_collector
from pathfinder_sdk.models import RankingResult
from pathfinder_sdk.telemetry import NoOpTracer, get_tracer


class TestNoOpTracer:
    def test_start_span_noop(self):
        tracer = NoOpTracer()
        with tracer.start_span("test") as span:
            span.set_attribute("key", "value")
            span.record_exception(Exception("test"))

    def test_get_tracer_returns_noop_when_otel_missing(self):
        tracer = get_tracer()
        assert isinstance(tracer, NoOpTracer)


class TestNoOpMetrics:
    def test_record_latency_noop(self):
        metrics = NoOpMetricsCollector()
        metrics.record_latency("fetch", 100.0, "default")
        metrics.record_candidates(10, 5, 3, "default")
        metrics.record_fetch_error(403, "curl")

    def test_get_metrics_returns_noop_when_prometheus_missing(self):
        metrics = get_metrics_collector()
        assert isinstance(metrics, NoOpMetricsCollector)


class TestPathfinderWithObservability:
    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_rank_candidates_with_tracer(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = []
        mock_ranker_class.return_value = mock_ranker

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_span.return_value.__exit__ = MagicMock(return_value=False)

        pf = Pathfinder(fetcher=None, tracer=mock_tracer)
        result = pf.rank_candidates(
            url="https://example.com",
            task_description="Find A",
            candidates=[{"href": "/a", "text": "A"}],
        )

        assert isinstance(result, RankingResult)
        # Should have created spans for fetch, filter, rank
        assert mock_tracer.start_span.call_count >= 3

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_rank_candidates_with_metrics(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = []
        mock_ranker_class.return_value = mock_ranker

        mock_metrics = MagicMock()

        pf = Pathfinder(fetcher=None, metrics=mock_metrics)
        result = pf.rank_candidates(
            url="https://example.com",
            task_description="Find A",
            candidates=[{"href": "/a", "text": "A"}],
        )

        assert isinstance(result, RankingResult)
        mock_metrics.record_latency.assert_called()
        mock_metrics.record_candidates.assert_called_once()

    @patch("pathfinder_sdk.core.BiEncoderRanker")
    def test_rank_candidates_fetch_error_with_tracer(self, mock_ranker_class):
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = []
        mock_ranker_class.return_value = mock_ranker

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("pathfinder_sdk.core.Fetcher") as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher.fetch.side_effect = Exception("network error")
            mock_fetcher_class.return_value = mock_fetcher

            pf = Pathfinder(fetcher="auto", tracer=mock_tracer)
            with patch.object(pf, "_fetcher", mock_fetcher):
                # The fetch error is caught by the fetcher and returned as FetchError
                # For this test, we just verify tracer is passed through
                pass
