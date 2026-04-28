"""Optional Prometheus metrics integration for Pathfinder SDK.

Provides zero-overhead no-op fallbacks when prometheus_client is not installed.
"""


class NoOpMetricsCollector:
    """No-op metrics collector that does nothing."""

    def record_latency(self, stage: str, latency_ms: float, model_tier: str) -> None:
        """Record stage latency (no-op)."""
        pass

    def record_candidates(
        self,
        input_count: int,
        filtered_count: int,
        output_count: int,
        model_tier: str,
    ) -> None:
        """Record candidate counts (no-op)."""
        pass

    def record_fetch_error(self, status_code: int, fetcher: str) -> None:
        """Record a fetch error (no-op)."""
        pass


class PrometheusMetricsCollector:
    """Prometheus metrics collector using prometheus_client.

    Args:
        registry: Optional prometheus_client registry. If None, uses default.
    """

    def __init__(self, registry=None):
        from prometheus_client import Counter, Histogram

        self._latency = Histogram(
            "pathfinder_rank_latency_seconds",
            "Per-stage latency in seconds",
            ["stage", "model_tier"],
            registry=registry,
        )
        self._candidates = Counter(
            "pathfinder_candidates_total",
            "Number of candidates at each stage",
            ["stage", "model_tier"],
            registry=registry,
        )
        self._fetch_errors = Counter(
            "pathfinder_fetch_errors_total",
            "Fetch errors by status code",
            ["status_code", "fetcher"],
            registry=registry,
        )

    def record_latency(self, stage: str, latency_ms: float, model_tier: str) -> None:
        """Record stage latency in seconds."""
        self._latency.labels(stage=stage, model_tier=model_tier).observe(
            latency_ms / 1000.0
        )

    def record_candidates(
        self,
        input_count: int,
        filtered_count: int,
        output_count: int,
        model_tier: str,
    ) -> None:
        """Record candidate counts at each stage."""
        self._candidates.labels(stage="input", model_tier=model_tier).inc(input_count)
        self._candidates.labels(stage="filtered", model_tier=model_tier).inc(
            filtered_count
        )
        self._candidates.labels(stage="output", model_tier=model_tier).inc(output_count)

    def record_fetch_error(self, status_code: int, fetcher: str) -> None:
        """Record a fetch error."""
        self._fetch_errors.labels(status_code=str(status_code), fetcher=fetcher).inc()


def get_metrics_collector(registry=None):
    """Get a metrics collector — returns Prometheus if available, else no-op."""
    import importlib.util

    if importlib.util.find_spec("prometheus_client") is not None:
        return PrometheusMetricsCollector(registry=registry)
    return NoOpMetricsCollector()
