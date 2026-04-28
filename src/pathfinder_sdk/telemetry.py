"""Optional OpenTelemetry integration for Pathfinder SDK.

Provides zero-overhead no-op fallbacks when opentelemetry is not installed.
"""

from typing import Any


class NoOpSpan:
    """No-op span that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute (no-op)."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span (no-op)."""
        pass

    def __enter__(self):
        """Enter span context."""
        return self

    def __exit__(self, *args):
        """Exit span context."""
        pass


class NoOpTracer:
    """No-op tracer that does nothing."""

    def start_span(self, name: str):
        """Start a span (no-op)."""
        return NoOpSpan()


def get_tracer(name: str = "pathfinder_sdk") -> Any:
    """Get a tracer — returns OTel tracer if available, else no-op."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return NoOpTracer()
