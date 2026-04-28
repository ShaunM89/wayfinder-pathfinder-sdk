"""Pydantic dataclasses for Pathfinder SDK result types and exceptions."""

from pydantic import BaseModel, Field


class PathfinderError(Exception):
    """Base exception for Pathfinder SDK."""


class ModelNotFoundError(PathfinderError):
    """Raised when a requested model tier is unavailable."""


class ConfigurationError(PathfinderError):
    """Raised when invalid configuration parameters are provided."""


class FetchError(PathfinderError):
    """Raised when page fetch fails (network, timeout, blocked)."""


class ModelLoadError(PathfinderError):
    """Raised when model files cannot be loaded."""


class CandidateRecommendation(BaseModel):
    """A single ranked candidate link.

    Attributes:
        rank: 1-indexed position in ranked list.
        href: Absolute, normalized URL.
        text: Anchor text.
        score: Cosine similarity score (0.0–1.0, higher is better).
        context_snippet: Surrounding text for LLM context.
    """

    rank: int = Field(..., ge=1, description="1-indexed rank position")
    href: str
    text: str
    score: float = Field(..., ge=0.0, le=1.0)
    context_snippet: str | None = None

    def to_dict(self) -> dict:
        """Dict serialization for programmatic use."""
        return self.model_dump()


class RankingResult(BaseModel):
    """Structured output from Pathfinder.rank_candidates().

    Attributes:
        task_description: The original natural-language task.
        source_url: The starting page URL.
        candidates: Top-N ranked links.
        latency_ms: Total latency of the rank call in milliseconds.
        total_links_analyzed: Number of links before filtering.
        total_links_after_filter: Number of links after heuristic filtering.
        model_tier: Which model tier was used (default, high, ultra).
        metadata: Optional extra metadata, including per-stage latency.
    """

    task_description: str
    source_url: str
    candidates: list[CandidateRecommendation]
    latency_ms: float = Field(..., ge=0.0)
    total_links_analyzed: int = Field(..., ge=0)
    total_links_after_filter: int = Field(..., ge=0)
    model_tier: str
    metadata: dict = Field(default_factory=dict)

    def to_json(self) -> str:
        """JSON serialization for API responses or LLM prompts."""
        return self.model_dump_json(indent=2)

    def to_dict(self) -> dict:
        """Dict serialization for programmatic use."""
        return self.model_dump()
