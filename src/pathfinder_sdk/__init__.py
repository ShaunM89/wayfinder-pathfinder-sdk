"""Pathfinder SDK — Local ranking engine for AI navigation agents.

Quickstart:
    >>> from pathfinder_sdk import Pathfinder
    >>> sdk = Pathfinder(model="default")
    >>> result = sdk.rank_candidates("https://example.com", "Find privacy policy")
    >>> print(result.candidates[0].href)
"""

from pathfinder_sdk.core import Pathfinder
from pathfinder_sdk.models import (
    CandidateRecommendation,
    ConfigurationError,
    FetchError,
    ModelLoadError,
    ModelNotFoundError,
    RankingResult,
)

__all__ = [
    "Pathfinder",
    "RankingResult",
    "CandidateRecommendation",
    "ModelNotFoundError",
    "ConfigurationError",
    "FetchError",
    "ModelLoadError",
]

__version__ = "0.1.0"
