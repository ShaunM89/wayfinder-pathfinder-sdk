"""Canonical entry point for Pathfinder SDK.

Orchestrates fetch → filter → rank → result packaging.
"""

import logging
import time

from pathfinder_sdk.fetcher import Fetcher
from pathfinder_sdk.filter import HeuristicFilter
from pathfinder_sdk.models import RankingResult
from pathfinder_sdk.ranker import BiEncoderRanker

logger = logging.getLogger(__name__)


class Pathfinder:
    """Local-first ranking engine for AI navigation agents.

    Args:
        model: Model tier — "default" (BGE-small, ~400MB),
               "high" (bge-m3, ~2.6GB), "ultra" (pplx-4b, largest).
        top_n: Number of ranked candidates to return (1–100).
        cache_dir: Directory for cached model downloads from HF Hub.
        fetcher: Fetcher backend — "auto" tries curl_cffi first,
                 falls back to Playwright shell for JS-rendered pages.
        device: Inference device — None auto-detects (CPU default).
    """

    def __init__(
        self,
        model: str = "default",
        top_n: int = 20,
        cache_dir: str = "~/.cache/pathfinder",
        fetcher: str | None = "auto",
        device: str | None = None,
    ):
        self.model_tier = model
        self.top_n = top_n
        self.cache_dir = cache_dir
        self.fetcher_backend = fetcher
        self.device = device

        self._fetcher = Fetcher(backend=fetcher) if fetcher else None
        self._filter = HeuristicFilter()
        self._ranker = BiEncoderRanker(
            model_tier=model,
            cache_dir=cache_dir,
            device=device,
        )

    def rank_candidates(
        self,
        url: str,
        task_description: str,
        candidates: list[dict] | None = None,
        top_n: int | None = None,
    ) -> RankingResult:
        """Rank candidate links by relevance to the task description.

        Args:
            url: Starting page URL (used for fetch + result metadata).
            task_description: What you're trying to find (natural language).
            candidates: Optional pre-extracted links; if None, SDK fetches page.
            top_n: Override constructor default (useful for per-call tuning).

        Returns:
            RankingResult with ranked candidates, scores, and metadata.
        """
        top_n = top_n if top_n is not None else self.top_n
        start_time = time.perf_counter()

        # Stage 0: Fetch (optional)
        if candidates is None:
            if self._fetcher is None:
                raise ValueError(
                    "candidates is None but no fetcher is configured. "
                    "Pass candidates or set fetcher to 'auto'."
                )
            logger.debug("Fetching %s", url)
            raw_candidates = self._fetcher.fetch(url)
        else:
            raw_candidates = candidates

        total_links = len(raw_candidates)

        # Stage 1: Heuristic Filter
        filtered = self._filter.filter(raw_candidates, base_url=url)
        total_after_filter = len(filtered)

        # Stage 2: Bi-Encoder Ranking
        ranked = self._ranker.rank(
            task=task_description,
            candidates=filtered,
            top_n=top_n,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return RankingResult(
            task_description=task_description,
            source_url=url,
            candidates=ranked,
            latency_ms=round(latency_ms, 2),
            total_links_analyzed=total_links,
            total_links_after_filter=total_after_filter,
            model_tier=self.model_tier,
        )

    def unload(self) -> None:
        """Unload model from memory to free RAM."""
        self._ranker.unload()
