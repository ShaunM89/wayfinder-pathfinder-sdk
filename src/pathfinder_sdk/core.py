"""Canonical entry point for Pathfinder SDK.

Orchestrates fetch → filter → rank → result packaging.
"""

import asyncio
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
        quiet: bool = False,
        cache=None,
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
            quiet=quiet,
            cache=cache,
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
        return self._rank_single(
            url=url,
            task_description=task_description,
            candidates=candidates,
            top_n=top_n,
        )

    def _rank_single(
        self,
        url: str,
        task_description: str,
        candidates: list[dict] | None = None,
        top_n: int | None = None,
    ) -> RankingResult:
        """Internal ranking implementation shared by sync, async, and batch."""
        top_n = top_n if top_n is not None else self.top_n
        overall_start = time.perf_counter()
        stage_latencies: dict[str, float] = {}

        # Stage 0: Fetch (optional)
        fetch_start = time.perf_counter()
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
        stage_latencies["fetch_ms"] = round(
            (time.perf_counter() - fetch_start) * 1000, 2
        )

        total_links = len(raw_candidates)
        logger.debug(
            "Fetched %d raw links from %s (%.2f ms)",
            total_links,
            url,
            stage_latencies["fetch_ms"],
        )

        # Stage 1: Heuristic Filter
        filter_start = time.perf_counter()
        filtered = self._filter.filter(raw_candidates, base_url=url)
        stage_latencies["filter_ms"] = round(
            (time.perf_counter() - filter_start) * 1000, 2
        )

        total_after_filter = len(filtered)
        logger.debug(
            "Filtered %d → %d links (%.2f ms)",
            total_links,
            total_after_filter,
            stage_latencies["filter_ms"],
        )

        # Stage 2: Bi-Encoder Ranking
        rank_start = time.perf_counter()
        ranked = self._ranker.rank(
            task=task_description,
            candidates=filtered,
            top_n=top_n,
        )
        stage_latencies["rank_ms"] = round((time.perf_counter() - rank_start) * 1000, 2)

        latency_ms = round((time.perf_counter() - overall_start) * 1000, 2)

        logger.info(
            "Ranked %d candidates for '%s...' on %s — "
            "total=%.2f ms (fetch=%.2f, filter=%.2f, rank=%.2f)",
            len(ranked),
            task_description[:40],
            url,
            latency_ms,
            stage_latencies["fetch_ms"],
            stage_latencies["filter_ms"],
            stage_latencies["rank_ms"],
        )

        return RankingResult(
            task_description=task_description,
            source_url=url,
            candidates=ranked,
            latency_ms=latency_ms,
            total_links_analyzed=total_links,
            total_links_after_filter=total_after_filter,
            model_tier=self.model_tier,
            metadata={"stage_latencies": stage_latencies},
        )

    def rank_multiple(
        self,
        requests: list,
        top_n: int | None = None,
    ) -> list[RankingResult]:
        """Rank multiple (URL, task) pairs efficiently.

        Processes requests sequentially to respect target servers.
        The model is loaded once and reused across all requests.

        Args:
            requests: List of tuples. Each tuple is either:
                (url, task_description) or
                (url, task_description, candidates).
            top_n: Override constructor default for all requests.

        Returns:
            List of RankingResult in the same order as requests.
        """
        if not requests:
            return []

        results: list[RankingResult] = []
        for req in requests:
            if len(req) == 2:
                url, task = req
                candidates = None
            elif len(req) == 3:
                url, task, candidates = req
            else:
                raise ValueError(
                    "Each request must be a tuple of (url, task) or "
                    "(url, task, candidates)"
                )
            result = self._rank_single(
                url=url,
                task_description=task,
                candidates=candidates,
                top_n=top_n,
            )
            results.append(result)

        return results

    def rank_stream(
        self,
        url: str,
        task_description: str,
        candidates: list[dict] | None = None,
        top_n: int | None = None,
        min_score: float | None = None,
    ):
        """Yield ranked candidates one at a time, highest score first.

        Performs the same fetch → filter → rank pipeline as
        rank_candidates(), but yields each result incrementally.
        Useful for large candidate sets where you want early results
        without waiting for the full ranking to complete.

        Args:
            url: Starting page URL.
            task_description: Natural-language task.
            candidates: Optional pre-extracted links.
            top_n: Maximum candidates to yield.
            min_score: Stop yielding if score drops below this threshold.

        Yields:
            CandidateRecommendation objects in rank order.
        """
        result = self._rank_single(
            url=url,
            task_description=task_description,
            candidates=candidates,
            top_n=top_n,
        )

        for cand in result.candidates:
            if min_score is not None and cand.score < min_score:
                break
            yield cand

    async def rank_candidates_async(
        self,
        url: str,
        task_description: str,
        candidates: list[dict] | None = None,
        top_n: int | None = None,
    ) -> RankingResult:
        """Async version of rank_candidates.

        Fetches and ranks candidate links asynchronously. Model inference
        runs in a thread executor since ONNX Runtime / PyTorch are sync.

        Args:
            url: Starting page URL.
            task_description: Natural-language task.
            candidates: Optional pre-extracted links.
            top_n: Override constructor default.

        Returns:
            RankingResult with ranked candidates.
        """
        top_n = top_n if top_n is not None else self.top_n
        overall_start = time.perf_counter()
        stage_latencies: dict[str, float] = {}

        # Stage 0: Fetch (optional)
        fetch_start = time.perf_counter()
        if candidates is None:
            if self._fetcher is None:
                raise ValueError(
                    "candidates is None but no fetcher is configured. "
                    "Pass candidates or set fetcher to 'auto'."
                )
            raw_candidates = await self._fetcher.fetch_async(url)
        else:
            raw_candidates = candidates
        stage_latencies["fetch_ms"] = round(
            (time.perf_counter() - fetch_start) * 1000, 2
        )

        total_links = len(raw_candidates)

        # Stage 1: Heuristic Filter
        filter_start = time.perf_counter()
        filtered = self._filter.filter(raw_candidates, base_url=url)
        stage_latencies["filter_ms"] = round(
            (time.perf_counter() - filter_start) * 1000, 2
        )

        total_after_filter = len(filtered)

        # Stage 2: Bi-Encoder Ranking (CPU-bound, run in thread executor)
        rank_start = time.perf_counter()
        loop = asyncio.get_running_loop()
        ranked = await loop.run_in_executor(
            None,
            self._ranker.rank,
            task_description,
            filtered,
            top_n,
        )
        stage_latencies["rank_ms"] = round((time.perf_counter() - rank_start) * 1000, 2)

        latency_ms = round((time.perf_counter() - overall_start) * 1000, 2)

        return RankingResult(
            task_description=task_description,
            source_url=url,
            candidates=ranked,
            latency_ms=latency_ms,
            total_links_analyzed=total_links,
            total_links_after_filter=total_after_filter,
            model_tier=self.model_tier,
            metadata={"stage_latencies": stage_latencies},
        )

    def unload(self) -> None:
        """Unload model from memory to free RAM."""
        self._ranker.unload()
