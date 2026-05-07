"""Canonical entry point for Pathfinder SDK.

Orchestrates fetch → filter → rank → result packaging.
"""

import asyncio
import logging
import time

from pathfinder_sdk.cache import InMemoryEmbeddingCache
from pathfinder_sdk.config import load_config
from pathfinder_sdk.fetcher import Fetcher
from pathfinder_sdk.filter import HeuristicFilter
from pathfinder_sdk.metrics import get_metrics_collector
from pathfinder_sdk.models import RankingResult
from pathfinder_sdk.politeness import PolitenessController
from pathfinder_sdk.ranker import BiEncoderRanker
from pathfinder_sdk.telemetry import get_tracer

logger = logging.getLogger(__name__)


_UNSET = object()


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
        model: str = _UNSET,  # type: ignore[assignment]
        top_n: int = _UNSET,  # type: ignore[assignment]
        cache_dir: str = _UNSET,  # type: ignore[assignment]
        fetcher: str | None = _UNSET,  # type: ignore[assignment]
        device: str | None = _UNSET,  # type: ignore[assignment]
        quiet: bool = _UNSET,  # type: ignore[assignment]
        cache: InMemoryEmbeddingCache | None = None,
        tracer=None,
        metrics=None,
        config_path: str | None = None,
        **kwargs,
    ):
        # Defaults for named params (used when not explicitly provided)
        _defaults = {
            "model": "default",
            "top_n": 20,
            "cache_dir": "~/.cache/pathfinder",
            "fetcher": "auto",
            "device": None,
            "quiet": False,
        }

        # Load config from file + env vars, then merge with kwargs
        file_config = load_config(path=config_path)
        overrides = {k: v for k, v in kwargs.items() if v is not None}

        # Add explicitly-provided named params to overrides so they win over
        # config file values.
        for key in _defaults:
            val = locals()[key]
            if val is not _UNSET:
                overrides[key] = val

        self.model_tier = overrides.get(
            "model", file_config.get("model", _defaults["model"])
        )
        self.top_n = overrides.get(
            "top_n", file_config.get("top_n", _defaults["top_n"])
        )
        self.cache_dir = overrides.get(
            "cache_dir", file_config.get("cache_dir", _defaults["cache_dir"])
        )
        self.device = overrides.get(
            "device", file_config.get("device", _defaults["device"])
        )
        quiet = overrides.get("quiet", file_config.get("quiet", _defaults["quiet"]))
        self._tracer = tracer if tracer is not None else get_tracer()
        self._metrics = metrics if metrics is not None else get_metrics_collector()

        # Fetcher config: support nested dict or flat string
        fetcher_val = file_config.get("fetcher", fetcher)
        fetcher_cfg: dict = {}
        if isinstance(fetcher_val, dict):
            fetcher_cfg = fetcher_val
            fetcher_backend = overrides.get(
                "fetcher", fetcher_cfg.get("backend", fetcher)
            )
        elif isinstance(fetcher_val, str):
            fetcher_backend = overrides.get("fetcher", fetcher_val)
        else:
            fetcher_backend = overrides.get("fetcher", fetcher)

        # Flat env-var fallbacks for fetcher settings
        fetcher_cfg.setdefault("timeout", file_config.get("fetcher_timeout", 10))
        fetcher_cfg.setdefault("max_retries", file_config.get("fetcher_max_retries", 3))
        fetcher_cfg.setdefault(
            "retry_delay", file_config.get("fetcher_retry_delay", 1.0)
        )
        fetcher_cfg.setdefault(
            "max_body_size",
            file_config.get("fetcher_max_body_size", 10 * 1024 * 1024),
        )
        fetcher_cfg.setdefault(
            "min_links_for_curl",
            file_config.get("fetcher_min_links_for_curl", 3),
        )
        fetcher_cfg.setdefault("user_agent", file_config.get("fetcher_user_agent"))

        self.fetcher_backend = fetcher_backend

        # Filter config
        filter_cfg = file_config.get("filter", {})
        filter_cfg.setdefault(
            "exclude_boilerplate",
            file_config.get("filter_exclude_boilerplate", False),
        )
        filter_cfg.setdefault(
            "min_anchor_length",
            file_config.get("filter_min_anchor_length", 1),
        )

        # Politeness config
        politeness_cfg = file_config.get("politeness", {})
        polite_enabled = overrides.get("polite", politeness_cfg.get("enabled", True))
        if polite_enabled:
            politeness_controller: PolitenessController | None = PolitenessController(
                polite=True,
                rate_limit=politeness_cfg.get("rate_limit", 1.0),
                max_requests_per_domain=politeness_cfg.get(
                    "max_requests_per_domain", 100
                ),
                user_agent=politeness_cfg.get("user_agent", "PathfinderSDK/0.1.0"),
            )
        else:
            politeness_controller = None

        # Inference config
        inference_cfg = file_config.get("inference", {})
        batch_size = inference_cfg.get("batch_size", 32)

        self._fetcher = (
            Fetcher(
                backend=self.fetcher_backend,
                min_links_for_curl=fetcher_cfg.get("min_links_for_curl", 3),
                timeout=fetcher_cfg.get("timeout", 10),
                max_body_size=fetcher_cfg.get("max_body_size", 10 * 1024 * 1024),
                max_retries=fetcher_cfg.get("max_retries", 3),
                retry_delay=fetcher_cfg.get("retry_delay", 1.0),
                user_agent=fetcher_cfg.get("user_agent"),
                politeness=politeness_controller,
            )
            if self.fetcher_backend
            else None
        )
        self._filter = HeuristicFilter(**filter_cfg)
        self._ranker = BiEncoderRanker(
            model_tier=self.model_tier,
            cache_dir=self.cache_dir,
            device=self.device,
            quiet=quiet,
            cache=cache,
            batch_size=batch_size,
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
        with self._tracer.start_span("fetch") as span:
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
            span.set_attribute("url", url)
            span.set_attribute("candidates_count", len(raw_candidates))

        total_links = len(raw_candidates)
        logger.debug(
            "Fetched %d raw links from %s (%.2f ms)",
            total_links,
            url,
            stage_latencies["fetch_ms"],
        )

        # Stage 1: Heuristic Filter
        with self._tracer.start_span("filter") as span:
            filter_start = time.perf_counter()
            filtered = self._filter.filter(raw_candidates, base_url=url)
            stage_latencies["filter_ms"] = round(
                (time.perf_counter() - filter_start) * 1000, 2
            )
            span.set_attribute("input_count", total_links)
            span.set_attribute("output_count", len(filtered))

        total_after_filter = len(filtered)
        logger.debug(
            "Filtered %d → %d links (%.2f ms)",
            total_links,
            total_after_filter,
            stage_latencies["filter_ms"],
        )

        # Stage 2: Bi-Encoder Ranking
        with self._tracer.start_span("rank") as span:
            rank_start = time.perf_counter()
            ranked = self._ranker.rank(
                task=task_description,
                candidates=filtered,
                top_n=top_n,
            )
            stage_latencies["rank_ms"] = round(
                (time.perf_counter() - rank_start) * 1000, 2
            )
            span.set_attribute("model_tier", self.model_tier)
            span.set_attribute("output_count", len(ranked))

        latency_ms = round((time.perf_counter() - overall_start) * 1000, 2)

        # Record metrics
        self._metrics.record_latency(
            "fetch", stage_latencies["fetch_ms"], self.model_tier
        )
        self._metrics.record_latency(
            "filter", stage_latencies["filter_ms"], self.model_tier
        )
        self._metrics.record_latency(
            "rank", stage_latencies["rank_ms"], self.model_tier
        )
        self._metrics.record_candidates(
            total_links, total_after_filter, len(ranked), self.model_tier
        )

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
