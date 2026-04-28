"""Tests for async support."""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pathfinder_sdk.core import Pathfinder
from pathfinder_sdk.fetcher import CurlFetcher, Fetcher, PlaywrightFetcher
from pathfinder_sdk.models import FetchError, RankingResult


@pytest.fixture(autouse=True)
def _inject_fake_playwright_module():
    """Inject fake playwright modules so async imports resolve."""
    fake_playwright = types.ModuleType("playwright")
    fake_sync_api = types.ModuleType("playwright.sync_api")
    fake_sync_api.sync_playwright = MagicMock()
    fake_sync_api.TimeoutError = TimeoutError
    fake_playwright.sync_api = fake_sync_api

    fake_async_api = types.ModuleType("playwright.async_api")
    fake_async_api.async_playwright = MagicMock()
    fake_async_api.TimeoutError = TimeoutError
    fake_playwright.async_api = fake_async_api

    sys.modules["playwright"] = fake_playwright
    sys.modules["playwright.sync_api"] = fake_sync_api
    sys.modules["playwright.async_api"] = fake_async_api
    yield
    sys.modules.pop("playwright.async_api", None)
    sys.modules.pop("playwright.sync_api", None)
    sys.modules.pop("playwright", None)


class TestCurlFetcherAsync:
    def test_fetch_async_returns_same_as_fetch(self):
        fetcher = CurlFetcher()
        with patch.object(fetcher, "fetch", return_value=[{"href": "/a", "text": "A"}]):
            result = asyncio.run(fetcher.fetch_async("https://example.com"))
            assert result == [{"href": "/a", "text": "A"}]


class TestPlaywrightFetcherAsync:
    @pytest.mark.asyncio
    async def test_fetch_async_uses_async_playwright(self):
        import pathfinder_sdk.fetcher as fetcher_module

        mock_async_pw = MagicMock()
        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_page = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200

        mock_page.goto = AsyncMock(return_value=mock_response)
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value='<a href="/about">About</a>')
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        mock_pw = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_async_pw.return_value.__aenter__ = AsyncMock(return_value=mock_pw)
        mock_async_pw.return_value.__aexit__ = AsyncMock(return_value=False)

        # The import inside fetch_async looks up 'playwright.async_api' in sys.modules
        sys.modules["playwright.async_api"].async_playwright = mock_async_pw
        fetcher_module.async_playwright = mock_async_pw
        try:
            fetcher = PlaywrightFetcher()
            candidates = await fetcher.fetch_async("https://example.com")

            assert len(candidates) == 1
            assert candidates[0]["text"] == "About"
        finally:
            if hasattr(fetcher_module, "async_playwright"):
                delattr(fetcher_module, "async_playwright")


class TestFetcherDispatcherAsync:
    @pytest.mark.asyncio
    async def test_auto_fallback_async(self):
        fetcher = Fetcher(backend="auto")
        fetcher._playwright = PlaywrightFetcher()
        with (
            patch.object(fetcher._curl, "fetch_async", side_effect=FetchError("fail")),
            patch.object(
                fetcher._playwright,
                "fetch_async",
                return_value=[{"href": "/a", "text": "A"}],
            ) as mock_pw,
        ):
            result = await fetcher.fetch_async("https://example.com")
            assert len(result) == 1
            mock_pw.assert_called_once_with("https://example.com")


class TestPathfinderAsync:
    @pytest.mark.asyncio
    async def test_rank_candidates_async_with_pre_extracted(self):
        with patch("pathfinder_sdk.core.BiEncoderRanker") as mock_ranker_class:
            mock_ranker = MagicMock()
            mock_ranker.rank.return_value = []
            mock_ranker_class.return_value = mock_ranker

            pf = Pathfinder(fetcher=None)
            result = await pf.rank_candidates_async(
                url="https://example.com",
                task_description="Find A",
                candidates=[{"href": "/a", "text": "A"}],
            )
            assert isinstance(result, RankingResult)

    @pytest.mark.asyncio
    async def test_rank_candidates_async_fetches_when_no_candidates(self):
        with (
            patch("pathfinder_sdk.core.BiEncoderRanker") as mock_ranker_class,
            patch("pathfinder_sdk.core.Fetcher") as mock_fetcher_class,
        ):
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_async = AsyncMock(
                return_value=[{"href": "/a", "text": "A"}]
            )
            mock_fetcher_class.return_value = mock_fetcher

            mock_ranker = MagicMock()
            mock_ranker.rank.return_value = []
            mock_ranker_class.return_value = mock_ranker

            pf = Pathfinder(fetcher="auto")
            result = await pf.rank_candidates_async(
                url="https://example.com",
                task_description="Find A",
            )
            assert isinstance(result, RankingResult)
            mock_fetcher.fetch_async.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_rank_candidates_async_top_n_override(self):
        with patch("pathfinder_sdk.core.BiEncoderRanker") as mock_ranker_class:
            mock_ranker = MagicMock()
            mock_ranker.rank.return_value = []
            mock_ranker_class.return_value = mock_ranker

            pf = Pathfinder(top_n=20)
            await pf.rank_candidates_async(
                url="https://example.com",
                task_description="Find A",
                candidates=[{"href": "/a", "text": "A"}],
                top_n=5,
            )

            call_args = mock_ranker.rank.call_args
            assert call_args[0][2] == 5  # positional: (task, candidates, top_n)
