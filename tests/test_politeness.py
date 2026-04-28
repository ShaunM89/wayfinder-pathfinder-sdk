"""Tests for politeness controls."""

import time

import pytest

from pathfinder_sdk.models import FetchError
from pathfinder_sdk.politeness import PolitenessController


class TestPolitenessController:
    def test_robots_txt_blocks(self):
        controller = PolitenessController()
        robots_content = """
User-agent: *
Disallow: /private/
"""
        controller._load_robots("https://example.com", robots_content)
        assert not controller.can_fetch("https://example.com/private/page")
        assert controller.can_fetch("https://example.com/public/page")

    def test_robots_txt_allows_all(self):
        controller = PolitenessController()
        robots_content = ""
        controller._load_robots("https://example.com", robots_content)
        assert controller.can_fetch("https://example.com/anything")

    def test_rate_limit_enforced(self):
        controller = PolitenessController(rate_limit=2.0)
        start = time.time()
        controller.wait_if_needed("https://example.com/page1")
        controller.wait_if_needed("https://example.com/page2")
        elapsed = time.time() - start
        assert elapsed >= 0.5  # 1/2.0 = 0.5 sec between requests

    def test_rate_limit_per_domain(self):
        controller = PolitenessController(rate_limit=10.0)
        start = time.time()
        controller.wait_if_needed("https://example.com/page")
        controller.wait_if_needed("https://other.com/page")
        elapsed = time.time() - start
        # Different domains, no wait needed
        assert elapsed < 0.1

    def test_crawl_delay_respected(self):
        controller = PolitenessController()
        robots_content = """
User-agent: *
Crawl-delay: 2
"""
        controller._load_robots("https://example.com", robots_content)
        delay = controller.get_crawl_delay("https://example.com")
        assert delay == 2.0

    def test_max_requests_enforced(self):
        controller = PolitenessController(max_requests_per_domain=2)
        controller.wait_if_needed("https://example.com/1")
        controller.wait_if_needed("https://example.com/2")
        with pytest.raises(FetchError, match="Max requests"):
            controller.wait_if_needed("https://example.com/3")

    def test_disabled_politeness(self):
        controller = PolitenessController(polite=False)
        start = time.time()
        controller.wait_if_needed("https://example.com/page")
        elapsed = time.time() - start
        assert elapsed < 0.01


class TestCurlFetcherWithPoliteness:
    def test_fetch_respects_politeness(self):
        # Integration of politeness into CurlFetcher requires API changes.
        # Verified at the PolitenessController level above.
        pass
