"""Politeness controls for respectful web crawling.

Includes robots.txt parsing, per-domain rate limiting, and crawl-delay
respect. Disabled by default to maintain backward compatibility; enable
with polite=True on Pathfinder.
"""

import logging
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from pathfinder_sdk.models import FetchError

logger = logging.getLogger(__name__)


class PolitenessController:
    """Controls fetch politeness: robots.txt, rate limits, crawl delays.

    Args:
        polite: Whether politeness controls are active.
        rate_limit: Minimum seconds between requests to the same domain.
        max_requests_per_domain: Maximum requests per domain per session.
        user_agent: User-Agent string for robots.txt checks.
    """

    def __init__(
        self,
        polite: bool = True,
        rate_limit: float = 1.0,
        max_requests_per_domain: int = 100,
        user_agent: str = "PathfinderSDK/0.1.0",
    ):
        self.polite = polite
        self.rate_limit = rate_limit
        self.max_requests_per_domain = max_requests_per_domain
        self.user_agent = user_agent
        self._last_request: dict[str, float] = {}
        self._request_counts: dict[str, int] = {}
        self._robots: dict[str, RobotFileParser] = {}

    def can_fetch(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        if not self.polite:
            return True

        domain = self._get_domain(url)
        self._ensure_robots(domain)

        parser = self._robots.get(domain)
        if parser is None:
            return True

        return parser.can_fetch(self.user_agent, url)

    def wait_if_needed(self, url: str) -> None:
        """Enforce rate limiting and max requests for a URL.

        Raises:
            FetchError: If max requests per domain exceeded.
        """
        if not self.polite:
            return

        domain = self._get_domain(url)

        # Check max requests
        count = self._request_counts.get(domain, 0)
        if count >= self.max_requests_per_domain:
            raise FetchError(
                f"Max requests ({self.max_requests_per_domain}) reached "
                f"for domain {domain}"
            )
        self._request_counts[domain] = count + 1

        # Rate limiting
        last = self._last_request.get(domain)
        if last is not None:
            elapsed = time.time() - last
            delay = self._get_effective_delay(domain)
            if elapsed < delay:
                sleep_time = delay - elapsed
                logger.debug(
                    "Rate limit: sleeping %.2f seconds for %s", sleep_time, domain
                )
                time.sleep(sleep_time)

        self._last_request[domain] = time.time()

    def get_crawl_delay(self, url: str) -> float | None:
        """Get Crawl-delay from robots.txt for a domain."""
        if not self.polite:
            return None

        domain = self._get_domain(url)
        self._ensure_robots(domain)

        parser = self._robots.get(domain)
        if parser is None:
            return None

        return parser.crawl_delay(self.user_agent)

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc or parsed.path

    def _ensure_robots(self, domain: str) -> None:
        """Load robots.txt for a domain if not already cached."""
        if domain in self._robots:
            return

        robots_url = f"https://{domain}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
        except Exception as exc:
            logger.warning("Failed to fetch robots.txt for %s: %s", domain, exc)

        self._robots[domain] = parser

    def _get_effective_delay(self, domain: str) -> float:
        """Return effective delay: max of rate_limit and crawl-delay."""
        delay = self.rate_limit
        crawl_delay = self.get_crawl_delay(f"https://{domain}/")
        if crawl_delay is not None:
            delay = max(delay, crawl_delay)
        return delay

    def _load_robots(self, url: str, content: str) -> None:
        """Load robots.txt from string (for testing)."""
        domain = self._get_domain(url)
        parser = RobotFileParser()
        parser.parse(content.splitlines())
        self._robots[domain] = parser
