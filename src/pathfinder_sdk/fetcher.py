"""Page fetchers for extracting candidate links.

Default: curl_cffi + BeautifulSoup
Optional fallback: Playwright headless shell

Adapted from compass_core.crawler.parser, fetcher, and dual_fetcher.
"""

import logging
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

from pathfinder_sdk.models import FetchError

logger = logging.getLogger(__name__)

# Maximum response body size (10 MB) — prevents OOM on oversized pages
_MAX_RESPONSE_SIZE_BYTES = 10 * 1024 * 1024

# Content types that indicate non-HTML responses
_NON_HTML_CONTENT_TYPES = {
    "application/pdf",
    "application/zip",
    "application/octet-stream",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/svg+xml",
    "audio/mpeg",
    "video/mp4",
    "application/json",
}


class CurlFetcher:
    """Fetch pages using curl_cffi for TLS fingerprint impersonation.

    Args:
        user_agent: User-Agent string.
        timeout: Request timeout in seconds.
        max_body_size: Maximum response body size in bytes.
    """

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: int = 10,
        max_body_size: int = _MAX_RESPONSE_SIZE_BYTES,
    ):
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/133.0.0.0 Safari/537.36"
        )
        self.timeout = timeout
        self.max_body_size = max_body_size

    def fetch(self, url: str) -> list[dict]:
        """Fetch URL and extract candidate links.

        Args:
            url: URL to fetch.

        Returns:
            List of candidate link dicts.

        Raises:
            FetchError: If the page cannot be fetched.
        """
        try:
            from curl_cffi import requests as cffi_requests
        except ImportError as exc:
            raise FetchError(
                "curl_cffi is not installed. Install it with: pip install curl-cffi"
            ) from exc

        try:
            response = cffi_requests.get(
                url,
                headers={"User-Agent": self.user_agent},
                impersonate="chrome120",
                timeout=self.timeout,
                allow_redirects=True,
            )
        except Exception as exc:
            raise FetchError(f"curl_cffi request failed for {url}: {exc}") from exc

        if response.status_code != 200:
            raise FetchError(
                f"HTTP {response.status_code} for {url}"
            )

        content_type = response.headers.get("content-type", "").lower()
        raw_ct = content_type.split(";")[0].strip()
        if raw_ct in _NON_HTML_CONTENT_TYPES:
            raise FetchError(f"Non-HTML content: {raw_ct}")

        return self._parse_html(response.text, str(response.url))

    def _parse_html(self, html: str, base_url: str) -> list[dict]:
        """Parse HTML and extract candidate links."""
        soup = BeautifulSoup(html, "html.parser")
        candidates: list[dict] = []
        position = 0

        for link_tag in soup.find_all("a", href=True):
            href = link_tag.get("href", "").strip()
            if not href:
                continue

            absolute_url = urljoin(base_url, href)
            anchor_text = _get_accessible_text(link_tag)
            title_attr = link_tag.get("title", "")
            surrounding_text = _get_surrounding_text(link_tag, chars=100)
            dom_path = _get_dom_path(link_tag)
            in_navigation = _is_in_navigation(link_tag)
            parent_tag = link_tag.parent.name if link_tag.parent else None

            candidates.append({
                "href": absolute_url,
                "text": anchor_text,
                "title": title_attr if title_attr else None,
                "surrounding_text": surrounding_text,
                "dom_path": dom_path,
                "position": position,
                "in_navigation": in_navigation,
                "parent_tag": parent_tag,
            })
            position += 1

        return candidates


class PlaywrightFetcher:
    """Fetch JS-rendered pages using Playwright headless shell.

    Playwright is an optional dependency.
    """

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: int = 15000,
        headless: bool = True,
    ):
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/133.0.0.0 Safari/537.36"
        )
        self.timeout = timeout
        self.headless = headless

    def fetch(self, url: str) -> list[dict]:
        """Fetch URL with Playwright and extract candidate links.

        Args:
            url: URL to fetch.

        Returns:
            List of candidate link dicts.

        Raises:
            FetchError: If Playwright is not installed or fetch fails.
        """
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        except ImportError as exc:
            raise FetchError(
                "Playwright is not installed. "
                "Install it with: pip install pathfinder-sdk[playwright]"
            ) from exc

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                context = browser.new_context(user_agent=self.user_agent)
                page = context.new_page()
                page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
                html = page.content()
                final_url = page.url
                context.close()
                browser.close()
        except PWTimeout as exc:
            raise FetchError(f"Playwright timeout for {url}") from exc
        except Exception as exc:
            raise FetchError(f"Playwright error for {url}: {exc}") from exc

        return self._parse_html(html, final_url)

    def _parse_html(self, html: str, base_url: str) -> list[dict]:
        """Parse HTML and extract candidate links."""
        # Reuse CurlFetcher's parser logic
        return CurlFetcher()._parse_html(html, base_url)


class Fetcher:
    """Dispatcher that selects the appropriate fetcher backend.

    Args:
        backend: "auto", "curl", "playwright", or None.
    """

    def __init__(self, backend: str | None = "auto"):
        self.backend = backend
        self._curl = CurlFetcher()
        self._playwright: PlaywrightFetcher | None = None

    def fetch(self, url: str) -> list[dict]:
        """Fetch URL and return candidate links.

        Args:
            url: URL to fetch.

        Returns:
            List of candidate link dicts.
        """
        if self.backend == "curl":
            return self._curl.fetch(url)
        if self.backend == "playwright":
            if self._playwright is None:
                self._playwright = PlaywrightFetcher()
            return self._playwright.fetch(url)
        if self.backend == "auto":
            candidates = self._curl.fetch(url)
            if len(candidates) < 3:
                logger.info(
                    "curl_cffi returned %d links for %s, trying Playwright",
                    len(candidates), url,
                )
                if self._playwright is None:
                    self._playwright = PlaywrightFetcher()
                try:
                    candidates = self._playwright.fetch(url)
                except FetchError:
                    logger.warning("Playwright fallback failed for %s", url)
            return candidates
        if self.backend is None:
            return []
        raise ValueError(f"Unknown fetcher backend: {self.backend}")


# --- BeautifulSoup helpers extracted from Compass ---

def _get_accessible_text(tag: Tag) -> str:
    """Extract visible text from a tag, including inside <template> elements.

    Adapted from compass_core.crawler.parser.ContentParser._get_accessible_text.
    """
    aria = tag.get("aria-label", "")
    if isinstance(aria, str) and aria.strip():
        return aria.strip()

    texts: list[str] = []
    for desc in tag.descendants:
        if isinstance(desc, NavigableString):
            stripped = desc.strip()
            if stripped:
                texts.append(stripped)
        elif isinstance(desc, Tag) and desc.name == "img":
            alt = desc.get("alt", "")
            if isinstance(alt, str) and alt.strip():
                texts.append(alt.strip())

    return " ".join(texts)


def _get_surrounding_text(tag: Tag, chars: int = 100) -> str:
    """Get text surrounding a tag.

    Adapted from compass_core.crawler.parser.ContentParser._get_surrounding_text.
    """
    try:
        if tag.parent:
            parent_text = _get_accessible_text(tag.parent)
            link_text = _get_accessible_text(tag)
            if link_text in parent_text:
                idx = parent_text.find(link_text)
                start = max(0, idx - chars)
                end = min(len(parent_text), idx + len(link_text) + chars)
                return parent_text[start:end]
        return ""
    except Exception:
        return ""


def _get_dom_path(tag: Tag) -> str:
    """Generate a simple DOM path for a tag.

    Adapted from compass_core.crawler.parser.ContentParser._get_dom_path.
    """
    try:
        path_parts: list[str] = []
        current: Tag | None = tag
        for _ in range(5):
            if current and current.name:
                part = current.name
                if current.get("id"):
                    part += f"#{current['id']}"
                elif current.get("class"):
                    classes = current["class"]
                    if classes:
                        part += f".{classes[0]}"
                path_parts.insert(0, part)
                current = current.parent
            else:
                break
        return " > ".join(path_parts)
    except Exception:
        return ""


def _is_in_navigation(tag: Tag) -> bool:
    """Check if a tag is within a navigation element.

    Adapted from compass_core.crawler.parser.ContentParser._is_in_navigation.
    """
    try:
        current = tag.parent
        while current:
            if current.name in ("nav", "header", "footer"):
                return True
            if current.get("class"):
                classes = " ".join(current["class"])
                if any(
                    nav_word in classes.lower()
                    for nav_word in ("nav", "menu", "header", "footer")
                ):
                    return True
            if current.get("id"):
                id_str = current["id"].lower()
                if any(
                    nav_word in id_str
                    for nav_word in ("nav", "menu", "header", "footer")
                ):
                    return True
            current = current.parent
        return False
    except Exception:
        return False
