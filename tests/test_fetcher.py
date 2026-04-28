"""Tests for fetcher module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from pathfinder_sdk.fetcher import (
    CurlFetcher,
    Fetcher,
    PlaywrightFetcher,
    _get_accessible_text,
    _get_dom_path,
    _is_in_navigation,
)
from pathfinder_sdk.models import FetchError


class TestAccessibleText:
    def test_plain_text(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup('<a href="/x">Click here</a>', "html.parser")
        assert _get_accessible_text(soup.a) == "Click here"

    def test_aria_label_preferred(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            '<a href="/x" aria-label="Better description">Click</a>', "html.parser"
        )
        assert _get_accessible_text(soup.a) == "Better description"

    def test_img_alt(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup('<a href="/x"><img alt="Logo"></a>', "html.parser")
        assert _get_accessible_text(soup.a) == "Logo"


class TestDomPath:
    def test_simple_path(self):
        from bs4 import BeautifulSoup

        html = '<nav><ul><li><a href="/x">Link</a></li></ul></nav>'
        soup = BeautifulSoup(html, "html.parser")
        path = _get_dom_path(soup.a)
        assert "nav" in path
        assert "ul" in path
        assert "li" in path
        assert "a" in path


class TestIsInNavigation:
    def test_inside_nav(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup('<nav><a href="/x">Link</a></nav>', "html.parser")
        assert _is_in_navigation(soup.a) is True

    def test_outside_nav(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup('<main><a href="/x">Link</a></main>', "html.parser")
        assert _is_in_navigation(soup.a) is False


class TestCurlFetcherParseHtml:
    def test_extracts_links(self):
        fetcher = CurlFetcher()
        html = """
        <html>
        <body>
            <a href="/about">About</a>
            <a href="/contact">Contact Us</a>
            <a href="mailto:test@example.com">Email</a>
        </body>
        </html>
        """
        candidates = fetcher._parse_html(html, "https://example.com")
        texts = [c["text"] for c in candidates]
        assert "About" in texts
        assert "Contact Us" in texts
        assert "Email" in texts
        # All should be absolute (parser does not filter schemes)
        for c in candidates:
            assert c["href"].startswith(("https://", "mailto:"))

    def test_extracts_context(self):
        fetcher = CurlFetcher()
        html = """
        <html><body>
        <div>For more information see our <a href="/privacy">Privacy Policy</a> and terms.</div>
        </body></html>
        """
        candidates = fetcher._parse_html(html, "https://example.com")
        assert len(candidates) == 1
        assert "Privacy Policy" in candidates[0]["surrounding_text"]


class TestCurlFetcherFetch:
    """Test CurlFetcher.fetch() with mocked curl_cffi."""

    @patch("pathfinder_sdk.fetcher._import_curl_cffi")
    def test_successful_fetch(self, mock_import):
        mock_cffi = MagicMock()
        mock_import.return_value = mock_cffi

        # Mock HEAD response
        mock_head = MagicMock()
        mock_head.status_code = 200
        mock_head.headers = {"content-type": "text/html; charset=utf-8"}
        mock_cffi.head.return_value = mock_head

        # Mock GET response
        mock_get = MagicMock()
        mock_get.status_code = 200
        mock_get.headers = {"content-type": "text/html; charset=utf-8"}
        mock_get.content = b'<a href="/about">About</a>'
        mock_get.text = '<a href="/about">About</a>'
        mock_get.url = "https://example.com"
        mock_cffi.get.return_value = mock_get

        fetcher = CurlFetcher()
        candidates = fetcher.fetch("https://example.com")

        assert len(candidates) == 1
        assert candidates[0]["text"] == "About"
        assert candidates[0]["href"] == "https://example.com/about"

    @patch("pathfinder_sdk.fetcher._import_curl_cffi")
    def test_non_html_preflight_rejection(self, mock_import):
        mock_cffi = MagicMock()
        mock_import.return_value = mock_cffi

        mock_head = MagicMock()
        mock_head.status_code = 200
        mock_head.headers = {"content-type": "application/pdf"}
        mock_cffi.head.return_value = mock_head

        fetcher = CurlFetcher()
        with pytest.raises(FetchError, match="Non-HTML content"):
            fetcher.fetch("https://example.com/doc.pdf")

    @patch("pathfinder_sdk.fetcher._import_curl_cffi")
    def test_size_limit_preflight_rejection(self, mock_import):
        mock_cffi = MagicMock()
        mock_import.return_value = mock_cffi

        mock_head = MagicMock()
        mock_head.status_code = 200
        mock_head.headers = {"content-type": "text/html", "content-length": "20971520"}
        mock_cffi.head.return_value = mock_head

        fetcher = CurlFetcher()
        with pytest.raises(FetchError, match="too large"):
            fetcher.fetch("https://example.com/huge")

    @patch("pathfinder_sdk.fetcher._import_curl_cffi")
    def test_block_status_fail_fast(self, mock_import):
        mock_cffi = MagicMock()
        mock_import.return_value = mock_cffi

        mock_head = MagicMock()
        mock_head.status_code = 200
        mock_head.headers = {}
        mock_cffi.head.return_value = mock_head

        mock_get = MagicMock()
        mock_get.status_code = 403
        mock_cffi.get.return_value = mock_get

        fetcher = CurlFetcher()
        with pytest.raises(FetchError, match="blocked"):
            fetcher.fetch("https://example.com")

    @patch("pathfinder_sdk.fetcher._import_curl_cffi")
    def test_retry_then_success(self, mock_import):
        mock_cffi = MagicMock()
        mock_import.return_value = mock_cffi

        mock_head = MagicMock()
        mock_head.status_code = 200
        mock_head.headers = {}
        mock_cffi.head.return_value = mock_head

        # First GET fails with 503, second succeeds
        mock_fail = MagicMock()
        mock_fail.status_code = 503
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.headers = {"content-type": "text/html"}
        mock_success.content = b"<a href='/'>Home</a>"
        mock_success.text = "<a href='/'>Home</a>"
        mock_success.url = "https://example.com"
        mock_cffi.get.side_effect = [mock_fail, mock_success]

        fetcher = CurlFetcher(max_retries=2, retry_delay=0.01)
        candidates = fetcher.fetch("https://example.com")
        assert len(candidates) == 1
        assert mock_cffi.get.call_count == 2

    @patch("pathfinder_sdk.fetcher._import_curl_cffi")
    def test_all_retries_exhausted(self, mock_import):
        mock_cffi = MagicMock()
        mock_import.return_value = mock_cffi

        mock_head = MagicMock()
        mock_head.status_code = 200
        mock_head.headers = {}
        mock_cffi.head.return_value = mock_head

        mock_fail = MagicMock()
        mock_fail.status_code = 503
        mock_cffi.get.return_value = mock_fail

        fetcher = CurlFetcher(max_retries=2, retry_delay=0.01)
        with pytest.raises(FetchError, match="All 2 fetch attempts failed"):
            fetcher.fetch("https://example.com")


class TestFetcherDispatcher:
    @patch("pathfinder_sdk.fetcher.CurlFetcher.fetch")
    @patch("pathfinder_sdk.fetcher.PlaywrightFetcher.fetch")
    def test_auto_fallback(self, mock_pw_fetch, mock_curl_fetch):
        mock_curl_fetch.return_value = []  # Too few links
        mock_pw_fetch.return_value = [{"href": "/a", "text": "A"}]

        fetcher = Fetcher(backend="auto")
        result = fetcher.fetch("https://example.com")

        assert len(result) == 1
        mock_curl_fetch.assert_called_once()
        mock_pw_fetch.assert_called_once()

    @patch("pathfinder_sdk.fetcher.CurlFetcher.fetch")
    @patch("pathfinder_sdk.fetcher.PlaywrightFetcher.fetch")
    def test_auto_no_fallback_when_enough_links(self, mock_pw_fetch, mock_curl_fetch):
        mock_curl_fetch.return_value = [
            {"href": "/a", "text": "A"},
            {"href": "/b", "text": "B"},
            {"href": "/c", "text": "C"},
        ]

        fetcher = Fetcher(backend="auto")
        result = fetcher.fetch("https://example.com")

        assert len(result) == 3
        mock_curl_fetch.assert_called_once()
        mock_pw_fetch.assert_not_called()

    def test_none_backend_returns_empty(self):
        fetcher = Fetcher(backend=None)
        assert fetcher.fetch("https://example.com") == []

    def test_unknown_backend_raises(self):
        fetcher = Fetcher(backend="unknown")
        with pytest.raises(ValueError, match="Unknown fetcher backend"):
            fetcher.fetch("https://example.com")


class TestPlaywrightFetcher:
    def test_missing_playwright_raises(self):
        """Playwright import fails — should raise FetchError."""
        real_import = __builtins__["__import__"]

        def fake_import(name, *args, **kwargs):
            if "playwright" in name:
                raise ImportError("No module named playwright")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            fetcher = PlaywrightFetcher()
            with pytest.raises(FetchError, match="Playwright is not installed"):
                fetcher.fetch("https://example.com")
