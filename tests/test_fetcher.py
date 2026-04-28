"""Tests for fetcher module."""

import pytest

from pathfinder_sdk.fetcher import CurlFetcher, _get_accessible_text, _get_dom_path, _is_in_navigation


class TestAccessibleText:
    def test_plain_text(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup('<a href="/x">Click here</a>', "html.parser")
        assert _get_accessible_text(soup.a) == "Click here"

    def test_aria_label_preferred(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup('<a href="/x" aria-label="Better description">Click</a>', "html.parser")
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
