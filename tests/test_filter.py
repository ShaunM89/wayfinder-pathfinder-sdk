"""Tests for HeuristicFilter."""

import pytest

from pathfinder_sdk.filter import HeuristicFilter


@pytest.fixture
def filter_default():
    return HeuristicFilter()


@pytest.fixture
def sample_candidates():
    return [
        {"href": "/about", "text": "About Us"},
        {"href": "mailto:test@example.com", "text": "Email"},
        {"href": "tel:+1234567890", "text": "Call"},
        {"href": "javascript:void(0)", "text": "Click"},
        {"href": "#section", "text": "Section"},
        {"href": "/about", "text": "About Us Duplicate"},
        {"href": "/contact.pdf", "text": "Contact PDF"},
        {"href": "/", "text": "Home"},
    ]


class TestHeuristicFilter:
    def test_removes_mailto_tel_javascript(self, filter_default, sample_candidates):
        result = filter_default.filter(sample_candidates, "https://example.com")
        hrefs = [c["href"] for c in result]
        assert "mailto:test@example.com" not in hrefs
        assert "tel:+1234567890" not in hrefs
        assert "javascript:void(0)" not in hrefs

    def test_removes_fragment_only(self, filter_default, sample_candidates):
        result = filter_default.filter(sample_candidates, "https://example.com")
        hrefs = [c["href"] for c in result]
        assert "https://example.com#section" not in hrefs

    def test_removes_non_html_extensions(self, filter_default, sample_candidates):
        result = filter_default.filter(sample_candidates, "https://example.com")
        hrefs = [c["href"] for c in result]
        assert "https://example.com/contact.pdf" not in hrefs

    def test_deduplicates(self, filter_default, sample_candidates):
        result = filter_default.filter(sample_candidates, "https://example.com")
        about_count = sum(
            1 for c in result if c["href"] == "https://example.com/about/"
        )
        assert about_count == 1

    def test_normalizes_relative_urls(self, filter_default, sample_candidates):
        result = filter_default.filter(sample_candidates, "https://example.com")
        hrefs = [c["href"] for c in result]
        assert "https://example.com/about/" in hrefs

    def test_empty_input(self, filter_default):
        result = filter_default.filter([], "https://example.com")
        assert result == []

    def test_exclude_boilerplate(self):
        filt = HeuristicFilter(exclude_boilerplate=True)
        candidates = [
            {"href": "/footer-link", "text": "Footer", "dom_path": "footer > div > a"},
            {"href": "/header-link", "text": "Header", "dom_path": "header > nav > a"},
            {"href": "/content", "text": "Content", "dom_path": "main > div > a"},
        ]
        result = filt.filter(candidates, "https://example.com")
        hrefs = [c["href"] for c in result]
        assert "https://example.com/footer-link" not in hrefs
        assert "https://example.com/header-link" not in hrefs
        assert "https://example.com/content" in hrefs
