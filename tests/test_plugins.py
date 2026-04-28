"""Tests for plugin system."""

from unittest.mock import patch

import pytest

from pathfinder_sdk.fetcher import CurlFetcher, Fetcher
from pathfinder_sdk.plugins import (
    _FETCHER_REGISTRY,
    discover_plugins,
    register_fetcher,
    resolve_fetcher,
)


class TestRegisterFetcher:
    def test_register_decorator(self):
        @register_fetcher("test_fetcher")
        class TestFetcher(CurlFetcher):
            pass

        assert "test_fetcher" in _FETCHER_REGISTRY
        assert _FETCHER_REGISTRY["test_fetcher"] is TestFetcher

        # Clean up
        del _FETCHER_REGISTRY["test_fetcher"]

    def test_register_duplicate_raises(self):
        @register_fetcher("dup_fetcher")
        class FetcherA(CurlFetcher):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_fetcher("dup_fetcher")
            class FetcherB(CurlFetcher):
                pass

        del _FETCHER_REGISTRY["dup_fetcher"]


class TestResolveFetcher:
    def test_resolve_builtin(self):
        # Built-in fetchers should resolve
        cls = resolve_fetcher("curl")
        assert cls is CurlFetcher

    def test_resolve_registered(self):
        @register_fetcher("my_fetcher")
        class MyFetcher(CurlFetcher):
            pass

        cls = resolve_fetcher("my_fetcher")
        assert cls is MyFetcher

        del _FETCHER_REGISTRY["my_fetcher"]

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown fetcher"):
            resolve_fetcher("nonexistent")


class TestDiscoverPlugins:
    def test_discover_no_entry_points(self):
        with patch("pathfinder_sdk.plugins.entry_points", return_value=[]) as mock_eps:
            discover_plugins()
            mock_eps.assert_called()


class TestFetcherDispatcherWithPlugins:
    def test_fetcher_uses_registered_plugin(self):
        @register_fetcher("mock_fetcher")
        class MockFetcher(CurlFetcher):
            def fetch(self, url):
                return [{"href": "/plugin", "text": "Plugin"}]

        fetcher = Fetcher(backend="mock_fetcher")
        result = fetcher.fetch("https://example.com")
        assert len(result) == 1
        assert result[0]["text"] == "Plugin"

        del _FETCHER_REGISTRY["mock_fetcher"]
