"""Tests for improved error messages with actionable suggestions."""

from unittest.mock import MagicMock, patch

import pytest

from pathfinder_sdk.fetcher import CurlFetcher
from pathfinder_sdk.models import FetchError, ModelLoadError, ModelNotFoundError
from pathfinder_sdk.ranker import BiEncoderRanker
from pathfinder_sdk.utils import did_you_mean


class TestDidYouMean:
    def test_exact_match(self):
        assert did_you_mean("default", ["default", "high", "ultra"]) == "default"

    def test_typo_one_char(self):
        assert did_you_mean("defualt", ["default", "high", "ultra"]) == "default"

    def test_typo_two_chars(self):
        assert did_you_mean("defualt", ["default", "high", "ultra"]) == "default"

    def test_no_close_match(self):
        assert did_you_mean("xyz", ["default", "high", "ultra"]) is None

    def test_empty_candidates(self):
        assert did_you_mean("default", []) is None


class TestModelNotFoundError:
    def test_unknown_tier_includes_valid_tiers(self):
        with pytest.raises(ModelNotFoundError, match="Valid tiers"):
            BiEncoderRanker(model_tier="nonexistent")

    def test_typo_suggests_correction(self):
        with pytest.raises(ModelNotFoundError, match='Did you mean "default"'):
            BiEncoderRanker(model_tier="defualt")

    def test_typo_high_tier(self):
        with pytest.raises(ModelNotFoundError, match='Did you mean "high"'):
            BiEncoderRanker(model_tier="hig")


class TestModelLoadError:
    @patch("pathfinder_sdk.ranker.SentenceTransformer")
    @patch("pathfinder_sdk.ranker._download_with_retry")
    def test_onnx_and_pytorch_fail_suggests_install(self, mock_download, mock_st_class):
        mock_download.return_value = "/fake/model"
        mock_st_class.side_effect = RuntimeError("ONNX failed")

        ranker = BiEncoderRanker(model_tier="default")
        with pytest.raises(ModelLoadError, match="pip install"):
            _ = ranker.model


class TestFetchErrorMessages:
    def test_missing_curl_cffi_suggests_install(self):
        with patch("pathfinder_sdk.fetcher._import_curl_cffi") as mock_import:
            mock_import.side_effect = FetchError(
                "curl_cffi is not installed. Install it with: pip install curl-cffi"
            )

            fetcher = CurlFetcher()
            with pytest.raises(FetchError, match="pip install"):
                fetcher.fetch("https://example.com")

    def test_fetch_403_suggests_playwright(self):
        response = MagicMock()
        response.status_code = 403
        response.headers = {"content-type": "text/html"}

        cffi_requests = MagicMock()
        cffi_requests.get.return_value = response
        cffi_requests.head.return_value = response

        with patch(
            "pathfinder_sdk.fetcher._import_curl_cffi", return_value=cffi_requests
        ):
            fetcher = CurlFetcher()
            with pytest.raises(FetchError, match="playwright"):
                fetcher.fetch("https://example.com")

    def test_fetch_429_suggests_rate_limit(self):
        response = MagicMock()
        response.status_code = 429
        response.headers = {"content-type": "text/html", "retry-after": "60"}

        cffi_requests = MagicMock()
        cffi_requests.get.return_value = response
        cffi_requests.head.return_value = response

        with patch(
            "pathfinder_sdk.fetcher._import_curl_cffi", return_value=cffi_requests
        ):
            fetcher = CurlFetcher()
            with pytest.raises(FetchError, match="Rate limit"):
                fetcher.fetch("https://example.com")
