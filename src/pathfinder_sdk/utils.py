"""Utility functions extracted and adapted from Compass.

Includes URL normalization, validation, deduplication, and cosine similarity.
"""

from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import numpy as np


class LinkNormalizer:
    """Static utility class for URL normalization and validation.

    Adapted from compass_core.crawler.link_normalizer.
    """

    @staticmethod
    def normalize(url: str, base_url: str) -> str:
        """Normalize a URL to absolute form with standardized format."""
        absolute_url = urljoin(base_url, url)
        parsed = urlparse(absolute_url)
        parsed = parsed._replace(fragment="")

        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=True)
            sorted_params = sorted(params.items())
            normalized_query = urlencode(sorted_params, doseq=True)
            parsed = parsed._replace(query=normalized_query)

        path = parsed.path
        if path and not path.endswith("/") and "." not in path.split("/")[-1]:
            path = path + "/"
            parsed = parsed._replace(path=path)

        return urlunparse(parsed)

    @staticmethod
    def is_valid_http_url(url: str) -> bool:
        """Check if URL is a valid HTTP or HTTPS URL."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            if not parsed.netloc:
                return False
            return parsed.scheme not in ("javascript", "mailto", "tel", "data", "file")
        except Exception:
            return False

    @staticmethod
    def is_same_domain(url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain."""
        try:
            return urlparse(url1).netloc == urlparse(url2).netloc
        except Exception:
            return False

    @staticmethod
    def remove_fragment(url: str) -> str:
        """Remove URL fragment (#section)."""
        try:
            parsed = urlparse(url)
            parsed = parsed._replace(fragment="")
            return urlunparse(parsed)
        except Exception:
            return url

    @staticmethod
    def clean_url(url: str) -> str:
        """Clean and normalize URL without requiring base_url."""
        try:
            parsed = urlparse(url)
            parsed = parsed._replace(fragment="")
            if parsed.query:
                params = parse_qs(parsed.query, keep_blank_values=True)
                sorted_params = sorted(params.items())
                normalized_query = urlencode(sorted_params, doseq=True)
                parsed = parsed._replace(query=normalized_query)
            return urlunparse(parsed)
        except Exception:
            return url

    @staticmethod
    def filter_valid_urls(urls: list[str]) -> list[str]:
        """Filter list of URLs to only valid HTTP/HTTPS URLs."""
        return [url for url in urls if LinkNormalizer.is_valid_http_url(url)]

    @staticmethod
    def deduplicate_urls(urls: list[str], base_url: str) -> list[str]:
        """Remove duplicate URLs after normalization."""
        seen: set[str] = set()
        unique_urls: list[str] = []
        for url in urls:
            normalized = LinkNormalizer.normalize(url, base_url)
            if normalized not in seen:
                seen.add(normalized)
                unique_urls.append(normalized)
        return unique_urls


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.

    Adapted from compass_core.features.semantic_features.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
    return max(0.0, min(1.0, similarity))
