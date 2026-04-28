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


def did_you_mean(query: str, candidates: list[str]) -> str | None:
    """Fuzzy-match a query string against a list of candidates.

    Uses a simple Levenshtein-distance-based approach to suggest the
    closest matching candidate.

    Args:
        query: The potentially misspelled input string.
        candidates: List of valid candidate strings.

    Returns:
        The closest candidate string, or None if no reasonable match.
    """
    if not candidates:
        return None

    def _levenshtein(a: str, b: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(a) < len(b):
            return _levenshtein(b, a)
        if len(b) == 0:
            return len(a)

        prev_row = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr_row = [i + 1]
            for j, cb in enumerate(b):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (0 if ca == cb else 1)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[len(b)]

    best_match: str | None = None
    best_score: int = float("inf")  # type: ignore[assignment]
    query_lower = query.lower()

    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower == query_lower:
            return cand
        dist = _levenshtein(query_lower, cand_lower)
        # Allow up to 2 typos or 30% of length, whichever is larger
        threshold = max(2, len(cand) // 3)
        if dist <= threshold and dist < best_score:
            best_score = dist
            best_match = cand

    return best_match


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
