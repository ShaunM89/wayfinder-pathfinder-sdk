"""Heuristic filtering stage for candidate links.

Removes non-navigable links before embedding to reduce noise and improve latency.
Target latency: <20ms for typical page (100–500 links).
"""

from urllib.parse import urlparse

from pathfinder_sdk.utils import LinkNormalizer

# Non-navigable URL schemes
_NON_NAVIGABLE_SCHEMES = frozenset({"mailto", "tel", "javascript", "data", "file"})

# Non-HTML file extensions that should be dropped
_NON_HTML_EXTENSIONS = frozenset({
    ".pdf", ".zip", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".wmv",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dmg", ".apk", ".iso",
})

# DOM paths that indicate boilerplate (configurable)
_BOILERPLATETAGS = frozenset({"footer", "header"})


class HeuristicFilter:
    """Filters candidate links using fast heuristics.

    Args:
        exclude_boilerplate: Whether to exclude footer/header links.
        min_anchor_length: Minimum anchor text length to keep.
    """

    def __init__(self, exclude_boilerplate: bool = False, min_anchor_length: int = 1):
        self.exclude_boilerplate = exclude_boilerplate
        self.min_anchor_length = min_anchor_length

    def filter(self, candidates: list[dict], base_url: str) -> list[dict]:
        """Filter candidates and return deduplicated, navigable list.

        Args:
            candidates: List of candidate link dicts.
            base_url: Base URL for resolving relative hrefs.

        Returns:
            Filtered list of unique candidate dicts with normalized hrefs.
        """
        seen_hrefs: set[str] = set()
        filtered: list[dict] = []

        for cand in candidates:
            href = cand.get("href", "").strip()
            if not href:
                continue

            # Normalize to absolute URL
            normalized = LinkNormalizer.normalize(href, base_url)

            # Filter by scheme
            parsed = urlparse(normalized)
            if parsed.scheme in _NON_NAVIGABLE_SCHEMES:
                continue

            # Filter fragment-only links (same-page anchors with no path)
            if not parsed.path or parsed.path == "/":
                if parsed.fragment:
                    continue

            # Filter non-HTML extensions
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in _NON_HTML_EXTENSIONS):
                continue

            # Filter by anchor text length
            text = cand.get("text", "").strip()
            if len(text) < self.min_anchor_length:
                continue

            # Optional: exclude boilerplate
            if self.exclude_boilerplate:
                dom_path = (cand.get("dom_path") or "").lower()
                if any(tag in dom_path for tag in _BOILERPLATETAGS):
                    continue

            # Deduplicate by normalized href
            deduped = LinkNormalizer.remove_fragment(normalized)
            if deduped in seen_hrefs:
                continue
            seen_hrefs.add(deduped)

            # Update candidate with normalized href
            cand_copy = dict(cand)
            cand_copy["href"] = normalized
            filtered.append(cand_copy)

        return filtered
