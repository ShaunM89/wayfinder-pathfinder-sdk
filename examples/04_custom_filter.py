"""Example 4: Custom heuristic filter.

Demonstrates subclassing HeuristicFilter to add domain-specific filtering
rules. In this example, we exclude links that contain "deprecated" in the
text and require a minimum anchor text length.

Run:
    python examples/04_custom_filter.py
"""

from pathfinder_sdk import Pathfinder
from pathfinder_sdk.filter import HeuristicFilter


class DocumentationFilter(HeuristicFilter):
    """Custom filter for documentation sites.

    Extends the base HeuristicFilter with:
    - Exclusion of links with "deprecated" or "obsolete" anchor text
    - Minimum anchor text length of 3 characters
    """

    def __init__(self, **kwargs):
        super().__init__(min_anchor_length=3, **kwargs)

    def filter(self, candidates: list[dict], base_url: str) -> list[dict]:
        """Filter candidates with custom rules."""
        # First apply base heuristic filtering
        filtered = super().filter(candidates, base_url)

        # Then apply custom rules
        result = []
        for cand in filtered:
            text = cand.get("text", "").lower()
            if "deprecated" in text or "obsolete" in text:
                continue
            result.append(cand)

        return result


def main() -> None:
    candidates = [
        {"href": "/tutorial", "text": "Tutorial"},
        {"href": "/api", "text": "API Reference"},
        {"href": "/old", "text": "Deprecated API (v1)"},
        {"href": "/new", "text": "New API (v2)"},
        {"href": "/x", "text": "x"},  # too short (min_anchor_length=3)
    ]

    # Use the custom filter directly
    custom_filter = DocumentationFilter()
    filtered = custom_filter.filter(candidates, base_url="https://example.com")

    print("Original candidates:")
    for c in candidates:
        print(f"  - {c['text']} -> {c['href']}")

    print(f"\nAfter custom filter ({len(filtered)} remaining):")
    for c in filtered:
        print(f"  - {c['text']} -> {c['href']}")


if __name__ == "__main__":
    main()
