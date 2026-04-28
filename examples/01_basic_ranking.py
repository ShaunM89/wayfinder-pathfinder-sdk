"""Example 1: Basic ranking with pre-extracted candidates.

This is the simplest possible usage of Pathfinder SDK — rank a list of
candidate links for a given task without any page fetching.

Run:
    python examples/01_basic_ranking.py
"""

from pathfinder_sdk import Pathfinder


def main() -> None:
    # Pre-extracted candidate links from a documentation page
    candidates = [
        {"href": "/tutorial", "text": "Tutorial"},
        {"href": "/library", "text": "Library Reference"},
        {"href": "/whatsnew", "text": "What's New"},
        {"href": "/install", "text": "Installation Guide"},
        {"href": "/faq", "text": "Frequently Asked Questions"},
    ]

    sdk = Pathfinder(model="default", fetcher=None)

    result = sdk.rank_candidates(
        url="https://docs.python.org/3/",
        task_description="Find the beginner tutorial",
        candidates=candidates,
        top_n=3,
    )

    print(f"Task: {result.task_description}")
    print(f"Latency: {result.latency_ms:.2f} ms\n")

    for c in result.candidates:
        print(f"  #{c.rank}: {c.text} ({c.score:.3f}) -> {c.href}")

    sdk.unload()


if __name__ == "__main__":
    main()
