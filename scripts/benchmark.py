"""Lightweight navigation benchmark for Pathfinder SDK.

Usage:
    python scripts/benchmark.py --tier default

Evaluates a model tier on synthetic navigation tasks and reports
top-k accuracy and latency.
"""

import argparse
import os
import sys
import time

# Allow running from repo root without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathfinder_sdk import Pathfinder
from pathfinder_sdk.ranker import _MODEL_REGISTRY
from pathfinder_sdk.utils import LinkNormalizer


# Synthetic navigation test cases: (task, candidates, expected_href)
_TEST_CASES: list[tuple[str, list[dict], str]] = [
    (
        "Find the privacy policy",
        [
            {"href": "/about", "text": "About Us"},
            {"href": "/privacy", "text": "Privacy Policy"},
            {"href": "/contact", "text": "Contact"},
            {"href": "/terms", "text": "Terms of Service"},
        ],
        "/privacy",
    ),
    (
        "Find pricing information",
        [
            {"href": "/features", "text": "Features"},
            {"href": "/pricing", "text": "Pricing"},
            {"href": "/blog", "text": "Blog"},
            {"href": "/docs", "text": "Documentation"},
        ],
        "/pricing",
    ),
    (
        "Get in touch with support",
        [
            {"href": "/help", "text": "Help Center"},
            {"href": "/contact", "text": "Contact Us"},
            {"href": "/faq", "text": "FAQ"},
            {"href": "/status", "text": "System Status"},
        ],
        "/contact",
    ),
    (
        "Read about the company",
        [
            {"href": "/team", "text": "Our Team"},
            {"href": "/about", "text": "About Us"},
            {"href": "/careers", "text": "Careers"},
            {"href": "/press", "text": "Press"},
        ],
        "/about",
    ),
    (
        "Find the blog",
        [
            {"href": "/resources", "text": "Resources"},
            {"href": "/blog", "text": "Blog"},
            {"href": "/newsletter", "text": "Newsletter"},
            {"href": "/events", "text": "Events"},
        ],
        "/blog",
    ),
]


def run_benchmark(tier: str, top_k: int = 3) -> dict:
    """Run benchmark and return metrics."""
    sdk = Pathfinder(model=tier, top_n=top_k, fetcher=None)

    correct = 0
    total_latency = 0.0
    ranks: list[int] = []

    for task, candidates, expected in _TEST_CASES:
        result = sdk.rank_candidates(
            url="https://example.com",
            task_description=task,
            candidates=candidates,
        )

        total_latency += result.latency_ms

        # Find rank of expected answer (compare normalized hrefs)
        expected_normalized = LinkNormalizer.normalize(expected, "https://example.com")
        found_rank = None
        for cand in result.candidates:
            if cand.href == expected_normalized:
                found_rank = cand.rank
                break

        ranks.append(found_rank)
        if found_rank is not None and found_rank <= top_k:
            correct += 1

    sdk.unload()

    return {
        "tier": tier,
        "top_k": top_k,
        "total_cases": len(_TEST_CASES),
        "correct": correct,
        "accuracy": correct / len(_TEST_CASES),
        "avg_latency_ms": total_latency / len(_TEST_CASES),
        "ranks": ranks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pathfinder navigation benchmark")
    parser.add_argument(
        "--tier",
        choices=list(_MODEL_REGISTRY.keys()),
        default="default",
        help="Model tier to benchmark",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Consider correct if expected is in top-k",
    )
    args = parser.parse_args()

    print(f"Benchmarking tier: {args.tier}")
    print(f"Model: {_MODEL_REGISTRY[args.tier]}")
    print(f"Test cases: {len(_TEST_CASES)}")
    print("-" * 40)

    metrics = run_benchmark(args.tier, top_k=args.top_k)

    print(f"\nResults:")
    print(f"  Top-{args.top_k} accuracy: {metrics['correct']}/{metrics['total_cases']} ({metrics['accuracy']:.0%})")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.1f} ms")
    print(f"\nPer-case ranks (None = not in top-{args.top_k}):")
    for (task, _, expected), rank in zip(_TEST_CASES, metrics["ranks"]):
        print(f"  '{task[:40]}...' → {expected}: rank={rank}")


if __name__ == "__main__":
    main()
