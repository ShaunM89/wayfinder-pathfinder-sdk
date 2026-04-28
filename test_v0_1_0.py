"""End-to-end validation script for Pathfinder SDK v0.1.0.

Run with:
    cd /home/shaun-myandee/wayfinder-pathfinder
    source .venv/bin/activate
    python test_v0_1_0.py
"""

import json
import time

from pathfinder_sdk import Pathfinder


def heading(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def test_1_preextracted() -> None:
    """Test 1: Pre-extracted candidates (no fetch, fastest)."""
    heading("Test 1: Pre-extracted candidates")

    sdk = Pathfinder(model="default", fetcher=None)
    candidates = [
        {"href": "/about", "text": "About Us"},
        {"href": "/pricing", "text": "Pricing Plans"},
        {"href": "/contact", "text": "Contact Support"},
        {"href": "/docs", "text": "API Documentation"},
        {"href": "/blog", "text": "Company Blog"},
    ]

    result = sdk.rank_candidates(
        url="https://example.com",
        task_description="How much does this service cost?",
        candidates=candidates,
    )

    print(f"Task: {result.task_description}")
    print(f"Total links: {result.total_links_analyzed}")
    print(f"After filter: {result.total_links_after_filter}")
    print(f"Total latency: {result.latency_ms:.1f} ms")
    print(f"Stage breakdown: {json.dumps(result.metadata.get('stage_latencies', {}), indent=2)}")
    print()
    for c in result.candidates:
        print(f"  rank={c.rank}  score={c.score:.3f}  text={c.text!r}")

    # Assertions
    assert len(result.candidates) > 0, "Should return at least one candidate"
    assert result.candidates[0].text == "Pricing Plans", f"Expected 'Pricing Plans' at #1, got {result.candidates[0].text!r}"
    print("  ✅ PASSED: Pricing Plans ranked #1")


def test_2_live_fetch() -> None:
    """Test 2: Live fetch against a real website."""
    heading("Test 2: Live fetch (docs.python.org)")

    sdk = Pathfinder(model="default", fetcher="curl")
    result = sdk.rank_candidates(
        url="https://docs.python.org/3/",
        task_description="Find the tutorial for beginners",
    )

    print(f"Task: {result.task_description}")
    print(f"Raw links found: {result.total_links_analyzed}")
    print(f"After filter: {result.total_links_after_filter}")
    print(f"Total latency: {result.latency_ms:.1f} ms")
    print()
    for c in result.candidates[:5]:
        print(f"  rank={c.rank:2d}  score={c.score:.3f}  text={c.text[:50]!r}")

    assert result.total_links_analyzed > 10, f"Expected >10 raw links, got {result.total_links_analyzed}"
    top_texts = [c.text for c in result.candidates[:3]]
    assert "Tutorial" in top_texts, f"Expected 'Tutorial' in top 3, got {top_texts}"
    print(f"  ✅ PASSED: Live fetch works, 'Tutorial' in top 3 (at rank {[c.rank for c in result.candidates if c.text == 'Tutorial'][0]})")


def test_3_json_for_llm() -> None:
    """Test 3: JSON output suitable for external LLM consumption."""
    heading("Test 3: JSON output for LLM prompt")

    sdk = Pathfinder(model="default", fetcher=None, top_n=3)
    candidates = [
        {"href": "/privacy", "text": "Privacy Policy", "surrounding_text": "Read our privacy policy for details"},
        {"href": "/terms", "text": "Terms of Service", "surrounding_text": "By using this site you agree to our terms"},
        {"href": "/cookies", "text": "Cookie Policy", "surrounding_text": "We use cookies to improve your experience"},
    ]

    result = sdk.rank_candidates(
        url="https://example.com",
        task_description="Find the privacy policy",
        candidates=candidates,
    )

    prompt = f"""Task: {result.task_description}

Here are the top candidate links ranked by relevance:
{json.dumps([c.model_dump() for c in result.candidates], indent=2)}

Which link should I click next? Return only the href."""

    print(prompt)
    assert result.candidates[0].href == "https://example.com/privacy/"
    print("  ✅ PASSED: JSON serialization works for LLM prompts")


def test_4_different_tasks_same_page() -> None:
    """Test 4: Same page, different tasks produce different rankings."""
    heading("Test 4: Task sensitivity")

    sdk = Pathfinder(model="default", fetcher=None)
    candidates = [
        {"href": "/pricing", "text": "Pricing"},
        {"href": "/support", "text": "Support"},
        {"href": "/features", "text": "Features"},
    ]

    result_pricing = sdk.rank_candidates(
        "https://example.com", "How much does it cost?", candidates=candidates
    )
    result_support = sdk.rank_candidates(
        "https://example.com", "I need help", candidates=candidates
    )

    pricing_top = result_pricing.candidates[0].text
    support_top = result_support.candidates[0].text

    print(f"  Task 'How much does it cost?' → top: {pricing_top!r}")
    print(f"  Task 'I need help' → top: {support_top!r}")

    assert pricing_top == "Pricing", f"Expected Pricing, got {pricing_top!r}"
    assert support_top == "Support", f"Expected Support, got {support_top!r}"
    print("  ✅ PASSED: Different tasks produce different rankings")


def test_5_backend_reporting() -> None:
    """Test 5: Verify which inference backend is active."""
    heading("Test 5: Backend reporting")

    sdk = Pathfinder(model="default", fetcher=None)
    _ = sdk.rank_candidates(
        "https://example.com",
        "test",
        candidates=[{"href": "/a", "text": "A"}],
    )

    backend = sdk._ranker.backend
    print(f"  Active inference backend: {backend}")
    assert backend in ("onnx", "pytorch"), f"Unexpected backend: {backend}"
    print(f"  ✅ PASSED: Backend is {backend}")


def main() -> None:
    print("Pathfinder SDK v0.1.0 — End-to-End Validation")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        test_1_preextracted()
        test_2_live_fetch()
        test_3_json_for_llm()
        test_4_different_tasks_same_page()
        test_5_backend_reporting()

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED ✅")
        print("=" * 60)

    except AssertionError as exc:
        print(f"\n❌ FAILED: {exc}")
        raise
    except Exception as exc:
        print(f"\n❌ ERROR: {exc}")
        raise


if __name__ == "__main__":
    main()
