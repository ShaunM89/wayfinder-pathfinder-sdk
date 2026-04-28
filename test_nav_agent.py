"""Navigation agent — emulate Compass trace flow using Pathfinder SDK + local LLM.

This script validates that the full navigation loop works end-to-end:
1. Fetch page → extract links
2. Pathfinder ranks candidates by task relevance
3. LLM (llama-swap) selects the best link
4. Navigate to selected link
5. Repeat until max depth reached or task resolved

Usage:
    source .venv/bin/activate
    python test_nav_agent.py \
        --url https://docs.python.org/3/ \
        --task "Find the tutorial for beginners" \
        --max-depth 3

Requires:
    pip install openai  (for OpenAI-compatible API to llama-swap)
"""

import argparse
import json
import os
import sys
import time
from typing import Any

# Ensure SDK is importable without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pathfinder_sdk import Pathfinder
from pathfinder_sdk.models import RankingResult


LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://gx10:9292/v1")
LLAMA_SWAP_MODEL = os.environ.get("LLAMA_SWAP_MODEL", "qwen3.6-35b-a3b")


def create_llm_client() -> Any:
    """Create an OpenAI-compatible client pointing at llama-swap."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package required. Install with: pip install openai"
        ) from exc

    return OpenAI(base_url=LLAMA_SWAP_URL, api_key="dummy")


def llm_select_link(
    client: Any, task: str, ranked_candidates: list[dict], model: str = LLAMA_SWAP_MODEL
) -> dict:
    """Ask the LLM to select the best link from ranked candidates.

    Returns a dict with 'href', 'reason', and 'done' flag.
    """
    candidates_json = json.dumps(ranked_candidates, indent=2)

    prompt = f"""You are a web navigation agent. Your task: {task}

Here are the top candidate links on the current page, ranked by relevance:
{candidates_json}

Select the SINGLE best link to click next to make progress on the task.
If the current page already satisfies the task, say you are done.

Respond with ONLY a JSON object in this exact format:
{{"href": "https://...", "reason": "brief explanation", "done": false}}

If the task is already complete on this page, respond:
{{"href": "", "reason": "Task is complete because...", "done": true}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
    )

    content = response.choices[0].message.content.strip()

    # Extract JSON from possible markdown fences
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        decision = json.loads(content)
    except json.JSONDecodeError as exc:
        print(f"  ⚠️  LLM returned non-JSON: {content[:200]}")
        raise

    return decision


def run_navigation_trace(
    start_url: str,
    task: str,
    max_depth: int = 3,
    top_n: int = 10,
    llm_model: str = LLAMA_SWAP_MODEL,
) -> list[dict]:
    """Run a full navigation trace and return the step log."""
    client = create_llm_client()
    sdk = Pathfinder(model="default", fetcher="curl", top_n=top_n)

    trace: list[dict] = []
    current_url = start_url

    for step in range(1, max_depth + 1):
        print(f"\n{'─' * 60}")
        print(f"Step {step}/{max_depth}: {current_url}")
        print(f"Task: {task}")
        print("─" * 60)

        # 1. Rank candidates
        try:
            result: RankingResult = sdk.rank_candidates(
                url=current_url,
                task_description=task,
            )
        except Exception as exc:
            print(f"  ❌ Fetch/rank failed: {exc}")
            trace.append({
                "step": step,
                "url": current_url,
                "error": str(exc),
            })
            break

        print(f"  Found {result.total_links_analyzed} links "
              f"(→ {result.total_links_after_filter} after filter)")
        print(f"  Rank latency: {result.latency_ms:.0f} ms")

        if not result.candidates:
            print("  ⚠️  No candidates found on this page")
            trace.append({
                "step": step,
                "url": current_url,
                "candidates": [],
                "action": "no_candidates",
            })
            break

        for c in result.candidates[:5]:
            print(f"    rank={c.rank} score={c.score:.3f} text={c.text[:45]!r}")

        # 2. Ask LLM to select
        ranked = [c.model_dump() for c in result.candidates]
        decision = llm_select_link(client, task, ranked, model=llm_model)

        print(f"  LLM decision: href={decision.get('href', '')[:60]}")
        print(f"  Reason: {decision.get('reason', 'N/A')}")

        trace.append({
            "step": step,
            "url": current_url,
            "candidates": ranked,
            "decision": decision,
            "latency_ms": result.latency_ms,
        })

        if decision.get("done"):
            print(f"\n  ✅ Task complete! LLM says: {decision.get('reason')}")
            break

        next_href = decision.get("href", "")
        if not next_href:
            print("  ⚠️  LLM provided no href, stopping")
            break

        current_url = next_href

    else:
        print(f"\n  ⏹️  Max depth ({max_depth}) reached")

    sdk.unload()
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Navigation agent using Pathfinder SDK + local LLM"
    )
    parser.add_argument(
        "--url",
        default="https://docs.python.org/3/",
        help="Starting URL",
    )
    parser.add_argument(
        "--task",
        default="Find the tutorial for beginners",
        help="Navigation task description",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum navigation steps",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of candidates to show LLM",
    )
    parser.add_argument(
        "--model",
        default=LLAMA_SWAP_MODEL,
        help="LLM model ID on llama-swap",
    )
    args = parser.parse_args()

    print("Pathfinder SDK v0.1.0 — Navigation Agent Test")
    print(f"LLM endpoint: {LLAMA_SWAP_URL}")
    print(f"LLM model: {LLAMA_SWAP_MODEL}")
    print(f"Start URL: {args.url}")
    print(f"Task: {args.task}")

    start_time = time.perf_counter()
    trace = run_navigation_trace(
        args.url, args.task, args.max_depth, args.top_n, args.model
    )
    elapsed = time.perf_counter() - start_time

    print(f"\n{'=' * 60}")
    print(f"Trace complete: {len(trace)} steps in {elapsed:.1f}s")
    print("=" * 60)

    # Save trace to file
    trace_file = "nav_trace.json"
    with open(trace_file, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Trace saved to: {trace_file}")


if __name__ == "__main__":
    main()
