"""Example 2: Navigation agent loop — SDK ranks, LLM decides, iterate.

This demonstrates the full Compass-style navigation loop using Pathfinder
for link ranking and a mock LLM for decision-making.

To use a real LLM, replace `mock_llm_decide()` with an OpenAI-compatible
API call (e.g., OpenAI, llama-swap, ollama, vLLM).

Run:
    python examples/02_with_llm_agent.py
"""

import json
from pathfinder_sdk import Pathfinder


def mock_llm_decide(task: str, candidates: list, url: str, history: list) -> tuple[str, str]:
    """Mock LLM that always picks the highest-ranked candidate.

    Replace this with a real LLM call:

        import openai
        client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
        response = client.chat.completions.create(
            model="qwen3.6-35b-a3b",
            messages=[{"role": "user", "content": prompt}],
        )
        href = parse_href_from_response(response.choices[0].message.content)
    """
    if not candidates:
        return None, "No candidates available"
    best = candidates[0]
    return best["href"], f"Selected top-ranked candidate: {best['text']}"


def build_prompt(task: str, candidates: list, url: str, history: list) -> str:
    """Build a prompt for the LLM decision step."""
    lines = [
        f"You are navigating a website to complete a task.",
        f"Current URL: {url}",
        f"Task: {task}",
        f"History: {history}",
        f"\nRanked candidate links:",
    ]
    for c in candidates:
        lines.append(f"  {c['rank']}. {c['text']} ({c['score']:.3f}) -> {c['href']}")
    lines.append("\nWhich link should we follow? Respond with ONLY the href.")
    return "\n".join(lines)


def navigate(task: str, start_url: str, max_depth: int = 2) -> list[dict]:
    """Navigate from start_url toward task completion."""
    sdk = Pathfinder(model="default", fetcher=None)
    url = start_url
    history = []

    # Simulated page content for the demo
    pages = {
        "https://docs.python.org/3/": [
            {"href": "/tutorial", "text": "Tutorial"},
            {"href": "/library", "text": "Library Reference"},
            {"href": "/whatsnew", "text": "What's New"},
        ],
        "https://docs.python.org/3/tutorial": [
            {"href": "/tutorial/introduction", "text": "An Informal Introduction to Python"},
            {"href": "/tutorial/controlflow", "text": "More Control Flow Tools"},
            {"href": "/tutorial/datastructures", "text": "Data Structures"},
        ],
    }

    for depth in range(max_depth):
        print(f"\n--- Step {depth + 1}: {url} ---")

        candidates = pages.get(url, [])
        if not candidates:
            print("No candidates found.")
            break

        result = sdk.rank_candidates(
            url=url,
            task_description=task,
            candidates=candidates,
            top_n=3,
        )

        # Convert to dict for LLM prompt
        cand_dicts = [c.to_dict() for c in result.candidates]
        prompt = build_prompt(task, cand_dicts, url, history)

        href, reason = mock_llm_decide(task, cand_dicts, url, history)
        print(f"LLM: {reason}")
        print(f" -> Navigate to: {href}")

        history.append({"url": url, "choice": href, "reason": reason})
        url = href

        if url not in pages:
            print("Reached leaf page.")
            break

    sdk.unload()
    return history


def main() -> None:
    history = navigate(
        task="Find the beginner tutorial",
        start_url="https://docs.python.org/3/",
        max_depth=2,
    )
    print("\n=== Navigation trace ===")
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
