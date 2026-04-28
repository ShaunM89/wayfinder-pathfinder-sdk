# Pathfinder SDK Examples

Copy-pasteable examples demonstrating common use cases.

## Quick Start

```bash
# Install the SDK
pip install pathfinder-sdk

# Run any example
python examples/01_basic_ranking.py
```

## Examples

| # | File | What it demonstrates |
|---|------|---------------------|
| 1 | [`01_basic_ranking.py`](01_basic_ranking.py) | Simplest usage: rank pre-extracted links |
| 2 | [`02_with_llm_agent.py`](02_with_llm_agent.py) | Full navigation loop: SDK ranks → LLM decides → iterate |
| 3 | [`03_batch_processing.py`](03_batch_processing.py) | Reuse a single SDK instance across multiple tasks |
| 4 | [`04_custom_filter.py`](04_custom_filter.py) | Subclass `HeuristicFilter` with domain-specific rules |
| 5 | [`05_cli_one_liner.sh`](05_cli_one_liner.sh) | Shell one-liners using the CLI |

## Notes

- Examples use **pre-extracted candidates** to avoid network dependencies.
- To test with live fetching, remove `fetcher=None` and the `candidates=` argument.
- Example 2 includes a **mock LLM**; replace `mock_llm_decide()` with a real OpenAI-compatible API call for production use.
