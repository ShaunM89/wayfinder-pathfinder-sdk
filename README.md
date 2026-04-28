# Pathfinder SDK

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Local ranking engine for AI navigation agents.**

Pathfinder SDK is a standalone, local-first Python SDK that takes a URL + task description and outputs a structured, ranked list of candidate links for external LLM consumption. Unlike full audit orchestrators, Pathfinder is a **pure ranking layer** — it makes no final navigation decisions, leaving that to your LLM of choice.

## Features

- **Single canonical API**: `Pathfinder().rank_candidates(url, task_description)`
- **Three model tiers**: `default` (BGE-small, ~400MB), `high` (bge-m3, ~2.6GB), `ultra` (pplx-4b, ~4GB+)
- **Bi-encoder batch ranking**: Fast cosine-similarity scoring with single forward-pass encoding
- **Lightweight fetcher**: `curl_cffi` + BeautifulSoup by default; optional Playwright headless shell fallback
- **Heuristic pre-filtering**: Removes non-navigable links (<20ms latency budget)
- **ONNX Runtime default**: Cross-platform CPU inference; PyTorch fallback available
- **Model weights downloaded on demand**: Wheel stays under 250MB

## Installation

```bash
pip install pathfinder-sdk
```

With optional Playwright fallback for JS-rendered pages:

```bash
pip install pathfinder-sdk[playwright]
```

## Quickstart

```python
from pathfinder_sdk import Pathfinder

sdk = Pathfinder(model="default")
result = sdk.rank_candidates("https://example.com", "Find the privacy policy page")

print(f"Top match: {result.candidates[0].href}")
print(f"Score: {result.candidates[0].score:.2f}")
```

## With Pre-Extracted Links

```python
from pathfinder_sdk import Pathfinder

sdk = Pathfinder(model="high", top_n=10)

my_links = [
    {"href": "/pricing", "text": "Pricing", "surrounding_text": "See our plans"},
    {"href": "/contact", "text": "Contact Us", "surrounding_text": "Get in touch"},
]

result = sdk.rank_candidates(
    url="https://example.com",
    task_description="Find pricing information",
    candidates=my_links,
)
```

## Passing Results to an External LLM

```python
import json
from pathfinder_sdk import Pathfinder

sdk = Pathfinder(model="high")
result = sdk.rank_candidates("https://example.com", "Find privacy policy")

prompt = f"""Task: {result.task_description}

Ranked candidates:
{json.dumps([c.model_dump() for c in result.candidates[:5]], indent=2)}

Which link should I click?"""
```

## Model Tiers

| Tier | Model | Size | Latency (100 links) | Use Case |
|---|---|---:|---:|---|
| `default` | `BAAI/bge-small-en-v1.5` | ~400MB | ~1.5s | Balanced quality and memory |
| `high` | `BAAI/bge-m3` | ~2.6GB | ~2.2s | Accuracy-first navigation |
| `ultra` | `perplexity/pplx-embed-context-v1-4b` | ~4GB+ | ~5s+ | Maximum quality |

*First call includes model download time (~10–60s depending on tier and connection).*

## Architecture

```
URL + Task Description
        │
        ▼
┌─────────────────────────────┐
│  Pathfinder.rank_candidates │
└─────────────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
Fetch    Pre-extracted
   │         │
   ▼         ▼
┌─────────────────────────────┐
│  HeuristicFilter            │
│  • Drop mailto/tel/#        │
│  • Deduplicate              │
│  • Drop non-HTML            │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  BiEncoderRanker            │
│  • Batch encode (CRITICAL)  │
│  • Cosine similarity        │
│  • Sort top-N               │
└─────────────────────────────┘
        │
        ▼
   RankingResult (JSON)
```

## Development

```bash
# Clone
git clone https://github.com/ShaunM89/wayfinder-pathfinder-sdk.git
cd wayfinder-pathfinder-sdk

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
black src tests
ruff check src tests
```

## License

MIT License — see [LICENSE](LICENSE).

## Design Document

See [DESIGN.md](DESIGN.md) for the full architecture specification.
