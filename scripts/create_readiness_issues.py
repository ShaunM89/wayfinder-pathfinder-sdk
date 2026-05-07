#!/usr/bin/env python3
"""Create GitHub issues for SDK v0.1.0 readiness items."""

import subprocess
import sys

REPO = "ShaunM89/wayfinder-pathfinder-sdk"
LABEL = "enhancement"

ISSUES = [
    # ── Tier 1: Must-have for adoption ──────────────────────────────
    {
        "title": "[SDK readiness] Add CLI entry point: `python -m pathfinder_sdk rank <url> <task>`",
        "body": """## Goal
Add a first-class CLI so developers can use Pathfinder without writing Python.

## Proposed interface
```bash
# Rank candidates on a page
python -m pathfinder_sdk rank https://docs.python.org/3/ "Find the tutorial"

# With options
python -m pathfinder_sdk rank https://docs.python.org/3/ "Find the tutorial" \
  --model high --top-n 10 --output json
```

## Deliverables
- [ ] `src/pathfinder_sdk/__main__.py` implementing `python -m pathfinder_sdk`
- [ ] `src/pathfinder_sdk/cli.py` with `argparse` subcommands
- [ ] Subcommand: `rank <url> <task>` → prints JSON or human-readable table
- [ ] Optional flags: `--model`, `--top-n`, `--output {json,table}`, `--cache-dir`, `--fetcher`
- [ ] Exit codes: `0` = success, `1` = fetch error, `2` = model error, `3` = no candidates

## Why this matters
This is the highest-impact UX improvement. It turns a library you have to import into a tool you can run in 10 seconds.
""",
    },
    {
        "title": "[SDK readiness] Add `py.typed` marker for type-checker support",
        "body": """## Goal
Make the package fully typed so that `mypy` and `pyright` recognize and enforce types in downstream projects.

## Problem
Currently, type checkers treat `pathfinder_sdk` as untyped because there's no `py.typed` marker file. This means users lose autocomplete, type inference, and static analysis for the entire SDK surface.

## Deliverables
- [ ] Add `src/pathfinder_sdk/py.typed` (empty marker file)
- [ ] Ensure `pyproject.toml` includes it in the wheel via `tool.hatch.build.targets.wheel.packages` or similar
- [ ] Verify with `pip install -e . && python -c "import pathfinder_sdk; print(pathfinder_sdk.__file__)"` that `py.typed` is present
- [ ] Add a CI check: `mypy --strict src/pathfinder_sdk` passes (or at least doesn't report "module is untyped")

## Effort
Tiny — one empty file + packaging config.
""",
    },
    {
        "title": "[SDK readiness] Add `examples/` directory with real-world scripts",
        "body": """## Goal
Provide copy-pasteable examples that demonstrate common use cases.

## Proposed examples
| File | Purpose |
|---|---|
| `examples/01_basic_ranking.py` | Simplest possible usage: `Pathfinder().rank_candidates(...)` |
| `examples/02_with_llm_agent.py` | Full navigation loop: SDK ranks → LLM decides → iterate |
| `examples/03_batch_processing.py` | Process multiple (url, task) pairs efficiently |
| `examples/04_custom_filter.py` | Plug in a custom `HeuristicFilter` subclass |
| `examples/05_cli_one_liner.sh` | Shell one-liners using the CLI (blocked on #27) |

## Deliverables
- [ ] Create `examples/` directory at repo root
- [ ] Each example has a docstring header explaining what it does
- [ ] Each example is runnable standalone (with `if __name__ == "__main__":` guard)
- [ ] Add `examples/README.md` with index and quick-start
- [ ] Add CI check that examples are syntactically valid (parse-only is fine)

## Notes
- Keep examples focused — one concept per file.
- Use `docs.python.org` or similar stable targets so examples don't break.
""",
    },
    {
        "title": "[SDK readiness] Improve error messages with actionable suggestions",
        "body": """## Goal
Replace cryptic exceptions with helpful, actionable error messages.

## Current pain points
- `ModelNotFoundError: "default"` → doesn't list available tiers
- `FetchError: 403` → doesn't suggest trying Playwright or checking robots.txt
- `ModelLoadError` → doesn't mention ONNX vs PyTorch fallback
- Missing `curl_cffi` → doesn't tell user how to install it

## Deliverables
- [ ] `ModelNotFoundError`: include list of valid tiers (`default`, `high`, `ultra`)
- [ ] `FetchError` on 403/429: suggest `--fetcher playwright` or checking rate limits
- [ ] `ModelLoadError`: mention ONNX primary / PyTorch fallback, and `pip install pathfinder-sdk[onnx]`
- [ ] `ImportError` for optional deps: print `pip install pathfinder-sdk[playwright]` etc.
- [ ] Add `did_you_mean()` utility for model tier name fuzzing (e.g. `"defualt"` → `"default"`)
- [ ] Unit tests for each improved error message

## Example target
```python
ModelNotFoundError(
    'Model tier "defualt" not found. Did you mean "default"? '
    'Valid tiers: default, high, ultra.'
)
```
""",
    },
    {
        "title": "[SDK readiness] Add progress bar for model download",
        "body": """## Goal
Show download progress when models are fetched from Hugging Face Hub for the first time.

## Problem
First use of `Pathfinder()` downloads ~400MB (default) or 2–4GB (high/ultra) with zero user feedback. Users think the process is hung.

## Options
1. **Rich progress bar** (`rich` library) — beautiful, but adds dependency
2. **tqdm** — standard, lightweight, already familiar to ML users
3. **hf-hub built-in** — `huggingface_hub` supports progress callbacks; we can wire our own

## Deliverables
- [ ] Wrap `hf_hub_download` calls with a progress callback
- [ ] Use `tqdm` (lightweight, common in ML ecosystem)
- [ ] Make it optional: if `tqdm` not installed, fall back to simple logging
- [ ] Add `quiet=True` kwarg to `Pathfinder()` to suppress progress for headless use
- [ ] Unit test: mock download and assert progress callback is invoked

## Dependency note
`tqdm` could be an optional dependency (`pip install pathfinder-sdk[progress]`) or a core dependency since it's only ~50KB.
""",
    },

    # ── Tier 2: Delightful additions ────────────────────────────────
    {
        "title": "[SDK readiness] Add async support: `rank_candidates_async()`",
        "body": """## Goal
Enable non-blocking ranking for web apps and async pipelines.

## Proposed API
```python
sdk = Pathfinder()

# Async version of rank_candidates
result = await sdk.rank_candidates_async(
    url="https://docs.python.org/3/",
    task_description="Find the tutorial"
)
```

## Scope
- **Model inference**: ONNX Runtime is synchronous on CPU; wrap in `asyncio.to_thread()` or `loop.run_in_executor()`
- **Fetching**: `curl_cffi` is sync; same executor pattern. Playwright *is* async-native — can use `playwright.async_api` directly.
- **Filter**: Pure CPU, negligible latency; can stay sync.

## Deliverables
- [ ] `Pathfinder.rank_candidates_async()` method
- [ ] `Fetcher.fetch_async()` protocol (optional on base class)
- [ ] `PlaywrightFetcher.fetch_async()` using `async_playwright()`
- [ ] `CurlFetcher.fetch_async()` via thread executor
- [ ] Unit tests with `pytest-asyncio`
- [ ] Example in `examples/` showing FastAPI integration

## Why this matters
Web frameworks (FastAPI, Starlette, Quart) are async-first. Without this, users must wrap the SDK themselves, which is error-prone.
""",
    },
    {
        "title": "[SDK readiness] Add batch API: `rank_multiple()` for efficient multi-page ranking",
        "body": """## Goal
Allow ranking multiple (url, task) pairs in a single call, sharing model load overhead.

## Proposed API
```python
sdk = Pathfinder()

results = sdk.rank_multiple([
    ("https://docs.python.org/3/", "Find the tutorial"),
    ("https://docs.python.org/3/library/", "Find os.path docs"),
])
# Returns list[RankingResult]
```

## Optimization opportunity
Model is loaded once. Embedding cache is warm across the batch.

## Deliverables
- [ ] `Pathfinder.rank_multiple(requests, top_n=None) -> list[RankingResult]`
- [ ] Sequential fetch (respect politeness) with parallel rank if safe
- [ ] Progress reporting if `tqdm` available (linked to #31)
- [ ] Unit tests: verify model loads only once, results are ordered correctly
- [ ] Benchmark: compare `rank_multiple` vs loop of `rank_candidates`

## Notes
- Don't parallelize fetches by default — could overwhelm target servers.
- Could add `max_workers` kwarg for parallel fetch if user opts in.
""",
    },
    {
        "title": "[SDK readiness] Add persistent embedding cache (SQLite-backed)",
        "body": """## Goal
Survive process restarts by persisting the embedding cache to disk.

## Current state
`BiEncoderRanker` has an in-memory LRU cache (10k entries). This is lost on every restart, meaning repeated tasks recompute embeddings.

## Proposed design
- Key: `hash(task_text + candidate_text + model_tier)`
- Value: serialized embedding vector (numpy bytes or JSON)
- Backend: SQLite (zero external deps, single file)
- TTL: optional expiration (e.g., 7 days) to handle model updates

## Deliverables
- [ ] `src/pathfinder_sdk/cache.py` with `EmbeddingCache` protocol
- [ ] `SQLiteEmbeddingCache` implementation using `sqlite3`
- [ ] `InMemoryEmbeddingCache` (current behavior, for testing)
- [ ] Integrate into `BiEncoderRanker` via `cache=` kwarg
- [ ] `cache_dir` parameter on `Pathfinder()` defaults to `~/.cache/pathfinder/embeddings.db`
- [ ] Unit tests: hit/miss, TTL expiry, thread safety
- [ ] Benchmark: show speedup on warm cache

## Size estimate
At 384 dims × float32 = 1.5KB per embedding. 10k entries ≈ 15MB on disk.
""",
    },
    {
        "title": "[SDK readiness] Add streaming API: `rank_stream()` for incremental results",
        "body": """## Goal
Yield candidates as they are scored, enabling progressive UIs and early termination.

## Proposed API
```python
for candidate in sdk.rank_stream(
    url="https://docs.python.org/3/",
    task_description="Find the tutorial",
    top_n=10
):
    print(candidate.rank, candidate.href, candidate.score)
    # Can break early if score > threshold
```

## Design considerations
- Bi-encoder does batch scoring, so true streaming requires partial batch yields
- Alternative: score all, sort, then yield — simpler, still useful for memory-bound cases
- Could yield `CandidateRecommendation` objects or a lighter `StreamingCandidate` dataclass

## Deliverables
- [ ] `Pathfinder.rank_stream()` generator method
- [ ] Yield candidates in rank order (highest score first)
- [ ] Support `min_score` threshold for early termination
- [ ] Unit tests: verify ordering, verify early break doesn't leak resources
- [ ] Example: real-time CLI output that updates as candidates are scored

## Why this matters
Large pages (500+ links) can take seconds. Streaming lets a UI show top results immediately while the rest compute.
""",
    },
    {
        "title": "[SDK readiness] Add pre-commit hooks for contributors",
        "body": """## Goal
Enforce code quality automatically before commits.

## Deliverables
- [ ] `.pre-commit-config.yaml` with:
  - `ruff` (lint + format)
  - `black` (formatter — or remove if ruff-format covers it)
  - `mypy --strict` (type check)
  - `pytest` (fast unit tests only, not integration)
- [ ] Document in `CONTRIBUTING.md`: `pip install pre-commit && pre-commit install`
- [ ] Add `pre-commit` to `dev` dependency group in `pyproject.toml`
- [ ] CI already runs these; pre-commit just catches issues earlier

## Notes
- Keep hooks fast (< 10s total) so developers don't bypass them.
- Integration tests should NOT run in pre-commit — too slow, require network.
""",
    },

    # ── Tier 3: Production-grade ────────────────────────────────────
    {
        "title": "[SDK readiness] Add observability hooks: OpenTelemetry + Prometheus metrics",
        "body": """## Goal
Expose per-stage latency and throughput metrics for production monitoring.

## Proposed metrics
| Metric | Type | Labels |
|---|---|---|
| `pathfinder_rank_latency_seconds` | Histogram | `stage={fetch,filter,rank}`, `model_tier` |
| `pathfinder_candidates_total` | Counter | `stage={input,filtered,output}` |
| `pathfinder_fetch_errors_total` | Counter | `status_code`, `fetcher` |
| `pathfinder_model_load_duration_seconds` | Gauge | `model_tier`, `backend={onnx,pytorch}` |

## Design
- **OpenTelemetry**: Add `@trace_span()` decorators around each stage; users inject their own tracer provider
- **Prometheus**: Expose a metrics registry; users can mount it on `/metrics` in their app
- **Zero-dependency default**: Only activate if `opentelemetry-api` or `prometheus-client` is installed

## Deliverables
- [ ] `src/pathfinder_sdk/telemetry.py` with optional OTel integration
- [ ] `src/pathfinder_sdk/metrics.py` with optional Prometheus integration
- [ ] `Pathfinder()` accepts optional `tracer` and `metrics_registry` kwargs
- [ ] Unit tests: verify spans are created when OTel is available, no-op when absent
- [ ] Example: FastAPI app with `/metrics` endpoint

## Effort
Medium — requires careful design to stay zero-overhead when not enabled.
""",
    },
    {
        "title": "[SDK readiness] Add politeness controls: robots.txt, rate limiting, crawl delay",
        "body": """## Goal
Be a respectful web citizen by default.

## Deliverables
- [ ] `robots.txt` parsing via `robotparser` (stdlib) or `protego`
- [ ] Per-domain rate limiting: default 1 req/sec, configurable
- [ ] Respect `Crawl-delay` directive from robots.txt
- [ ] `User-Agent` rotation / custom UA string support
- [ ] `polite=True` default on `Pathfinder()`; can be disabled with `polite=False`
- [ ] Configurable `max_requests_per_domain` to prevent accidental hammering
- [ ] Unit tests: mock robots.txt, verify delays are enforced
- [ ] Document politeness defaults in README

## Why this matters
A navigation agent that fetches aggressively will get blocked. Being polite by default protects users and the web.
""",
    },
    {
        "title": "[SDK readiness] Add Docker image for containerized usage",
        "body": """## Goal
Provide an official Docker image so users can run Pathfinder without installing Python locally.

## Proposed interface
```bash
docker run --rm ghcr.io/shaunm89/pathfinder-sdk:latest \
  rank https://docs.python.org/3/ "Find the tutorial"
```

## Dockerfile requirements
- Based on `python:3.11-slim` (good balance of size and compatibility)
- Pre-installs `curl_cffi` system deps (libcurl)
- Optional Playwright layer via build arg
- Multi-stage build to minimize final image size
- Non-root user for security

## Deliverables
- [ ] `Dockerfile` at repo root
- [ ] `.dockerignore`
- [ ] GitHub Actions workflow to build + push to GHCR on release tags
- [ ] Document in README: `docker pull ghcr.io/shaunm89/pathfinder-sdk`
- [ ] Size target: < 200MB for base image (ONNX Runtime + BGE-small)

## Notes
- Model weights download at runtime, so base image stays small.
- Could provide `pathfinder-sdk:latest` (default) and `pathfinder-sdk:playwright` (with browser).
""",
    },
    {
        "title": "[SDK readiness] Add plugin system for custom fetchers and rankers",
        "body": """## Goal
Let users inject their own fetcher and ranker implementations without forking.

## Proposed API
```python
from pathfinder_sdk import register_fetcher, Fetcher

@register_fetcher("scrapy")
class ScrapyFetcher(Fetcher):
    def fetch(self, url):
        # Custom implementation
        ...

# Usage
sdk = Pathfinder(fetcher="scrapy")
```

## Design
- Entry-point based discovery: `pathfinder_sdk.fetchers` entry point group
- Decorator-based registration for simple cases
- Protocol/class-based interface: `Fetcher.fetch(url) -> list[dict]`
- Same pattern for rankers: `pathfinder_sdk.rankers` entry point group

## Deliverables
- [ ] `src/pathfinder_sdk/plugins.py` with registration + discovery logic
- [ ] `register_fetcher(name)` and `register_ranker(name)` decorators
- [ ] `Pathfinder()` accepts `fetcher="custom_name"` and resolves via registry
- [ ] Document the `Fetcher` and `Ranker` protocols in `PLUGINS.md`
- [ ] Example plugin package in `examples/plugin_example/`
- [ ] Unit tests: verify registration, verify resolution, verify error on unknown plugin

## Why this matters
Compass has internal fetchers that are hard to swap. A plugin system makes Pathfinder extensible for researchers and integrators.
""",
    },
    {
        "title": "[SDK readiness] Add configuration file support: `.pathfinder.yaml`",
        "body": """## Goal
Let users set defaults via config file instead of repeating kwargs.

## Search path (in order of priority)
1. `./.pathfinder.yaml` (project-local)
2. `~/.config/pathfinder/config.yaml` (user-global)
3. Environment variables (`PATHFINDER_MODEL`, `PATHFINDER_CACHE_DIR`, etc.)
4. Constructor kwargs (highest priority)

## Proposed schema
```yaml
model: default
top_n: 20
cache_dir: ~/.cache/pathfinder
fetcher: auto

# Fetcher settings
fetcher_config:
  timeout: 10
  max_retries: 3
  user_agent: "PathfinderSDK/0.1.0"

# Politeness (linked to #35)
polite: true
rate_limit: 1.0  # requests per second

# Telemetry (linked to #34)
telemetry:
  enabled: false
  exporter: otlp  # or prometheus
```

## Deliverables
- [ ] `src/pathfinder_sdk/config.py` with `load_config(path=None)` function
- [ ] Support YAML (primary) and JSON (fallback)
- [ ] `Pathfinder()` auto-loads config and merges with kwargs
- [ ] `--config` CLI flag (blocked on #27)
- [ ] JSON Schema for config validation
- [ ] Unit tests: priority merging, missing file handling, invalid schema errors
- [ ] Document in README with example config

## Dependency note
`PyYAML` could be optional (`pip install pathfinder-sdk[yaml]`); if absent, only JSON/ENV/kwargs work.
""",
    },
]


def create_issue(title: str, body: str) -> str:
    cmd = [
        "gh", "issue", "create",
        "--repo", REPO,
        "--title", title,
        "--label", LABEL,
        "--body", body,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR creating issue '{title}':\n{result.stderr}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def main():
    print(f"Creating {len(ISSUES)} issues in {REPO}...\n")
    for i, issue in enumerate(ISSUES, 1):
        tier = "Tier 1" if i <= 5 else "Tier 2" if i <= 10 else "Tier 3"
        print(f"[{tier}] {issue['title'][:60]}... ", end="", flush=True)
        url = create_issue(issue["title"], issue["body"])
        if url:
            print(f"✅ {url}")
        else:
            print("❌ FAILED")
    print("\nDone.")


if __name__ == "__main__":
    main()
