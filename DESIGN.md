# Pathfinder SDK v1 — Master Design Document

**Version:** 1.0.0 (Consolidated)  
**Date:** 2026-04-28  
**Status:** Ready for Implementation  
**Supersedes:**
- `Pathfinder-SDK-API-Surface-Design-v1.2.md`
- `Pathfinder-SDK-Design-TaskB-Architecture-v1.2.md`
- `Pathfinder-SDK-Design-TaskC-DistillationPipeline-v1.2.md`
- `2026-04-23-pathfinder-sdk-v01-architecture-finalisation.md`

---

## 1. Executive Summary

Pathfinder SDK is a **standalone, local-first Python SDK** that extracts the core navigation/ranking engine from Compass (Wayfinder's full-stack audit platform) and packages it for enterprise/agency integration. Unlike Compass — which is a cloud-hosted, end-to-end audit orchestrator — Pathfinder is a **pure ranking layer** that consumes a URL + task description and outputs a structured, ranked list of candidate links for external LLM consumption.

**Key design principles:**
- **Single canonical entry point:** `Pathfinder().rank_candidates(url, task_description)` — all use cases flow through this.
- **No final navigation decision:** The SDK ranks candidates only; an external LLM (Anthropic, OpenAI, Gemini, local) makes the choice. This preserves portability across providers and enterprise policies.
- **Bi-encoder-only architecture:** Cross-encoder removed entirely after SOTA spike showed it hurt Hit@1 by 29% on navigation tasks.
- **Three model tiers** with clear quality/memory trade-offs (revised from original 2-model design based on research evidence).
- **ONNX Runtime default** for cross-platform CPU inference; CoreML Mac optimization deferred to Phase 2.
- **Model weights not included in wheel** — downloaded separately via HuggingFace Hub.
- **Optional lightweight fetcher** (`curl_cffi` + BeautifulSoup as default; Playwright headless shell for JS-rendered fallback). No full browser required by default.
- **Batch encoding is critical** — must use `model.encode(texts, batch_size=...)` not per-link loops.
- **Memory target:** v0.1 will exceed the original 500MB budget; distillation (v0.2) is the path to reduction.

**What Pathfinder is NOT:**
- Not a full audit orchestrator (that's Compass)
- Not a cloud service (runs fully locally)
- Not an LLM client (no API keys required for core ranking)
- Not a browser automation tool (lightweight fetch only; optional Playwright shell)

---

## 2. Product Positioning

| Dimension | Compass (Existing) | Pathfinder SDK (New) |
|---|---|---|
| **Target user** | SMBs, marketers, non-technical users | Serious agencies, enterprises, integrators |
| **Deployment** | Cloud-hosted (Railway + Vercel) | Local / self-hosted |
| **Entry point** | Web app + API | Python SDK (`pip install pathfinder-sdk`) |
| **Scope** | Full audit: crawl → score → report → deck | Link ranking only: URL + task → ranked candidates |
| **LLM dependency** | Required (Claude Sonnet default) | Optional (SDK outputs JSON for any LLM) |
| **Decision authority** | Compass makes final navigation decisions | External LLM makes final choice |
| **Model weights** | Shipped separately (XGBoost + MiniLM) | Downloaded via HF Hub (bi-encoder only) |
| **Memory budget** | ~500MB–1.5GB (full stack) | ~400MB–2.6GB (bi-encoder only, tier-dependent) |
| **Pricing** | Token-metered via Stripe | Free / open-source (MIT license) |

**Relationship:** Pathfinder is a **component** that serious Compass users may graduate to. Compass remains the accessible starting point; Pathfinder is for integrators wanting the core engine in their own products.

---

## 3. Architecture Overview

### 3.1 High-Level Flow

```
URL + Task Description
        │
        ▼
┌─────────────────────────────────────────────┐
│  Pathfinder.rank_candidates()               │
│  (Canonical entry point)                    │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 0: Fetch & Parse (Optional)          │
│  • curl_cffi + BeautifulSoup (default)      │
│  • Optional: Playwright headless shell      │
│  • Extract: href, text, surrounding_text    │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 1: Heuristic Filter                  │
│  • Remove non-navigable (mailto, tel, #)    │
│  • Deduplicate by normalized href           │
│  • Optional: footer/header exclusion        │
│  • Latency budget: <20ms                    │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 2: Bi-Encoder Batch Ranking          │
│  • Encode task_description (1 text)         │
│  • Encode all candidate links (batch)       │
│  • Cosine similarity scoring                │
│  • Sort descending, return top-N            │
│  • Latency: 650ms–2.2s (tier-dependent)     │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  RankingResult (JSON output)                │
│  {task, candidates[], scores[], metadata}   │
│  → External LLM for final decision          │
└─────────────────────────────────────────────┘
```

### 3.2 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Pathfinder SDK Public API                 │
│                                                              │
│   Pathfinder(model="default", top_n=20, cache_dir=...)      │
│   └── rank_candidates(url, task_description, candidates)     │
│        → RankingResult                                       │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Fetcher    │    │ HeuristicFilter │    │  BiEncoderRanker │
│  (Optional)  │    │   (Stage 1)     │    │    (Stage 2)     │
│              │    │                 │    │                  │
│ • curl_cffi  │    │ • URL filter    │    │ • ONNX Runtime   │
│ • Beautiful  │    │ • Deduplicate   │    │ • Batch encode   │
│ • Playwright │    │ • DOM filter    │    │ • Cosine sim     │
│   (optional) │    │                 │    │ • Top-N sort     │
└──────────────┘    └─────────────────┘    └──────────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                              ▼
                   ┌────────────────────┐
                   │   Model Registry   │
                   │  (HF Hub download) │
                   │                    │
                   │ • default: BGE-sm  │
                   │ • high: bge-m3     │
                   │ • ultra: pplx-4b   │
                   └────────────────────┘
```

### 3.3 Design Decisions & Rationale

| Decision | Rationale | Source |
|---|---|---|
| **Bi-encoder only** | Cross-encoder hurt Hit@1 by 29% on navigation tasks (MS-MARCO semantic mismatch) | SOTA Spike Report [4] |
| **Three model tiers** | Research showed 7 models with 2 Pareto frontiers; 3 tiers cover the quality spectrum | Study 02 [research] |
| **ONNX Runtime default** | Cross-platform CPU inference; no GPU required; mature HF integration | Architecture v1.2 [2] |
| **No model weights in wheel** | Wheel stays <250MB; models downloaded on first use via HF Hub | SOTA Spike [4] |
| **curl_cffi default fetcher** | Lighter than httpx + Playwright; no full browser required | User consensus |
| **Batch encoding mandatory** | Per-link encoding causes 10–100× latency regression (Compass v1.2.0 lesson) | SDK-ENGINE-MAP §5.4 |
| **JSON output for LLMs** | Preserves portability across Anthropic/OpenAI/Gemini/local | User requirement |
| **Memory >500MB accepted for v0.1** | Distillation is v0.2 path; BGE-small at ~400MB is closest v0.1 gets | User consensus |

---

## 4. Public API Specification

### 4.1 Canonical Entry Point

```python
from pathfinder_sdk import Pathfinder, RankingResult

# Initialize with model tier
sdk = Pathfinder(
    model="default",      # "default" | "high" | "ultra"
    top_n=20,             # Number of candidates to return
    cache_dir="~/.cache/pathfinder",  # Model download cache
)

# Rank candidates for external LLM consumption
result: RankingResult = sdk.rank_candidates(
    url="https://example.com",
    task_description="Find the privacy policy page",
    candidates=None,      # Optional: pre-extracted links; None → auto-fetch
)

# Pass to external LLM
import json
llm_prompt = f"""Task: {result.task_description}

Here are the top candidate links ranked by relevance:
{json.dumps([c.model_dump() for c in result.candidates], indent=2)}

Which link should I click next? Return only the href."""
```

### 4.2 `Pathfinder.__init__()`

```python
def __init__(
    self,
    model: str = "default",                    # Model tier selection
    top_n: int = 20,                           # Default candidates to return
    cache_dir: str = "~/.cache/pathfinder",    # HF Hub cache location
    fetcher: str | None = "auto",              # "auto" | "curl" | "playwright" | None
    device: str | None = None,                 # "cpu" | "cuda" | None (auto)
) -> None:
    """
    Initialize Pathfinder SDK with specified model tier.
    
    Args:
        model: Model tier — "default" (BGE-small, ~400MB), 
               "high" (bge-m3, ~2.6GB), "ultra" (pplx-4b, largest)
        top_n: Number of ranked candidates to return (1–100)
        cache_dir: Directory for cached model downloads from HF Hub
        fetcher: Fetcher backend — "auto" tries curl_cffi first, 
                 falls back to Playwright shell for JS-rendered pages
        device: Inference device — None auto-detects (CPU default)
    
    Raises:
        ModelNotFoundError: If specified model tier unavailable
        ConfigurationError: If invalid parameters provided
    """
```

### 4.3 `rank_candidates()` Method

```python
def rank_candidates(
    self,
    url: str,                                  # Required: Starting URL
    task_description: str,                     # Required: Natural language task
    candidates: list[dict] | None = None,      # Optional: Pre-extracted links
    top_n: int | None = None,                  # Optional: Override default top-N
) -> RankingResult:
    """
    Rank candidate links by relevance to the task description.
    
    Args:
        url: Starting page URL (used for fetch + result metadata)
        task_description: What you're trying to find (natural language)
        candidates: Optional pre-extracted links; if None, SDK fetches page
        top_n: Override constructor default (useful for per-call tuning)
    
    Returns:
        RankingResult with ranked candidates, scores, and metadata
    
    Raises:
        ValueError: If URL invalid or unreachable
        FetchError: If page fetch fails (network, timeout, blocked)
        ModelLoadError: If model files cannot be loaded
        MemoryError: If system RAM insufficient for model
    """
```

### 4.4 Pre-Extracted Candidates Format

When providing `candidates`, each link must include:

```python
{
    "href": str,                    # Required: Absolute or relative URL
    "text": str,                    # Required: Link anchor text
    "title": str | None,            # Optional: Link title attribute
    "surrounding_text": str,        # Recommended: Text within ~100 chars
    "dom_path": str | None,         # Optional: DOM path (e.g., "footer")
    "position": int | None,         # Optional: Position on page (0-indexed)
}
```

### 4.5 Output: `RankingResult`

```python
from pydantic import BaseModel
from typing import Optional

class CandidateRecommendation(BaseModel):
    rank: int                    # 1, 2, 3...
    href: str                    # Link URL (absolute, normalized)
    text: str                    # Anchor text
    score: float                 # Cosine similarity (0–1, higher = better)
    context_snippet: str | None  # Surrounding text for LLM context

class RankingResult(BaseModel):
    task_description: str
    source_url: str
    candidates: list[CandidateRecommendation]  # Top-N ranked links
    latency_ms: float
    total_links_analyzed: int
    total_links_after_filter: int
    model_tier: str              # Which model tier was used
    
    def to_json(self) -> str:
        """JSON serialization for API responses or LLM prompts."""
        return self.model_dump_json(indent=2)
    
    def to_dict(self) -> dict:
        """Dict serialization for programmatic use."""
        return self.model_dump()
```

### 4.6 Example Output

```json
{
  "task_description": "Find privacy policy page",
  "source_url": "https://example.com",
  "candidates": [
    {
      "rank": 1,
      "href": "https://example.com/privacy-policy",
      "text": "Privacy Policy",
      "score": 0.89,
      "context_snippet": "For more information, see our Privacy Policy and Terms."
    },
    {
      "rank": 2,
      "href": "https://example.com/legal",
      "text": "Legal Information",
      "score": 0.72,
      "context_snippet": "Read our legal terms and privacy information"
    }
  ],
  "latency_ms": 653.2,
  "total_links_analyzed": 141,
  "total_links_after_filter": 128,
  "model_tier": "default"
}
```

---

## 5. Model Tiers

### 5.1 Three-Tier Selection (Revised from Original Design)

The original design specified 2 models (`bge-m3` and `mpnet-base-v2`). Research (Study 02) evaluated 7 embedders and revealed 2 Pareto frontiers. The revised 3-tier design covers the full quality spectrum:

| Tier | Model | Size (FP32) | Mean Judge* | Hit@50** | Latency (100 links) | Use Case |
|---|---|---:|---:|---:|---:|---|
| **default** | `BGE-small-en-v1.5` | ~400MB | 0.71 | TBD | ~1.5s | Balanced: good quality, manageable memory |
| **high** | `bge-m3` | ~2.6GB | TBD | 0.980 | ~2.2s | Accuracy-first: best measured Hit@50 |
| **ultra** | `pplx-embed-context-v1-4b` | ~4GB+ | 0.86 | TBD | ~5s+ | Maximum quality: highest research score |

\* Mean judge from Study 02 (RAG retrieval benchmark, not navigation-specific).  
\*\* Hit@50 from SOTA spike (navigation-specific). BGE-small and pplx Hit@50 not yet measured on navigation tasks.

### 5.2 Model Tier Rationale

**Why 3 tiers instead of 2?**
- Study 02 showed `BGE-small-en-v1.5` (0.71 mean judge) is a clear step up from `all-MiniLM-L6-v2` (0.63) and is the storage-frontier leader (1,536 bytes/chunk).
- `bge-m3` has the only navigation-specific Hit@50 measurement (0.980) and is the accuracy leader.
- `pplx-embed-context-v1-4b` has the highest research score (0.86 mean judge) but was not tested on navigation tasks; included as "ultra" for serious local deployments.

**Why not mpnet-base-v2?**
- `mpnet-base-v2` was the original "fast" fallback but has no direct measurement in Study 02.
- `BGE-small-en-v1.5` is comparable in size (~400MB vs ~440MB for mpnet) with measured quality (0.71 vs mpnet's unmeasured).
- If mpnet proves faster in practice, it can be added as an additional tier later.

### 5.3 Model Download & Caching

```python
from huggingface_hub import snapshot_download

def _download_model(tier: str, cache_dir: str) -> str:
    """Download model weights from HF Hub."""
    model_ids = {
        "default": "BAAI/bge-small-en-v1.5",
        "high": "BAAI/bge-m3",
        "ultra": "perplexity-ai/pplx-embed-context-v1-4b",
    }
    return snapshot_download(
        repo_id=model_ids[tier],
        cache_dir=cache_dir,
        local_files_only=False,
    )
```

- First call downloads model; subsequent calls use cache.
- Models are **not** included in the wheel (wheel stays <250MB).
- Total package size with one model: ~400MB (default) to ~4GB+ (ultra).

---

## 6. Inference Pipeline

### 6.1 Critical: Batch Encoding

**This is the #1 performance invariant.** Compass v1.2.0 suffered a 551s → 5s regression fix caused by serial per-link `model.encode()` calls. Pathfinder MUST batch all candidate texts in a single `encode()` call.

```python
# CORRECT: Batch encoding (MANDATORY)
all_texts = [task_description] + [c["text"] for c in filtered_candidates]
embeddings = model.encode(
    all_texts,
    batch_size=32,
    convert_to_numpy=True,
    show_progress_bar=False,
)
task_embedding = embeddings[0]
candidate_embeddings = embeddings[1:]
scores = cosine_similarity([task_embedding], candidate_embeddings)[0]

# WRONG: Per-link loop (DO NOT IMPLEMENT)
for candidate in candidates:
    score = model.encode(task_description + candidate["text"])  # CATASTROPHIC
```

### 6.2 Heuristic Filter (Stage 1)

Applied before embedding to reduce noise and improve latency:

| Filter | Action | Rationale |
|---|---|---|
| **URL scheme** | Drop `mailto:`, `tel:`, `javascript:` | Non-navigable |
| **Fragment-only** | Drop `#anchor` links (same-page) | No navigation value |
| **Deduplicate** | Normalize href (resolve relative, strip fragments) | Avoid duplicate candidates |
| **Non-HTML** | Drop `.pdf`, `.zip`, `.jpg`, etc. | Compass lesson: non-HTML crashes downstream |
| **Optional DOM** | Exclude `footer`, `header` if flagged | Reduce boilerplate (configurable) |

Latency budget: **<20ms** for typical page (100–500 links).

### 6.3 Bi-Encoder Ranking (Stage 2)

```python
class BiEncoderRanker:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_path, device=device)
        self.embedding_cache = LRUCache(maxsize=10000)
    
    def rank(
        self,
        task: str,
        candidates: list[dict],
        top_n: int = 20,
    ) -> list[CandidateRecommendation]:
        # Batch encode task + all candidates
        texts = [task] + [c["text"] for c in candidates]
        embeddings = self.model.encode(texts, batch_size=32, convert_to_numpy=True)
        
        # Cosine similarity
        task_emb = embeddings[0]
        cand_embs = embeddings[1:]
        scores = cosine_similarity([task_emb], cand_embs)[0]
        
        # Sort and package top-N
        top_indices = scores.argsort()[-top_n:][::-1]
        return [
            CandidateRecommendation(
                rank=i+1,
                href=candidates[idx]["href"],
                text=candidates[idx]["text"],
                score=float(scores[idx]),
                context_snippet=candidates[idx].get("surrounding_text"),
            )
            for i, idx in enumerate(top_indices)
        ]
```

### 6.4 ONNX Runtime Backend

Default inference uses ONNX Runtime for cross-platform compatibility:

```python
import onnxruntime as ort

class ONNXBiEncoder:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            f"{model_path}/model.onnx",
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="np")
            outputs = self.session.run(None, dict(inputs))
            all_embeddings.append(outputs[0])
        return np.vstack(all_embeddings)
```

**Fallback:** If ONNX model unavailable, fall back to PyTorch CPU inference via `sentence-transformers`.

---

## 7. Fetcher Architecture

### 7.1 Default: `curl_cffi` + BeautifulSoup

```python
from curl_cffi import requests
from bs4 import BeautifulSoup

class CurlFetcher:
    def fetch(self, url: str) -> list[dict]:
        resp = requests.get(url, impersonate="chrome")
        soup = BeautifulSoup(resp.content, "html.parser")
        
        candidates = []
        for link in soup.find_all("a", href=True):
            candidates.append({
                "href": urljoin(url, link["href"]),
                "text": link.get_text(strip=True),
                "title": link.get("title"),
                "surrounding_text": self._extract_context(link),
                "dom_path": self._get_dom_path(link),
            })
        return candidates
```

**Why `curl_cffi` over `httpx`?**
- Better bot-detection evasion (impersonates real browser TLS/JA3 fingerprint).
- Lighter than Playwright (no browser process).
- Used successfully in Compass preflight checker.

### 7.2 Optional: Playwright Headless Shell

For JS-rendered pages where `curl_cffi` returns insufficient content:

```python
from playwright.sync_api import sync_playwright

class PlaywrightFetcher:
    def fetch(self, url: str) -> list[dict]:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, shell=True)  # shell=True = headless_shell
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            # Extract links same as CurlFetcher...
            browser.close()
```

**Requirements:**
- Playwright is an **optional dependency** (`pip install pathfinder-sdk[playwright]`).
- Uses `headless_shell` (not full Chromium) where available.
- Documented as fallback, not default.

### 7.3 Fetcher Selection Logic

```python
class Fetcher:
    def __init__(self, backend: str = "auto"):
        self.backend = backend
    
    def fetch(self, url: str) -> list[dict]:
        if self.backend == "curl":
            return CurlFetcher().fetch(url)
        elif self.backend == "playwright":
            return PlaywrightFetcher().fetch(url)
        elif self.backend == "auto":
            # Try curl first; fallback to playwright if <3 usable links
            candidates = CurlFetcher().fetch(url)
            if len(candidates) < 3:
                candidates = PlaywrightFetcher().fetch(url)
            return candidates
        else:
            raise ValueError(f"Unknown fetcher: {self.backend}")
```

---

## 8. Distillation Pipeline (v0.2, Parallel Workstream)

### 8.1 Goal

Train a smaller student model to match `bge-m3` quality at <500MB memory (vs. ~2.6GB currently).

### 8.2 Student Model Selection

**Revised from MiniLM-L6 to BGE-small:**
- Study 02 shows `BGE-small-en-v1.5` scores 0.71 mean judge vs. `all-MiniLM-L6-v2`'s 0.63.
- BGE-small is a better starting point for distillation (already closer to target).
- Target: ~400MB → ~200MB distilled (50% reduction) while maintaining ≥90% of bge-m3 Hit@50.

### 8.3 Training Data

| Metric | Value | Source |
|---|---|---|
| Traces audited | 2,749 | Spike 0 audit |
| Usable eval steps | 1,100 | Spike 0 audit (success=True, non-terminal) |
| Noise flag rate | 9.0% | Spike 0 audit (exclude flagged steps) |
| Pairwise samples | ~10,000–20,000 | 1,100 steps × 10 negatives each |

### 8.4 Loss Function

Start with **pairwise ranking loss** (existing labels). Add **knowledge distillation loss** if using `bge-m3` soft labels:

```python
# Pairwise loss
loss = max(0, margin - (score_positive - score_negative))

# Knowledge distillation (optional)
loss_kd = KL_divergence(student_logits, teacher_softmax(teacher_logits))
loss_total = loss_pairwise + alpha * loss_kd
```

### 8.5 Go/No-Go Gates

| Scenario | Action |
|---|---|
| **Hit@50 ≥ 0.94 AND Memory ≤ 500MB** | ✅ GO: Ship as v0.2 |
| **Hit@50 ≥ 0.94 BUT Memory > 500MB** | ⚠️ PARTIAL: Try smaller student (MiniLM-L6) |
| **Hit@50 < 0.94 AND Memory ≤ 500MB** | ⚠️ PARTIAL: Add teacher augmentation |
| **Hit@50 < 0.94 OR Memory > 1GB** | ❌ NO-GO: Stick with off-the-shelf tiers |

### 8.6 Timeline

- **Weeks 1–2:** Data extraction + initial training (BGE-small student)
- **Week 3:** Validation + hyperparameter tuning
- **Week 4:** Quantization (INT8) + benchmarking vs. bge-m3
- **Total:** 4–6 weeks parallel to v0.1 development

---

## 9. Dependency Management

### 9.1 Core Dependencies (`pyproject.toml`)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pathfinder-sdk"
version = "0.1.0"
description = "Local ranking engine for AI navigation agents"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
dependencies = [
    "onnxruntime>=1.16.0",           # CPU inference
    "transformers>=4.35.0",          # Model loading + ONNX export
    "sentence-transformers>=2.3.0",  # Bi-encoder logic
    "torch>=2.0.0,<3.0.0",           # CPU-only
    "numpy>=1.24.0",
    "pydantic>=2.0.0",               # Result dataclasses
    "curl-cffi>=0.6.0",              # Default fetcher
    "beautifulsoup4>=4.12.0",        # HTML parsing
    "huggingface-hub>=0.19.0",       # Model download
]

[project.optional-dependencies]
playwright = ["playwright>=1.40.0"]  # JS-rendered fallback
mac = ["coremltools>=7.0"]           # Phase 2 CoreML
 dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
```

### 9.2 Wheel Size

- **SDK code + deps:** ≤250MB
- **Model weights:** Downloaded separately (~400MB–4GB+ depending on tier)
- **Total installed:** ~650MB (default tier) to ~4.5GB+ (ultra tier)

---

## 10. Error Handling & Fallbacks

| Error Type | Action | Documentation |
|---|---|---|
| **ONNX load fails** | Fall back to PyTorch CPU via `sentence-transformers` | Document as known v0.1 limitation |
| **Model download fails** | Retry 3× with exponential backoff; allow local path override | Provide HF Hub download script |
| **Memory pressure (OOM)** | Auto-unload model after rank call; raise clear error | Document min spec per tier |
| **HF Hub rate limit** | Retry with backoff; cache locally | Document in deployment guide |
| **Fetch fails (curl)** | Auto-fallback to Playwright if installed | Log fallback event |
| **Page has no links** | Return empty candidates list (not an error) | Document expected behavior |
| **All candidates filtered out** | Return empty list with metadata | Document filter criteria |

---

## 11. Performance Targets

### 11.1 Latency Budgets

| Tier | Model | 100 links | 200 links | 500 links |
|---|---|---|---|---|
| default | BGE-small | ~1.5s | ~2.5s | ~5s |
| high | bge-m3 | ~2.2s | ~3.5s | ~7s |
| ultra | pplx-4b | ~5s+ | ~8s+ | ~15s+ |

*Note: First call includes model download + load time (~10–60s depending on tier and connection).*

### 11.2 Memory Budgets

| Tier | Model | Model Size | Total Stack* |
|---|---|---|---|
| default | BGE-small | ~400MB | ~600MB |
| high | bge-m3 | ~2.6GB | ~3.0GB |
| ultra | pplx-4b | ~4GB+ | ~4.5GB+ |

\* Including ONNX Runtime overhead + embedding cache (10k entries ≈ 15MB for 384-dim).

### 11.3 Throughput

| Tier | Model | Chunks/sec (batched-32)** |
|---|---|---|
| default | BGE-small | ~3,652 |
| high | bge-m3 | TBD |
| ultra | pplx-4b | ~75 |

\*\* Measured on NVIDIA GB10 (Blackwell-class) GPU. CPU throughput will be lower.

---

## 12. Integration Examples

### 12.1 Quickstart (5 Lines)

```python
from pathfinder_sdk import Pathfinder

sdk = Pathfinder(model="default")
result = sdk.rank_candidates("https://example.com", "Find privacy policy")
print(f"Top match: {result.candidates[0].href}")
```

### 12.2 With Pre-Extracted Links

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

### 12.3 Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
from pathfinder_sdk import Pathfinder

def rank_page(url: str, task: str):
    sdk = Pathfinder(model="default")
    return sdk.rank_candidates(url=url, task_description=task)

urls_tasks = [
    ("https://site1.com", "Find contact information"),
    ("https://site2.com", "Find contact information"),
]

with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(lambda x: rank_page(*x), urls_tasks))
```

### 12.4 Passing to External LLM

```python
import json
from pathfinder_sdk import Pathfinder
from anthropic import Anthropic

sdk = Pathfinder(model="high")
result = sdk.rank_candidates("https://example.com", "Find privacy policy")

prompt = f"""Task: {result.task_description}

Ranked candidates:
{json.dumps([c.model_dump() for c in result.candidates[:5]], indent=2)}

Which link should I click? Respond with JSON: {{"href": "...", "reason": "..."}}"""

client = Anthropic(api_key="sk-...")
response = client.messages.create(model="claude-3-haiku", messages=[{"role": "user", "content": prompt}])
```

---

## 13. Breaking Changes Policy

| Version Range | Scope | Breaking Changes |
|---|---|---|
| **0.x.x** | Beta (no stability guarantees) | API changes allowed; document in changelog |
| **1.0.0+** | Production stable | Only non-breaking additions; deprecation notices for 2 versions |

---

## 14. Known Gaps & Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Sub-second latency only on ≤100 links** | Pages with >200 links may exceed 2s | Document latency per link count; tune top_n |
| **v0.1 exceeds 500MB memory target** | ~600MB (default) to ~4.5GB (ultra) | Accept for v0.1; distillation is v0.2 path |
| **BGE-small Hit@50 not yet measured** | Navigation-specific accuracy unknown | Run navigation benchmark before v0.1 ship |
| **pplx-4b not tested on navigation** | Ultra tier quality is theoretical | Add navigation benchmark for ultra tier |
| **8GB machine support** | bge-m3 may OOM on 8GB | Document min specs; default tier works on 8GB |
| **HF Hub rate limiting** | Model downloads may fail | Retry logic + local path override |
| **Playwright headless_shell availability** | May not be installed on all systems | Document install command; curl fallback |

---

## 15. Implementation Checklist

### Phase 1: Core Pipeline (Week 1)
- [ ] SDK package structure (`pathfinder_sdk/`)
- [ ] `Pathfinder` class with `__init__` and `rank_candidates`
- [ ] Model registry + HF Hub download
- [ ] Bi-encoder batch encoding (CRITICAL: single batch call)
- [ ] Heuristic filter (URL dedupe, non-navigable removal)
- [ ] `RankingResult` Pydantic models
- [ ] Basic tests passing

### Phase 2: Fetcher & Polish (Week 2)
- [ ] `curl_cffi` + BeautifulSoup fetcher
- [ ] Optional Playwright headless shell fallback
- [ ] Surrounding text extraction
- [ ] Error handling + fallback strategies
- [ ] Logging + latency breakdown
- [ ] Documentation + README

### Phase 3: Release Prep (Week 3)
- [ ] ONNX Runtime export for all tiers
- [ ] PyTorch fallback for missing ONNX
- [ ] Model download mechanism finalized
- [ ] API docs (docstrings → auto-generated)
- [ ] v0.1 release candidate
- [ ] Navigation benchmark for default + high tiers

### Parallel: Distillation (Weeks 1–4)
- [ ] Data extraction from Compass traces
- [ ] BGE-small student training
- [ ] Validation vs. bge-m3 baseline
- [ ] INT8 quantization
- [ ] Go/No-Go decision

---

## 16. References

| ID | Document | Description |
|---|---|---|
| [1] | `Pathfinder-SDK-API-Surface-Design-v1.2.md` | Original API surface design (superseded) |
| [2] | `Pathfinder-SDK-Design-TaskB-Architecture-v1.2.md` | Original architecture design (superseded) |
| [3] | `Pathfinder-SDK-Design-TaskC-DistillationPipeline-v1.2.md` | Original distillation design (superseded) |
| [4] | `2026-04-23-pathfinder-sdk-v01-architecture-finalisation.md` | Architecture finalisation session |
| [5] | `SDK-ENGINE-MAP.md` | Compass v1.7.0 deep technical map |
| [6] | `CANONICAL-STATE.md` | Compass high-level state mirror |
| [7] | `02-embedder-strategy-PAPER-DRAFT.md` | Study 02: 7 embedders on live-web RAG |
| [8] | `01-chunking-strategy-PAPER-DRAFT.md` | Study 01: 9 chunking strategies |
| [9] | SOTA_SPIKE_REPORT.md | Bi-encoder SOTA validation (Hit@50=0.980) |
| [10] | SPIKE_REPORT.md | Accuracy targets + memory budget analysis |

---

## 17. Document History

| Version | Date | Changes |
|---|---|---|
| v1.0.0 | 2026-04-28 | Consolidated from 4 source docs + research papers + Compass codebase review. Revised model tiers from 2 to 3 based on Study 02. Switched default fetcher to curl_cffi. Switched distillation student from MiniLM-L6 to BGE-small. |

---

*Document ends. This is the single source of truth for Pathfinder SDK v1 implementation.*
