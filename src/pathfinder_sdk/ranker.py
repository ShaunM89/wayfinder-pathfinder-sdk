"""Bi-encoder ranking stage with batch encoding.

Adapted from compass_core.features.semantic_features.
Critical invariant: batch encode all texts in a single forward pass.
"""

import logging
import threading
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from pathfinder_sdk.cache import InMemoryEmbeddingCache
from pathfinder_sdk.models import (
    CandidateRecommendation,
    ModelLoadError,
    ModelNotFoundError,
)
from pathfinder_sdk.utils import did_you_mean


class _NoOpTqdm:
    """No-op tqdm replacement for quiet mode."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass

    def set_description(self, desc):
        pass


logger = logging.getLogger(__name__)

# Model tier registry (HF Hub repo IDs)
_MODEL_REGISTRY: dict[str, str] = {
    "default": "BAAI/bge-small-en-v1.5",
    "high": "BAAI/bge-m3",
    "ultra": "perplexity/pplx-embed-context-v1-4b",
}

# Retry config for HF Hub downloads
_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_RETRY_DELAY = 2.0


def _download_with_retry(
    repo_id: str,
    cache_dir: str,
    max_retries: int = _DOWNLOAD_MAX_RETRIES,
    base_delay: float = _DOWNLOAD_RETRY_DELAY,
    tqdm_class=None,
) -> str:
    """Download model from HF Hub with exponential backoff retries.

    Args:
        repo_id: HuggingFace Hub repository ID.
        cache_dir: Local cache directory.
        max_retries: Maximum retry attempts.
        base_delay: Base delay in seconds between retries.
        tqdm_class: Optional tqdm class for progress display.

    Returns:
        Local path to downloaded model.

    Raises:
        ModelLoadError: If all retries are exhausted.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=False,
                tqdm_class=tqdm_class,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Model download attempt %d/%d failed for %s: %s",
                attempt,
                max_retries,
                repo_id,
                exc,
            )
            if attempt < max_retries:
                sleep_time = base_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)

    raise ModelLoadError(
        f"Failed to download model '{repo_id}' after {max_retries} "
        f"attempts: {last_error}"
    )


class BiEncoderRanker:
    """Ranks candidate links using bi-encoder cosine similarity.

    Attempts ONNX Runtime backend first (faster CPU inference), falling
    back to PyTorch via sentence-transformers if ONNX is unavailable.

    Args:
        model_tier: One of "default", "high", "ultra".
        cache_dir: Directory for HF Hub model cache.
        device: Inference device ("cpu", "cuda", or None for auto).
        local_model_path: Optional local path to bypass HF Hub download.
    """

    def __init__(
        self,
        model_tier: str = "default",
        cache_dir: str = "~/.cache/pathfinder",
        device: str | None = None,
        local_model_path: str | None = None,
        quiet: bool = False,
        cache=None,
    ):
        if model_tier not in _MODEL_REGISTRY:
            valid = list(_MODEL_REGISTRY.keys())
            suggestion = did_you_mean(model_tier, valid)
            msg = f'Unknown model tier: "{model_tier}". '
            if suggestion:
                msg += f'Did you mean "{suggestion}"? '
            msg += f"Valid tiers: {', '.join(valid)}."
            raise ModelNotFoundError(msg)
        self.model_tier = model_tier
        self.cache_dir = str(Path(cache_dir).expanduser())
        self.device = device or "cpu"
        self.local_model_path = local_model_path
        self.quiet = quiet
        self._cache = cache if cache is not None else InMemoryEmbeddingCache()
        self._model: SentenceTransformer | None = None
        self._backend: str = "pytorch"  # Tracks which backend is active
        self._lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def backend(self) -> str:
        """Return active inference backend ('onnx' or 'pytorch')."""
        return self._backend

    def _load_model(self) -> None:
        """Download model from HF Hub and load into memory.

        Tries ONNX Runtime first, then falls back to PyTorch.
        """
        if self.local_model_path:
            model_path = self.local_model_path
            logger.info("Using local model path: %s", model_path)
        else:
            repo_id = _MODEL_REGISTRY[self.model_tier]
            if not self.quiet:
                logger.info("Downloading model '%s' from HF Hub...", repo_id)
            tqdm_class = _NoOpTqdm if self.quiet else None
            model_path = _download_with_retry(
                repo_id, self.cache_dir, tqdm_class=tqdm_class
            )

        logger.info("Loading model from %s", model_path)

        # Attempt 1: ONNX Runtime backend (faster CPU inference)
        onnx_error: Exception | None = None
        try:
            self._model = SentenceTransformer(
                model_path,
                device=self.device,
                backend="onnx",
            )
            self._backend = "onnx"
            logger.info("Model loaded with ONNX Runtime backend")
            return
        except Exception as exc:
            onnx_error = exc
            logger.warning("ONNX backend failed (%s), falling back to PyTorch", exc)

        # Attempt 2: PyTorch fallback (always works if model files are present)
        try:
            self._model = SentenceTransformer(model_path, device=self.device)
            self._backend = "pytorch"
            logger.info("Model loaded with PyTorch backend")
        except Exception as pt_exc:
            msg = (
                f"Failed to load model from {model_path}. "
                f"ONNX error: {onnx_error}; PyTorch error: {pt_exc}\n"
                f"Hints:\n"
                f"  - Ensure model files are complete "
                f"(re-download with rm -rf {self.cache_dir})\n"
                f"  - ONNX backend: pip install pathfinder-sdk[onnx]\n"
                f"  - PyTorch backend should work out of the box"
            )
            raise ModelLoadError(msg) from pt_exc

    def rank(
        self,
        task: str,
        candidates: list[dict],
        top_n: int = 20,
    ) -> list[CandidateRecommendation]:
        """Rank candidates by cosine similarity to the task description.

        Uses batch encoding for all texts in a single forward pass.
        Caches individual embeddings for reuse across calls.

        Args:
            task: Natural language task description.
            candidates: Filtered candidate link dicts.
            top_n: Number of top results to return.

        Returns:
            List of CandidateRecommendation sorted by relevance.
        """
        if not candidates:
            return []

        # Prepare texts: task + all candidate anchor texts
        texts = [task] + [c.get("text", "").strip() for c in candidates]

        # Try to retrieve all embeddings from cache
        cached_embeddings: list[np.ndarray | None] = []
        for text in texts:
            emb = self._cache.get(text)
            cached_embeddings.append(emb)

        if all(e is not None for e in cached_embeddings):
            # Cache hit for all texts — skip model inference
            embeddings = np.stack(cached_embeddings)
        else:
            # Batch encode (CRITICAL: single forward pass)
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            # Store individual embeddings in cache
            for text, emb in zip(texts, embeddings):
                self._cache.put(text, emb)

        task_emb = embeddings[0]
        cand_embs = embeddings[1:]

        # Cosine similarity via sklearn for vectorized efficiency
        scores = sklearn_cosine_similarity([task_emb], cand_embs)[0]

        # Get top-N indices (descending)
        top_indices = scores.argsort()[-top_n:][::-1]

        recommendations: list[CandidateRecommendation] = []
        for rank_pos, idx in enumerate(top_indices, start=1):
            cand = candidates[idx]
            recommendations.append(
                CandidateRecommendation(
                    rank=rank_pos,
                    href=cand.get("href", ""),
                    text=cand.get("text", ""),
                    score=float(scores[idx]),
                    context_snippet=cand.get("surrounding_text"),
                )
            )

        return recommendations

    def unload(self) -> None:
        """Unload model from memory to free RAM."""
        self._model = None
        self._backend = "pytorch"

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text with caching."""
        with self._lock:
            if text in self._cache:
                return self._cache[text]

        embedding = self.model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )

        with self._lock:
            self._cache[text] = embedding

        return embedding
