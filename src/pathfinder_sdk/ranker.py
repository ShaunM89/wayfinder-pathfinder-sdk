"""Bi-encoder ranking stage with batch encoding.

Adapted from compass_core.features.semantic_features.
Critical invariant: batch encode all texts in a single forward pass.
"""

import logging
import threading
import time
from pathlib import Path

import numpy as np
from cachetools import LRUCache
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from pathfinder_sdk.models import CandidateRecommendation, ModelLoadError, ModelNotFoundError

logger = logging.getLogger(__name__)

# Model tier registry (HF Hub repo IDs)
_MODEL_REGISTRY: dict[str, str] = {
    "default": "BAAI/bge-small-en-v1.5",
    "high": "BAAI/bge-m3",
    "ultra": "perplexity/pplx-embed-context-v1-4b",
}


class BiEncoderRanker:
    """Ranks candidate links using bi-encoder cosine similarity.

    Args:
        model_tier: One of "default", "high", "ultra".
        cache_dir: Directory for HF Hub model cache.
        device: Inference device ("cpu", "cuda", or None for auto).
    """

    def __init__(
        self,
        model_tier: str = "default",
        cache_dir: str = "~/.cache/pathfinder",
        device: str | None = None,
    ):
        if model_tier not in _MODEL_REGISTRY:
            raise ModelNotFoundError(
                f"Unknown model tier: {model_tier}. "
                f"Available: {list(_MODEL_REGISTRY.keys())}"
            )
        self.model_tier = model_tier
        self.cache_dir = str(Path(cache_dir).expanduser())
        self.device = device or "cpu"
        self._model: SentenceTransformer | None = None
        # 10_000 entries ≈ 15 MB for 384-dim float32 embeddings
        self._cache: LRUCache = LRUCache(maxsize=10000)
        self._lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """Download model from HF Hub and load into memory."""
        repo_id = _MODEL_REGISTRY[self.model_tier]
        logger.info("Downloading model '%s' from HF Hub...", repo_id)
        try:
            model_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=self.cache_dir,
                local_files_only=False,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to download model '{repo_id}': {exc}"
            ) from exc

        logger.info("Loading model from %s", model_path)
        try:
            self._model = SentenceTransformer(model_path, device=self.device)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model from {model_path}: {exc}"
            ) from exc

    def rank(
        self,
        task: str,
        candidates: list[dict],
        top_n: int = 20,
    ) -> list[CandidateRecommendation]:
        """Rank candidates by cosine similarity to the task description.

        Uses batch encoding for all texts in a single forward pass.

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

        # Batch encode (CRITICAL: single forward pass)
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

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
