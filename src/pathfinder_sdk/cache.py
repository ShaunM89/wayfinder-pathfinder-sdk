"""Embedding cache implementations for Pathfinder SDK.

Provides in-memory and SQLite-backed caches for embedding vectors
to avoid recomputing them across sessions.
"""

import logging
import sqlite3
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class InMemoryEmbeddingCache:
    """Simple in-memory embedding cache (non-persistent)."""

    def __init__(self):
        self._data: dict[str, np.ndarray] = {}

    def get(self, key: str) -> np.ndarray | None:
        """Retrieve embedding by key."""
        return self._data.get(key)

    def put(self, key: str, embedding: np.ndarray) -> None:
        """Store embedding by key."""
        self._data[key] = embedding


class SQLiteEmbeddingCache:
    """Persistent SQLite-backed embedding cache.

    Args:
        db_path: Path to SQLite database file.
        ttl_seconds: Time-to-live in seconds. None means no expiry.
    """

    def __init__(self, db_path: str, ttl_seconds: float | None = None):
        self.db_path = str(Path(db_path).expanduser())
        self.ttl_seconds = ttl_seconds
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    dtype TEXT NOT NULL,
                    shape TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """)
            conn.commit()

    def get(self, key: str) -> np.ndarray | None:
        """Retrieve embedding by key, respecting TTL."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data, dtype, shape, created_at FROM embeddings WHERE key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None

        data_bytes, dtype, shape_str, created_at = row
        if self.ttl_seconds is not None and time.time() - created_at > self.ttl_seconds:
            self._delete(key)
            return None

        shape = tuple(int(x) for x in shape_str.split(","))
        arr = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
        return arr

    def put(self, key: str, embedding: np.ndarray) -> None:
        """Store embedding by key."""
        data_bytes = embedding.tobytes()
        dtype = str(embedding.dtype)
        shape = ",".join(str(x) for x in embedding.shape)
        created_at = time.time()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (key, data, dtype, shape, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, data_bytes, dtype, shape, created_at),
            )
            conn.commit()

    def _delete(self, key: str) -> None:
        """Delete a cached entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM embeddings WHERE key = ?", (key,))
            conn.commit()
