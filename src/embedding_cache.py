"""
Embedding cache utility for semantic similarity computations.

Uses OpenAI text-embedding-3-small with persistent caching to avoid redundant API calls.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for OpenAI embeddings with persistent storage."""

    def __init__(
        self,
        cache_dir: Path = Path(".cache/embeddings"),
        model: str = "text-embedding-3-small",
    ):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings
            model: OpenAI embedding model name
        """
        self.cache_dir = cache_dir
        self.model = model
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI()
        self._memory_cache: dict[str, np.ndarray] = {}

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        # Hash text + model name for unique key
        content = f"{self.model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached embedding."""
        return self.cache_dir / f"{cache_key}.npy"

    def _load_from_disk(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from disk cache."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                embedding = np.load(cache_path)
                logger.debug(f"Loaded embedding from disk cache: {cache_key[:8]}...")
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
                return None
        return None

    def _save_to_disk(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to disk cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            np.save(cache_path, embedding)
            logger.debug(f"Saved embedding to disk cache: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text, using cache if available.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        cache_key = self._get_cache_key(text)

        # Check memory cache first
        if cache_key in self._memory_cache:
            logger.debug(f"Loaded embedding from memory cache: {text[:50]}...")
            return self._memory_cache[cache_key]

        # Check disk cache
        embedding = self._load_from_disk(cache_key)
        if embedding is not None:
            self._memory_cache[cache_key] = embedding
            return embedding

        # Call API
        logger.debug(f"Calling embedding API for: {text[:50]}...")
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Cache it
            self._memory_cache[cache_key] = embedding
            self._save_to_disk(cache_key, embedding)

            return embedding

        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    def get_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Get embeddings for multiple texts, batching API calls efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Check which texts need embedding
        embeddings = []
        texts_to_fetch = []
        fetch_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)

            # Check memory cache
            if cache_key in self._memory_cache:
                embeddings.append(self._memory_cache[cache_key])
                continue

            # Check disk cache
            cached = self._load_from_disk(cache_key)
            if cached is not None:
                self._memory_cache[cache_key] = cached
                embeddings.append(cached)
                continue

            # Need to fetch
            texts_to_fetch.append(text)
            fetch_indices.append(i)
            embeddings.append(None)  # Placeholder

        # Batch API call for texts that need fetching
        if texts_to_fetch:
            logger.info(f"Fetching {len(texts_to_fetch)} embeddings from API...")
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts_to_fetch,
                )

                # Store fetched embeddings
                for text, emb_data, idx in zip(texts_to_fetch, response.data, fetch_indices):
                    embedding = np.array(emb_data.embedding, dtype=np.float32)
                    cache_key = self._get_cache_key(text)

                    # Cache it
                    self._memory_cache[cache_key] = embedding
                    self._save_to_disk(cache_key, embedding)

                    # Update result list
                    embeddings[idx] = embedding

            except Exception as e:
                logger.error(f"Failed to get batch embeddings: {e}")
                raise

        return embeddings

    def clear_memory_cache(self):
        """Clear in-memory cache (disk cache persists)."""
        self._memory_cache.clear()
        logger.info("Cleared embedding memory cache")

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        disk_cache_files = list(self.cache_dir.glob("*.npy"))
        return {
            "memory_cache_size": len(self._memory_cache),
            "disk_cache_size": len(disk_cache_files),
            "cache_dir": str(self.cache_dir),
            "model": self.model,
        }


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        emb1: First embedding vector
        emb2: Second embedding vector

    Returns:
        Cosine similarity (0 to 1)
    """
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    return float(dot_product / (norm1 * norm2))


def batch_cosine_similarity(emb: np.ndarray, emb_list: list[np.ndarray]) -> np.ndarray:
    """
    Compute cosine similarity between one embedding and a list of embeddings.

    Args:
        emb: Query embedding
        emb_list: List of embeddings to compare against

    Returns:
        Array of cosine similarities
    """
    # Stack embeddings into matrix
    emb_matrix = np.stack(emb_list)

    # Compute similarities
    dot_products = np.dot(emb_matrix, emb)
    norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(emb)

    return dot_products / norms
