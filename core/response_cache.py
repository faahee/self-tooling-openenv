"""Response Cache — LRU + semantic cache for fast repeated queries."""
from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ResponseCache:
    """Two-level cache: exact string match (LRU) and semantic similarity.

    Level 1 — exact match in O(1).
    Level 2 — cosine similarity > threshold returns cached response.
    """

    def __init__(self, config: dict, embedder=None) -> None:
        """Initialize the response cache.

        Args:
            config: Full JARVIS configuration dictionary.
            embedder: Optional shared SentenceTransformer instance.
        """
        self.max_entries: int = config["memory"]["max_cache_entries"]
        self.similarity_threshold: float = config["memory"]["cache_similarity_threshold"]
        self.exact_cache: OrderedDict[str, str] = OrderedDict()
        self.semantic_cache: list[tuple[np.ndarray, str, str]] = []  # (embedding, query, response)
        if embedder is not None:
            self.embedder = embedder
        else:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def get(self, query: str) -> str | None:
        """Look up a cached response.

        Args:
            query: The user's input text.

        Returns:
            Cached response string, or None if no hit.
        """
        # Level 1: exact match
        if query in self.exact_cache:
            self.exact_cache.move_to_end(query)
            logger.debug("Cache hit (exact): %s", query[:60])
            return self.exact_cache[query]

        # Level 2: semantic match
        if not self.semantic_cache:
            return None

        query_emb = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        best_score = -1.0
        best_response: str | None = None

        for emb, _cached_query, response in self.semantic_cache:
            score = float(np.dot(query_emb, emb))
            if score > best_score:
                best_score = score
                best_response = response

        if best_score >= self.similarity_threshold:
            logger.debug("Cache hit (semantic, score=%.3f): %s", best_score, query[:60])
            return best_response

        return None

    def set(self, query: str, response: str) -> None:
        """Store a response in both cache levels.

        Args:
            query: The user's input text.
            response: The generated response.
        """
        # Exact cache
        self.exact_cache[query] = response
        if len(self.exact_cache) > self.max_entries:
            self.exact_cache.popitem(last=False)

        # Semantic cache
        embedding = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        self.semantic_cache.append((embedding, query, response))
        if len(self.semantic_cache) > self.max_entries:
            self.semantic_cache.pop(0)

    def invalidate_system_state_cache(self) -> None:
        """Remove entries likely to be stale (system query responses)."""
        system_keywords = {"ram", "cpu", "gpu", "memory", "disk", "process", "running", "temperature", "vram"}
        # Purge exact cache
        keys_to_remove = [
            k for k in self.exact_cache
            if any(kw in k.lower() for kw in system_keywords)
        ]
        for k in keys_to_remove:
            del self.exact_cache[k]

        # Purge semantic cache
        self.semantic_cache = [
            (emb, q, r) for emb, q, r in self.semantic_cache
            if not any(kw in q.lower() for kw in system_keywords)
        ]
