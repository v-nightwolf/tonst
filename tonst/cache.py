"""
cache.py
--------
Local semantic cache: if a new query is close enough in meaning to one
we've already paid the cloud model to answer, skip the API call entirely.

For this proof-of-concept, similarity is computed with a simple bag-of-words
cosine similarity (numpy only, no downloads). In a production build you'd
swap this for embeddings from a small local model pulled via Ollama
(e.g. `nomic-embed-text`) -- the cache interface below doesn't change either
way, only how `_embed()` is implemented.
"""

from __future__ import annotations
import re
import time
import numpy as np
from dataclasses import dataclass, field


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class CacheEntry:
    query: str
    response: str
    vocab: dict
    vector: np.ndarray
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.90, max_entries: int = 5000):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: list[CacheEntry] = []

    def _embed(self, text: str) -> tuple[dict, np.ndarray]:
        tokens = _tokenize(text)
        vocab: dict[str, int] = {}
        for t in tokens:
            vocab[t] = vocab.get(t, 0) + 1
        vec = np.array(list(vocab.values()), dtype=float)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vocab, vec

    @staticmethod
    def _cosine(vocab_a: dict, vec_a: np.ndarray, vocab_b: dict, vec_b: np.ndarray) -> float:
        shared = set(vocab_a) & set(vocab_b)
        if not shared:
            return 0.0
        # Project both vectors onto the shared-vocab dimensions for a fair comparison
        a_vals = np.array([vocab_a[k] for k in shared], dtype=float)
        b_vals = np.array([vocab_b[k] for k in shared], dtype=float)
        a_vals /= (np.linalg.norm(list(vocab_a.values())) or 1)
        b_vals /= (np.linalg.norm(list(vocab_b.values())) or 1)
        return float(np.dot(a_vals, b_vals))

    def lookup(self, query: str) -> str | None:
        if not self._entries:
            return None
        vocab, vec = self._embed(query)
        best_score, best_entry = 0.0, None
        for entry in self._entries:
            score = self._cosine(vocab, vec, entry.vocab, entry.vector)
            if score > best_score:
                best_score, best_entry = score, entry
        if best_entry and best_score >= self.similarity_threshold:
            best_entry.hits += 1
            return best_entry.response
        return None

    def store(self, query: str, response: str) -> None:
        vocab, vec = self._embed(query)
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)  # simple FIFO eviction for the POC
        self._entries.append(CacheEntry(query=query, response=response, vocab=vocab, vector=vec))

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "total_hits": sum(e.hits for e in self._entries),
        }
