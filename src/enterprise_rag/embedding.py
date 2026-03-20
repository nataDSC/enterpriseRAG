from __future__ import annotations

import functools
from typing import Protocol

import numpy as np

from enterprise_rag.text_utils import tokenize


class EmbedderProtocol(Protocol):
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...


class HashingEmbedder:
    """Fast, deterministic embedding for demo use (no model downloads)."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            idx = hash(token) % self.dimension
            sign = 1.0 if (hash(token + "_sign") % 2 == 0) else -1.0
            vector[idx] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


@functools.lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        return SentenceTransformer(model_name)
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install it with: pip install sentence-transformers"
        ) from exc


class SentenceTransformerEmbedder:
    """Semantic embedder backed by a sentence-transformers model.

    Defaults to all-MiniLM-L6-v2 (384-dim, fast on CPU).
    The underlying model is process-cached across instances.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    @property
    def _model(self):
        return _load_sentence_transformer(self.model_name)

    def embed(self, text: str) -> np.ndarray:
        vector = self._model.encode(text, normalize_embeddings=True)
        return np.array(vector, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=64)
        return [np.array(v, dtype=np.float32) for v in vectors]
