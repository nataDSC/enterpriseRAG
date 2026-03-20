from __future__ import annotations

from typing import Protocol

import numpy as np

from enterprise_rag.models import CatalogItem


class VectorStoreProtocol(Protocol):
    def build(self, items: list[CatalogItem], vectors: list[np.ndarray]) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[CatalogItem, float]]: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.items: list[CatalogItem] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)

    def build(self, items: list[CatalogItem], vectors: list[np.ndarray]) -> None:
        if not items:
            raise ValueError("Cannot build vector store with no items.")
        self.items = items
        self.vectors = np.vstack(vectors).astype(np.float32, copy=False)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[CatalogItem, float]]:
        if self.vectors.size == 0:
            return []

        scores = self.vectors @ query_vector.astype(np.float32, copy=False)
        k = min(top_k, len(self.items))
        if k <= 0:
            return []

        # Argpartition keeps retrieval fast for large candidate sets.
        candidate_idx = np.argpartition(scores, -k)[-k:]
        sorted_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
        return [(self.items[i], float(scores[i])) for i in sorted_idx]


class FAISSVectorStoreAdapter:
    """FAISS-backed vector store using a flat inner-product index.

    Exact cosine search on L2-normalised vectors.
    Requires: pip install faiss-cpu
    """

    def __init__(self) -> None:
        self._index = None
        self.items: list[CatalogItem] = []

    def build(self, items: list[CatalogItem], vectors: list[np.ndarray]) -> None:
        if not items:
            raise ValueError("Cannot build FAISS index with no items.")
        try:
            import faiss  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. Install it with: pip install faiss-cpu"
            ) from exc
        self.items = items
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        dimension = matrix.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(matrix)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[CatalogItem, float]]:
        if self._index is None or not self.items:
            return []
        k = min(top_k, len(self.items))
        q = query_vector.astype(np.float32, copy=False).reshape(1, -1)
        scores, indices = self._index.search(q, k)
        return [
            (self.items[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]
