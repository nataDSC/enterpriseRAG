import numpy as np

from enterprise_rag.models import CatalogItem


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
