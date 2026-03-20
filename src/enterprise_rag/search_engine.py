from __future__ import annotations

from enterprise_rag.embedding import EmbedderProtocol, HashingEmbedder
from enterprise_rag.keyword_index import KeywordIndex
from enterprise_rag.models import CatalogItem, SearchResult
from enterprise_rag.reranker import LightweightReranker
from enterprise_rag.vector_store import InMemoryVectorStore, VectorStoreProtocol


class HybridSearchEngine:
    def __init__(
        self,
        items: list[CatalogItem],
        vector_weight: float = 0.55,
        keyword_weight: float = 0.45,
        embedder: EmbedderProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
    ) -> None:
        self.items = items
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.embedder = embedder if embedder is not None else HashingEmbedder()
        self.vector_store = vector_store if vector_store is not None else InMemoryVectorStore()
        self.keyword_index = KeywordIndex()
        self.reranker = LightweightReranker()

        vectors = self.embedder.embed_batch([item.to_index_text() for item in items])
        self.vector_store.build(items, vectors)
        self.keyword_index.build(items)

    def _normalize_hits(self, hits: list[tuple[CatalogItem, float]]) -> dict[str, float]:
        if not hits:
            return {}
        max_score = max(score for _, score in hits)
        if max_score <= 0:
            max_score = 1.0
        return {item.item_id: score / max_score for item, score in hits}

    def search(self, query: str, top_k: int = 5, candidate_k: int = 15) -> list[SearchResult]:
        query_vector = self.embedder.embed(query)
        vector_hits = self.vector_store.search(query_vector, top_k=candidate_k)
        keyword_hits = self.keyword_index.search(query, top_k=candidate_k)

        norm_vector = self._normalize_hits(vector_hits)
        norm_keyword = self._normalize_hits(keyword_hits)

        item_lookup = {item.item_id: item for item in self.items}
        all_item_ids = set(norm_vector) | set(norm_keyword)

        merged: list[SearchResult] = []
        for item_id in all_item_ids:
            vector_score = norm_vector.get(item_id, 0.0)
            keyword_score = norm_keyword.get(item_id, 0.0)
            hybrid_score = self.vector_weight * vector_score + self.keyword_weight * keyword_score

            item = item_lookup[item_id]
            rerank_score = self.reranker.score(query, item)
            final_score = 0.7 * hybrid_score + 0.3 * rerank_score

            merged.append(
                SearchResult(
                    item=item,
                    score=final_score,
                    component_scores={
                        "vector": vector_score,
                        "keyword": keyword_score,
                        "rerank": rerank_score,
                        "hybrid": hybrid_score,
                    },
                )
            )

        merged.sort(key=lambda result: result.score, reverse=True)
        return merged[:top_k]
