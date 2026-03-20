from collections import Counter
from math import log

from enterprise_rag.models import CatalogItem
from enterprise_rag.text_utils import tokenize


class KeywordIndex:
    def __init__(self) -> None:
        self.items: list[CatalogItem] = []
        self.tf_by_id: dict[str, Counter[str]] = {}
        self.doc_len: dict[str, int] = {}
        self.df: Counter[str] = Counter()
        self.avg_doc_len = 0.0

    def build(self, items: list[CatalogItem]) -> None:
        self.items = items
        self.tf_by_id.clear()
        self.doc_len.clear()
        self.df = Counter()

        total_len = 0
        for item in items:
            tokens = tokenize(item.to_index_text())
            tf = Counter(tokens)
            self.tf_by_id[item.item_id] = tf
            self.doc_len[item.item_id] = len(tokens)
            total_len += len(tokens)
            self.df.update(tf.keys())

        self.avg_doc_len = (total_len / len(items)) if items else 0.0

    def _idf(self, term: str) -> float:
        n_docs = len(self.items)
        return log((n_docs + 1) / (self.df.get(term, 0) + 1)) + 1.0

    def search(self, query: str, top_k: int, k1: float = 1.5, b: float = 0.75) -> list[tuple[CatalogItem, float]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: list[tuple[CatalogItem, float]] = []
        for item in self.items:
            tf = self.tf_by_id[item.item_id]
            doc_length = self.doc_len[item.item_id]
            score = 0.0
            for term in query_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term)
                denom = f + k1 * (1.0 - b + b * doc_length / max(self.avg_doc_len, 1.0))
                score += idf * ((f * (k1 + 1.0)) / denom)
            if score > 0:
                scores.append((item, score))

        scores.sort(key=lambda hit: hit[1], reverse=True)
        return scores[:top_k]
