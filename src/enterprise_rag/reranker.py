from enterprise_rag.models import CatalogItem
from enterprise_rag.text_utils import tokenize


class LightweightReranker:
    """Lexical overlap re-ranker to simulate cross-encoder style refinement."""

    def score(self, query: str, item: CatalogItem) -> float:
        query_tokens = set(tokenize(query))
        item_tokens = set(tokenize(item.to_index_text()))
        if not query_tokens:
            return 0.0

        overlap = len(query_tokens & item_tokens) / len(query_tokens)
        phrase_boost = 0.15 if query.lower() in item.to_index_text().lower() else 0.0
        category_boost = 0.10 if item.category.lower() in query.lower() else 0.0
        return overlap + phrase_boost + category_boost
