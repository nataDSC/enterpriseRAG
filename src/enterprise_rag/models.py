from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogItem:
    item_id: str
    sku: str
    name: str
    category: str
    description: str
    features: list[str]

    def to_index_text(self) -> str:
        feature_text = " ".join(self.features)
        return f"{self.name} {self.category} {self.description} {feature_text}"


@dataclass(frozen=True)
class SearchResult:
    item: CatalogItem
    score: float
    component_scores: dict[str, float]
