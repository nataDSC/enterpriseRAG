
import argparse

from enterprise_rag.mock_catalog import load_mock_catalog
from enterprise_rag.search_engine import HybridSearchEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sales Engineer AI hybrid-search demo")
    parser.add_argument("query", nargs="?", default="sso saml enterprise identity")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    catalog = load_mock_catalog()
    engine = HybridSearchEngine(catalog)

    print(f"Query: {args.query}")
    results = engine.search(args.query, top_k=args.top_k)
    for rank, result in enumerate(results, start=1):
        item = result.item
        print(
            f"{rank}. {item.name} ({item.sku}) | score={result.score:.3f} | "
            f"vector={result.component_scores['vector']:.3f} "
            f"keyword={result.component_scores['keyword']:.3f} "
            f"rerank={result.component_scores['rerank']:.3f}"
        )
        print(f"   Category: {item.category}")
        print(f"   {item.description}")


if __name__ == "__main__":
    main()
