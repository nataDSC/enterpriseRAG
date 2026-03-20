from enterprise_rag.mock_catalog import load_mock_catalog
from enterprise_rag.search_engine import HybridSearchEngine


def test_hybrid_search_returns_sso_for_identity_query():
    engine = HybridSearchEngine(load_mock_catalog())
    results = engine.search("enterprise saml sso identity", top_k=3)

    assert results
    assert results[0].item.sku == "SEC-IAM-001"


def test_hybrid_search_returns_vector_db_for_latency_query():
    engine = HybridSearchEngine(load_mock_catalog())
    results = engine.search("low latency vector database", top_k=3)

    assert results
    top_skus = [result.item.sku for result in results]
    assert "DATA-DB-011" in top_skus
