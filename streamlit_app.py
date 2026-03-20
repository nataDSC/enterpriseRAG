from __future__ import annotations

import time

import streamlit as st

from enterprise_rag.mock_catalog import load_mock_catalog
from enterprise_rag.search_engine import HybridSearchEngine


@st.cache_resource
def get_catalog() -> list:
    return load_mock_catalog()


def run_query(
    query: str,
    top_k: int,
    candidate_k: int,
    vector_weight: float,
    keyword_weight: float,
):
    engine = HybridSearchEngine(
        items=get_catalog(),
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )

    start = time.perf_counter()
    results = engine.search(query, top_k=top_k, candidate_k=candidate_k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return results, elapsed_ms


def main() -> None:
    st.set_page_config(page_title="Sales Engineer AI Demo", page_icon="🔎", layout="wide")

    st.title("Sales Engineer AI Demo")
    st.caption("Hybrid Search (Vector + Keyword) with Re-ranking on mock catalog data")

    with st.sidebar:
        st.header("Search Settings")
        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)
        candidate_k = st.slider("Candidate Pool Size", min_value=5, max_value=30, value=15)

        st.subheader("Hybrid Weights")
        vector_weight = st.slider("Vector Weight", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
        keyword_weight = 1.0 - vector_weight
        st.write(f"Keyword Weight: {keyword_weight:.2f}")

        st.subheader("Try Examples")
        st.code("enterprise saml sso identity")
        st.code("low latency vector database")
        st.code("edge gpu private inference")

    query = st.text_input(
        "Ask about products",
        value="low latency vector database",
        help="Enter customer requirements or technical constraints.",
    )

    run_clicked = st.button("Run Search", type="primary")

    if run_clicked and query.strip():
        results, elapsed_ms = run_query(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Retrieved Results", value=len(results))
        with col2:
            st.metric("Retrieval Latency", value=f"{elapsed_ms:.2f} ms")

        if not results:
            st.warning("No results found. Try broadening the query.")
            return

        st.subheader("Ranked Results")
        for rank, result in enumerate(results, start=1):
            item = result.item
            with st.container(border=True):
                st.markdown(f"### {rank}. {item.name} ({item.sku})")
                st.write(item.description)
                st.caption(f"Category: {item.category}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final", f"{result.score:.3f}")
                c2.metric("Vector", f"{result.component_scores['vector']:.3f}")
                c3.metric("Keyword", f"{result.component_scores['keyword']:.3f}")
                c4.metric("Re-rank", f"{result.component_scores['rerank']:.3f}")

                st.progress(min(max(result.score, 0.0), 1.0), text="Final score")

        st.subheader("Debug View")
        st.dataframe(
            [
                {
                    "rank": idx + 1,
                    "sku": r.item.sku,
                    "name": r.item.name,
                    "final": round(r.score, 4),
                    "vector": round(r.component_scores["vector"], 4),
                    "keyword": round(r.component_scores["keyword"], 4),
                    "rerank": round(r.component_scores["rerank"], 4),
                    "hybrid": round(r.component_scores["hybrid"], 4),
                }
                for idx, r in enumerate(results)
            ],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
