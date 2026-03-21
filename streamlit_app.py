from __future__ import annotations

import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from enterprise_rag.mock_catalog import load_mock_catalog
from enterprise_rag.models import CatalogItem
from enterprise_rag.search_engine import HybridSearchEngine
from enterprise_rag.llm_synthesizer import get_synthesizer_from_env, TemplateSynthesizer

load_dotenv()


def get_config_value(key: str, default: str = "") -> str:
    """Resolve config from env vars first, then Streamlit secrets."""
    env_val = os.environ.get(key)
    if env_val:
        return env_val
    try:
        val = st.secrets.get(key, default)
        return str(val) if val is not None else default
    except Exception:
        return default


@st.cache_data(ttl=30, show_spinner=False)
def check_supabase_health(db_url: str) -> tuple[bool, str]:
    """Return (is_healthy, detail) for Supabase/Postgres connectivity."""
    if not db_url:
        return False, "SUPABASE_DB_URL not set"

    try:
        import psycopg2  # noqa: PLC0415

        conn = psycopg2.connect(db_url, connect_timeout=3)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
            return True, "Connected"
        finally:
            conn.close()
    except Exception as exc:
        return False, str(exc)

EMBEDDER_OPTIONS = {
    "Hashing (fast, no download)": "hashing",
    "Sentence-Transformers (semantic, ~80 MB)": "sentence-transformers",
}
VECTOR_STORE_OPTIONS = {
    "In-Memory (numpy)": "inmemory",
    "FAISS (flat IP index)": "faiss",
    "Supabase / pgvector": "supabase",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "enterprise saml sso identity",
    "low latency vector database",
    "edge gpu private inference",
    "api gateway waf traffic control",
    "rag document grounding citations",
    "distributed tracing observability slo",
]

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_catalog() -> list[CatalogItem]:
    return load_mock_catalog()


def get_categories() -> list[str]:
    return sorted({item.category for item in get_catalog()})


@st.cache_resource(show_spinner="Loading backend…")
def get_engine(
    embedder_key: str,
    vector_store_key: str,
    items_hash: int,
) -> HybridSearchEngine:
    """Build and cache an engine per (embedder, vector_store) combo."""
    from enterprise_rag.embedding import HashingEmbedder, SentenceTransformerEmbedder  # noqa: PLC0415
    from enterprise_rag.vector_store import InMemoryVectorStore, FAISSVectorStoreAdapter, SupabaseVectorStoreAdapter  # noqa: PLC0415

    embedder = SentenceTransformerEmbedder() if embedder_key == "sentence-transformers" else HashingEmbedder()
    if vector_store_key == "faiss":
        try:
            store = FAISSVectorStoreAdapter()
        except Exception:
            st.warning("faiss-cpu not installed — falling back to In-Memory store.")
            store = InMemoryVectorStore()
    elif vector_store_key == "supabase":
        db_url = get_config_value("SUPABASE_DB_URL")
        if not db_url:
            st.error("SUPABASE_DB_URL is not set in .env — falling back to In-Memory store.")
            store = InMemoryVectorStore()
        else:
            try:
                store = SupabaseVectorStoreAdapter(db_url=db_url)
            except Exception as exc:
                st.error(f"Supabase connection failed: {exc} — falling back to In-Memory store.")
                store = InMemoryVectorStore()
    else:
        store = InMemoryVectorStore()
    return HybridSearchEngine(items=get_catalog(), embedder=embedder, vector_store=store)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def run_query(
    query: str,
    top_k: int,
    candidate_k: int,
    vector_weight: float,
    keyword_weight: float,
    allowed_categories: list[str] | None,
    sku_filter: str,
    embedder_key: str,
    vector_store_key: str,
):
    catalog = get_catalog()
    filtered: list[CatalogItem] = [
        item
        for item in catalog
        if (not allowed_categories or item.category in allowed_categories)
        and (not sku_filter or sku_filter.upper() in item.sku.upper())
    ]
    if not filtered:
        return [], 0.0

    engine = get_engine(embedder_key, vector_store_key, hash(tuple(i.item_id for i in filtered)))
    engine.vector_weight = vector_weight
    engine.keyword_weight = keyword_weight
    start = time.perf_counter()
    results = engine.search(query, top_k=top_k, candidate_k=candidate_k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return results, elapsed_ms


# ---------------------------------------------------------------------------
# Answer synthesis (delegates to llm_synthesizer module)
# ---------------------------------------------------------------------------


def synthesize_answer(query: str, results, openai_key: str = "") -> str:
    synth = get_synthesizer_from_env() if not openai_key else __import__(
        "enterprise_rag.llm_synthesizer", fromlist=["OpenAISynthesizer"]
    ).OpenAISynthesizer(openai_key)
    try:
        return synth.synthesize(query, results)
    except Exception as exc:
        fallback = TemplateSynthesizer()
        return f"[LLM error: {exc}]\n\n{fallback.synthesize(query, results)}"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Sales Engineer AI Demo", page_icon="🔎", layout="wide")

    # Initialise session state
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []
    if "active_query" not in st.session_state:
        st.session_state["active_query"] = "low latency vector database"

    st.title("Sales Engineer AI Demo")
    st.caption("Hybrid Search (Vector + Keyword) with Re-ranking on mock catalog data")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Search Settings")
        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)
        candidate_k = st.slider("Candidate Pool Size", min_value=5, max_value=30, value=15)

        st.subheader("Hybrid Weights")
        vector_weight = st.slider("Vector Weight", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
        keyword_weight = 1.0 - vector_weight
        st.caption(f"Keyword Weight: {keyword_weight:.2f}")

        st.subheader("Backend")
        embedder_label = st.selectbox("Embedder", list(EMBEDDER_OPTIONS.keys()), index=0)
        embedder_key = EMBEDDER_OPTIONS[embedder_label]
        vector_store_label = st.selectbox("Vector Store", list(VECTOR_STORE_OPTIONS.keys()), index=0)
        vector_store_key = VECTOR_STORE_OPTIONS[vector_store_label]

        st.caption(f"Active: `{embedder_key}` / `{vector_store_key}`")

        if vector_store_key == "supabase":
            ok, detail = check_supabase_health(get_config_value("SUPABASE_DB_URL"))
            if ok:
                st.success("Supabase DB: Connected")
            else:
                st.warning(f"Supabase DB: Unreachable ({detail})")

        has_openai_key = bool(get_config_value("OPENAI_API_KEY").strip())
        if has_openai_key:
            st.success("OpenAI key detected")
        else:
            st.warning("OpenAI key not found")

        st.subheader("Filters")
        all_categories = get_categories()
        selected_categories = st.multiselect(
            "Category",
            options=all_categories,
            default=all_categories,
            help="Narrow search to specific product categories.",
        )
        sku_filter = st.text_input("SKU contains", value="", placeholder="e.g. SEC, AI, DATA")

        st.subheader("Example Queries")
        for example in EXAMPLE_QUERIES:
            if st.button(example, use_container_width=True):
                st.session_state["active_query"] = example
                st.rerun()

        if st.session_state["query_history"]:
            st.subheader("Query History")
            for entry in reversed(st.session_state["query_history"][-10:]):
                label = entry["query"][:32] + ("…" if len(entry["query"]) > 32 else "")
                if st.button(f"↩ {label}", key=f"hist_{entry['ts']}", use_container_width=True):
                    st.session_state["active_query"] = entry["query"]
                    st.rerun()
            if st.button("Clear History", type="secondary", use_container_width=True):
                st.session_state["query_history"] = []
                st.rerun()

    # ---- Main query area ----
    query = st.text_input(
        "Ask about products",
        value=st.session_state["active_query"],
        help="Enter customer requirements or technical constraints.",
    )

    col_run, _ = st.columns([1, 5])
    with col_run:
        run_clicked = st.button("Run Search", type="primary")

    if run_clicked and query.strip():
        st.session_state["active_query"] = query
        history = st.session_state["query_history"]
        if not history or history[-1]["query"] != query:
            history.append({"query": query, "ts": datetime.now().isoformat(timespec="seconds")})

        results, elapsed_ms = run_query(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            allowed_categories=selected_categories if selected_categories != all_categories else None,
            sku_filter=sku_filter.strip(),
            embedder_key=embedder_key,
            vector_store_key=vector_store_key,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieved Results", len(results))
        col2.metric("Retrieval Latency", f"{elapsed_ms:.2f} ms")
        col3.metric("Backend", f"{embedder_key} / {vector_store_key}")

        if not results:
            st.warning("No results found. Try broadening the query or adjusting filters.")
            return

        # ---- AI Answer panel ----
        st.subheader("AI Answer")
        with st.spinner("Generating answer…"):
            answer = synthesize_answer(query, results, get_config_value("OPENAI_API_KEY"))
        with st.container(border=True):
            st.markdown(answer)
            st.caption(
                "Citations: "
                + " · ".join(f"[{i + 1}] {r.item.sku}" for i, r in enumerate(results[:3]))
            )
            if not get_config_value("OPENAI_API_KEY"):
                st.caption("Tip: set `OPENAI_API_KEY` in your environment for GPT-4o-mini answers.")

        # ---- Ranked results ----
        st.subheader("Ranked Results")
        for rank, result in enumerate(results, start=1):
            item = result.item
            with st.container(border=True):
                st.markdown(f"### {rank}. {item.name} `{item.sku}`")
                st.write(item.description)
                col_cat, col_feat = st.columns([1, 3])
                col_cat.caption(f"Category: **{item.category}**")
                col_feat.caption("Features: " + " · ".join(item.features))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final", f"{result.score:.3f}")
                c2.metric("Vector", f"{result.component_scores['vector']:.3f}")
                c3.metric("Keyword", f"{result.component_scores['keyword']:.3f}")
                c4.metric("Re-rank", f"{result.component_scores['rerank']:.3f}")
                st.progress(min(max(result.score, 0.0), 1.0), text="Final score")

        # ---- Debug table ----
        with st.expander("Debug: Score Breakdown"):
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

