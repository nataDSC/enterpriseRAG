from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from enterprise_rag.mock_catalog import load_mock_catalog
from enterprise_rag.telemetry import capture_exception, get_tracer, init_all, langfuse_log_event
from enterprise_rag.models import CatalogItem
from enterprise_rag.search_engine import HybridSearchEngine
from enterprise_rag.llm_synthesizer import get_synthesizer_from_env, TemplateSynthesizer

load_dotenv()

logger = logging.getLogger("enterprise_rag.observability")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def obs_enabled() -> bool:
    return get_config_value("OBSERVABILITY_ENABLED", "true").lower() == "true"


def log_event(event: str, **fields) -> None:
    if not obs_enabled():
        return
    payload = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event": event,
        **fields,
    }
    logger.info(json.dumps(payload, ensure_ascii=True))
    events = st.session_state.get("obs_events", [])
    events.append(payload)
    st.session_state["obs_events"] = events[-50:]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((q / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def init_observability_state() -> None:
    st.session_state.setdefault("obs_events", [])
    st.session_state.setdefault("obs_queries_total", 0)
    st.session_state.setdefault("obs_zero_results", 0)
    st.session_state.setdefault("obs_search_errors", 0)
    st.session_state.setdefault("obs_llm_errors", 0)
    st.session_state.setdefault("obs_latencies_ms", [])


def send_test_telemetry() -> tuple[bool, str]:
    """Emit a synthetic event/span/message to configured telemetry backends."""
    test_id = f"test-{str(uuid4())[:8]}"
    sent = []

    # Local structured event log (always available if observability is enabled)
    log_event("telemetry_test", request_id=test_id, source="manual_button")
    sent.append("local")

    # OpenTelemetry test span
    try:
        tracer = get_tracer()
        with tracer.start_as_current_span("telemetry_test", attributes={"request_id": test_id}) as span:
            span.set_attribute("test", True)
        sent.append("otel")
    except Exception as exc:
        capture_exception(exc)

    # Sentry test message
    try:
        import sentry_sdk  # noqa: PLC0415

        sentry_sdk.capture_message(f"Telemetry test event ({test_id})", level="info")
        sent.append("sentry")
    except Exception:
        pass

    # Langfuse test event
    try:
        if langfuse_log_event(
            "telemetry_test",
            input={"request_id": test_id, "action": "test_button"},
            output={"status": "ok"},
            metadata={"source": "manual_button"},
        ):
            sent.append("langfuse")
    except Exception as exc:
        capture_exception(exc)

    if len(sent) <= 1:
        return False, f"No external backend accepted the test. id={test_id}"
    return True, f"Sent test event to: {', '.join(sent)} (id={test_id})"


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


def render_deployment_readiness(tel: dict[str, bool]) -> None:
    """Render deployment checks to speed up production verification."""
    app_env = get_config_value("APP_ENV", "development")
    openai_ok = bool(get_config_value("OPENAI_API_KEY").strip())
    db_url = get_config_value("SUPABASE_DB_URL")
    supa_ok, supa_detail = check_supabase_health(db_url)
    obs_ok = obs_enabled()
    telemetry_ok = any(tel.values())

    with st.expander("Deployment Readiness", expanded=False):
        if app_env == "production":
            st.success("APP_ENV=production")
        else:
            st.warning(f"APP_ENV={app_env} (set to production for deploy)")

        if openai_ok:
            st.success("OPENAI_API_KEY present")
        else:
            st.warning("OPENAI_API_KEY missing")

        if db_url:
            st.success("SUPABASE_DB_URL present")
        else:
            st.warning("SUPABASE_DB_URL missing")

        if supa_ok:
            st.success("Supabase reachable")
        else:
            st.warning(f"Supabase unreachable ({supa_detail})")

        if obs_ok:
            st.success("Observability enabled")
        else:
            st.warning("Observability disabled")

        if telemetry_ok:
            active = ", ".join(name for name, is_on in tel.items() if is_on)
            st.success(f"Telemetry active: {active}")
        else:
            st.warning("No telemetry backend active")

        deploy_ready = app_env == "production" and openai_ok and db_url and supa_ok
        if deploy_ready:
            st.success("Ready to deploy")
        else:
            st.info("Not deployment-ready yet. Resolve warnings above.")

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
        app_env = get_config_value("APP_ENV", "development")
        if not db_url:
            st.error("SUPABASE_DB_URL is not set in .env — falling back to In-Memory store.")
            store = InMemoryVectorStore()
        else:
            try:
                store = SupabaseVectorStoreAdapter(
                    db_url=db_url,
                    allow_writes=app_env != "production",
                )
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


def synthesize_answer(query: str, results, openai_key: str = "") -> tuple[str, bool, str]:
    synth = get_synthesizer_from_env() if not openai_key else __import__(
        "enterprise_rag.llm_synthesizer", fromlist=["OpenAISynthesizer"]
    ).OpenAISynthesizer(openai_key)
    try:
        return synth.synthesize(query, results), False, ""
    except Exception as exc:
        fallback = TemplateSynthesizer()
        return f"[LLM error: {exc}]\n\n{fallback.synthesize(query, results)}", True, str(exc)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


_TELEMETRY_ENV_KEYS = (
    "SENTRY_DSN",
    "SENTRY_TRACES_SAMPLE_RATE",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_SERVICE_NAME",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
)


@st.cache_resource(show_spinner=False)
def _init_telemetry() -> dict[str, bool]:
    # On Streamlit Cloud, secrets live in st.secrets but may not be in os.environ.
    # Inject any missing telemetry keys so telemetry.py can read them normally.
    try:
        for key in _TELEMETRY_ENV_KEYS:
            if key not in os.environ:
                val = st.secrets.get(key)
                if val:
                    os.environ[key] = str(val)
    except Exception:
        pass
    return init_all()


SIDEBAR_WIDTH = 420


def main() -> None:
    st.set_page_config(page_title="Sales Engineer AI Demo", page_icon="🔎", layout="wide")
    init_observability_state()

    # Wider sidebar
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            min-width: {SIDEBAR_WIDTH}px;
            max-width: {SIDEBAR_WIDTH}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    tel = _init_telemetry()

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

        # Telemetry status
        with st.expander("Integrations", expanded=False):
            labels = {"sentry": "Sentry", "otel": "OpenTelemetry", "langfuse": "Langfuse"}
            for key, label in labels.items():
                if tel.get(key):
                    st.success(f"{label}: active")
                else:
                    st.caption(f"{label}: not configured")

            if st.button("Send test telemetry", use_container_width=True):
                ok, msg = send_test_telemetry()
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)

        render_deployment_readiness(tel)

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

        with st.expander("Observability", expanded=False):
            latencies = st.session_state.get("obs_latencies_ms", [])
            total = st.session_state.get("obs_queries_total", 0)
            zero = st.session_state.get("obs_zero_results", 0)
            search_errors = st.session_state.get("obs_search_errors", 0)
            llm_errors = st.session_state.get("obs_llm_errors", 0)

            c1, c2 = st.columns(2)
            c1.metric("Queries", total)
            c2.metric("Zero results", zero)

            c3, c4 = st.columns(2)
            c3.metric("Search errors", search_errors)
            c4.metric("LLM fallback/errors", llm_errors)

            if latencies:
                c5, c6, c7 = st.columns(3)
                c5.metric("Latency p50", f"{percentile(latencies, 50):.1f} ms")
                c6.metric("Latency p95", f"{percentile(latencies, 95):.1f} ms")
                c7.metric("Latency max", f"{max(latencies):.1f} ms")

            if st.button("Clear observability stats", use_container_width=True):
                for key, default in (
                    ("obs_events", []),
                    ("obs_queries_total", 0),
                    ("obs_zero_results", 0),
                    ("obs_search_errors", 0),
                    ("obs_llm_errors", 0),
                    ("obs_latencies_ms", []),
                ):
                    st.session_state[key] = default
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
        request_id = str(uuid4())[:8]
        st.session_state["active_query"] = query
        history = st.session_state["query_history"]
        if not history or history[-1]["query"] != query:
            history.append({"query": query, "ts": datetime.now().isoformat(timespec="seconds")})

        st.session_state["obs_queries_total"] += 1
        log_event(
            "query_started",
            request_id=request_id,
            query=query,
            embedder=embedder_key,
            vector_store=vector_store_key,
        )

        tracer = get_tracer()
        with tracer.start_as_current_span(
            "hybrid_search",
            attributes={"query": query, "embedder": embedder_key, "vector_store": vector_store_key},
        ) as span:
            try:
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
                span.set_attribute("results_count", len(results))
                span.set_attribute("latency_ms", round(elapsed_ms, 2))
            except Exception as exc:
                span.record_exception(exc)
                st.session_state["obs_search_errors"] += 1
                capture_exception(exc)
                log_event("query_failed", request_id=request_id, error=str(exc))
                st.error(f"Search failed: {exc}")
                return

        st.session_state["obs_latencies_ms"].append(elapsed_ms)
        st.session_state["obs_latencies_ms"] = st.session_state["obs_latencies_ms"][-200:]
        log_event(
            "query_finished",
            request_id=request_id,
            latency_ms=round(elapsed_ms, 2),
            results_count=len(results),
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieved Results", len(results))
        col2.metric("Retrieval Latency", f"{elapsed_ms:.2f} ms")
        col3.metric("Backend", f"{embedder_key} / {vector_store_key}")

        if not results:
            st.session_state["obs_zero_results"] += 1
            st.warning("No results found. Try broadening the query or adjusting filters.")
            return

        # ---- AI Answer panel ----
        st.subheader("AI Answer")
        with st.spinner("Generating answer…"):
            answer, had_llm_error, llm_error_detail = synthesize_answer(
                query, results, get_config_value("OPENAI_API_KEY")
            )
            if had_llm_error:
                st.session_state["obs_llm_errors"] += 1
                log_event(
                    "llm_fallback_used",
                    request_id=request_id,
                    error=llm_error_detail,
                )
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

