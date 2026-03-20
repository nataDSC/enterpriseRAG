# enterpriseRAG

Demo "Sales Engineer" AI for complex catalog retrieval.

This MVP uses:

- Hybrid retrieval: vector similarity + keyword BM25-like scoring
- Re-ranking: lightweight lexical overlap layer
- Low-latency vector search: in-memory matrix ops with top-k partitioning
- Mock catalog data so the full flow works end-to-end before a real database

## Quick start

1. Activate virtual environment:
   - macOS/Linux: `source .venv/bin/activate`
2. Install project in editable mode (with dev deps):
   - `pip install -e '.[dev]'`
3. Run the demo query:
   - `python -m enterprise_rag.main`
4. Run your own query:
   - `python -m enterprise_rag.main "edge gpu low latency inference" --top-k 3`
5. Launch the Streamlit dashboard:
   - `streamlit run streamlit_app.py`
6. Run tests:
   - `pytest`

## Dashboard

The Streamlit app provides:

- Clickable UI for search and ranking output
- Real-time latency metric for retrieval performance checks
- Tunable hybrid weights (`vector` vs `keyword`) and candidate pool size
- Per-result score breakdown (`final`, `vector`, `keyword`, `rerank`)

## Project structure

- `src/enterprise_rag/search_engine.py` hybrid pipeline orchestration
- `src/enterprise_rag/vector_store.py` low-latency in-memory vector search
- `src/enterprise_rag/keyword_index.py` lexical keyword scoring
- `src/enterprise_rag/reranker.py` re-ranking layer
- `src/enterprise_rag/mock_catalog.py` mock product catalog data
