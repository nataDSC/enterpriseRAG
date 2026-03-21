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
5. Configure local environment variables:
   - `cp .env.example .env`
   - Add your key in `.env`: `OPENAI_API_KEY=your_key_here`
6. Launch the Streamlit dashboard:
   - `streamlit run streamlit_app.py`
7. Run tests:
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

## Deploy to Streamlit Community Cloud

### Checklist

1. Push this project to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select your repo/branch and set the app entry file to `streamlit_app.py`.
4. In app settings, add secrets:

```toml
OPENAI_API_KEY = "your_openai_key"
SUPABASE_DB_URL = "your_supabase_postgres_url"
APP_ENV = "production"
```

5. Deploy and verify the app opens successfully.

### Supabase connection options

You can use either of these for `SUPABASE_DB_URL`:

1. Public/reachable Supabase Postgres URL (recommended for cloud hosting).
2. ngrok-exposed Postgres URL (works for demos if your local database is tunneled).

### If using ngrok for Supabase

Use your ngrok hostname/port in the connection string, for example:

```text
postgresql://postgres:your_password@your-ngrok-hostname:your-ngrok-port/postgres
```

Important notes:

1. Streamlit Community Cloud cannot access `localhost` on your machine, so local Docker URLs will not work directly.
2. Keep the ngrok tunnel running continuously or the app will lose DB connectivity.
3. Free ngrok endpoints may rotate; update `SUPABASE_DB_URL` in Streamlit secrets when that happens.
4. Prefer a stable hosted database for long-term deployments.
