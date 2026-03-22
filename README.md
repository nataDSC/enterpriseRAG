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

## Deploy Smoke Test Checklist

Use this quick checklist after each deploy or reboot:

1. App boot
   - App loads without red error banner.
   - Deployment Readiness panel shows Supabase reachable.
2. Retrieval path
   - Select Supabase / pgvector in the sidebar.
   - Run query: "enterprise saml sso identity".
   - Confirm at least one result card is returned.
3. Synthesis path
   - Confirm a synthesized answer is shown for the query.
   - If OpenAI key is missing, verify Template mode still returns an answer.
4. Telemetry path
   - Open Integrations expander and click Send test telemetry.
   - Confirm success toast includes at least one external backend.
5. Access mode
   - Verify APP_ENV is production in app settings.
   - Confirm runtime uses restricted DB role URL (enterprise_rag_app.<project-ref>).
6. Regression sanity
   - Try one no-results query (for example: "quantum toaster") and confirm graceful handling.
   - Confirm latency metric and result score breakdown render correctly.

## Supabase Production Setup

### Why use two DB roles?

This app has two distinct database phases:

1. Bootstrap/setup: create or refresh the `enterprise_rag_catalog` table and load catalog data.
2. Runtime search: query the already-loaded table from the deployed Streamlit app.

For production, use:

1. An admin DB URL for one-time bootstrap tasks.
2. A restricted runtime DB URL for the deployed app.

### Step 1: Create a restricted Postgres role in Supabase SQL Editor

Open Supabase Dashboard → SQL Editor and run:

```sql
create role enterprise_rag_app
login
password 'replace-with-a-strong-generated-password';

grant usage on schema public to enterprise_rag_app;
grant select on table public.enterprise_rag_catalog to enterprise_rag_app;
```

If `enterprise_rag_catalog` does not exist yet, create the role first, then do bootstrap in Step 3, then re-run the `grant select ...` statement.

### Step 2: Understand the two connection strings

Admin/bootstrap connection string:

```text
postgresql://postgres.<project-ref>:<admin-password>@db.<project-ref>.supabase.co:6543/postgres?sslmode=require
```

Restricted runtime connection string:

```text
postgresql://enterprise_rag_app:<app-password>@db.<project-ref>.supabase.co:6543/postgres?sslmode=require
```

After you create the restricted role, your deployed app should use the second one as `SUPABASE_DB_URL`.

### Step 3: Bootstrap the table with the admin URL

Set your local environment:

```bash
export SUPABASE_ADMIN_DB_URL='postgresql://postgres.<project-ref>:<admin-password>@db.<project-ref>.supabase.co:6543/postgres?sslmode=require'
```

Then run:

```bash
./.venv/bin/python bootstrap_supabase.py
```

This script:

1. Creates/updates the pgvector table if needed.
2. Loads the current mock catalog into Supabase.
3. Builds the vector index when the catalog is large enough.

### Step 4: Configure the deployed app to use the restricted role

In Streamlit Community Cloud secrets, set:

```toml
SUPABASE_DB_URL = "postgresql://enterprise_rag_app:<app-password>@db.<project-ref>.supabase.co:6543/postgres?sslmode=require"
APP_ENV = "production"
```

In production mode, the app will:

1. Use the existing `enterprise_rag_catalog` table.
2. Skip schema-changing operations.
3. Run searches with the restricted runtime DB role.
