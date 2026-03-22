from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from enterprise_rag.embedding import HashingEmbedder
from enterprise_rag.mock_catalog import load_mock_catalog
from enterprise_rag.vector_store import SupabaseVectorStoreAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Supabase pgvector table with catalog data.")
    parser.add_argument(
        "--db-url-env",
        default="SUPABASE_ADMIN_DB_URL",
        help="Environment variable containing admin DB URL (default: SUPABASE_ADMIN_DB_URL)",
    )
    args = parser.parse_args()

    db_url = os.environ.get(args.db_url_env) or os.environ.get("SUPABASE_DB_URL", "")
    if not db_url:
        raise SystemExit(
            f"No DB URL found. Set {args.db_url_env} or SUPABASE_DB_URL before bootstrapping."
        )

    catalog = load_mock_catalog()
    embedder = HashingEmbedder()
    vectors = embedder.embed_batch([item.to_index_text() for item in catalog])

    store = SupabaseVectorStoreAdapter(db_url=db_url, allow_writes=True)
    store.bootstrap(catalog, vectors)
    print(f"Bootstrapped {len(catalog)} items into Supabase table '{store.TABLE}'.")


if __name__ == "__main__":
    main()
