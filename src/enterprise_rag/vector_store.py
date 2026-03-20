from __future__ import annotations

from typing import Protocol

import numpy as np

from enterprise_rag.models import CatalogItem


class VectorStoreProtocol(Protocol):
    def build(self, items: list[CatalogItem], vectors: list[np.ndarray]) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[CatalogItem, float]]: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.items: list[CatalogItem] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)

    def build(self, items: list[CatalogItem], vectors: list[np.ndarray]) -> None:
        if not items:
            raise ValueError("Cannot build vector store with no items.")
        self.items = items
        self.vectors = np.vstack(vectors).astype(np.float32, copy=False)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[CatalogItem, float]]:
        if self.vectors.size == 0:
            return []

        scores = self.vectors @ query_vector.astype(np.float32, copy=False)
        k = min(top_k, len(self.items))
        if k <= 0:
            return []

        # Argpartition keeps retrieval fast for large candidate sets.
        candidate_idx = np.argpartition(scores, -k)[-k:]
        sorted_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
        return [(self.items[i], float(scores[i])) for i in sorted_idx]


class FAISSVectorStoreAdapter:
    """FAISS-backed vector store using a flat inner-product index.

    Exact cosine search on L2-normalised vectors.
    Requires: pip install faiss-cpu
    """

    def __init__(self) -> None:
        self._index = None
        self.items: list[CatalogItem] = []

    def build(self, items: list[CatalogItem], vectors: list[np.ndarray]) -> None:
        if not items:
            raise ValueError("Cannot build FAISS index with no items.")
        try:
            import faiss  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. Install it with: pip install faiss-cpu"
            ) from exc
        self.items = items
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        dimension = matrix.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(matrix)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[CatalogItem, float]]:
        if self._index is None or not self.items:
            return []
        k = min(top_k, len(self.items))
        q = query_vector.astype(np.float32, copy=False).reshape(1, -1)
        scores, indices = self._index.search(q, k)
        return [
            (self.items[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]


class SupabaseVectorStoreAdapter:
    """pgvector-backed vector store via a local or hosted Supabase PostgreSQL instance.

    Uses cosine similarity (<=> operator) on L2-normalised vectors.
    Requires: pip install psycopg2-binary pgvector

    Connection is read from the SUPABASE_DB_URL environment variable:
        postgresql://postgres:postgres@localhost:54322/postgres
    """

    TABLE = "enterprise_rag_catalog"

    def __init__(self, db_url: str = "") -> None:
        import os  # noqa: PLC0415

        self._db_url = db_url or os.environ.get("SUPABASE_DB_URL", "")
        if not self._db_url:
            raise ValueError(
                "SUPABASE_DB_URL is not set. "
                "Add it to your .env file or pass db_url explicitly."
            )
        self._item_lookup: dict[str, CatalogItem] = {}

    def _connect(self):
        try:
            import psycopg2  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "psycopg2-binary is not installed. "
                "Install it with: pip install psycopg2-binary pgvector"
            ) from exc
        return psycopg2.connect(self._db_url)

    def _ensure_schema(self, conn, dimension: int) -> None:
        """Create extension and table if they don't exist."""
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Drop and recreate if dimension changed so the column type stays in sync.
            cur.execute(f"""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{self.TABLE}'
                          AND column_name = 'embedding'
                          AND udt_name != 'vector'
                    ) THEN
                        DROP TABLE {self.TABLE};
                    END IF;
                END
                $$;
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    item_id     TEXT PRIMARY KEY,
                    sku         TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    category    TEXT NOT NULL,
                    description TEXT NOT NULL,
                    features    TEXT[] NOT NULL,
                    embedding   vector({dimension}) NOT NULL
                );
            """)
            # Drop any stale IVFFlat index — we rebuild it after data is loaded.
            cur.execute(
                f"DROP INDEX IF EXISTS {self.TABLE}_embedding_idx;"
            )
        conn.commit()

    def build(self, items: list[CatalogItem], vectors: list[np.ndarray]) -> None:
        if not items:
            raise ValueError("Cannot build Supabase index with no items.")

        from pgvector.psycopg2 import register_vector  # noqa: PLC0415
        from psycopg2.extras import execute_values   # noqa: PLC0415

        dimension = vectors[0].shape[0]
        self._item_lookup = {item.item_id: item for item in items}

        conn = self._connect()
        try:
            register_vector(conn)
            self._ensure_schema(conn, dimension)
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.TABLE};")
                execute_values(
                    cur,
                    f"""INSERT INTO {self.TABLE}
                        (item_id, sku, name, category, description, features, embedding)
                        VALUES %s""",
                    [
                        (
                            item.item_id,
                            item.sku,
                            item.name,
                            item.category,
                            item.description,
                            item.features,
                            vec,  # pass np.ndarray directly — registered adapter serialises to vector
                        )
                        for item, vec in zip(items, vectors)
                    ],
                )
            conn.commit()
            # Build IVFFlat index only when catalog has enough rows to train properly.
            # pgvector recommends lists << sqrt(n_rows); we require at least 100 rows.
            if len(items) >= 100:
                lists = max(1, int(len(items) ** 0.5))
                with conn.cursor() as cur:
                    cur.execute(
                        f"""CREATE INDEX {self.TABLE}_embedding_idx
                            ON {self.TABLE} USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = {lists});"""
                    )
                conn.commit()
        finally:
            conn.close()

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[CatalogItem, float]]:
        if not self._item_lookup:
            return []

        from pgvector.psycopg2 import register_vector  # noqa: PLC0415

        conn = self._connect()
        try:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT item_id, 1 - (embedding <=> %s) AS score
                    FROM {self.TABLE}
                    ORDER BY embedding <=> %s
                    LIMIT %s;
                    """,
                    (query_vector, query_vector, top_k),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        return [
            (self._item_lookup[item_id], float(score))
            for item_id, score in rows
            if item_id in self._item_lookup
        ]
