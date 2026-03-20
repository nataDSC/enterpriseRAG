"""
Generate ~50 realistic enterprise software products via GPT-4o-mini,
write them to src/enterprise_rag/mock_catalog.py, and load into Supabase.

Usage:
    ./.venv/bin/python populate_catalog.py
    ./.venv/bin/python populate_catalog.py --dry-run      # print only, no file/DB writes
    ./.venv/bin/python populate_catalog.py --no-supabase  # update file only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Existing 10 items we keep (to preserve any existing tests / demos)
# ---------------------------------------------------------------------------

SEED_ITEMS = [
    {
        "item_id": "p1", "sku": "SEC-IAM-001",
        "name": "IdentityHub Enterprise SSO", "category": "Security",
        "description": "Enterprise single sign-on with SAML, OIDC, and SCIM provisioning.",
        "features": ["mfa", "sso", "saml", "oidc", "scim", "zero trust"],
    },
    {
        "item_id": "p2", "sku": "SEC-IAM-002",
        "name": "Privileged Access Vault", "category": "Security",
        "description": "Secure vault for rotating privileged credentials and session recording.",
        "features": ["pam", "credential rotation", "audit", "session replay"],
    },
    {
        "item_id": "p3", "sku": "DATA-DB-011",
        "name": "Velocity Vector Database", "category": "Data Platform",
        "description": "Low-latency vector search with IVF-HNSW and scalar metadata filtering.",
        "features": ["vector", "ann", "hnsw", "ivf", "millisecond latency"],
    },
    {
        "item_id": "p4", "sku": "DATA-DB-012",
        "name": "Relational Lakehouse Connector", "category": "Data Platform",
        "description": "Federated SQL queries across cloud warehouses and operational stores.",
        "features": ["sql", "federation", "analytics", "bi"],
    },
    {
        "item_id": "p5", "sku": "AI-RAG-201",
        "name": "Document Grounding Service", "category": "AI / ML",
        "description": "RAG-ready chunking, metadata enrichment, and citation generation.",
        "features": ["rag", "chunking", "metadata", "citations"],
    },
    {
        "item_id": "p6", "sku": "AI-OBS-203",
        "name": "Prompt Safety Firewall", "category": "AI / ML",
        "description": "Policy enforcement for prompt injection defense and sensitive data masking.",
        "features": ["guardrails", "prompt injection", "pii redaction", "policy"],
    },
    {
        "item_id": "p7", "sku": "NET-EDGE-301",
        "name": "Edge GPU Inference Appliance", "category": "Edge",
        "description": "On-prem edge appliance with GPU acceleration for private AI inference.",
        "features": ["edge", "gpu", "low latency", "air-gapped"],
    },
    {
        "item_id": "p8", "sku": "NET-CDN-302",
        "name": "Adaptive API Gateway", "category": "Networking",
        "description": "Smart API gateway with rate limiting, WAF rules, and traffic shaping.",
        "features": ["api", "gateway", "waf", "traffic control"],
    },
    {
        "item_id": "p9", "sku": "CRM-SE-401",
        "name": "Sales Engineer Assistant", "category": "Sales Enablement",
        "description": "Conversational assistant for mapping customer requirements to SKUs.",
        "features": ["catalog qa", "requirement mapping", "proposal assist"],
    },
    {
        "item_id": "p10", "sku": "OBS-OPS-501",
        "name": "Observability Mesh", "category": "Operations",
        "description": "Distributed traces, logs, and metrics with SLO-driven alerting.",
        "features": ["otel", "tracing", "metrics", "slo"],
    },
]

# ---------------------------------------------------------------------------
# GPT generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a product catalog generator for an enterprise software company.
Generate realistic B2B software product entries. Each product must be
distinct and cover a specific technical niche that a Sales Engineer might
encounter when working with large enterprise customers.
"""

USER_PROMPT = """\
Generate exactly 50 enterprise software products. Do NOT repeat or
closely paraphrase these already-existing products:
{existing_names}

Return a JSON array (no markdown, no explanation) where each element is:
{{
  "sku": "CATEGORY-SUBCATEGORY-NNN",   // e.g. SEC-PKI-003, DATA-STRM-015
  "name": "Product Name",
  "category": "one of: Security | Data Platform | AI / ML | Networking | Edge | Operations | DevOps | Compliance | Sales Enablement | Infrastructure",
  "description": "One sentence (15-25 words) describing the product's core value.",
  "features": ["tag1", "tag2", "tag3", "tag4"]  // 3-6 lowercase tags
}}
"""


def generate_products(api_key: str) -> list[dict]:
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=api_key)
    existing_names = ", ".join(p["name"] for p in SEED_ITEMS)

    print("Calling GPT-4o-mini to generate 50 products…")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.9,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(existing_names=existing_names)},
        ],
    )
    raw = response.choices[0].message.content.strip()

    # Strip optional markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    products = json.loads(raw)
    print(f"  Received {len(products)} products from API.")
    return products


# ---------------------------------------------------------------------------
# Assign stable item_ids and merge with seed
# ---------------------------------------------------------------------------

def build_full_catalog(generated: list[dict]) -> list[dict]:
    all_items = list(SEED_ITEMS)
    existing_skus = {p["sku"] for p in all_items}
    next_id = len(all_items) + 1

    for prod in generated:
        sku = prod.get("sku", "").strip()
        if not sku or sku in existing_skus:
            continue  # skip duplicates
        all_items.append({
            "item_id": f"p{next_id}",
            "sku": sku,
            "name": prod.get("name", "").strip(),
            "category": prod.get("category", "General").strip(),
            "description": prod.get("description", "").strip(),
            "features": [f.strip().lower() for f in prod.get("features", [])],
        })
        existing_skus.add(sku)
        next_id += 1

    return all_items


# ---------------------------------------------------------------------------
# Write mock_catalog.py
# ---------------------------------------------------------------------------

CATALOG_TEMPLATE = '''\
# AUTO-GENERATED by populate_catalog.py — do not edit manually.
from enterprise_rag.models import CatalogItem


def load_mock_catalog() -> list[CatalogItem]:
    return [
{entries}
    ]
'''

ITEM_TEMPLATE = '''\
        CatalogItem(
            item_id={item_id!r},
            sku={sku!r},
            name={name!r},
            category={category!r},
            description={description!r},
            features={features!r},
        ),'''


def write_catalog_file(items: list[dict], target: Path) -> None:
    entries = "\n".join(ITEM_TEMPLATE.format(**item) for item in items)
    target.write_text(CATALOG_TEMPLATE.format(entries=entries), encoding="utf-8")
    print(f"  Wrote {len(items)} items to {target}")


# ---------------------------------------------------------------------------
# Load into Supabase
# ---------------------------------------------------------------------------

def load_into_supabase(items: list[dict]) -> None:
    import numpy as np  # noqa: PLC0415
    from enterprise_rag.embedding import HashingEmbedder  # noqa: PLC0415
    from enterprise_rag.models import CatalogItem  # noqa: PLC0415
    from enterprise_rag.vector_store import SupabaseVectorStoreAdapter  # noqa: PLC0415

    catalog = [CatalogItem(**item) for item in items]
    embedder = HashingEmbedder()
    vectors = embedder.embed_batch([c.to_index_text() for c in catalog])

    store = SupabaseVectorStoreAdapter()
    print(f"  Loading {len(catalog)} items into Supabase…")
    store.build(catalog, vectors)
    print("  Done — Supabase table updated.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Populate enterprise RAG catalog with GPT-generated data.")
    parser.add_argument("--dry-run", action="store_true", help="Print products only; don't write files or DB.")
    parser.add_argument("--no-supabase", action="store_true", help="Update mock_catalog.py only; skip Supabase.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    generated = generate_products(api_key)
    all_items = build_full_catalog(generated)

    print(f"\nTotal catalog size: {len(all_items)} items "
          f"({len(SEED_ITEMS)} seed + {len(all_items) - len(SEED_ITEMS)} generated)\n")

    if args.dry_run:
        for item in all_items:
            print(f"  [{item['sku']:<22}] {item['name']} — {item['category']}")
        return

    catalog_path = Path(__file__).parent / "src" / "enterprise_rag" / "mock_catalog.py"
    write_catalog_file(all_items, catalog_path)

    if not args.no_supabase:
        load_into_supabase(all_items)

    print("\nAll done! Restart Streamlit to pick up the new catalog.")


if __name__ == "__main__":
    main()
