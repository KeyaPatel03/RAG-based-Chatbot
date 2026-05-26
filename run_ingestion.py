#!/usr/bin/env python3
"""
Run only the ingestion steps (clean → chunk → embed) without re-scraping.
Use this when outputs_aiohttp/ already has .txt files from a previous scrape.
Run from project root: python3 run_ingestion.py
"""
import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent))

from ingestion.clean import clean_new_docs
from ingestion.chunk import chunk_new_docs
from ingestion.embed import embed_new_chunks

def run_ingestion():
    print("=== INGESTION START (skip scraping) ===")

    raw_txt = list(Path("outputs_aiohttp").glob("*.txt"))
    print(f"[info] Found {len(raw_txt)} .txt files in outputs_aiohttp/")
    if not raw_txt:
        print("[stop] No .txt files found in outputs_aiohttp/ — run scraper first")
        return

    # ── Step 1: Clean ──────────────────────────────────────────────
    print("\n[Step 1] Cleaning docs...")
    clean_new_docs()

    cleaned_files = list(Path("data/cleaned").glob("*.txt"))
    print(f"[info] data/cleaned/ has {len(cleaned_files)} files")
    if not cleaned_files:
        print("[stop] No cleaned files produced — check clean.py logs above")
        return

    # ── Step 2: Chunk ──────────────────────────────────────────────
    print("\n[Step 2] Chunking docs...")
    chunk_new_docs()

    chunk_file = Path("chunk_metadata.json")
    if not chunk_file.exists():
        print("[stop] chunk_metadata.json was not created")
        return

    chunks = json.loads(chunk_file.read_text())
    print(f"[info] chunk_metadata.json has {len(chunks)} chunks")
    if not chunks:
        print("[stop] chunk_metadata.json is empty — check chunk.py logs above")
        return

    # ── Step 3: Embed ──────────────────────────────────────────────
    print("\n[Step 3] Embedding and storing in Qdrant...")
    embed_new_chunks()

    # ── Final verify ───────────────────────────────────────────────
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.getenv("QDRANT_URL", "") or "http://localhost:6333",
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )
        count = client.count(collection_name="track2college_docs", exact=True).count
        print(f"\n[✓] Qdrant 'track2college_docs' → {count} documents stored")
    except Exception as e:
        print(f"\n[warning] Could not verify Qdrant count: {e}")

    print("\n=== INGESTION COMPLETE ===")

if __name__ == "__main__":
    run_ingestion()
