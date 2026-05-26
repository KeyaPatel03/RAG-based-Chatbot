#!/usr/bin/env python3
import sys
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import scrap
from ingestion.clean import clean_new_docs
from ingestion.chunk import chunk_new_docs
from ingestion.embed import embed_new_chunks

def run_pipeline():
    print("=== PIPELINE START ===")

    print("\n[Step 1] Scraping URLs (Cluster Side)...")
    urls = scrap.load_urls_from_file()
    if not urls:
        print("[stop] No URLs found in urls.txt or clgUrls.txt")
        return

    try:
        asyncio.run(scrap.main(urls))
    except Exception as e:
        print(f"[error] Scraping step failed: {e}")
        return

    # Sanity check after scraping
    raw_txt = list(Path("outputs_aiohttp").glob("*.txt"))
    print(f"[info]  outputs_aiohttp/ has {len(raw_txt)} .txt files after scraping")
    if not raw_txt:
        print("[stop] No .txt files produced by scraper — aborting pipeline")
        return

    print("\n[Step 2] Cleaning docs...")
    clean_new_docs()

    # Sanity check after cleaning
    cleaned_files = list(Path("data/cleaned").glob("*.txt"))
    print(f"[info]  data/cleaned/ has {len(cleaned_files)} files after cleaning")
    if not cleaned_files:
        print("[stop] No cleaned files produced — aborting pipeline")
        return

    print("\n[Step 3] Chunking docs...")
    chunk_new_docs()

    # Sanity check after chunking
    chunk_file = Path("chunk_metadata.json")
    if chunk_file.exists():
        import json
        chunks = json.loads(chunk_file.read_text())
        print(f"[info]  chunk_metadata.json has {len(chunks)} chunks after chunking")
        if not chunks:
            print("[stop] chunk_metadata.json is empty — aborting pipeline")
            return
    else:
        print("[stop] chunk_metadata.json was not created — aborting pipeline")
        return

    print("\n[Step 4] Embedding and Storing in Qdrant...")
    embed_new_chunks()

    # Final check
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.getenv("QDRANT_URL", "") or "http://localhost:6333",
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )
        count = client.count(collection_name="track2college_docs", exact=True).count
        print(f"[info]  Qdrant collection 'track2college_docs' has {count} documents")
    except Exception as e:
        print(f"[warning] Could not verify Qdrant count: {e}")

    print("\n=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    run_pipeline()
