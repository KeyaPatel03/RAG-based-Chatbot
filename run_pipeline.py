#!/usr/bin/env python3
import sys
import asyncio
from pathlib import Path

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

    # Run scraping
    try:
        asyncio.run(scrap.main(urls))
    except Exception as e:
        print(f"[error] Scraping step failed: {e}")
        return
    
    print("\n[Step 2] Cleaning docs...")
    clean_new_docs()
    
    print("\n[Step 3] Chunking docs...")
    chunk_new_docs()
    
    print("\n[Step 4] Embedding and Storing in ChromaDB...")
    embed_new_chunks()
    
    print("\n=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    run_pipeline()
