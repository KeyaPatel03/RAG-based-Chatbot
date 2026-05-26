#!/usr/bin/env python3
"""
Diagnostic script: checks every stage of the pipeline and Qdrant state.
Run from the project root: python3 check_db.py
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("PIPELINE DIAGNOSTIC")
print("=" * 60)

# --- Step 1: Raw files ---
raw_dir = Path("outputs_aiohttp")
txt_files = list(raw_dir.glob("*.txt")) if raw_dir.exists() else []
print(f"\n[1] outputs_aiohttp/*.txt files : {len(txt_files)}")

# --- Step 2: scraped_metadata.json ---
sm = Path("scraped_metadata.json")
if sm.exists():
    data = json.loads(sm.read_text())
    success = [d for d in data if d.get("status") == "success" and d.get("file_path")]
    print(f"[2] scraped_metadata.json       : {len(data)} total entries, {len(success)} with file_path+success")
else:
    print("[2] scraped_metadata.json       : NOT FOUND")

# --- Step 3: cleaned_metadata.json ---
cm = Path("cleaned_metadata.json")
if cm.exists():
    cleaned = json.loads(cm.read_text())
    print(f"[3] cleaned_metadata.json       : {len(cleaned)} entries")
else:
    print("[3] cleaned_metadata.json       : NOT FOUND")
    cleaned = []

# --- Step 4: data/cleaned/ ---
cleaned_dir = Path("data/cleaned")
cleaned_files = list(cleaned_dir.glob("*.txt")) if cleaned_dir.exists() else []
print(f"[4] data/cleaned/*.txt files    : {len(cleaned_files)}")

# Check path consistency
if cleaned:
    missing_paths = [d for d in cleaned if not Path(d["cleaned_path"]).exists()]
    print(f"    → paths in cleaned_metadata that DON'T exist on disk: {len(missing_paths)}")

# --- Step 5: chunk_metadata.json ---
chk = Path("chunk_metadata.json")
if chk.exists():
    chunks = json.loads(chk.read_text())
    print(f"[5] chunk_metadata.json         : {len(chunks)} chunks")
else:
    print("[5] chunk_metadata.json         : NOT FOUND")

# --- Step 6: Qdrant ---
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
print(f"[6] Qdrant URL                  : {qdrant_url}")
try:
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=qdrant_url or "http://localhost:6333",
        api_key=qdrant_api_key or None,
    )
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    print(f"    → Collections: {names}")
    for name in names:
        count = client.count(collection_name=name, exact=True).count
        print(f"    → '{name}': {count} documents")
except Exception as e:
    print(f"    → Qdrant error: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
