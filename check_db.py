#!/usr/bin/env python3
"""
Diagnostic script: checks every stage of the pipeline and ChromaDB state.
Run from the project root: python3 check_db.py
"""
import json
from pathlib import Path

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

# --- Step 6: ChromaDB ---
chroma_dir = Path("data/chroma_db")
print(f"[6] data/chroma_db/ exists      : {chroma_dir.exists()}")
if chroma_dir.exists():
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collections = client.list_collections()
        print(f"    → Collections: {[c.name for c in collections]}")
        for c in collections:
            col = client.get_collection(c.name)
            print(f"    → '{c.name}': {col.count()} documents")
    except Exception as e:
        print(f"    → ChromaDB error: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
