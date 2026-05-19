import os
import json
import re
import hashlib
from pathlib import Path

RAW_DIR = Path("outputs_aiohttp")
METADATA_FILE = Path("scraped_metadata.json")   # Optional: used to pull url/source metadata
CLEANED_DIR = Path("data/cleaned")
CLEANED_METADATA_FILE = Path("cleaned_metadata.json")

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def clean_new_docs():
    """
    Primary source of truth: scan ALL .txt files in outputs_aiohttp/.
    Use scraped_metadata.json only to look up url/source for known files.
    Any file not in scraped_metadata.json is treated as a manual upload.
    """
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # Build a lookup: filename → metadata entry (url, source, doc_id)
    meta_by_filename = {}
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r") as f:
                raw_metadata = json.load(f)
            for item in raw_metadata:
                if item.get("status") != "success":
                    continue
                fp = item.get("file_path", "")
                if not fp:
                    continue
                fname = Path(fp).name
                # Only register a filename once (first occurrence wins)
                if fname not in meta_by_filename:
                    meta_by_filename[fname] = {
                        "doc_id": item["doc_id"],
                        "url": item.get("url", ""),
                        "source": item.get("source", "web"),
                    }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[warning] Could not fully parse {METADATA_FILE}: {e}")

    # Load existing cleaned_metadata to update in-place
    cleaned_metadata = []
    if CLEANED_METADATA_FILE.exists():
        try:
            with open(CLEANED_METADATA_FILE, "r") as f:
                cleaned_metadata = json.load(f)
        except json.JSONDecodeError:
            cleaned_metadata = []

    existing_map = {item["doc_id"]: i for i, item in enumerate(cleaned_metadata)}

    new_items_count = 0
    updated_items_count = 0
    skipped_count = 0

    if not RAW_DIR.exists():
        print(f"[error] Raw directory not found: {RAW_DIR}")
        return

    txt_files = sorted(RAW_DIR.glob("*.txt"))
    print(f"[clean] Found {len(txt_files)} .txt files in {RAW_DIR}")

    for file_path in txt_files:
        fname = file_path.name

        if fname in meta_by_filename:
            # Known file from scraped_metadata.json
            meta = meta_by_filename[fname]
            doc_id = meta["doc_id"]
            url = meta["url"]
            source = meta["source"]
        else:
            # Unknown file — treat as manual upload; generate stable ID from filename
            file_id_hash = hashlib.md5(fname.encode("utf-8")).hexdigest()
            doc_id = f"manual_{file_id_hash}"
            url = f"file://{fname}"
            source = "manual_upload"

        text = file_path.read_text(encoding="utf-8", errors="replace")
        cleaned_text = clean_text(text)

        if not cleaned_text.strip():
            skipped_count += 1
            continue  # Skip effectively empty files

        cleaned_filename = f"{doc_id}.txt"
        cleaned_path = CLEANED_DIR / cleaned_filename
        cleaned_path.write_text(cleaned_text, encoding="utf-8")

        new_entry = {
            "doc_id": doc_id,
            "url": url,
            "source": source,
            "cleaned_path": str(cleaned_path),
        }

        if doc_id in existing_map:
            cleaned_metadata[existing_map[doc_id]] = new_entry
            updated_items_count += 1
        else:
            cleaned_metadata.append(new_entry)
            existing_map[doc_id] = len(cleaned_metadata) - 1
            new_items_count += 1

    with open(CLEANED_METADATA_FILE, "w") as f:
        json.dump(cleaned_metadata, f, indent=2)

    print(
        f"[clean] Done — {new_items_count} new, {updated_items_count} updated, "
        f"{skipped_count} skipped (empty) → {CLEANED_METADATA_FILE}"
    )
    print(f"[clean] Total entries in cleaned_metadata: {len(cleaned_metadata)}")

if __name__ == "__main__":
    clean_new_docs()
