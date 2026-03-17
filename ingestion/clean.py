import os
import json
import re
from pathlib import Path

RAW_DIR = Path("outputs_aiohttp") # Expects uploaded files here
METADATA_FILE = Path("scraped_metadata.json") # Expects uploaded metadata here
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
    """Reads raw files from metadata, cleans them, and maps them."""
    # Removed check for metadata existence to allow for manual-only runs
    # if not METADATA_FILE.exists():
    #    print(f"[skip] No metadata found at {METADATA_FILE}")
    #    return

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            raw_metadata = json.load(f)
    else:
        raw_metadata = []
    
    cleaned_metadata = []
    if CLEANED_METADATA_FILE.exists():
        try:
            with open(CLEANED_METADATA_FILE, "r") as f:
                cleaned_metadata = json.load(f)
        except json.JSONDecodeError:
            cleaned_metadata = []
            
    # We process all items in the new metadata to allow updates.
    # We maintain a map of existing items to update them in place or append.
    existing_map = {item['doc_id']: i for i, item in enumerate(cleaned_metadata)}
    
    new_items_count = 0
    updated_items_count = 0
    
    
    # Track which filenames we've seen from metadata to identify orphans
    seen_filenames = set()

    for item in raw_metadata:
        doc_id = item['doc_id']
        raw_path = Path(item['file_path'])
        
        # Determine actual path
        filename = raw_path.name
        actual_raw_path = RAW_DIR / filename
        seen_filenames.add(filename)
        
        if not actual_raw_path.exists():
            print(f"[warning] Raw file missing: {actual_raw_path}")
            continue
            
        text = actual_raw_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(text)
        
        cleaned_filename = f"{doc_id}.txt"
        cleaned_path = CLEANED_DIR / cleaned_filename
        cleaned_path.write_text(cleaned_text, encoding="utf-8")
        
        new_entry = {
            "doc_id": doc_id,
            "url": item['url'],
            "source": item['source'],
            "cleaned_path": str(cleaned_path)
        }
        
        if doc_id in existing_map:
            cleaned_metadata[existing_map[doc_id]] = new_entry
            updated_items_count += 1
        else:
            cleaned_metadata.append(new_entry)
            existing_map[doc_id] = len(cleaned_metadata) - 1
            new_items_count += 1

    # Scan for orphan files (manually added) in RAW_DIR
    if RAW_DIR.exists():
        for file_path in RAW_DIR.glob("*.txt"):
            if file_path.name in seen_filenames:
                continue
                
            print(f"[notice] Found manual file: {file_path.name}")
            
            # Generate ID and Metadata for manual file
            import hashlib
            file_id_hash = hashlib.md5(file_path.name.encode("utf-8")).hexdigest()
            doc_id = f"manual_{file_id_hash}"
            
            text = file_path.read_text(encoding="utf-8")
            cleaned_text = clean_text(text)
            
            cleaned_filename = f"{doc_id}.txt"
            cleaned_path = CLEANED_DIR / cleaned_filename
            cleaned_path.write_text(cleaned_text, encoding="utf-8")
            
            new_entry = {
                "doc_id": doc_id,
                "url": f"file://{file_path.name}", # Placeholder URL
                "source": "manual_upload",
                "cleaned_path": str(cleaned_path)
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
        
    print(f"[clean] Processed {new_items_count} new documents -> {CLEANED_METADATA_FILE}")

if __name__ == "__main__":
    clean_new_docs()
