import json
from pathlib import Path
import tiktoken

# =========================
# Paths
# =========================
CLEANED_METADATA_FILE = Path("cleaned_metadata.json")
CHUNK_METADATA_FILE = Path("chunk_metadata.json")

# =========================
# Chunking config (RAG-optimized)
# =========================
CHUNK_SIZE = 250        # LLM tokens
OVERLAP = 60
MIN_TOKENS = 50

# =========================
# LLM tokenizer (generation-aligned)
# =========================
ENCODING_NAME = "cl100k_base"  # GPT-4 / GPT-3.5 compatible
tokenizer = tiktoken.get_encoding(ENCODING_NAME)

# =========================
# Section-aware splitting
# =========================
def split_by_sections(text: str):
    """
    Split text by markdown-style headings.
    Keeps semantic blocks intact.
    """
    sections = []
    current = []

    for line in text.splitlines():
        if line.strip().startswith(("#", "##", "###")):
            if current:
                sections.append("\n".join(current))
                current = []
        current.append(line)

    if current:
        sections.append("\n".join(current))

    return sections

# =========================
# Token-based chunking (LLM tokenizer)
# =========================
def chunk_text(text: str):
    tokens = tokenizer.encode(text)

    if len(tokens) < MIN_TOKENS:
        return []

    if len(tokens) <= CHUNK_SIZE:
        return [text]

    chunks = []
    stride = max(CHUNK_SIZE - OVERLAP, CHUNK_SIZE // 2)

    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        if i + CHUNK_SIZE >= len(tokens):
            break

    return chunks

# =========================
# Main chunking pipeline
# =========================
def chunk_new_docs():
    if not CLEANED_METADATA_FILE.exists():
        print(f"[skip] No cleaned metadata found at {CLEANED_METADATA_FILE}")
        return

    with open(CLEANED_METADATA_FILE, "r", encoding="utf-8") as f:
        cleaned_docs = json.load(f)

    chunk_metadata = []
    seen_chunks = set()  # 🔹 Step 7: deduplication
    global_chunk_counter = 0

    for doc in cleaned_docs:
        doc_id = doc["doc_id"]
        path = Path(doc["cleaned_path"])

        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")

        # Step 1: section-aware split
        sections = split_by_sections(text)

        for section in sections:
            chunks = chunk_text(section)
            total_chunks = len(chunks)

            for i, chunk_content in enumerate(chunks):
                # 🔹 Dedup near-identical chunks (overlap artifacts)
                dedup_key = hash(chunk_content[:200])
                if dedup_key in seen_chunks:
                    continue
                seen_chunks.add(dedup_key)

                chunk_metadata.append({
                    "chunk_id": f"{doc_id}_chunk_{global_chunk_counter}",
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "url": doc.get("url", ""),
                    "source": doc.get("source", ""),
                    "text": chunk_content
                })

                global_chunk_counter += 1

    with open(CHUNK_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, indent=2)

    print(
        f"[chunk] Generated {len(chunk_metadata)} chunks "
        f"from {len(cleaned_docs)} docs → {CHUNK_METADATA_FILE}"
    )

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    chunk_new_docs()

