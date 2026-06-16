import json
import gc
import os
import uuid
from pathlib import Path
import sys
import torch

# Add parent directory to path to import load_model if needed, 
# but usually run_pipeline.py runs from root so imports work.
# We will assume this is run from the project root.

def embed_new_chunks():
    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        print("[error] qdrant-client not installed. Run: pip install qdrant-client")
        return

    CHUNK_METADATA_FILE = Path("chunk_metadata.json")
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME = "track2college_docs"

    if not CHUNK_METADATA_FILE.exists():
        print(f"[skip] No chunk metadata found at {CHUNK_METADATA_FILE}")
        return

    print("[embed] Loading model for embedding: sentence-transformers/multi-qa-mpnet-base-cos-v1")
    from transformers import AutoTokenizer, AutoModel
    
    model_id = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"[embed] Connecting to Qdrant at {QDRANT_URL or 'http://localhost:6333'}...")
    try:
        client_kwargs = {"url": QDRANT_URL or "http://localhost:6333"}
        if QDRANT_API_KEY:
            client_kwargs["api_key"] = QDRANT_API_KEY
        client = QdrantClient(**client_kwargs)
    except Exception as e:
        print(f"[error] Failed to connect to Qdrant: {e}")
        return

    with open(CHUNK_METADATA_FILE, "r") as f:
        chunks = json.load(f)

    print(f"[embed] Processing {len(chunks)} chunks...")

    # Smaller batch size to reduce peak memory usage
    BATCH_SIZE = 16
    import torch.nn.functional as F
    from dotenv import load_dotenv

    load_dotenv()

    total_upserted = 0
    collection_ready = False
    doc_id_index_ready = False

    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _ensure_doc_id_index() -> None:
        nonlocal doc_id_index_ready

        if doc_id_index_ready or not client.collection_exists(COLLECTION_NAME):
            return

        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception as index_err:
            message = str(index_err).lower()
            if "already exists" not in message and "duplicate" not in message:
                print(f"[warning] failed to create doc_id payload index: {index_err}")
            else:
                doc_id_index_ready = True
                return

        doc_id_index_ready = True

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_texts = [c['text'] for c in batch]
        batch_ids = [c['chunk_id'] for c in batch]
        batch_metadatas = [{"doc_id": c['doc_id'], "url": c['url'], "source": c['source']} for c in batch]

        _ensure_doc_id_index()

        # Delete existing chunks for these doc_ids to avoid duplicates/ghosts.
        doc_ids_to_clean = set(c['doc_id'] for c in batch)
        for d_id in doc_ids_to_clean:
            try:
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=d_id),
                            )
                        ]
                    ),
                    wait=True,
                )
            except Exception as del_err:
                # Safe to ignore if collection is not created yet on first batch.
                if "doesn't exist" not in str(del_err).lower():
                    print(f"[warning] delete failed for doc_id={d_id}: {del_err}")

        # Tokenize and embed
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)

            # Move to CPU immediately and free GPU memory
            batch_embeddings = mean_embeddings.cpu().tolist()

        # Free GPU tensors right away
        del inputs, outputs, token_embeddings, attention_mask
        del input_mask_expanded, sum_embeddings, sum_mask, mean_embeddings
        if device == "cuda":
            torch.cuda.empty_cache()

        if not collection_ready:
            vector_size = len(batch_embeddings[0]) if batch_embeddings else 0
            if vector_size == 0:
                print("[warning] Empty embedding batch; skipping")
                continue
            if not client.collection_exists(COLLECTION_NAME):
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                doc_id_index_ready = False
            collection_ready = True

        # Upsert this batch immediately (no accumulation in RAM).
        points = []
        for cid, emb, meta, text in zip(batch_ids, batch_embeddings, batch_metadatas, batch_texts):
            payload = {
                "chunk_id": cid,
                "doc_id": meta["doc_id"],
                "url": meta["url"],
                "source": meta["source"],
                "text": text,
            }
            points.append(
                models.PointStruct(
                    id=_point_id(cid),
                    vector=emb,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )
        total_upserted += len(batch_ids)

        # Free CPU lists and run GC
        del batch_embeddings, batch_texts, batch_ids, batch_metadatas, batch
        gc.collect()

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"   Upserted {total_upserted} / {len(chunks)} chunks...")

    print(f"[embed] Successfully embedded {total_upserted} chunks into Qdrant collection '{COLLECTION_NAME}'")

if __name__ == "__main__":
    embed_new_chunks()
