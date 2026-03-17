import json
from pathlib import Path
import sys
import torch

# Add parent directory to path to import load_model if needed, 
# but usually run_pipeline.py runs from root so imports work.
# We will assume this is run from the project root.

def embed_new_chunks():
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("[error] chromadb not installed. Run: pip install chromadb")
        return

    CHUNK_METADATA_FILE = Path("chunk_metadata.json")
    CHROMA_DB_DIR = Path("data/chroma_db")
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

    print("[embed] Connecting to ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"[error] Failed to connect to ChromaDB: {e}")
        return

    with open(CHUNK_METADATA_FILE, "r") as f:
        chunks = json.load(f)

    print(f"[embed] Processing {len(chunks)} chunks...")
    
    # Batch processing
    BATCH_SIZE = 32 # Increased batch size for smaller model
    
    ids = []
    documents = [] # The raw text to store (for retrieval)
    metadatas = []
    embeddings = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_texts = [c['text'] for c in batch]
        
        # Determine IDs and Metadata
        batch_ids = [c['chunk_id'] for c in batch]
        batch_metadatas = [{"doc_id": c['doc_id'], "url": c['url'], "source": c['source']} for c in batch]

        # Delete existing chunks for these doc_ids to avoid duplicates/ghosts
        doc_ids_to_clean = set(c['doc_id'] for c in batch)
        for d_id in doc_ids_to_clean:
            try:
                collection.delete(where={"doc_id": d_id})
            except Exception:
                pass 

        # Generate Embeddings
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean Pooling - Take attention mask into account for correct averaging
            token_embeddings = outputs.last_hidden_state # Contains all token embeddings
            attention_mask = inputs['attention_mask']
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            mean_embeddings = sum_embeddings / sum_mask
            
            # Normalize embeddings (optional but good for cosine similarity models)
            import torch.nn.functional as F
            mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)
            
            # Move to CPU list
            batch_embeddings = mean_embeddings.tolist()

        ids.extend(batch_ids)
        documents.extend(batch_texts)
        metadatas.extend(batch_metadatas)
        embeddings.extend(batch_embeddings)
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i} / {len(chunks)}")

    # Upsert to Chroma
    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"[embed] Successfully embedded {len(ids)} chunks into ChromaDB collection '{COLLECTION_NAME}'")

if __name__ == "__main__":
    embed_new_chunks()
