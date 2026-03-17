__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sys
import torch
import chromadb
from pathlib import Path
from collections import deque
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Import generation model
# We assume load_model.py is in the same directory
try:
    from load_model import model as gen_model, tokenizer as gen_tokenizer
except ImportError:
    print("[error] Could not import load_model.py. Make sure it is in the same directory.")
    sys.exit(1)

# Configuration
CHROMA_DB_DIR = Path("data/chroma_db")
COLLECTION_NAME = "track2college_docs"
EMBED_MODEL_ID = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
TOP_K = 6

embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model.to(device)

client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def get_query_embedding(query_text):
    """Generates embedding for the query using the same logic as ingestion."""
    inputs = embed_tokenizer(
        [query_text], 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = embed_model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        mean_embeddings = sum_embeddings / sum_mask
        mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)
        
        return mean_embeddings[0].tolist()

def format_context(results):
    """Formats retrieved results into a context string."""
    context_parts = []
    sources = set()
    
    # chromadb results structure: {'documents': [[...]], 'metadatas': [[...]], ...}
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        url = meta.get('url', 'Unknown')
        sources.add(url)
        context_parts.append(f"--- Passage {i+1} from {url} ---\n{doc}\n")
        
    return "\n".join(context_parts), list(sources)




def _format_conversation_history(conversation_history):
    if not conversation_history:
        return "None"

    formatted_turns = []
    for idx, turn in enumerate(conversation_history, start=1):
        user_q = str(turn.get("user", "")).strip()
        bot_a = str(turn.get("bot", "")).strip()
        formatted_turns.append(
            f"Turn {idx}:\nUser: {user_q}\nAssistant: {bot_a}"
        )
    return "\n\n".join(formatted_turns)


def generate_answer(query, return_context=False, conversation_history=None):
    # 1. Retrieve
    query_emb = get_query_embedding(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K
    )
    
    context_str, sources = format_context(results)
    
    if not context_str.strip():
        response = _enforce_response_format(
            "I don't have this information right now. Please check trusted sources online or talk to your school counselor.",
            sources,
            [],
        )
        if return_context:
            return response, sources, []
        return response, sources

    # 2. Construct Prompt with History
    system_prompt = """You are a Retrieval-Augmented Generation (RAG) assistant.

You will be given:
1. A user question
2. Retrieved context passages
3. Conversation history (optional)

Your task is to:
- Answer the user's question using ONLY the information present in the context
- Use the conversation history to understand the context of the question if needed
- Generate a clear, accurate, and well-structured response
- Avoid hallucinations

Answering rules:
- If information is missing, say: "I don't have this information right now. Please check trusted sources online or talk to your school counselor."
- Do NOT invent facts or URLs
- Do NOT say phrases like "mentioned in the context", "not mentioned in the context", or "based on the context"
- Output specific Source URLs from the context at the end
- Provide 1-3 follow-up questions or related topics that the user might be interested in, based on the context and the answer.

Output format (use exactly these section headers):

Answer:
<your answer>

Sources:
- <url>

Follow-up Questions:
- <question 1>
- <question 2>
"""
    
    history_text = _format_conversation_history(conversation_history)
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"Context:\n{context_str}\n"
        f"User Question: {query}\n\n"
        f"Answer:\n"
    )

    # 3. Generate
    inputs = gen_tokenizer(full_prompt, return_tensors="pt").to(gen_model.device)
    
    if inputs.input_ids.shape[1] > 3000:
        pass
    
    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        pad_token_id=gen_tokenizer.pad_token_id
    )
    
    decoded = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    # Robust extraction: split by "Answer:"
    if "Answer:" in decoded:
        response = decoded.split("Answer:")[-1].strip()
    else:
        # Fallback if "Answer:" is missing but generation happened
        # Remove prompt from decoded
        response = decoded[len(full_prompt):]
        # Or if prompt is not exactly in decoded due to tokenization quirks:
        response = gen_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    answer_text, generated_sources, followups = _extract_sections(response)
    source_urls = generated_sources if generated_sources else sources
    response = _enforce_response_format(answer_text, source_urls, followups)
    
    if return_context:
        return response, sources, results['documents'][0]
    return response, sources

import re


def _extract_sections(response_text: str):
    answer_text = response_text.strip()
    source_urls = []
    followups = []

    answer_match = re.search(
        r"Answer:\s*(.*?)(?:\n\s*Sources:|\n\s*Follow-up Questions:|$)",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        split_match = re.split(
            r"\n\s*(?:Sources:|Follow-up Questions:)\s*",
            response_text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        if split_match:
            answer_text = split_match[0].strip()

    sources_match = re.search(
        r"Sources:\s*(.*?)(?:\n\s*Follow-up Questions:|$)",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if sources_match:
        source_block = sources_match.group(1)
        source_urls = re.findall(r"https?://[^\s)\]>]+", source_block)
        if not source_urls:
            for line in source_block.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    source_urls.append(line[2:].strip())

    follow_match = re.search(
        r"Follow-up Questions:\s*(.*)$",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if follow_match:
        follow_block = re.split(
            r"\n\s*Sources:\s*",
            follow_match.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        for line in follow_block.splitlines():
            line = line.strip()
            if line.startswith("- "):
                followups.append(line[2:].strip())

    return answer_text, source_urls, followups


def _enforce_response_format(answer_text: str, source_urls, followups):
    deduped_sources = []
    seen_sources = set()
    for src in source_urls or []:
        normalized = str(src).strip()
        if not normalized:
            continue
        key = normalized.lower().rstrip("/")
        if key in seen_sources:
            continue
        seen_sources.add(key)
        deduped_sources.append(normalized)

    if not deduped_sources:
        deduped_sources = ["No source URL available"]

    clean_followups = []
    seen_followups = set()
    for question in followups or []:
        text = str(question).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen_followups:
            continue
        seen_followups.add(key)
        clean_followups.append(text)

    if not clean_followups:
        clean_followups = [
            "Would you like a shorter summary of this answer?",
            "Do you want me to explain eligibility or application steps related to this topic?",
        ]

    clean_answer = _sanitize_answer_text(answer_text)

    sources_block = "\n".join([f"- {s}" for s in deduped_sources])
    followups_block = "\n".join([f"- {q}" for q in clean_followups[:3]])

    return (
        f"Answer:\n{clean_answer}\n\n"
        f"Sources:\n{sources_block}\n\n"
        f"Follow-up Questions:\n{followups_block}"
    )


def _sanitize_answer_text(answer_text: str) -> str:
    text = (answer_text or "").strip()
    if not text:
        return "I don't have this information right now. Please check trusted sources online or talk to your school counselor."

    fallback = "I don't have this information right now. Please check trusted sources online or talk to your school counselor."

    missing_info_patterns = [
        r"\bnot\s+(?:mentioned|provided|available)\b",
        r"\binsufficient\s+information\b",
        r"\bdo(?:es)?\s+not\s+contain\s+sufficient\s+information\b",
        r"\bcannot\s+answer\b",
        r"\bcan['’]?t\s+answer\b",
        r"\bunable\s+to\s+answer\b",
    ]

    lower_text = text.lower()
    if any(re.search(pattern, lower_text, flags=re.IGNORECASE) for pattern in missing_info_patterns):
        return fallback

    context_phrases = [
        r"\bin\s+the\s+provided\s+context\b",
        r"\bin\s+this\s+context\b",
        r"\bbased\s+on\s+the\s+context\b",
        r"\bmentioned\s+in\s+the\s+context\b",
        r"\bnot\s+mentioned\s+in\s+the\s+context\b",
    ]
    for pattern in context_phrases:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text).strip()
    return text or fallback

def _deduplicate_sources(response_text: str) -> str:
    """
    Removes duplicate URLs from the Sources section of the response.
    """
    if "Sources:" not in response_text:
        return response_text

    parts = response_text.rpartition("Sources:")
    main_text = parts[0]
    # The part after "Sources:" might contain the related questions section now
    sources_and_related = parts[2]
    
    RELATED_HEADER = "I can also help you with related questions you might have:"
    
    # Split sources and related questions
    if RELATED_HEADER in sources_and_related:
        src_parts = sources_and_related.split(RELATED_HEADER)
        sources_text = src_parts[0]
        related_text = RELATED_HEADER + src_parts[1]
    else:
        sources_text = sources_and_related
        related_text = ""

    grouped_sources = {}
    lines = sources_text.strip().split('\n')
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        if stripped.startswith("- ") or stripped.startswith("* "):
            content = stripped[2:].strip()
            
            # Simple URL extraction
            url_raw = content.split(' ')[0] # Assume URL is first, ignore passage info for dedupe key
            url_norm = url_raw.lower().rstrip('/')
            
            if url_norm not in grouped_sources:
                grouped_sources[url_norm] = content # Keep original content
                
    unique_sources = []
    for content in grouped_sources.values():
        unique_sources.append(f"- {content}")

    if not unique_sources:
        # If no sources, just return main text + related questions (if any)
        if related_text:
             return f"{main_text.strip()}\n\n{related_text.strip()}"
        return main_text.strip()

    final_sources = "\n".join(unique_sources)
    
    result = f"{main_text}Sources:\n{final_sources}"
    if related_text:
        result += f"\n\n{related_text.strip()}"
        
    return result


def get_answer(question: str) -> str:
    """Minimal callable used by API routes to get final answer text."""
    response, _ = generate_answer(question)
    return response


if __name__ == "__main__":
    conversation_memory = deque(maxlen=4)

    while True:
        user_input = input("Enter your question: ")
        if user_input.lower() in ('exit', 'quit'):
            break

        try:
            memory_for_prompt = list(conversation_memory)
            answer, sources = generate_answer(
                user_input,
                conversation_history=memory_for_prompt,
            )
            print(answer)
            conversation_memory.append({"user": user_input, "bot": answer})

        except Exception as e:
            print(f"Answer:\nAn error occurred while generating the response.\n\nSources:\n- No source URL available\n\nFollow-up Questions:\n- Please try rephrasing your question.\n- Would you like to ask a more specific question?")

