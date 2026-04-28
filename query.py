__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import re
import time
import sys
import torch
import chromadb
import requests
import os
from pathlib import Path
from collections import deque
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment / Config
# ---------------------------------------------------------------------------
load_dotenv()
RUNPOD_API_KEY      = os.environ.get("RunPod_API_Key", "") or os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_URL = os.environ.get("RUNPOD_ENDPOINT_URL", "")

if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_URL:
    print("[warning] Ensure RunPod_API_Key and RUNPOD_ENDPOINT_URL are set in .env")

# RunPod expects /runsync for synchronous calls
_RUNPOD_SYNC_URL = (
    RUNPOD_ENDPOINT_URL.rstrip("/").rsplit("/run", 1)[0] + "/runsync"
    if RUNPOD_ENDPOINT_URL.endswith("/run")
    else RUNPOD_ENDPOINT_URL
)

# ---------------------------------------------------------------------------
# ChromaDB / Embedding config
# ---------------------------------------------------------------------------
CHROMA_DB_DIR   = Path("data/chroma_db")
COLLECTION_NAME = "track2college_docs"
EMBED_MODEL_ID  = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
TOP_K           = 6

# Cosine distance threshold: documents with distance > this are considered
# too dissimilar from the query → out-of-scope response.
# ChromaDB cosine distance: 0 = identical, 2 = opposite.
# multi-qa-mpnet-base-cos-v1 typical range: relevant=0.1-0.9, off-topic=1.2-2.0
# Set conservatively high so we never block legitimate questions.
RELEVANCE_THRESHOLD = 1.4

# ---------------------------------------------------------------------------
# Load embedding model once at startup
# ---------------------------------------------------------------------------
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
embed_model     = AutoModel.from_pretrained(EMBED_MODEL_ID)
device          = "cuda" if torch.cuda.is_available() else "cpu"
embed_model.to(device)

client     = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# ---------------------------------------------------------------------------
# Out-of-scope fallback response
# ---------------------------------------------------------------------------
OUT_OF_SCOPE_RESPONSE = (
    "I'm sorry, I can only help with topics related to college admissions, "
    "financial aid, scholarships, and related college planning information. "
    "I don't have reliable information on this topic in my knowledge base.\n\n"
    "For help with this question, you may:\n"
    "- Contact your school counselor directly\n"
    "- Visit a trusted online resource such as collegeboard.org or studentaid.gov"
)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------
def get_query_embedding(query_text: str) -> list:
    """Mean-pooled, normalised embedding — same logic as ingestion."""
    inputs = embed_tokenizer(
        [query_text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        token_embeddings = embed_model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        mean_emb = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        mean_emb = F.normalize(mean_emb, p=2, dim=1)

    return mean_emb[0].tolist()


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------
def format_context(results: dict) -> tuple[str, list[str]]:
    """Build a readable context string and deduplicated source URL list."""
    context_parts = []
    seen_urls: set[str] = set()
    sources: list[str] = []

    docs   = results["documents"][0]
    metas  = results["metadatas"][0]

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        url = meta.get("url", "").strip()
        context_parts.append(f"--- Passage {i + 1} ---\n{doc}\n")
        if url:
            key = url.lower().rstrip("/")
            if key not in seen_urls:
                seen_urls.add(key)
                sources.append(url)

    return "\n".join(context_parts), sources


def _top_sources_from_results(results: dict, n: int = 3) -> list[str]:
    """
    Return source URLs from only the n closest (most relevant) retrieved
    passages, sorted by ascending cosine distance.

    This is used as the fallback source list when the LLM either produces
    no URLs or produces hallucinated ones that fail the allowlist check.
    Using only the top-n closest passages (instead of all TOP_K) avoids
    surfacing topically-adjacent but irrelevant links (e.g. financial-aid
    pages appearing as sources for a university-admissions question).
    """
    distances = results.get("distances", [[]])[0]
    metas     = results.get("metadatas",  [[]])[0]

    if not distances or not metas:
        return []

    # Pair each passage with its distance and sort closest-first
    ranked = sorted(zip(distances, metas), key=lambda x: x[0])

    seen: set[str] = set()
    top_urls: list[str] = []
    for _dist, meta in ranked[:n]:
        url = meta.get("url", "").strip()
        if url:
            key = url.lower().rstrip("/")
            if key not in seen:
                seen.add(key)
                top_urls.append(url)

    return top_urls


def _is_out_of_scope(results: dict) -> bool:
    """
    Return True if ALL retrieved documents are too far from the query,
    meaning the question is outside the dataset's scope.
    ChromaDB returns cosine distances in results['distances'][0].

    IMPORTANT: If distances are not returned (old chromadb version or include
    param unsupported), we assume IN-scope and let the LLM decide — the
    LLM-signal check in generate_answer() acts as the second guard.
    """
    raw = results.get("distances")
    if not raw or not isinstance(raw, list) or not raw[0]:
        # Distances unavailable → assume in-scope, rely on LLM guard instead
        return False
    distances = raw[0]
    best = min(distances)
    print(f"[scope] best cosine distance = {best:.4f} (threshold={RELEVANCE_THRESHOLD})")
    return best > RELEVANCE_THRESHOLD


# ---------------------------------------------------------------------------
# Conversation history formatting
# ---------------------------------------------------------------------------
def _format_conversation_history(history: list | None) -> str:
    """Format up to the last 4 turns as a compact block for the prompt.
    Returns an empty string when there is no history so the caller can
    skip the section header entirely and save tokens.
    """
    if not history:
        return ""
    turns = []
    for idx, turn in enumerate(history, start=1):
        user_q = str(turn.get("user", "")).strip()
        bot_a  = str(turn.get("bot", "")).strip()
        if user_q or bot_a:
            turns.append(f"Turn {idx}:\nUser: {user_q}\nAssistant: {bot_a}")
    return "\n\n".join(turns)


# ---------------------------------------------------------------------------
# RunPod vLLM call
# ---------------------------------------------------------------------------
def _call_runpod(prompt: str, max_tokens: int = 600) -> str:
    """
    Send the fully-assembled RAG prompt to the RunPod vLLM serverless worker
    and return ONLY the new generated text (completion, not the echoed prompt).

    IMPORTANT: RunPod worker-vllm reads generation parameters directly from
    the top level of the "input" dict.  Nesting them inside "sampling_params"
    is silently ignored by the current worker, causing it to fall back to its
    hardcoded default of 100 tokens — which is why responses were truncated
    even after setting the DEFAULT_MAX_TOKENS env variable on RunPod.
    We also send "sampling_params" as a secondary key for forward-compat with
    any future worker build that switches to that schema.

    RunPod vLLM worker response formats:
      Newer: {"output": {"choices": [{"text": "<prompt+completion>", ...}]}}
      Older: {"output": [{"choices": [{"text": "<prompt+completion>"}]}]}
    vLLM always echoes the full prompt in the text field, so we strip it.
    """
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "prompt": prompt,
            # ── Primary: top-level keys read by worker-vllm ──────────────────
            "max_tokens": max_tokens,
            "max_new_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            # ── Secondary: sampling_params for forward-compat ────────────────
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
            },
        }
    }

    try:
        resp = requests.post(_RUNPOD_SYNC_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[RunPod] Request error: {exc}")
        return ""

    # Top-level error
    if isinstance(data, dict) and data.get("error"):
        print(f"[RunPod] Error from endpoint: {data['error']}")
        return ""

    # ── Handle async / cold-start: runsync may return IN_QUEUE or IN_PROGRESS
    #    instead of the completed result when the worker is cold-starting or
    #    the job takes longer than RunPod's runsync timeout.
    #    In that case, poll /status/{job_id} until COMPLETED.
    status = str(data.get("status", "")).upper()
    job_id = data.get("id", "")

    if status in ("IN_QUEUE", "IN_PROGRESS") and job_id:
        # Derive the base endpoint URL (strip /runsync or /run suffix)
        base_url = _RUNPOD_SYNC_URL.rstrip("/")
        for suffix in ("/runsync", "/run"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break

        print(f"[RunPod] Job queued ({job_id}), polling for completion...")
        deadline = time.time() + 300  # wait up to 5 minutes
        while time.time() < deadline:
            time.sleep(3)
            try:
                poll_resp = requests.get(
                    f"{base_url}/status/{job_id}",
                    headers=headers,
                    timeout=30,
                )
                poll_resp.raise_for_status()
                data = poll_resp.json()
            except Exception as poll_exc:
                print(f"[RunPod] Polling error: {poll_exc}")
                return ""

            poll_status = str(data.get("status", "")).upper()
            print(f"[RunPod] Job status: {poll_status}")

            if poll_status == "COMPLETED":
                break
            if poll_status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                print(f"[RunPod] Job ended with status: {poll_status}")
                return ""
        else:
            print(f"[RunPod] Timed out waiting for job {job_id}")
            return ""

    output = data.get("output", data)

    full_text = _extract_vllm_text(output)
    if not full_text:
        print(f"[RunPod] Unexpected output structure: {data}")
        return ""

    # vLLM echoes the full prompt — strip it so we get only the completion.
    if full_text.startswith(prompt):
        completion = full_text[len(prompt):]
    else:
        # Prompt echo not found at start (shouldn't normally happen).
        # Return as-is; generate_answer() will still prepend "Answer:\n".
        completion = full_text

    return completion.strip()


def _extract_vllm_text(output) -> str:
    """
    Extract the raw generated text from any known RunPod vLLM output shape.

    RunPod vLLM worker (worker-vllm) actual format:
      {"output": [{"choices": [{"tokens": ["tok1", "tok2", ...]}], "usage": {...}}]}
    Text lives in choices[0]["tokens"] as a LIST that must be joined.
    Some builds use choices[0]["text"] as a plain string instead.
    """
    if isinstance(output, str):
        return output.strip()

    def _from_choice(choice: dict) -> str:
        # "text" key — plain string (newer worker builds)
        text = choice.get("text")
        if text and isinstance(text, str):
            return text.strip()
        # "tokens" key — list of strings (current RunPod vLLM worker)
        tokens = choice.get("tokens")
        if isinstance(tokens, list) and tokens:
            return "".join(str(t) for t in tokens).strip()
        return ""

    # ── 1. List output: [{"choices": [...], "usage": {...}}]  ────────────────
    #    This is what the current RunPod vLLM worker actually returns.
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            choices = first.get("choices")
            if isinstance(choices, list) and choices:
                txt = _from_choice(choices[0] if isinstance(choices[0], dict) else {})
                if txt:
                    return txt
            # Direct text keys on the list item
            for key in ("text", "generated_text", "content"):
                val = first.get(key)
                if val and isinstance(val, str):
                    return val.strip()

    # ── 2. Dict output: {"choices": [...]}  ─────────────────────────────────
    if isinstance(output, dict):
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            txt = _from_choice(choices[0] if isinstance(choices[0], dict) else {})
            if txt:
                return txt
        # Direct text keys on the dict
        for key in ("text", "generated_text", "response", "content"):
            val = output.get(key)
            if val:
                if isinstance(val, list) and val:
                    return "".join(str(t) for t in val).strip()
                if isinstance(val, str):
                    return val.strip()

    return ""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def _parse_llm_output(
    raw_text: str,
    fallback_sources: list[str],
    context_urls: set[str] | None = None,
) -> tuple[str, list[str], list[str]]:
    """
    Parse the LLM completion into (answer_text, source_urls, followup_questions).

    context_urls: normalised set of URLs extracted from the retrieved context
    passages.  Any URL the LLM emits that is NOT in this set is silently
    dropped — this prevents hallucinated or training-data links from leaking
    into the response regardless of what the model outputs.
    """
    text = raw_text.strip()

    # ── Answer ──
    answer_match = re.search(
        r"Answer:\s*(.*?)(?:\n\s*Sources:|\n\s*Follow-up Questions|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    answer_text = answer_match.group(1).strip() if answer_match else text

    # ── Sources ──
    sources_match = re.search(
        r"Sources:\s*(.*?)(?:\n\s*Follow-up Questions|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    raw_urls: list[str] = []
    if sources_match:
        block = sources_match.group(1)
        raw_urls = re.findall(r"https?://[^\s)\]>\"]+", block)
        if not raw_urls:
            for line in block.splitlines():
                line = line.strip().lstrip("-•* ").strip()
                if line:
                    raw_urls.append(line)

    # ── Allowlist filter ──────────────────────────────────────────────────
    # Keep only URLs that literally appeared in the retrieved context passages.
    # This is the hard enforcement layer — the system prompt alone is not
    # sufficient because models can still hallucinate URLs.
    if context_urls:
        source_urls = [
            u for u in raw_urls
            if u.lower().rstrip("/") in context_urls
        ]
        if not source_urls and raw_urls:
            print(f"[sources] LLM produced {len(raw_urls)} URL(s) not in context — discarded")
    else:
        source_urls = raw_urls

    # Fall back to ChromaDB-derived URLs if nothing valid remains
    if not source_urls:
        source_urls = fallback_sources

    # ── Follow-up questions ──
    fup_match = re.search(
        r"Follow-up Questions.*?:\s*(.*)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    followups: list[str] = []
    if fup_match:
        fup_block = fup_match.group(1)
        for line in fup_block.splitlines():
            line = re.sub(r"^[-•*Q\d.]+\s*", "", line.strip()).strip()
            if line:
                followups.append(line)

    return answer_text, source_urls, followups


# ---------------------------------------------------------------------------
# Context-phrase sanitisation
# ---------------------------------------------------------------------------
_CONTEXT_PHRASES = re.compile(
    r"\b(in the provided context|based on the (provided )?context|"
    r"(as |as per |according to the )?(provided |given )?context|"
    r"mentioned in the context|not mentioned in the context|"
    r"the context (does not|doesn't) (mention|contain|include|provide))\b",
    re.IGNORECASE,
)

_OUT_OF_SCOPE_SIGNALS = re.compile(
    r"\b(I (don't|do not|cannot|can't) (find|have|provide|answer|help with|address)|"
    r"not (in|within|part of|covered by) (my|the) (knowledge|context|database|scope)|"
    r"outside (my|the) (knowledge|scope|context)|"
    r"no information (about|on|regarding)|"
    r"I('m| am) not (able|designed|trained|equipped))\b",
    re.IGNORECASE,
)


def _sanitize_answer(text: str) -> str:
    """Remove context-reference phrases and trim whitespace."""
    text = _CONTEXT_PHRASES.sub("", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _answer_is_out_of_scope(answer_text: str) -> bool:
    """Return True if the LLM itself signalled it had no relevant answer."""
    return bool(_OUT_OF_SCOPE_SIGNALS.search(answer_text))


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------
def _build_response(answer_text: str, source_urls: list[str], followups: list[str]) -> str:
    """
    Assemble the final 3-section response with the exact format:

        Answer:
        <text>

        Sources:
        - <url>

        Follow-up Questions you might have:
        Q1. <question>
        Q2. <question>
        Q3. <question>
    """
    # Deduplicate sources
    seen: set[str] = set()
    clean_sources: list[str] = []
    for s in source_urls:
        key = s.strip().lower().rstrip("/")
        if key and key not in seen:
            seen.add(key)
            clean_sources.append(s.strip())
    if not clean_sources:
        clean_sources = ["No relevant source available"]

    # Deduplicate & cap follow-ups
    seen_fup: set[str] = set()
    clean_fups: list[str] = []
    for q in followups:
        q = q.strip()
        if q and q.lower() not in seen_fup:
            seen_fup.add(q.lower())
            clean_fups.append(q)
        if len(clean_fups) == 3:
            break

    # Default follow-ups if LLM produced none — kept intentionally vague so
    # they don't mislead; the real fix is the system-prompt rule that forces
    # the LLM to always generate on-topic questions.
    if not clean_fups:
        clean_fups = [
            "Can you tell me more about this topic?",
            "Where can I find more detailed information about this?",
            "Who should I contact if I have more questions about this?",
        ]

    sources_block = "\n".join(f"- {s}" for s in clean_sources)
    fups_block    = "\n".join(f"Q{i}. {q}" for i, q in enumerate(clean_fups, 1))

    clean_answer = _sanitize_answer(answer_text)
    if not clean_answer:
        return OUT_OF_SCOPE_RESPONSE

    return (
        f"Answer:\n{clean_answer}\n\n"
        f"Sources:\n{sources_block}\n\n"
        f"Follow-up Questions you might have:\n{fups_block}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_answer(
    query: str,
    return_context: bool = False,
    conversation_history: list | None = None,
) -> tuple:
    """
    Full RAG pipeline:
      1. Embed query locally
      2. Retrieve top-K passages from ChromaDB (with distances)
      3. Check relevance threshold → return out-of-scope if needed
      4. Assemble strict prompt
      5. Call RunPod vLLM for completion
      6. Parse & format 3-section response
    """

    # ── 1. Embed & Retrieve ──────────────────────────────────────────────────
    query_emb = get_query_embedding(query)
    results   = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    # ── 2. Out-of-scope check (distance-based) ───────────────────────────────
    if _is_out_of_scope(results):
        print(f"[scope] Query out of scope (min dist={min(results['distances'][0], default=99):.3f}): {query!r}")
        if return_context:
            return OUT_OF_SCOPE_RESPONSE, [], []
        return OUT_OF_SCOPE_RESPONSE, []

    context_str, sources = format_context(results)

    if not context_str.strip():
        if return_context:
            return OUT_OF_SCOPE_RESPONSE, [], []
        return OUT_OF_SCOPE_RESPONSE, []

    # ── 3. Build prompt using Mistral [INST] chat template ───────────────────
    history_text = _format_conversation_history(conversation_history)

    system_prompt = """\
You are a college-planning assistant for the Track2College platform.

STRICT RULES — follow every rule without exception:
1. Use ONLY information from the CONTEXT PASSAGES below. Do NOT use your own knowledge.
2. Do NOT invent facts, statistics, or URLs.
3. If the question cannot be answered from the context, write exactly:
   "I'm sorry, I can only help with topics covered in my knowledge base. Please contact your school counselor or visit collegeboard.org."
4. Do NOT say "based on the context", "the context mentions", or similar phrases.
5. Write in plain, friendly language a high school student can understand.
6. Use bullet points (•) or numbered steps when listing multiple items in the answer.
7. SOURCES — obey every sub-rule:
   a. List ONLY URLs that literally appear in the CONTEXT PASSAGES provided below.
   b. Do NOT fabricate, guess, or paraphrase any URL.
   c. Do NOT include any URL from your training data or general knowledge.
   d. If a URL was NOT shown in the context passages, you MUST omit it entirely.
   e. Only include a source if it is directly relevant to the specific answer you gave.
8. Follow-up questions rules (follow ALL of these):
   a. Each follow-up MUST be directly related to the user's current question and the specific
      topic of your answer — do NOT suggest unrelated college topics.
   b. Do NOT include any question that is already answered (fully or partially) in your Answer
      section. Only ask things the student still doesn't know after reading your answer.
   c. Aim for 1–3 questions. It is perfectly fine to write only 1 or 2 if fewer genuinely
      unanswered follow-ups exist. Never exceed 5.
   d. Each follow-up should be a natural next question a student would ask after reading your answer.

REQUIRED OUTPUT FORMAT — copy these exact headers, in this exact order.
Each follow-up question MUST be on its own separate line:

Answer:
<clear, well-structured answer with bullet points where helpful>

Sources:
- <url from context>
- <url from context>

Follow-up Questions you might have:
Q1. <unanswered follow-up — same topic as current question>
Q2. <unanswered follow-up — same topic as current question>  (omit if nothing genuine remains)
Q3. <unanswered follow-up — same topic as current question>  (omit if nothing genuine remains)
EXAMPLE of correct format (content is fictional — note how all 3 follow-ups stay on-topic):
Answer:
To apply for financial aid you need to:
• Complete the FAFSA at studentaid.gov before your state's deadline.
• Gather your family's tax returns and Social Security numbers beforehand.
• Submit the form separately to every college on your list.

Sources:
- https://studentaid.gov/apply-for-aid/fafsa/filling-out

Follow-up Questions you might have:
Q1. What is the FAFSA deadline for my state?
Q2. Do I need to reapply for financial aid every year?
Q3. What happens if my family's income changed since filing taxes?
"""

    # Only include the history block when there are prior turns — omitting it
    # when empty avoids wasting tokens on a "None" placeholder.
    if history_text:
        history_section = f"=== Conversation History ===\n{history_text}\n\n"
    else:
        history_section = ""

    user_message = (
        f"{history_section}"
        f"=== Context Passages ===\n{context_str}\n"
        f"=== Current Question ===\n{query}"
    )

    # Mistral-Instruct chat template — dramatically improves instruction following.
    # We prime the response with "Answer:\n" so the model starts the answer body directly.
    full_prompt = f"<s>[INST] {system_prompt}\n\n{user_message} [/INST] Answer:\n"

    # ── 4. Call RunPod vLLM ──────────────────────────────────────────────────
    raw_completion = _call_runpod(full_prompt, max_tokens=1000)

    if not raw_completion:
        print("[RunPod] Empty completion — returning fallback")
        if return_context:
            return OUT_OF_SCOPE_RESPONSE, sources, results["documents"][0]
        return OUT_OF_SCOPE_RESPONSE, sources

    # _call_runpod strips the prompt echo; raw_completion is the answer body.
    # Prepend the "Answer:" header so _parse_llm_output can locate sections.
    llm_output = "Answer:\n" + raw_completion

    # Build a normalised allowlist of URLs from the retrieved context passages.
    # _parse_llm_output uses this to discard any URL the LLM hallucinated.
    context_url_allowlist: set[str] = {
        url.lower().rstrip("/") for url in sources
    }

    # Fallback source list: only the top-3 most relevant passages by cosine
    # distance.  Using all TOP_K=6 passages would surface topically-adjacent
    # but off-topic URLs (e.g. financial-aid pages for an SCU-admissions query).
    top_sources = _top_sources_from_results(results, n=3)

    # ── 5. Parse sections ────────────────────────────────────────────────────
    answer_text, llm_sources, followups = _parse_llm_output(
        llm_output, top_sources, context_urls=context_url_allowlist
    )

    # ── 6. Check if LLM signalled out-of-scope ───────────────────────────────
    if _answer_is_out_of_scope(answer_text):
        print(f"[scope] LLM signalled out-of-scope for: {query!r}")
        if return_context:
            return OUT_OF_SCOPE_RESPONSE, sources, results["documents"][0]
        return OUT_OF_SCOPE_RESPONSE, sources

    # ── 7. Assemble final structured response ────────────────────────────────
    final_sources = llm_sources if llm_sources else top_sources
    response      = _build_response(answer_text, final_sources, followups)

    if return_context:
        return response, final_sources, results["documents"][0]
    return response, final_sources


# ---------------------------------------------------------------------------
# Convenience wrapper used by api.py
# ---------------------------------------------------------------------------
def get_answer(question: str) -> str:
    response, _ = generate_answer(question)
    return response


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------
def _answer_text_only(full_response: str) -> str:
    """Extract just the Answer section text for compact memory storage.
    Storing only the answer (not Sources/Follow-ups) keeps history turns
    concise so they don't consume the model's limited context window.
    """
    match = re.search(
        r"Answer:\s*(.*?)(?:\n\s*Sources:|\n\s*Follow-up Questions|\Z)",
        full_response,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else full_response.strip()


def _extract_followup_questions(full_response: str) -> list[str]:
    """Parse the Follow-up Questions block from a formatted response.
    Returns a list of question strings (without the Q1./Q2. prefix).
    """
    match = re.search(
        r"Follow-up Questions.*?:\s*(.*)$",
        full_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    questions: list[str] = []
    for line in match.group(1).splitlines():
        line = re.sub(r"^[-•*Q\d.]+\s*", "", line.strip()).strip()
        if line:
            questions.append(line)
    return questions


# Matches user inputs like: "Q1", "answer Q2", "Q3 please", "the second one",
# "first question", "tell me more about Q3", "2nd follow-up", etc.
_FOLLOWUP_REF_RE = re.compile(
    r"""
    (?:                          # optional leading verb phrase
        (?:answer|tell\s+me(?:\s+about)?|explain|elaborate\s+on|what\s+about|give\s+me)\s+
    )?
    (?:
        [Qq](?P<qnum>[1-5])\b               # Q1 … Q5
      | (?P<word>first|second|third|fourth|fifth)  # "the first one"
        \s+(?:question|one|follow[- ]?up)?
      | (?P<num>[1-5])(?:st|nd|rd|th)?      # "1st question" / "2"
        \s+(?:question|one|follow[- ]?up)?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_WORD_TO_IDX = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}


def _resolve_followup_query(user_input: str, memory: list) -> str:
    """
    If the user's input is a shorthand reference to a previous follow-up
    question (e.g. "Q1", "answer Q2", "the second one"), resolve it to the
    actual question text stored in the most recent memory turn.

    Returns the original user_input unchanged when:
    - no follow-up reference is detected
    - the referenced question index is out of range
    - no follow-up questions are stored in memory
    """
    stripped = user_input.strip()
    m = _FOLLOWUP_REF_RE.search(stripped)
    if not m:
        return user_input

    # Determine 0-based index
    if m.group("qnum"):
        idx = int(m.group("qnum")) - 1
    elif m.group("word"):
        idx = _WORD_TO_IDX.get(m.group("word").lower(), -1)
    elif m.group("num"):
        idx = int(m.group("num")) - 1
    else:
        return user_input

    if idx < 0:
        return user_input

    # Find the most recent memory turn that has stored follow-up questions
    followups: list[str] = []
    for turn in reversed(memory):
        if isinstance(turn, dict) and turn.get("followups"):
            followups = turn["followups"]
            break

    if followups and 0 <= idx < len(followups):
        resolved = followups[idx]
        print(f"[memory] Follow-up reference '{stripped}' → '{resolved}'")
        return resolved

    return user_input


if __name__ == "__main__":
    conversation_memory: deque = deque(maxlen=4)

    print("Track2College chatbot — type 'quit' to exit.")
    print("Conversation memory: last 4 turns")
    print("Tip: type 'Q1', 'Q2', or 'Q3' to ask a follow-up question from the previous answer.\n")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"): 
            break
        if not user_input:
            continue

        try:
            memory_snapshot = list(conversation_memory)

            # Resolve shorthand follow-up references (Q1/Q2/Q3, "first one", etc.)
            resolved_input = _resolve_followup_query(user_input, memory_snapshot)

            answer, answer_sources = generate_answer(resolved_input, conversation_history=memory_snapshot)
            print("\n" + answer)

            # Store: compact answer text + the extracted follow-up questions so
            # the next turn can resolve Q1/Q2/Q3 references correctly.
            conversation_memory.append({
                "user": resolved_input,
                "bot": _answer_text_only(answer),
                "followups": _extract_followup_questions(answer),
            })
        except Exception as exc:
            print(f"[error] {exc}")
