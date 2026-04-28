"""
RunPod serverless handler.

NOTE: This file is only used if you deploy a CUSTOM RunPod handler image.
If you are using the RunPod vLLM Serverless template (pre-built worker),
this file is NOT executed on RunPod — the vLLM worker template handles
inference directly.

For the vLLM template setup, the local query.py sends the fully-assembled
RAG prompt straight to the RunPod vLLM endpoint and parses the response.
No custom handler deployment is needed.
"""

import runpod
import os
import sys
import traceback


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure parent directory is in path for relative imports
sys.path.append(_project_root())

try:
    from query import generate_answer
except ImportError as error:
    print(f"Error importing query module: {error}")
    traceback.print_exc()
    sys.exit(1)

def _normalize_session_memory(raw_session_memory):
    if not isinstance(raw_session_memory, list):
        return []

    normalized = []
    for turn in raw_session_memory:
        if not isinstance(turn, dict):
            continue
        normalized.append({
            "user": str(turn.get("user", "")).strip(),
            "bot": str(turn.get("bot", "")).strip(),
        })
    return normalized


def _error_payload(error: Exception):
    return {
        "error": {
            "message": str(error),
            "type": error.__class__.__name__,
        }
    }

def handler(job):
    """
    Runpod serverless handler.
    `job` is a dictionary containing the payload.
    Expected schema in job['input']:
    {
        "query": "What courses are available?",
        "session_memory": []
    }
    """
    job_input = job.get("input") if isinstance(job, dict) else {}
    if not isinstance(job_input, dict):
        job_input = {}

    query_text = str(job_input.get("query", "")).strip()
    session_memory = _normalize_session_memory(
        job_input.get("session_memory") or job_input.get("conversation_history")
    )
    return_context = bool(job_input.get("return_context", False))

    if not query_text:
        return _error_payload(ValueError("Query cannot be empty"))

    try:
        if return_context:
            raw_response, sources, context_documents = generate_answer(
                query_text,
                return_context=True,
                conversation_history=session_memory,
            )
            return {
                "answer": raw_response,
                "sources": sources,
                "raw_response": raw_response,
                "context_documents": context_documents,
            }

        raw_response, sources = generate_answer(
            query_text,
            conversation_history=session_memory,
        )

        return {
            "answer": raw_response,
            "sources": sources,
            "raw_response": raw_response,
        }

    except Exception as error:
        print(f"RunPod handler error: {error}")
        traceback.print_exc()
        return _error_payload(error)

# Start the Runpod serverless function
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
