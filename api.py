from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Deque, Dict, List, Optional
import os
import requests
import secrets
import sys
import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure existing modules can be imported.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(dotenv_path: str) -> None:
    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path, "r", encoding="utf-8") as dotenv_file:
        for raw_line in dotenv_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_load_dotenv(os.path.join(BASE_DIR, ".env"))
API_KEY = os.getenv("API_KEY", "").strip()
INFERENCE_PROVIDER_ENV = os.getenv("INFERENCE_PROVIDER", "auto").strip().lower()
RUNPOD_API_KEY = (
    os.getenv("RUNPOD_API_KEY", "").strip()
    or os.getenv("RunPod_API_Key", "").strip()
    or os.getenv("RUNPODAPI_KEY", "").strip()
    or os.getenv("runpodapi_key", "").strip()
)
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "").strip()
RUNPOD_ENDPOINT_URL = os.getenv("RUNPOD_ENDPOINT_URL", "").strip()
RUNPOD_TIMEOUT_SEC = float(os.getenv("RUNPOD_TIMEOUT_SEC", "90").strip())
RUNPOD_ASYNC_MAX_WAIT_SEC = float(os.getenv("RUNPOD_ASYNC_MAX_WAIT_SEC", "300").strip())
RUNPOD_POLL_INTERVAL_SEC = float(os.getenv("RUNPOD_POLL_INTERVAL_SEC", "2").strip())
RUNPOD_PAYLOAD_MODE = os.getenv("RUNPOD_PAYLOAD_MODE", "auto").strip().lower()


def _resolve_runpod_url() -> str:
    if RUNPOD_ENDPOINT_URL:
        url = RUNPOD_ENDPOINT_URL.rstrip("/")
        if url.endswith("/run"):
            return f"{url[:-4]}/runsync"
        if url.endswith("/runsync"):
            return url
        return url
    if RUNPOD_ENDPOINT_ID:
        if RUNPOD_ENDPOINT_ID.startswith("http://") or RUNPOD_ENDPOINT_ID.startswith("https://"):
            url = RUNPOD_ENDPOINT_ID.rstrip("/")
            if url.endswith("/run"):
                return f"{url[:-4]}/runsync"
            if url.endswith("/runsync"):
                return url
            return url
        return f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"
    return ""


RUNPOD_URL = _resolve_runpod_url()

if not API_KEY:
    raise RuntimeError("Missing API_KEY in .env")

if INFERENCE_PROVIDER_ENV not in {"auto", "local", "runpod"}:
    raise RuntimeError("INFERENCE_PROVIDER must be one of: auto, local, runpod")

if RUNPOD_PAYLOAD_MODE not in {"auto", "handler", "vllm"}:
    raise RuntimeError("RUNPOD_PAYLOAD_MODE must be one of: auto, handler, vllm")

if INFERENCE_PROVIDER_ENV == "auto":
    INFERENCE_PROVIDER = "runpod" if RUNPOD_API_KEY and RUNPOD_URL else "local"
else:
    INFERENCE_PROVIDER = INFERENCE_PROVIDER_ENV

if INFERENCE_PROVIDER == "runpod" and (not RUNPOD_API_KEY or not RUNPOD_URL):
    raise RuntimeError(
        "RunPod mode requires RUNPOD_API_KEY (or RunPod_API_Key) and RUNPOD_ENDPOINT_ID or RUNPOD_ENDPOINT_URL"
    )

generate_answer = None
get_answer = None

try:
    from query import (
        generate_answer,
        get_answer,
        _answer_text_only,
        _extract_followup_questions,
        _resolve_followup_query,
    )
except ImportError as error:
    print(f"Error importing query module: {error}")
    sys.exit(1)

app = FastAPI(
    title="Track2College Chatbot API",
    version="1.0.0",
    description="RAG chatbot API for frontend integration.",
)

# Allow local frontend apps (including file:// pages with Origin: null).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://127.0.0.1",
        "http://localhost",
        "http://127.0.0.1:5500",
        "http://0.0.0.0:5500",
        "http://localhost:5500",
        "null", 
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_CONVERSATION_TURNS = 4
_session_lock = Lock()
_session_store: Dict[str, Deque[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    session_id: Optional[str] = Field(
        None,
        description="Optional chat session id. Create one if omitted.",
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[str]
    raw_response: str
    response_time_ms: int
    timestamp_utc: str


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class AnalyzeResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str
    service: str
    inference_provider: str
    timestamp_utc: str


def _call_runpod_generate(query_text: str, conversation_history: Optional[List[Dict[str, str]]] = None):
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    payload_candidates = _build_runpod_payload_candidates(query_text, conversation_history)
    last_error = "RunPod request failed"

    for mode_name, payload in payload_candidates:
        try:
            body = _invoke_runpod(payload, headers)
        except Exception as error:
            last_error = f"{mode_name}: {error}"
            continue

        answer, sources, parse_error = _extract_runpod_answer(body)
        if answer:
            return answer, sources

        if parse_error:
            last_error = f"{mode_name}: {parse_error}"

    raise RuntimeError(last_error)


def _invoke_runpod(payload: Dict, headers: Dict[str, str]):
    try:
        response = requests.post(
            RUNPOD_URL,
            json=payload,
            headers=headers,
            timeout=RUNPOD_TIMEOUT_SEC,
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        base_url = RUNPOD_URL.rstrip("/")
        if base_url.endswith("/runsync"):
            base_url = base_url[: -len("/runsync")]
        elif base_url.endswith("/run"):
            base_url = base_url[: -len("/run")]

        run_response = requests.post(
            f"{base_url}/run",
            json=payload,
            headers=headers,
            timeout=min(30, RUNPOD_TIMEOUT_SEC),
        )
        run_response.raise_for_status()
        run_body = run_response.json()
        job_id = run_body.get("id")
        if not job_id:
            raise RuntimeError("RunPod async fallback did not return a job id")

        deadline = time.time() + RUNPOD_ASYNC_MAX_WAIT_SEC
        while True:
            status_response = requests.get(
                f"{base_url}/status/{job_id}",
                headers=headers,
                timeout=min(30, RUNPOD_TIMEOUT_SEC),
            )
            status_response.raise_for_status()
            body = status_response.json()
            status = str(body.get("status", "")).upper()

            if status == "COMPLETED":
                break
            if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                error_detail = body.get("error") if isinstance(body, dict) else None
                if error_detail:
                    raise RuntimeError(f"RunPod async job ended with status {status}: {error_detail}")
                raise RuntimeError(f"RunPod async job ended with status: {status}")
            if time.time() > deadline:
                raise RuntimeError("RunPod async job timed out while waiting for completion")

            time.sleep(max(0.5, RUNPOD_POLL_INTERVAL_SEC))

        return body


def _format_history_for_vllm(conversation_history: Optional[List[Dict[str, str]]]) -> str:
    if not conversation_history:
        return ""

    lines = []
    for idx, turn in enumerate(conversation_history, start=1):
        user_text = str(turn.get("user", "")).strip()
        bot_text = str(turn.get("bot", "")).strip()
        if user_text:
            lines.append(f"Turn {idx} User: {user_text}")
        if bot_text:
            lines.append(f"Turn {idx} Assistant: {bot_text}")
    return "\n".join(lines)


def _build_runpod_payload_candidates(
    query_text: str,
    conversation_history: Optional[List[Dict[str, str]]],
):
    handler_payload = {
        "input": {
            "query": query_text,
            "session_memory": conversation_history or [],
        }
    }

    history_text = _format_history_for_vllm(conversation_history)
    if history_text:
        prompt = (
            "You are a helpful college assistant. Use conversation history when relevant.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"User question: {query_text}\n"
            "Assistant:"
        )
    else:
        prompt = query_text

    vllm_payload = {
        "input": {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9,
        }
    }

    if RUNPOD_PAYLOAD_MODE == "handler":
        return [("handler", handler_payload)]
    if RUNPOD_PAYLOAD_MODE == "vllm":
        return [("vllm", vllm_payload)]
    return [("handler", handler_payload), ("vllm", vllm_payload)]


def _extract_runpod_answer(body: Dict):
    if not isinstance(body, dict):
        text = str(body).strip()
        if text:
            return text, [], None
        return "", [], "RunPod returned an empty, non-dict payload"

    status = str(body.get("status", "")).upper().strip()
    error_detail = body.get("error")

    if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
        return "", [], f"RunPod job status {status}: {error_detail}"

    output = body.get("output")
    if output is None:
        output = body

    if isinstance(output, dict) and output.get("error"):
        return "", [], str(output.get("error"))

    if isinstance(output, dict):
        answer = str(output.get("answer") or output.get("raw_response") or "").strip()
        sources = output.get("sources") if isinstance(output.get("sources"), list) else []
        if answer:
            return answer, sources, None

        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            choice_text = str(first_choice.get("text") or "").strip()
            if not choice_text:
                message = first_choice.get("message")
                if isinstance(message, dict):
                    choice_text = str(message.get("content") or "").strip()
            if not choice_text:
                tokens = first_choice.get("tokens")
                if isinstance(tokens, list):
                    choice_text = "".join([str(tok) for tok in tokens]).strip()
            if choice_text:
                return choice_text, [], None

        for key in ("text", "generated_text", "response", "content"):
            value = output.get(key)
            if value is None:
                continue
            rendered = str(value).strip()
            if rendered:
                return rendered, [], None

    if isinstance(output, list) and output:
        if isinstance(output[0], dict):
            txt = str(output[0].get("text") or output[0].get("generated_text") or "").strip()
            if not txt:
                choices = output[0].get("choices")
                if isinstance(choices, list) and choices:
                    first_choice = choices[0] if isinstance(choices[0], dict) else {}
                    txt = str(first_choice.get("text") or "").strip()
                    if not txt:
                        tokens = first_choice.get("tokens")
                        if isinstance(tokens, list):
                            txt = "".join([str(tok) for tok in tokens]).strip()
            if txt:
                return txt, [], None
        rendered = str(output[0]).strip()
        if rendered:
            return rendered, [], None

    rendered_output = str(output).strip()
    if rendered_output and rendered_output != "{}":
        return rendered_output, [], None

    return "", [], f"RunPod returned an empty answer payload: {body}"


def _generate_answer_backend(query_text: str, conversation_history: Optional[List[Dict[str, str]]] = None):
    return generate_answer(query_text, conversation_history=conversation_history)


def _get_answer_backend(question: str) -> str:
    answer, _ = _generate_answer_backend(question, conversation_history=[])
    return answer


def _verify_api_key(authorization: Optional[str]) -> None:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    if not secrets.compare_digest(token.strip(), API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _get_or_create_session_id(session_id: Optional[str]) -> str:
    if session_id and session_id.strip():
        return session_id.strip()
    return str(uuid.uuid4())


def _get_session_memory(session_id: str) -> Deque[Dict[str, str]]:
    with _session_lock:
        if session_id not in _session_store:
            _session_store[session_id] = deque(maxlen=MAX_CONVERSATION_TURNS)
        return _session_store[session_id]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="track2college-rag-api",
        inference_provider=INFERENCE_PROVIDER,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the chatbot page at "/"
@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = _get_or_create_session_id(request.session_id)
    session_memory = _get_session_memory(session_id)
    memory_snapshot = list(session_memory)
    start_time = time.perf_counter()

    # Resolve shorthand follow-up references (e.g. "Q1", "answer Q2",
    # "the second one") to the actual stored question text before the
    # RAG pipeline runs.
    resolved_query = _resolve_followup_query(query_text, memory_snapshot)

    try:
        raw_response, sources = _generate_answer_backend(
            resolved_query,
            conversation_history=memory_snapshot,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {error}")

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    with _session_lock:
        # Store: compact answer text + extracted follow-up questions so the
        # next request can resolve Q1/Q2/Q3 references correctly.
        # Use the resolved query as the stored user turn so the conversation
        # history stays semantically complete.
        session_memory.append({
            "user": resolved_query,
            "bot": _answer_text_only(raw_response),
            "followups": _extract_followup_questions(raw_response),
        })

    return ChatResponse(
        session_id=session_id,
        answer=raw_response,
        sources=sources,
        raw_response=raw_response,
        response_time_ms=elapsed_ms,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    with _session_lock:
        existed = session_id in _session_store
        _session_store.pop(session_id, None)
    return {
        "session_id": session_id,
        "cleared": existed,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    authorization: Optional[str] = Header(default=None),
) -> AnalyzeResponse:
    _verify_api_key(authorization)

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer = _get_answer_backend(question)
        return AnalyzeResponse(answer=answer)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)