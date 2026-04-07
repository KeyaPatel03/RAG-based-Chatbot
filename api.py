from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Deque, Dict, List, Optional
import os
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

try:
    from query import generate_answer, get_answer
except ImportError as error:
    print(f"Error importing query module: {error}")
    sys.exit(1)


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

if not API_KEY:
    raise RuntimeError("Missing API_KEY in .env")

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
    timestamp_utc: str


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

    try:
        raw_response, sources = generate_answer(
            query_text,
            conversation_history=memory_snapshot,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {error}")

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    with _session_lock:
        session_memory.append({"user": query_text, "bot": raw_response})

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
        answer = get_answer(question)
        return AnalyzeResponse(answer=answer)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)