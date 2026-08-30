"""FastAPI backend that puts the agentic verifier behind an HTTP API.

Designed for the course setup, exactly like WE6: **each participant runs this locally**
and **supplies their own OpenRouter API key**, which the frontend sends on every request
in the ``X-OpenRouter-Key`` header. The key builds the AI clients for that one request and
is never written to disk or logged.

Run it:

    uv run uvicorn server:app --reload

The policy store is a single persistent Qdrant (``./qdrant_data`` by default) shared for the
life of the process. It is opened on startup but **not** populated there: embedding the
corpus needs a real key/model, and the server has none of its own. The bundled policy docs
are therefore ingested lazily, on the first ``/verify`` request, with that caller's
embedder; every later request finds the store already populated and skips ingestion. A
fresh ``VerifierPipeline`` is built per request over the shared store (the injected ``db``
plus hydrate-on-init mean it immediately sees everything already ingested), so the
read-only endpoints work without a key as soon as the corpus is in the store.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager, contextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AuthenticationError, OpenAIError
from pydantic import BaseModel

import solutions
from agentic.action_types import ACTION_TYPES
from agentic.models import Action
from agentic.pipeline import ActionValidationError, VerifierPipeline
from clients import build_clients
from rag.db_manager import DBManager

# Run the app on the participant's exported solutions where present; fall back to the
# reference implementation otherwise. AGENT_IMPL=student is swallowed (like WE6) so the
# server still boots and /status shows exactly which functions are still missing.
try:
    solutions.apply()
except RuntimeError:
    pass

# Optionally run policy retrieval on the participant's own WE6 code (no-op otherwise).
import rag.solutions  # noqa: E402

rag.solutions.apply(verbose=False)

DB_PATH = os.environ.get("AGENT_DB_PATH", "./qdrant_data")
COLLECTION = os.environ.get("AGENT_COLLECTION", "documents")

# The store is opened in the lifespan (startup), NOT at import time — on-disk Qdrant is
# single-writer and importing happens in more than one process under `uvicorn --reload`.
_db: DBManager | None = None
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db
    _db = DBManager(path=DB_PATH)
    # NOTE: the corpus is NOT ingested here — embeddings must be created with the caller's
    # real key/model, not a placeholder. The first /verify request ingests rag/data with
    # the real embedder (auto_ingest below); later requests find the store already populated.
    try:
        yield
    finally:
        _db.close()
        _db = None


def store() -> DBManager:
    if _db is None:
        raise HTTPException(status_code=503, detail="Store is not ready")
    return _db


app = FastAPI(title="WE6 Agentic Verifier API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class ActionRequest(BaseModel):
    action_type: str
    fields: dict = {}
    context: dict = {}


@contextmanager
def _provider_errors():
    """Map provider failures onto the API's status codes.

    Used around *both* places that can talk to OpenRouter — building the pipeline (the
    first request embeds the corpus) and verifying — so a bad key is a 401 either way
    instead of an unhandled 500 from inside the dependency.
    """
    try:
        yield
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenRouter API key")
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}")


# --------------------------------------------------------------------------- #
# Per-request pipeline built from the caller's key
# --------------------------------------------------------------------------- #
def get_pipeline(x_openrouter_key: str | None = Header(default=None)) -> VerifierPipeline:
    if not x_openrouter_key:
        raise HTTPException(status_code=401, detail="Missing X-OpenRouter-Key header")
    embedder, llm = build_clients(x_openrouter_key)
    # auto_ingest=True: on the first request the shared store is empty, so the corpus is
    # ingested with THIS caller's real embedder. Subsequent requests find it populated and
    # skip ingestion (RAGCore hydrates from the store on init). That ingestion is a *write*,
    # so it takes the same lock as /verify — on-disk Qdrant is single-writer, and two
    # concurrent first requests would otherwise both ingest the corpus.
    with _lock, _provider_errors():
        db = store()
        # # Uncomment to drop existing db collection; otherwise keep this commented to avoid re-ingesting
        # if db.collection_exists(COLLECTION):
        #     db.client.delete_collection(collection_name=COLLECTION)
        #     print(f"==========Deleted existing collection {COLLECTION} in {DB_PATH}==========")
        return VerifierPipeline(
            embedder, llm, db=db, collection_name=COLLECTION, auto_ingest=True
        )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
def implementation_status() -> dict:
    """Which notebook-exported functions are implemented — drives the progress dashboard."""
    return solutions.status()


@app.get("/action-types")
def action_types() -> dict:
    """The action catalogue + field schemas that drive the dashboard forms."""
    return {"action_types": ACTION_TYPES}


@app.get("/policies")
def list_policies() -> dict:
    """The policy corpus the verifier grounds on (one entry per document)."""
    db = store()
    if not db.collection_exists(COLLECTION):
        return {"policies": []}
    summaries: dict[str, dict] = {}
    with _lock:
        for chunk in db.get_all_chunks(COLLECTION):
            source = chunk.metadata.get("source", "unknown")
            entry = summaries.setdefault(
                source,
                {"source": source, "title": chunk.metadata.get("document_title", source), "chunks": 0},
            )
            entry["chunks"] += 1
    return {"policies": list(summaries.values())}


@app.post("/verify")
def verify(req: ActionRequest, pipeline: VerifierPipeline = Depends(get_pipeline)) -> dict:
    """Verify a dashboard action against company policy and return UI feedback."""
    action = Action(action_type=req.action_type, fields=req.fields, context=req.context)
    with _lock, _provider_errors():
        try:
            result = pipeline.verify(action)
        except ActionValidationError as exc:
            raise HTTPException(status_code=400, detail={"errors": exc.errors})
    return result.to_dict()
