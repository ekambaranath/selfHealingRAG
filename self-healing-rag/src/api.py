"""
FastAPI server for the Self-Healing RAG Pipeline.
Fully local — Ollama + ChromaDB + HuggingFace embeddings.

Endpoints:
  GET  /              → Chat UI
  POST /query         → Run the RAG pipeline
  GET  /health        → Health check (Ollama + doc count)
  POST /ingest        → Ingest raw text
  GET  /graph/diagram → ASCII pipeline diagram
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chains.graph import run_pipeline
from src.retrieval.vectorstore import get_doc_count, ingest_documents
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger("api")

app = FastAPI(
    title="Self-Healing RAG API",
    description="Fully local RAG pipeline — Ollama + ChromaDB + LangGraph",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)


class QueryResponse(BaseModel):
    query: str
    answer: str
    steps: list[str]
    retries: int
    success: bool
    query_used: str
    elapsed_seconds: float


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    source: str = Field(default="api_upload")


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    ollama_online: bool
    ollama_model: str
    max_retries: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = Path(__file__).parent.parent / "static" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>Self-Healing RAG</h1><p>UI not found.</p>")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        doc_count = get_doc_count()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector store error: {e}")

    ollama_ok = settings.ollama_configured

    return HealthResponse(
        status="ok" if ollama_ok else "ollama_offline",
        documents_indexed=doc_count,
        ollama_online=ollama_ok,
        ollama_model=settings.ollama_model,
        max_retries=settings.max_retries,
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not settings.ollama_configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not running. Start it with: ollama serve\n"
                f"Then pull the model: ollama pull {settings.ollama_model}"
            ),
        )

    log.info("api_query", query=request.query[:80])
    start = time.perf_counter()

    try:
        result = run_pipeline(request.query)
    except Exception as e:
        log.error("pipeline_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    elapsed = round(time.perf_counter() - start, 2)

    return QueryResponse(
        query=request.query,
        answer=result["answer"],
        steps=result["steps"],
        retries=result["retries"],
        success=result["success"],
        query_used=result["query_used"],
        elapsed_seconds=elapsed,
    )


@app.post("/ingest")
async def ingest_text(request: IngestRequest):
    from langchain_core.documents import Document
    doc = Document(page_content=request.text, metadata={"source": request.source})
    try:
        count = ingest_documents([doc])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    return {"status": "ok", "chunks_added": count, "source": request.source}


@app.get("/graph/diagram")
async def graph_diagram():
    diagram = """
Self-Healing RAG — Local Stack (Ollama + ChromaDB)
═══════════════════════════════════════════════════

  [User Query]
       │
       ▼
  ┌────────────────┐
  │ rewrite_query  │ ◄──────────────────────┐
  └───────┬────────┘                        │ (retry: reformulate)
          │                                 │
          ▼                                 │
  ┌────────────────┐                        │
  │    retrieve    │ ← ChromaDB + MiniLM    │
  └───────┬────────┘                        │
          │                                 │
          ▼                                 │
  ┌────────────────┐   FAIL                 │
  │grade_relevance │ ──────────────────────►┤
  │  (Ollama LLM)  │                        │
  └───────┬────────┘                        │
          │ PASS                            │
          ▼                                 │
  ┌────────────────┐                        │
  │    generate    │ ← Ollama LLM           │
  └───────┬────────┘                        │
          │                                 │
          ▼                                 │
  ┌──────────────────────┐   FAIL           │
  │ grade_hallucination  │ ────────────────►┤
  │     (Ollama LLM)     │                  │
  └──────────┬───────────┘                  │
             │ PASS          max retries ───┘
             ▼               exceeded → fallback
       ┌──────────┐
       │ finalize │
       └────┬─────┘
            ▼
       [Final Answer]

Model    : {model}
Retries  : {retries}
Cost     : $0.00
Privacy  : 100%% local
""".format(model=settings.ollama_model, retries=settings.max_retries)
    return {"diagram": diagram}
