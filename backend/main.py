# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
import shutil, os

from ingestion.document_loader import DocumentLoader
from ingestion.chunker import SmartChunker
from retrieval.vector_store import VectorStore
from rag.orchestrator import RAGOrchestrator

app = FastAPI(title="Local RAG Coding Assistant")

# ── Initialize components ──────────────────────────────────────────────
llm = Llama(
    model_path="./models/your-model.gguf",
    n_gpu_layers=35,    # tune based on your GPU VRAM
    n_ctx=4096,
    n_batch=512,
    verbose=False,
)

vector_store = VectorStore(persist_dir="./chroma_db")
orchestrator = RAGOrchestrator(llm, vector_store)
loader = DocumentLoader()
chunker = SmartChunker()

# ── Request/Response Models ────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    stream: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: list
    chunks_used: int

# ── Endpoints ──────────────────────────────────────────────────────────
@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """Upload and index a document."""
    upload_path = f"/tmp/{file.filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        docs = loader.load_file(upload_path)
        chunks = chunker.chunk(docs)
        count = vector_store.add_documents(chunks)
        return {"status": "success", "chunks_indexed": count, "file": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """RAG query with full response."""
    answer, sources = orchestrator.query(
        request.question, top_k_rerank=request.top_k
    )
    return QueryResponse(
        answer=answer,
        sources=[s["metadata"].get("source") for s in sources],
        chunks_used=len(sources),
    )

@app.post("/stream")
async def stream_query(request: QueryRequest):
    """RAG query with streaming response."""
    generator, _ = orchestrator.query(request.question, stream=True)

    def event_stream():
        for token in generator:
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/stats")
async def stats():
    return {"indexed_chunks": vector_store.collection_size()}