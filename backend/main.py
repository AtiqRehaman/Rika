# main.py
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import sys
sys.path.insert(0, "/content/Rika/backend")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading
import subprocess
import time
import shutil
import re

from llama_cpp import Llama
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import SmartChunker
from retrieval.vector_store import VectorStore
from rag.orchestrator import RAGOrchestrator

# ── App ────────────────────────────────────────────────────
app = FastAPI(title="Rika RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Models ─────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    stream: bool = False

# ── Load LLM ───────────────────────────────────────────────
print("[1/4] Loading LLM...")
llm = Llama(
    model_path="/content/glm-4-9b-chat-Q4_K_M.gguf",
    n_gpu_layers=35,
    n_ctx=8192,
    n_batch=512,
    verbose=False,
)
print("      LLM loaded ✓")

# ── Init RAG ───────────────────────────────────────────────
print("[2/4] Initializing RAG components...")
vector_store = VectorStore(persist_dir="/content/drive/MyDrive/chroma_db")
orchestrator = RAGOrchestrator(llm, vector_store)
loader       = DocumentLoader()
chunker      = SmartChunker()
print("      RAG ready ✓")

# ── Routes ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "chunks": vector_store.collection_size()}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    upload_path = f"/tmp/{file.filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        docs   = loader.load_file(upload_path)
        chunks = chunker.chunk(docs)
        count  = vector_store.add_documents(chunks)
        return {"status": "success", "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(req: QueryRequest):
    answer, sources = orchestrator.query(req.question, top_k_rerank=req.top_k)
    return {
        "answer": answer,   # newlines here are fine — JSON encodes them as \n automatically
        "sources": [s["metadata"].get("source") for s in sources],
        "chunks_used": len(sources),
    }

    
@app.post("/stream")
async def stream(req: QueryRequest):
    generator, _ = orchestrator.query(req.question, stream=True)

    def event_stream():
        for token in generator:
            if not token:
                continue
            # CRITICAL: escape real newlines inside tokens.
            # SSE uses \n\n as event separator — a raw \n in a token
            # breaks the frame and corrupts the stream into one line.
            safe = token.replace("\\", "\\\\").replace("\n", "\\n")
            yield f"data: {safe}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",    # stops Cloudflare buffering tokens
            "Connection":        "keep-alive",
        }
    )
# ── Cloudflare Tunnel (no login needed) ────────────────────
def start_cloudflare(port: int = 8000):
    """Starts cloudflared quick tunnel and extracts the public URL."""

    # Download cloudflared if not present
    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("      Downloading cloudflared...")
        os.system(
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/"
            "download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared "
            "&& chmod +x /usr/local/bin/cloudflared"
        )

    # Start tunnel and capture output to extract URL
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Read output until URL appears
    url = None
    for line in proc.stdout:
        print(f"      [cloudflare] {line.strip()}")
        match = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            break

    return proc, url

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":

    # Start uvicorn in background thread
    print("[3/4] Starting FastAPI server...")
    t = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning"),
        daemon=True,
    )
    t.start()
    time.sleep(3)
    print("      Server running on port 8000 ✓")

    # Start Cloudflare tunnel
    print("[4/4] Opening Cloudflare tunnel (no login needed)...")
    cf_proc, public_url = start_cloudflare(port=8000)

    if public_url:
        print(f"\n{'='*55}")
        print(f"  🚀 Backend live at:")
        print(f"  {public_url}")
        print(f"\n  ⚠️  Save this URL — it changes each session")
        print(f"  Tip: update config.js on your frontend with this URL")
        print(f"{'='*55}\n")
    else:
        print("✗ Could not extract URL — check cloudflare logs above")

    # Keep alive
    try:
        while True:
            time.sleep(60)
            print("  ✓ Server running...")
    except KeyboardInterrupt:
        print("Shutting down...")
        cf_proc.terminate()