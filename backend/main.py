# main.py
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import sys
sys.path.insert(0, "/content/Rika/backend")

# ── Standard library ───────────────────────────────────────
import re
import time
import uuid
import shutil
import mimetypes
import threading
import subprocess

# ── FastAPI ────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ── LLM ───────────────────────────────────────────────────
from llama_cpp import Llama

# ── RAG components ────────────────────────────────────────
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import SmartChunker
from retrieval.vector_store import VectorStore
from rag.orchestrator import RAGOrchestrator

# ══════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════
app = FastAPI(title="Rika RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ══════════════════════════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════════════════════════
class QueryRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None
    top_k:      int = 3
    stream:     bool = False

class SessionUpdate(BaseModel):
    title: str

# ══════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════
MODEL_PATH = "/content/models/glm-4-9b-chat-Q4_K_M.gguf"

# Fallback paths to try if primary not found
FALLBACK_PATHS = [
    "/content/glm-4-9b-chat-Q4_K_M.gguf",
    "/content/drive/MyDrive/models/glm-4-9b-chat-Q4_K_M.gguf",
]

def resolve_model_path():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    for p in FALLBACK_PATHS:
        if os.path.exists(p):
            print(f"      Model found at fallback: {p}")
            return p
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH} or any fallback path.\n"
        f"Tried: {[MODEL_PATH] + FALLBACK_PATHS}"
    )

print("[1/4] Loading LLM...")
llm = Llama(
    model_path=resolve_model_path(),
    n_gpu_layers=35,
    n_ctx=8192,
    n_batch=512,
    verbose=False,
)
print("      LLM loaded ✓")

# ══════════════════════════════════════════════════════════
# INIT RAG
# ══════════════════════════════════════════════════════════
print("[2/4] Initializing RAG components...")
vector_store = VectorStore(persist_dir="/content/drive/MyDrive/chroma_db")
orchestrator = RAGOrchestrator(llm, vector_store)
loader       = DocumentLoader()
chunker      = SmartChunker()
print("      RAG ready ✓")

# ══════════════════════════════════════════════════════════
# OPTIONAL: Supabase (graceful if not configured)
# ══════════════════════════════════════════════════════════
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
db = None

if SUPABASE_URL and SUPABASE_KEY and "your-project" not in SUPABASE_URL:
    try:
        from supabase import create_client
        db = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("      Supabase connected ✓")
    except Exception as e:
        print(f"      Supabase skipped: {e}")
else:
    print("      Supabase not configured — running without DB (set SUPABASE_URL + SUPABASE_KEY to enable)")

# ── DB helpers (no-op if db is None) ──────────────────────
def db_save_message(session_id, role, content):
    if not db or not session_id:
        return
    try:
        db.table("messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
        }).execute()
        db.table("sessions").update({"updated_at": "now()"}).eq("id", session_id).execute()
    except Exception as e:
        print(f"      [db] save_message failed: {e}")

def db_save_document(name, storage_path, chunks, session_id, file_size, mime_type):
    if not db:
        return None
    try:
        res = db.table("documents").insert({
            "name": name,
            "storage_path": storage_path,
            "chunks_indexed": chunks,
            "session_id": session_id,
            "file_size": file_size,
            "mime_type": mime_type,
        }).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"      [db] save_document failed: {e}")
        return None

def db_upload_file(local_path, storage_path, mime_type):
    if not db:
        return
    try:
        with open(local_path, "rb") as f:
            db.storage.from_("rika-documents").upload(
                path=storage_path,
                file=f,
                file_options={"content-type": mime_type or "application/octet-stream"}
            )
    except Exception as e:
        print(f"      [db] upload_file failed: {e}")

# ══════════════════════════════════════════════════════════
# ROUTES — HEALTH
# ══════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": vector_store.collection_size(),
        "supabase": db is not None,
    }

# ══════════════════════════════════════════════════════════
# ROUTES — SESSIONS (only work if Supabase is connected)
# ══════════════════════════════════════════════════════════
@app.post("/sessions")
def create_session():
    if not db:
        raise HTTPException(503, "Supabase not configured")
    res = db.table("sessions").insert({"title": "New Chat"}).execute()
    return res.data[0]

@app.get("/sessions")
def get_sessions():
    if not db:
        return []
    res = db.table("sessions").select("*").order("updated_at", desc=True).execute()
    return res.data

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    if not db:
        raise HTTPException(503, "Supabase not configured")
    res = db.table("sessions").select("*").eq("id", session_id).single().execute()
    if not res.data:
        raise HTTPException(404, "Session not found")
    return res.data

@app.patch("/sessions/{session_id}")
def rename_session(session_id: str, body: SessionUpdate):
    if not db:
        raise HTTPException(503, "Supabase not configured")
    res = db.table("sessions").update({"title": body.title}).eq("id", session_id).execute()
    return res.data[0]

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if not db:
        raise HTTPException(503, "Supabase not configured")
    db.table("sessions").delete().eq("id", session_id).execute()
    return {"deleted": session_id}

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    if not db:
        return []
    res = (db.table("messages")
             .select("*")
             .eq("session_id", session_id)
             .order("created_at")
             .execute())
    return res.data

@app.delete("/messages/{message_id}")
def delete_message(message_id: str):
    if not db:
        raise HTTPException(503, "Supabase not configured")
    db.table("messages").delete().eq("id", message_id).execute()
    return {"deleted": message_id}

# ══════════════════════════════════════════════════════════
# ROUTES — DOCUMENTS
# ══════════════════════════════════════════════════════════
@app.get("/documents")
def get_documents(session_id: Optional[str] = None):
    if not db:
        return []
    q = db.table("documents").select("*").order("created_at", desc=True)
    if session_id:
        q = q.eq("session_id", session_id)
    return q.execute().data

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not db:
        raise HTTPException(503, "Supabase not configured")
    # Remove from storage first
    try:
        res = db.table("documents").select("storage_path").eq("id", doc_id).single().execute()
        if res.data:
            db.storage.from_("rika-documents").remove([res.data["storage_path"]])
    except Exception:
        pass
    db.table("documents").delete().eq("id", doc_id).execute()
    return {"deleted": doc_id}

@app.get("/documents/{doc_id}/download")
def download_document(doc_id: str):
    if not db:
        raise HTTPException(503, "Supabase not configured")
    res = db.table("documents").select("storage_path").eq("id", doc_id).single().execute()
    if not res.data:
        raise HTTPException(404, "Document not found")
    signed = db.storage.from_("rika-documents").create_signed_url(
        res.data["storage_path"], 3600
    )
    return {"url": signed["signedURL"]}

# ══════════════════════════════════════════════════════════
# ROUTES — INGEST
# ══════════════════════════════════════════════════════════
@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
):
    local_path = f"/tmp/{file.filename}"
    with open(local_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 1. Index into ChromaDB
        docs   = loader.load_file(local_path)
        chunks = chunker.chunk(docs)
        count  = vector_store.add_documents(chunks)

        # 2. Upload to Supabase Storage (if configured)
        storage_path = f"{uuid.uuid4()}/{file.filename}"
        mime, _      = mimetypes.guess_type(file.filename)
        db_upload_file(local_path, storage_path, mime)

        # 3. Save metadata (if configured)
        file_size = os.path.getsize(local_path)
        doc_record = db_save_document(
            name=file.filename,
            storage_path=storage_path,
            chunks=count,
            session_id=session_id,
            file_size=file_size,
            mime_type=mime,
        )

        return {
            "status": "success",
            "chunks_indexed": count,
            "document": doc_record,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

# ══════════════════════════════════════════════════════════
# ROUTES — QUERY (non-streaming)
# ══════════════════════════════════════════════════════════
@app.post("/query")
async def query(req: QueryRequest):
    try:
        result = orchestrator.query(req.question, top_k_rerank=req.top_k)

        # orchestrator.query() returns (answer, sources) for non-stream
        if isinstance(result, tuple):
            answer, sources = result
        else:
            answer  = str(result)
            sources = []

        # Save to Supabase if session provided
        if req.session_id:
            db_save_message(req.session_id, "user",  req.question)
            db_save_message(req.session_id, "rika",  answer)

        return {
            "answer":      answer,
            "sources":     [s["metadata"].get("source") for s in sources],
            "chunks_used": len(sources),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════
# ROUTES — STREAM
# ══════════════════════════════════════════════════════════
@app.post("/stream")
async def stream(req: QueryRequest):
    try:
        result = orchestrator.query(req.question, stream=True)

        # Handle both return styles from orchestrator
        if isinstance(result, tuple):
            generator, _ = result
        else:
            generator = result

        # Save user message
        if req.session_id:
            db_save_message(req.session_id, "user", req.question)

        collected = []

        def event_stream():
            for token in generator:
                if not token:
                    continue
                collected.append(token)
                # Escape newlines so SSE framing stays intact
                # Frontend unescapes \n → real newline
                safe = token.replace("\\", "\\\\").replace("\n", "\\n")
                yield f"data: {safe}\n\n"
            yield "data: [DONE]\n\n"

            # Save complete Rika response after stream finishes
            if req.session_id and collected:
                db_save_message(req.session_id, "rika", "".join(collected))

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":     "no-cache",
                "X-Accel-Buffering": "no",
                "Connection":        "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════
# NGROK TUNNEL (replaces Cloudflare)
# ══════════════════════════════════════════════════════════
def start_ngrok(port: int = 8000):
    """
    Start ngrok tunnel and return the public URL.
    Requires ngrok authtoken set in environment variable NGROK_AUTHTOKEN.
    """
    # Check if ngrok is installed
    if not os.path.exists("/usr/local/bin/ngrok"):
        print("      Downloading ngrok...")
        os.system(
            "wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip "
            "-O /tmp/ngrok.zip && unzip -q /tmp/ngrok.zip -d /usr/local/bin/ && "
            "chmod +x /usr/local/bin/ngrok"
        )
    
    # Get auth token from environment
    auth_token = os.environ.get("NGROK_AUTHTOKEN")
    if not auth_token:
        print("      ⚠️  NGROK_AUTHTOKEN not set in environment")
        print("      Get your token from https://dashboard.ngrok.com/get-started/your-authtoken")
        print("      Set it as: os.environ['NGROK_AUTHTOKEN'] = 'your_token'")
        return None, None
    
    # Configure auth token
    subprocess.run(["ngrok", "config", "add-authtoken", auth_token], 
                   capture_output=True, text=True)
    
    # Start ngrok tunnel
    proc = subprocess.Popen(
        ["ngrok", "http", str(port), "--log=stdout"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    # Wait for URL
    url = None
    import json
    for line in proc.stdout:
        print(f"      [ngrok] {line.strip()}")
        # ngrok logs JSON lines - look for the URL
        try:
            data = json.loads(line)
            if data.get("msg") == "started tunnel" and "url" in data:
                url = data["url"]
                break
        except json.JSONDecodeError:
            # Fallback: look for URL in plain text output
            match = re.search(r"https://[a-z0-9\-]+\.ngrok\.io", line)
            if match:
                url = match.group(0)
                break
    
    return proc, url

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("[3/4] Starting FastAPI server...")
    threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning"),
        daemon=True,
    ).start()
    time.sleep(3)
    print("      Server running on port 8000 ✓")

    print("[4/4] Opening ngrok tunnel...")
    ngrok_proc, public_url = start_ngrok(port=8000)

    if public_url:
        print(f"\n{'='*55}")
        print(f"  🚀 Backend live at: {public_url}")
        print(f"  ⚠️  Update src/config.js with this URL")
        print(f"{'='*55}\n")
    else:
        print("✗ Could not get tunnel URL — check ngrok logs above")
        print("  Make sure NGROK_AUTHTOKEN is set in environment")

    try:
        while True:
            time.sleep(60)
            print("  ✓ Server alive...")
    except KeyboardInterrupt:
        print("Shutting down...")
        if ngrok_proc:
            ngrok_proc.terminate()
