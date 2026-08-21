# main.py  —  Rika RAG Backend
# uvicorn main:app --host 0.0.0.0 --port 8000

import os, sys, uuid, shutil, mimetypes
from contextlib import asynccontextmanager
from typing import Optional

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY",     "False")

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from llama_cpp import Llama
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import SmartChunker
from retrieval.vector_store import VectorStore
from rag.orchestrator import RAGOrchestrator

# ══════════════════════════════════════════════════════════════════════
# APP  +  CORS
# The order matters: CORSMiddleware must be added FIRST.
# We also add a manual middleware that stamps CORS on every response
# including errors — because ngrok/Cloudflare error pages strip headers.
# ══════════════════════════════════════════════════════════════════════
app = FastAPI(title="Rika RAG Assistant")

# 1. Standard CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when origins=["*"]
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# 2. Manual fallback — stamps headers even on 4xx/5xx error responses
@app.middleware("http")
async def force_cors(request: Request, call_next):
    # Handle preflight here too (belt-and-suspenders)
    if request.method == "OPTIONS":
        return JSONResponse(
            content={},
            headers={
                "Access-Control-Allow-Origin":  "*",
                "Access-Control-Allow-Methods": "GET,POST,PATCH,DELETE,OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age":       "600",
            }
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PATCH,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# 3. Explicit OPTIONS catch-all (some clients bypass middleware)
@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse(content={}, headers={
        "Access-Control-Allow-Origin":  "*",
        "Access-Control-Allow-Methods": "GET,POST,PATCH,DELETE,OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Max-Age":       "600",
    })

# ══════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════
class QueryRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None
    top_k:      int = 3
    stream:     bool = False

class SessionUpdate(BaseModel):
    title: str

# ══════════════════════════════════════════════════════════════════════
# GLOBALS — populated in lifespan
# ══════════════════════════════════════════════════════════════════════
llm = vector_store = orchestrator = loader = chunker = db = None

# ══════════════════════════════════════════════════════════════════════
# LIFESPAN — all heavy init here so errors appear in uvicorn logs
# ══════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_llm()
    _init_rag()
    _init_supabase()
    print("\n✅ Rika backend ready\n", flush=True)
    yield

app.router.lifespan_context = lifespan

def _find_model():
    for p in [
        "/content/models/glm-4-9b-chat-Q4_K_M.gguf",
        "/content/glm-4-9b-chat-Q4_K_M.gguf",
        "/content/drive/MyDrive/models/glm-4-9b-chat-Q4_K_M.gguf",
    ]:
        if os.path.exists(p):
            print(f"      Model: {p}", flush=True); return p
    raise FileNotFoundError("GGUF not found — check model path")

def _init_llm():
    global llm
    print("[1/3] Loading LLM…", flush=True)
    llm = Llama(model_path=_find_model(), n_gpu_layers=35,
                n_ctx=8192, n_batch=512, verbose=False)
    print("      LLM ✓", flush=True)

def _init_rag():
    global vector_store, orchestrator, loader, chunker
    print("[2/3] Initialising RAG…", flush=True)
    vector_store = VectorStore(persist_dir="/content/drive/MyDrive/chroma_db")
    orchestrator = RAGOrchestrator(llm, vector_store)
    loader, chunker = DocumentLoader(), SmartChunker()
    print("      RAG ✓", flush=True)

def _init_supabase():
    global db
    print("[3/3] Connecting Supabase…", flush=True)
    url = os.environ.get("SUPABASE_URL","").strip()
    key = os.environ.get("SUPABASE_KEY","").strip()
    print(f"      URL={url[:40]!r}", flush=True)
    print(f"      KEY={'set' if key else 'EMPTY'}", flush=True)
    if not url or not key or "your-project" in url:
        print("      ⚠️  Supabase disabled — env vars missing", flush=True)
        return
    try:
        from supabase import create_client
        db = create_client(url, key)
        db.table("sessions").select("id").limit(1).execute()
        print("      Supabase ✓", flush=True)
    except Exception as e:
        print(f"      Supabase failed: {e}", flush=True)
        db = None

# ══════════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════════
def _save_msg(sid, role, content):
    if not db or not sid: return
    try:
        db.table("messages").insert({"session_id":sid,"role":role,"content":content}).execute()
        db.table("sessions").update({"updated_at":"now()"}).eq("id",sid).execute()
    except Exception as e: print(f"[db] msg: {e}")

def _save_doc(name, spath, chunks, sid, fsize, mime):
    if not db: return None
    try:
        r = db.table("documents").insert({
            "name":name,"storage_path":spath,"chunks_indexed":chunks,
            "session_id":sid,"file_size":fsize,"mime_type":mime,
        }).execute()
        return r.data[0] if r.data else None
    except Exception as e: print(f"[db] doc: {e}"); return None

def _upload(local, spath, mime):
    if not db: return
    try:
        with open(local,"rb") as f:
            db.storage.from_("rika-documents").upload(
                path=spath, file=f,
                file_options={"content-type": mime or "application/octet-stream"})
    except Exception as e: print(f"[db] upload: {e}")

# ══════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {"status":"ok","chunks":vector_store.collection_size() if vector_store else 0,
            "supabase":db is not None,"llm":llm is not None}

# Sessions
@app.post("/sessions")
def create_session():
    if not db: raise HTTPException(503,"Supabase not connected")
    return db.table("sessions").insert({"title":"New Chat"}).execute().data[0]

@app.get("/sessions")
def get_sessions():
    if not db: return []
    return db.table("sessions").select("*").order("updated_at",desc=True).execute().data

@app.get("/sessions/{sid}")
def get_session(sid:str):
    if not db: raise HTTPException(503,"Supabase not connected")
    r = db.table("sessions").select("*").eq("id",sid).single().execute()
    if not r.data: raise HTTPException(404,"Not found")
    return r.data

@app.patch("/sessions/{sid}")
def rename_session(sid:str, body:SessionUpdate):
    if not db: raise HTTPException(503,"Supabase not connected")
    return db.table("sessions").update({"title":body.title}).eq("id",sid).execute().data[0]

@app.delete("/sessions/{sid}")
def delete_session(sid:str):
    if not db: raise HTTPException(503,"Supabase not connected")
    db.table("sessions").delete().eq("id",sid).execute()
    return {"deleted":sid}

@app.get("/sessions/{sid}/messages")
def get_messages(sid:str):
    if not db: return []
    return db.table("messages").select("*").eq("session_id",sid).order("created_at").execute().data

@app.delete("/messages/{mid}")
def delete_message(mid:str):
    if not db: raise HTTPException(503,"Supabase not connected")
    db.table("messages").delete().eq("id",mid).execute()
    return {"deleted":mid}

# Documents
@app.get("/documents")
def get_documents(session_id:Optional[str]=None):
    if not db: return []
    q = db.table("documents").select("*").order("created_at",desc=True)
    if session_id: q = q.eq("session_id",session_id)
    return q.execute().data

@app.delete("/documents/{did}")
def delete_document(did:str):
    if not db: raise HTTPException(503,"Supabase not connected")
    try:
        r = db.table("documents").select("storage_path").eq("id",did).single().execute()
        if r.data: db.storage.from_("rika-documents").remove([r.data["storage_path"]])
    except: pass
    db.table("documents").delete().eq("id",did).execute()
    return {"deleted":did}

@app.get("/documents/{did}/download")
def download_document(did:str):
    if not db: raise HTTPException(503,"Supabase not connected")
    r = db.table("documents").select("storage_path").eq("id",did).single().execute()
    if not r.data: raise HTTPException(404,"Not found")
    s = db.storage.from_("rika-documents").create_signed_url(r.data["storage_path"],3600)
    return {"url":s["signedURL"]}

# Ingest
@app.post("/ingest")
async def ingest(file:UploadFile=File(...), session_id:Optional[str]=None):
    local = f"/tmp/{file.filename}"
    with open(local,"wb") as f: shutil.copyfileobj(file.file,f)
    try:
        docs   = loader.load_file(local)
        chunks = chunker.chunk(docs)
        count  = vector_store.add_documents(chunks)
        spath  = f"{uuid.uuid4()}/{file.filename}"
        mime,_ = mimetypes.guess_type(file.filename)
        _upload(local, spath, mime)
        doc = _save_doc(file.filename, spath, count, session_id,
                        os.path.getsize(local), mime)
        return {"status":"success","chunks_indexed":count,"document":doc}
    except Exception as e: raise HTTPException(500,str(e))
    finally:
        if os.path.exists(local): os.remove(local)

# Query
@app.post("/query")
async def query(req:QueryRequest):
    try:
        result = orchestrator.query(req.question, top_k_rerank=req.top_k)
        answer, sources = result if isinstance(result,tuple) else (str(result),[])
        if req.session_id:
            _save_msg(req.session_id,"user",req.question)
            _save_msg(req.session_id,"rika",answer)
        return {"answer":answer,
                "sources":[s["metadata"].get("source") for s in sources],
                "chunks_used":len(sources)}
    except Exception as e: raise HTTPException(500,str(e))

# Stream
@app.post("/stream")
async def stream(req:QueryRequest):
    try:
        result    = orchestrator.query(req.question, stream=True)
        generator = result[0] if isinstance(result,tuple) else result
        if req.session_id:
            _save_msg(req.session_id,"user",req.question)
        collected = []

        def event_stream():
            for token in generator:
                if not token: continue
                collected.append(token)
                safe = token.replace("\\","\\\\").replace("\n","\\n")
                yield f"data: {safe}\n\n"
            yield "data: [DONE]\n\n"
            if req.session_id and collected:
                _save_msg(req.session_id,"rika","".join(collected))

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                # Explicit headers on the stream response itself —
                # middleware headers can be lost when streaming
                "Access-Control-Allow-Origin":  "*",
                "Access-Control-Allow-Headers": "*",
                "Cache-Control":                "no-cache, no-transform",
                "X-Accel-Buffering":            "no",
                "Connection":                   "keep-alive",
                # ngrok free tier browser warning bypass
                "ngrok-skip-browser-warning":   "true",
            }
        )
    except Exception as e: raise HTTPException(500,str(e))
