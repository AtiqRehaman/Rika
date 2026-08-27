# main.py  —  Rika RAG Backend
# uvicorn main:app --host 0.0.0.0 --port 8000
# NO CORS — frontend uses Vite proxy, all requests go to localhost

import os, sys, uuid, shutil, mimetypes, asyncio, threading
from contextlib import asynccontextmanager
from typing import Optional

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY",     "False")

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from llama_cpp import Llama
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import SmartChunker
from retrieval.vector_store import VectorStore
from rag.orchestrator import RAGOrchestrator

# ══════════════════════════════════════════════════════════════════
# APP  —  no CORS middleware at all
# ══════════════════════════════════════════════════════════════════
app = FastAPI(title="Rika RAG Assistant")

# Global exception handler — prevents uvicorn from dying on errors
from fastapi.responses import JSONResponse as _JR
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_handler(request: Request, exc: StarletteHTTPException):
    return _JR({"detail": exc.detail}, status_code=exc.status_code)

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return _JR({"detail": str(exc)}, status_code=422)

@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    print(f"[ERROR] Unhandled exception: {exc}", flush=True)
    import traceback; traceback.print_exc()
    return _JR({"detail": str(exc)}, status_code=500)

# ══════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════
class QueryRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None
    top_k:      int = 3

class SessionUpdate(BaseModel):
    title: str

# ══════════════════════════════════════════════════════════════════
# GLOBALS
# ══════════════════════════════════════════════════════════════════
llm = vector_store = orchestrator = loader = chunker = db = None

# ══════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════
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
            print(f"      Model: {p}", flush=True)
            return p
    raise FileNotFoundError("GGUF not found")

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
    print(f"      URL={url[:50]!r}", flush=True)
    print(f"      KEY starts: {key[:14]!r}", flush=True)
    if not url or not key or "your-project" in url:
        print("      ⚠️  Supabase disabled", flush=True); return
    if not key.startswith("eyJ"):
        print("      ⚠️  KEY wrong format — get anon key from supabase.com → Settings → API", flush=True); return
    try:
        from supabase import create_client
        db = create_client(url, key)
        db.table("sessions").select("id").limit(1).execute()
        print("      Supabase ✓", flush=True)
    except Exception as e:
        print(f"      Supabase failed: {e}", flush=True); db = None

# ══════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {"status":"ok",
            "chunks": vector_store.collection_size() if vector_store else 0,
            "supabase": db is not None,
            "llm": llm is not None}

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

# Stream  —  async generator with correct loop/thread pattern
@app.post("/stream")
async def stream(req: QueryRequest):
    try:
        result    = orchestrator.query(req.question, stream=True)
        generator = result[0] if isinstance(result, tuple) else result

        if req.session_id:
            _save_msg(req.session_id, "user", req.question)

        collected = []
        # get_running_loop() is correct in async context (not get_event_loop())
        loop  = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def _run():
            """Runs in a background thread — pushes tokens into the async queue."""
            try:
                for token in generator:
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("__error__", str(e)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        threading.Thread(target=_run, daemon=True).start()

        async def event_stream():
            # Immediate comment line — keeps tunnel alive before first token
            yield ": ping\n\n"

            while True:
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # keep-alive comment — ignored by SSE parsers
                    yield ": keep-alive\n\n"
                    continue

                if token is None:
                    # Sentinel — generation finished cleanly
                    break

                if isinstance(token, tuple) and token[0] == "__error__":
                    yield f"data: [ERROR] {token[1]}\n\n"
                    break

                collected.append(token)
                # Escape backslashes first, then newlines
                safe = token.replace("\\", "\\\\").replace("\n", "\\n")
                yield f"data: {safe}\n\n"

            yield "data: [DONE]\n\n"

            if req.session_id and collected:
                _save_msg(req.session_id, "rika", "".join(collected))

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":     "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection":        "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(500, str(e))
