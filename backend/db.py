# backend/db.py
# ── Supabase CRUD layer ──────────────────────────────────────────────────────
import os
from supabase import create_client, Client
from typing import List, Optional, Dict, Any
from datetime import datetime

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "your-anon-key")

_client: Optional[Client] = None

def get_db() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client

# ═══════════════════════════════════════════════════
# SESSIONS
# ═══════════════════════════════════════════════════

def create_session(title: str = "New Chat") -> Dict:
    """Create a new chat session."""
    res = get_db().table("sessions").insert({"title": title}).execute()
    return res.data[0]

def get_all_sessions() -> List[Dict]:
    """Get all sessions ordered by most recent."""
    res = (get_db().table("sessions")
           .select("*")
           .order("updated_at", desc=True)
           .execute())
    return res.data

def get_session(session_id: str) -> Optional[Dict]:
    """Get a single session by ID."""
    res = (get_db().table("sessions")
           .select("*")
           .eq("id", session_id)
           .single()
           .execute())
    return res.data

def update_session_title(session_id: str, title: str) -> Dict:
    """Rename a session."""
    res = (get_db().table("sessions")
           .update({"title": title})
           .eq("id", session_id)
           .execute())
    return res.data[0]

def delete_session(session_id: str) -> bool:
    """Delete a session and all its messages (CASCADE)."""
    get_db().table("sessions").delete().eq("id", session_id).execute()
    return True

# ═══════════════════════════════════════════════════
# MESSAGES
# ═══════════════════════════════════════════════════

def save_message(session_id: str, role: str, content: str) -> Dict:
    """Save a single message to a session."""
    res = (get_db().table("messages")
           .insert({
               "session_id": session_id,
               "role": role,
               "content": content,
           })
           .execute())
    # Touch the session updated_at
    get_db().table("sessions").update({"updated_at": datetime.utcnow().isoformat()}).eq("id", session_id).execute()
    return res.data[0]

def get_messages(session_id: str) -> List[Dict]:
    """Get all messages in a session, ordered by time."""
    res = (get_db().table("messages")
           .select("*")
           .eq("session_id", session_id)
           .order("created_at")
           .execute())
    return res.data

def delete_message(message_id: str) -> bool:
    """Delete a single message."""
    get_db().table("messages").delete().eq("id", message_id).execute()
    return True

# ═══════════════════════════════════════════════════
# DOCUMENTS
# ═══════════════════════════════════════════════════

def save_document_record(
    name: str,
    storage_path: str,
    chunks_indexed: int,
    session_id: Optional[str] = None,
    file_size: Optional[int] = None,
    mime_type: Optional[str] = None,
) -> Dict:
    """Save document metadata after indexing."""
    res = (get_db().table("documents")
           .insert({
               "name": name,
               "storage_path": storage_path,
               "chunks_indexed": chunks_indexed,
               "session_id": session_id,
               "file_size": file_size,
               "mime_type": mime_type,
           })
           .execute())
    return res.data[0]

def get_all_documents(session_id: Optional[str] = None) -> List[Dict]:
    """Get all documents, optionally filtered by session."""
    q = get_db().table("documents").select("*").order("created_at", desc=True)
    if session_id:
        q = q.eq("session_id", session_id)
    return q.execute().data

def delete_document(doc_id: str) -> bool:
    """Delete document record (and storage file separately)."""
    res = get_db().table("documents").select("storage_path").eq("id", doc_id).single().execute()
    if res.data:
        # Delete from storage too
        try:
            get_db().storage.from_("rika-documents").remove([res.data["storage_path"]])
        except Exception:
            pass
    get_db().table("documents").delete().eq("id", doc_id).execute()
    return True

# ═══════════════════════════════════════════════════
# STORAGE
# ═══════════════════════════════════════════════════

def upload_file_to_storage(local_path: str, storage_path: str, mime_type: str) -> str:
    """Upload a file to Supabase Storage. Returns the storage path."""
    with open(local_path, "rb") as f:
        get_db().storage.from_("rika-documents").upload(
            path=storage_path,
            file=f,
            file_options={"content-type": mime_type}
        )
    return storage_path

def get_file_url(storage_path: str, expires_in: int = 3600) -> str:
    """Get a signed download URL for a stored file."""
    res = get_db().storage.from_("rika-documents").create_signed_url(
        storage_path, expires_in
    )
    return res["signedURL"]
