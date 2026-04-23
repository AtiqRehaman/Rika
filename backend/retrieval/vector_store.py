# retrieval/vector_store.py

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import hashlib
from ingestion.document_loader import Document

class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # all-MiniLM-L6-v2: only 80MB, runs well on CPU, great for technical text
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.collection = self.client.get_or_create_collection(
            name="coding_assistant",
            metadata={"hnsw:space": "cosine"},  # cosine better than L2 for text
        )

    def add_documents(self, chunks: List[Document]) -> int:
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Stable IDs prevent duplicates on re-ingestion
        ids = [
            hashlib.md5(f"{m.get('source','')}_{m.get('chunk_index',0)}".encode()).hexdigest()
            for m in metadatas
        ]

        embeddings = self.embed_model.encode(
            texts, 
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,   # needed for cosine similarity
        ).tolist()

        # Upsert avoids duplicate errors
        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "content": doc,
                "metadata": meta,
                "score": 1 - dist,   # convert distance → similarity
            })

        return retrieved

    def collection_size(self) -> int:
        return self.collection.count()