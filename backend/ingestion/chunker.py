# ingestion/chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from .document_loader import Document
import re

class SmartChunker:
    """
    Uses different chunking strategies per content type.
    
    Research insight: Code needs larger chunks (preserve function scope),
    prose needs smaller chunks (dense retrieval). This is a publishable
    optimization over naive fixed-size chunking.
    """

    def __init__(self):
        # For code files — preserve function/class boundaries
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
            separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
        )
        # For prose/documentation
        self.prose_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ". ", " "],
        )

    def chunk(self, docs: List[Document]) -> List[Document]:
        chunks = []
        for doc in docs:
            splitter = (
                self.code_splitter
                if doc.metadata.get("type") in ("py", "code")
                else self.prose_splitter
            )
            splits = splitter.split_text(doc.content)
            for i, split in enumerate(splits):
                chunks.append(Document(
                    content=split,
                    metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(splits)}
                ))
        return chunks