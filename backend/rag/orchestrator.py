# rag/orchestrator.py

from retrieval.vector_store import VectorStore
from retrieval.reranker import Reranker
from rag.prompt_builder import PromptBuilder
from typing import List, Dict, Any, Generator

class RAGOrchestrator:
    def __init__(self, llm, vector_store: VectorStore):
        self.llm = llm                      # your llama-cpp Llama instance
        self.vector_store = vector_store
        self.reranker = Reranker()
        self.prompt_builder = PromptBuilder()

    def query(
        self, 
        question: str, 
        top_k_retrieve: int = 8,   # retrieve more, rerank down to fewer
        top_k_rerank: int = 3,
        stream: bool = False,
    ):
        # Stage 1: Dense retrieval
        candidates = self.vector_store.retrieve(question, top_k=top_k_retrieve)

        # Stage 2: Rerank (skip if no candidates)
        if candidates:
            final_chunks = self.reranker.rerank(question, candidates, top_k=top_k_rerank)
        else:
            final_chunks = []

        # Stage 3: Build grounded prompt
        prompt = self.prompt_builder.build(question, final_chunks)

        # Stage 4: Generate
        if stream:
            return self._stream(prompt), final_chunks
        else:
            return self._generate(prompt), final_chunks

    def _generate(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.2,       # lower temp = more deterministic code
            top_p=0.95,
            repeat_penalty=1.1,
            stop=["<|end|>", "<|user|>"],
        )
        return output["choices"][0]["text"].strip()

    def _stream(self, prompt: str) -> Generator:
        for chunk in self.llm(
            prompt,
            max_tokens=512,
            temperature=0.2,
            top_p=0.95,
            stream=True,
            stop=["<|end|>", "<|user|>"],
        ):
            yield chunk["choices"][0]["text"]