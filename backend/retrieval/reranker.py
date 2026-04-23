# retrieval/reranker.py

from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class Reranker:
    """
    Two-stage retrieval: vector search (fast, approximate) → 
    cross-encoder rerank (slow, accurate on small set).
    
    This is a well-established research pattern (Nogueira et al. 2019).
    Including it makes your system architecturally sound.
    """

    def __init__(self):
        # Small cross-encoder, runs on CPU in ~200ms for 10 candidates
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, c["content"]) for c in candidates]
        scores = self.model.predict(pairs)

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]