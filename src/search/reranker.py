# src/search/reranker.py
from sentence_transformers import CrossEncoder
from typing import List, Tuple

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # This downloads the cross-encoder model the first time
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidate_texts: List[Tuple[str, str]], top_k: int):
        """
        candidate_texts: list of tuples (doc_id, text)
        returns: list of dicts with doc_id, score (cross-encoder), preview
        """
        # Prepare input pairs
        pairs = [[query, text] for (_, text) in candidate_texts]
        scores = self.model.predict(pairs)  # higher = better
        scored = []
        for (doc_id, text), score in zip(candidate_texts, scores):
            scored.append({"doc_id": doc_id, "score": float(score), "preview": text[:200] + "..."})
        # sort descending
        scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
        return scored_sorted[:top_k]
