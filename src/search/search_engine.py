import numpy as np
import faiss

class SearchEngine:
    def __init__(self, index, doc_ids):
        self.index = index
        self.doc_ids = doc_ids

    def search(self, query_emb, top_k=5):
        q = np.array(query_emb).astype("float32")
        q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            doc_id = self.doc_ids[idx]
            results.append({"doc_id": doc_id, "score": float(score)})
        return results
