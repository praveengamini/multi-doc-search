from fastapi import FastAPI
from pydantic import BaseModel
from src.embeddings.embedder import EmbeddingGenerator
from src.search.index_manager import IndexManager
from src.search.search_engine import SearchEngine
from src.preprocess.cleaner import clean_text
from src.search.ranking import explain_match


import json

app = FastAPI()

# Load index
index_manager = IndexManager()
index_manager.load_index("vector.index")
embedder = EmbeddingGenerator()
search_engine = SearchEngine(index_manager.index, index_manager.doc_ids)

class Query(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search_api(payload: Query):
    cleaned = clean_text(payload.query)
    query_emb = embedder.embed(cleaned)

    results = search_engine.search(query_emb, payload.top_k)

    final = []
    for r in results:
        doc_id = r["doc_id"]
        text = open(f"data/docs/{doc_id}", "r", encoding="utf-8", errors="ignore").read()
        preview = text[:200] + "..."

        explanation = explain_match(cleaned, text)

        final.append({
            "doc_id": doc_id,
            "score": r["score"],
            "preview": preview,
            "explanation": explanation
        })

    return {"results": final}
