from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.embeddings.embedder import EmbeddingGenerator
from src.search.index_manager import IndexManager
from src.search.search_engine import SearchEngine
from src.preprocess.cleaner import clean_text
from src.search.ranking import explain_match
from src.search.query_expansion import expand_query_text
from src.search.reranker import Reranker

import os
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals (filled on startup)
index_manager = IndexManager()
search_engine = None
embedder = EmbeddingGenerator()
INDEX_PATH = "vector.index"


# ------------------------
#  LOAD INDEX SAFELY
# ------------------------

def load_faiss_index():
    """Try loading FAISS index. If missing → index_ready=False but API will not crash."""
    global search_engine

    if os.path.exists(INDEX_PATH):
        try:
            index_manager.load_index(INDEX_PATH)
            search_engine = SearchEngine(index_manager.index, index_manager.doc_ids)
            app.state.index_ready = True
            print("✔ FAISS index loaded.")
        except Exception as e:
            app.state.index_ready = False
            print("❌ Failed to load FAISS index:", e)
    else:
        app.state.index_ready = False
        print("⚠ No FAISS index found at startup.")


@app.on_event("startup")
def on_startup():
    # Try loading existing index
    load_faiss_index()


# ------------------------
#     REQUEST MODEL
# ------------------------

class Query(BaseModel):
    query: str
    top_k: int = 5
    use_expansion: bool = True
    rerank_multiplier: int = 5


# ------------------------
#       SEARCH API
# ------------------------

@app.post("/search")
def search_api(payload: Query):
    # If index is missing → don't crash. Tell user to build index.
    if not getattr(app.state, "index_ready", False):
        raise HTTPException(
            status_code=503,
            detail="Vector index is not ready. Please build it first using python src/main.py"
        )

    # 1. Clean and expand
    cleaned = clean_text(payload.query)
    expanded_text = expand_query_text(cleaned) if payload.use_expansion else cleaned

    # 2. Embed query
    query_emb = embedder.embed(expanded_text)

    # 3. Initial search pool
    candidate_count = max(payload.top_k * payload.rerank_multiplier, payload.top_k)
    initial = search_engine.search(query_emb, candidate_count)

    # 4. Load candidate texts
    candidates = []
    for r in initial:
        try:
            text = open(f"data/docs/{r['doc_id']}", "r", encoding="utf-8").read()
        except FileNotFoundError:
            text = ""
        candidates.append((r["doc_id"], text))

    # 5. Rerank using cross-encoder
    reranker = Reranker()
    reranked = reranker.rerank(cleaned, candidates, payload.top_k)

    # 6. Build final output
    final = []
    for item in reranked:
        doc_id = item["doc_id"]
        score = item["score"]
        preview = item["preview"]

        try:
            full_text = open(f"data/docs/{doc_id}", "r", encoding="utf-8").read()
        except FileNotFoundError:
            full_text = ""

        explanation = explain_match(cleaned, full_text)

        final.append({
            "doc_id": doc_id,
            "score": score,
            "preview": preview,
            "explanation": explanation
        })

    return {"results": final}
