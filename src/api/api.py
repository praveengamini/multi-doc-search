from fastapi import FastAPI
from pydantic import BaseModel
from src.embeddings.embedder import EmbeddingGenerator
from src.search.index_manager import IndexManager
from src.search.search_engine import SearchEngine
from src.preprocess.cleaner import clean_text
from src.search.ranking import explain_match
from src.search.query_expansion import expand_query_text
from src.search.reranker import Reranker
from fastapi.middleware.cors import CORSMiddleware

import json


import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load index
index_manager = IndexManager()
index_manager.load_index("vector.index")
embedder = EmbeddingGenerator()
search_engine = SearchEngine(index_manager.index, index_manager.doc_ids)

class Query(BaseModel):
    query: str
    top_k: int = 5
    use_expansion: bool = True       # toggle query expansion
    rerank_multiplier: int = 5      # initial candidates = top_k * multiplier

@app.post("/search")
def search_api(payload: Query):
    # 1. Clean + expand (optional)
    cleaned = clean_text(payload.query)
    expanded_text = expand_query_text(cleaned) if payload.use_expansion else cleaned

    # 2. Embed the original/expanded query with embedder
    query_emb = embedder.embed(expanded_text)

    # 3. Initial search: get a larger candidate pool
    candidate_count = max(payload.top_k * payload.rerank_multiplier, payload.top_k)
    initial = search_engine.search(query_emb, candidate_count)  # returns doc_id + score

    # 4. Load candidate texts for reranking
    candidates = []
    for r in initial:
        try:
            text = open(f"data/docs/{r['doc_id']}", "r", encoding="utf-8", errors="ignore").read()
        except FileNotFoundError:
            text = ""
        candidates.append((r["doc_id"], text))

    # 5. Rerank using cross-encoder
    reranker = Reranker()
    reranked = reranker.rerank(cleaned, candidates, payload.top_k)  # uses original cleaned query for semantic pair scoring

    # 6. Build final output with explanations
    final = []
    for item in reranked:
        doc_id = item["doc_id"]
        score = item["score"]
        preview = item["preview"]
        # load full text for explanation tokens
        full_text = ""
        try:
            full_text = open(f"data/docs/{doc_id}", "r", encoding="utf-8", errors="ignore").read()
        except FileNotFoundError:
            pass

        explanation = explain_match(cleaned, full_text)

        final.append({
            "doc_id": doc_id,
            "score": score,
            "preview": preview,
            "explanation": explanation
        })

    return {"results": final}

