from preprocess.loader import load_documents
from preprocess.cleaner import clean_text
from preprocess.metadata import get_doc_metadata
from embeddings.embedder import EmbeddingGenerator
from cache.cache_manager import CacheManager
from search.index_manager import IndexManager
import os

def main():
    docs_path = "data/docs"
    documents = load_documents(docs_path)

    cache = CacheManager("cache_db.json")
    embedder = EmbeddingGenerator()
    index_manager = IndexManager()

    embeddings = []
    ids = []

    for doc_id, text in documents.items():
        cleaned = clean_text(text)
        meta = get_doc_metadata(cleaned)

        cached = cache.get(doc_id, meta["hash"])
        if cached:
            embedding = cached["embedding"]
        else:
            embedding = embedder.embed(cleaned)
            cache.save(doc_id, embedding, meta["hash"])

        ids.append(doc_id)
        embeddings.append(embedding)

    index_manager.build_index(embeddings, ids)
    index_manager.save_index("vector.index")

    print("Index building complete!")

if __name__ == "__main__":
    main()
