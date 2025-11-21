import faiss
import numpy as np
import pickle
import os

class IndexManager:
    def __init__(self):
        self.index = None
        self.doc_ids = []

    def build_index(self, embeddings, ids):
        vectors = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.doc_ids = ids

    def save_index(self, path):
        faiss.write_index(self.index, path)
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.doc_ids, f)

    def load_index(self, path):
        self.index = faiss.read_index(path)
        self.doc_ids = pickle.load(open(path + ".meta", "rb"))
