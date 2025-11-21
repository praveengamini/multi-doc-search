# Optional bonus for multiprocessing
def batch_embed(docs, embedder):
    return [embedder.embed(text) for text in docs]
