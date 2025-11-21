import json
import os
import time

class CacheManager:
    def __init__(self, db_path):
        self.db_path = db_path
        if not os.path.exists(db_path):
            with open(db_path, "w") as f:
                json.dump({}, f)

    def get(self, doc_id, new_hash):
        db = json.load(open(self.db_path))
        entry = db.get(doc_id)

        if entry and entry["hash"] == new_hash:
            return entry
        return None

    def save(self, doc_id, embedding, hash_value):
        db = json.load(open(self.db_path))
        db[doc_id] = {
            "embedding": embedding,
            "hash": hash_value,
            "updated_at": time.time()
        }
        json.dump(db, open(self.db_path, "w"), indent=2)
