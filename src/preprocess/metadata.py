import hashlib

def get_doc_metadata(text):
    return {
        "length": len(text.split()),
        "hash": hashlib.sha256(text.encode("utf-8")).hexdigest()
    }
