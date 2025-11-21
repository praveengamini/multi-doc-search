import hashlib

def compute_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()
