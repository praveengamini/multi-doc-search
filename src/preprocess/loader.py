import os

def load_documents(folder_path):
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8", errors="ignore") as f:
                docs[filename] = f.read()
    return docs
