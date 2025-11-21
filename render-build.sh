#!/bin/bash
set -e

echo ">>> Setting PYTHONPATH"
export PYTHONPATH=/opt/render/project/src

echo ">>> Installing dependencies..."
pip install -r requirements.txt

echo ">>> Downloading dataset..."
python3 src/preprocess/download_data.py

echo ">>> Building embeddings + FAISS index..."
python3 src/main.py

echo ">>> Build complete!"
