#!/usr/bin/env bash

set -e

echo "===================================="
echo " Chatbot LLM + RAG - Render Start"
echo "===================================="
echo ""

echo "[1/3] Checking Python version..."
python --version

echo ""
echo "[2/3] Checking FAISS index..."

if [ ! -d "data/index" ]; then
    echo "WARNING: data/index directory not found."
    echo "Trying to build FAISS index before start..."
    python scripts/ingest_documents.py
fi

if [ ! -f "data/index/faiss.index" ]; then
    echo "WARNING: data/index/faiss.index not found."
    echo "Trying to build FAISS index before start..."
    python scripts/ingest_documents.py
fi

echo "FAISS index is ready."

echo ""
echo "[3/3] Starting FastAPI server..."
echo "Render PORT: ${PORT:-8000}"
echo ""

exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
