#!/usr/bin/env bash

set -e

echo "===================================="
echo " Chatbot LLM + RAG - Render Start"
echo "===================================="
echo ""

echo "[1/4] Checking Python version..."
python --version

echo ""
echo "[2/4] Checking project files..."

if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found."
    exit 1
fi

if [ ! -d "app" ]; then
    echo "ERROR: app directory not found."
    exit 1
fi

if [ ! -d "scripts" ]; then
    echo "ERROR: scripts directory not found."
    exit 1
fi

if [ ! -d "data/raw" ]; then
    echo "ERROR: data/raw directory not found."
    echo "The chatbot needs source documents in data/raw."
    exit 1
fi

echo "Project files found."

echo ""
echo "[3/4] Building FAISS index from data/raw..."
python scripts/ingest_documents.py

echo ""
echo "[4/4] Starting FastAPI server..."
echo "Render PORT: ${PORT:-8000}"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
