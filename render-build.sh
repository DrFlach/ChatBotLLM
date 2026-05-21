#!/usr/bin/env bash

set -e

echo "===================================="
echo " Chatbot LLM + RAG - Render Build"
echo "===================================="
echo ""

echo "[1/3] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/3] Building FAISS index..."
python scripts/ingest_documents.py

echo ""
echo "[3/3] Build completed successfully."
