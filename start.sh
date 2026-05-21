#!/usr/bin/env bash

set -e

echo "===================================="
echo " Chatbot LLM + RAG - Render Start"
echo "===================================="
echo ""

echo "[1/2] Python version:"
python --version

echo ""
echo "[2/2] Starting FastAPI server..."
echo "Render PORT: ${PORT:-8000}"
echo ""

exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
