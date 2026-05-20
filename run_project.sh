#!/usr/bin/env bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "===================================="
echo " Chatbot LLM + RAG - Auto Launcher"
echo "===================================="
echo ""

cd "$PROJECT_DIR"

echo "[1/7] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed."
    echo "Install it with:"
    echo "sudo apt update && sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

echo "[2/7] Creating virtual environment if needed..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo "[3/7] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "[4/7] Upgrading pip..."
python -m pip install --upgrade pip

echo "[5/7] Installing requirements..."
pip install -r requirements.txt

echo "[6/7] Building FAISS index from documents..."
python scripts/ingest_documents.py

echo "[7/7] Running tests..."
pytest

echo ""
echo "===================================="
echo " Project is ready!"
echo "===================================="
echo ""
echo "Frontend:"
echo "http://127.0.0.1:8000"
echo ""
echo "API documentation:"
echo "http://127.0.0.1:8000/docs"
echo ""
echo "Health check:"
echo "http://127.0.0.1:8000/health"
echo ""
echo "Starting server..."
echo ""

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
