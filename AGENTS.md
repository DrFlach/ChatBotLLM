# AGENTS.md

## Project Notes

This is a university MVP project for a FastAPI chatbot using Retrieval-Augmented Generation.

## Commands

- Install dependencies: `pip install -r requirements.txt`
- Build vector index: `python scripts/ingest_documents.py`
- Run API and UI: `uvicorn app.main:app --reload`
- Run tests: `pytest`

## Implementation Rules

- Keep code simple and readable for academic review.
- Answers must be grounded in retrieved context and include sources.
- The fallback generator must work without an OpenAI API key.
- Use the multilingual sentence-transformers model required by the project.
