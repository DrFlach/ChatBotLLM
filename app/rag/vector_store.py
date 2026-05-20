import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.chunker import TextChunk


INDEX_FILE = "faiss.index"
DOCS_FILE = "documents.json"


class VectorStore:
    """Small FAISS wrapper that stores vectors plus chunk text and metadata."""

    def __init__(self, index: faiss.Index, documents: list[dict], model: SentenceTransformer):
        self.index = index
        self.documents = documents
        self.model = model

    @classmethod
    def build(cls, chunks: list[TextChunk], model_name: str) -> "VectorStore":
        if not chunks:
            raise ValueError("Cannot build a vector store without chunks")

        model = _load_embedding_model(model_name)
        texts = [chunk.text for chunk in chunks]
        embeddings = _embed(model, texts)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        documents = [{"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks]
        return cls(index=index, documents=documents, model=model)

    @classmethod
    def load(cls, index_dir: Path, model_name: str) -> "VectorStore":
        index_path = index_dir / INDEX_FILE
        docs_path = index_dir / DOCS_FILE

        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(
                "Vector index not found. Run: python scripts/ingest_documents.py"
            )

        model = _load_embedding_model(model_name)
        index = faiss.read_index(str(index_path))
        documents = json.loads(docs_path.read_text(encoding="utf-8"))
        return cls(index=index, documents=documents, model=model)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / INDEX_FILE))
        (index_dir / DOCS_FILE).write_text(
            json.dumps(self.documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        query_embedding = _embed(self.model, [query])
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results: list[dict] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index == -1:
                continue
            document = self.documents[int(index)]
            results.append(
                {
                    "text": document["text"],
                    "metadata": document["metadata"],
                    "score": float(score),
                }
            )
        return results


def _embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def _load_embedding_model(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name)
    except Exception:
        try:
            return SentenceTransformer(model_name, local_files_only=True)
        except Exception as offline_error:
            raise RuntimeError(
                "Could not load embedding model. Check internet access for the first run "
                "or verify that the model is already cached locally."
            ) from offline_error
