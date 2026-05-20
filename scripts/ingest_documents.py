import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.rag.chunker import chunk_text
from app.rag.ingest import load_documents
from app.rag.vector_store import VectorStore


def main() -> None:
    settings = get_settings()
    records = load_documents(settings.raw_data_dir)

    chunks = []
    for record in records:
        chunks.extend(
            chunk_text(
                record["text"],
                metadata=record["metadata"],
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )

    if not chunks:
        raise RuntimeError(f"No documents were loaded from {settings.raw_data_dir}")

    store = VectorStore.build(chunks, settings.embedding_model_name)
    store.save(settings.index_dir)
    print(f"Indexed {len(chunks)} chunks from {len(records)} document records into {settings.index_dir}")


if __name__ == "__main__":
    main()
