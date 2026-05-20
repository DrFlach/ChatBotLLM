from functools import lru_cache

from app.core.config import get_settings
from app.rag.vector_store import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore.load(settings.index_dir, settings.embedding_model_name)


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    settings = get_settings()
    store = get_vector_store()
    return store.search(question, top_k or settings.top_k)
