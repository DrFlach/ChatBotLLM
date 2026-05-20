from app.rag.chunker import TextChunk
import app.rag.vector_store as vector_store_module
from app.rag.vector_store import VectorStore


class FakeModel:
    pass


def fake_embed(model, texts):
    import numpy as np

    vectors = []
    for text in texts:
        normalized = text.lower()
        if "program" in normalized or "informatyka" in normalized or "semestr" in normalized:
            vectors.append([1.0, 0.0])
        else:
            vectors.append([0.0, 1.0])
    return np.array(vectors, dtype="float32")


def test_vector_store_search_returns_relevant_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(vector_store_module, "SentenceTransformer", lambda model_name: FakeModel())
    monkeypatch.setattr(vector_store_module, "_embed", fake_embed)

    chunks = [
        TextChunk("Program Informatyka semestr 1 zawiera Matematyke i Programowanie.", {"source": "program.txt"}),
        TextChunk("Konsultacje dr Kowalskiej odbywaja sie w poniedzialki.", {"source": "konsultacje.html"}),
    ]
    store = VectorStore.build(chunks, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    results = loaded.search("Jakie przedmioty sa na pierwszym semestrze informatyki?", top_k=1)

    assert results
    assert results[0]["metadata"]["source"] == "program.txt"
