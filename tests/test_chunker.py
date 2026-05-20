import pytest

from app.rag.chunker import chunk_text


def test_chunk_text_creates_overlapping_chunks() -> None:
    text = "Ala ma kota. " * 80
    chunks = chunk_text(text, {"source": "test.txt"}, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "test.txt"
    assert chunks[0].metadata["chunk_id"] == 0
    assert all(chunk.text for chunk in chunks)


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("tekst", chunk_size=100, chunk_overlap=100)
