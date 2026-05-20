import pytest

from app.rag.chunker import chunk_text


def test_chunk_text_creates_overlapping_chunks() -> None:
    text = "\n\n".join(f"Akapit {index} zawiera opis przedmiotu i zasady zaliczenia." for index in range(30))
    chunks = chunk_text(text, {"source": "test.txt"}, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "test.txt"
    assert chunks[0].metadata["chunk_id"] == 0
    assert all(chunk.text for chunk in chunks)


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("tekst", chunk_size=100, chunk_overlap=100)


def test_chunk_text_does_not_start_in_middle_of_word() -> None:
    text = "\n\n".join(
        [
            "Pierwszy akapit opisuje program studiow oraz kierunek Informatyka.",
            "Drugi akapit opisuje przedmioty na drugim semestrze.",
            "Trzeci akapit opisuje egzaminy i zaliczenia.",
        ]
        * 5
    )

    chunks = chunk_text(text, {"source": "test.txt"}, chunk_size=150, chunk_overlap=40)

    assert len(chunks) > 1
    assert all(chunk.text[0].isupper() or chunk.text[0].isdigit() for chunk in chunks)
