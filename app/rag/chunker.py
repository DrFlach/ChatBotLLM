from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    metadata: dict


def chunk_text(
    text: str,
    metadata: dict | None = None,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[TextChunk]:
    """Split text into overlapping chunks for better retrieval."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[TextChunk] = []
    start = 0
    base_metadata = metadata or {}

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))

        # Prefer ending a chunk at a sentence boundary when one is nearby.
        if end < len(cleaned):
            sentence_end = max(cleaned.rfind(". ", start, end), cleaned.rfind("? ", start, end), cleaned.rfind("! ", start, end))
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1

        chunk = cleaned[start:end].strip()
        if chunk:
            chunk_metadata = dict(base_metadata)
            chunk_metadata["chunk_id"] = len(chunks)
            chunks.append(TextChunk(text=chunk, metadata=chunk_metadata))

        if end >= len(cleaned):
            break
        start = max(0, end - chunk_overlap)

    return chunks
