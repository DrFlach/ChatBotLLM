from dataclasses import dataclass
import re


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
    """Split text into paragraph-aware chunks for better retrieval."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    blocks = _split_blocks(text)
    if not blocks:
        return []

    chunks: list[TextChunk] = []
    base_metadata = metadata or {}
    current: list[str] = []
    current_size = 0

    for block in blocks:
        if len(block) > chunk_size:
            if current:
                _append_chunk(chunks, current, base_metadata)
                current = []
                current_size = 0
            for part in _split_long_block(block, chunk_size):
                _append_chunk(chunks, [part], base_metadata)
            continue

        next_size = current_size + len(block) + (2 if current else 0)
        if current and next_size > chunk_size:
            _append_chunk(chunks, current, base_metadata)
            current = _overlap_blocks(current, chunk_overlap)
            current_size = len("\n\n".join(current))

        current.append(block)
        current_size = len("\n\n".join(current))

    if current:
        _append_chunk(chunks, current, base_metadata)

    return chunks


def _split_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", normalized)
    blocks: list[str] = []
    for paragraph in paragraphs:
        lines = [" ".join(line.split()) for line in paragraph.splitlines()]
        block = " ".join(line for line in lines if line).strip()
        if block:
            blocks.append(block)
    return blocks


def _split_long_block(text: str, chunk_size: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            if current:
                parts.append(current)
                current = ""
            parts.extend(_split_on_words(sentence, chunk_size))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > chunk_size:
            parts.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        parts.append(current)
    return parts


def _split_on_words(text: str, chunk_size: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    current_size = 0
    for word in text.split():
        additional = len(word) + (1 if current else 0)
        if current and current_size + additional > chunk_size:
            parts.append(" ".join(current))
            current = []
            current_size = 0
        current.append(word)
        current_size += additional
    if current:
        parts.append(" ".join(current))
    return parts


def _append_chunk(chunks: list[TextChunk], blocks: list[str], base_metadata: dict) -> None:
    text = "\n\n".join(blocks).strip()
    if not text:
        return
    chunk_metadata = dict(base_metadata)
    chunk_metadata["chunk_id"] = len(chunks)
    chunks.append(TextChunk(text=text, metadata=chunk_metadata))


def _overlap_blocks(blocks: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap == 0:
        return []
    overlap: list[str] = []
    size = 0
    for block in reversed(blocks):
        additional = len(block) + (2 if overlap else 0)
        if overlap and size + additional > chunk_overlap:
            break
        overlap.insert(0, block)
        size += additional
    return overlap
