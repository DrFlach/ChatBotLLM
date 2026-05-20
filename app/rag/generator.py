import re

from openai import OpenAI

from app.core.config import get_settings


SYSTEM_PROMPT = (
    "You are a university study-program assistant. Answer only from the provided context. "
    "If the context does not contain the answer, say that the documents do not contain enough information. "
    "Answer in the same language as the user's question when possible."
)


def generate_answer(question: str, contexts: list[dict]) -> str:
    """Generate a grounded answer using OpenAI when configured, otherwise use extractive fallback."""

    settings = get_settings()
    if not contexts:
        return "Nie znaleziono informacji w dokumentach. / No relevant information was found in the documents."

    if settings.openai_api_key:
        return _generate_with_openai(question, contexts)
    return _extractive_fallback(question, contexts)


def _generate_with_openai(question: str, contexts: list[dict]) -> str:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    context_text = _format_context(contexts)

    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion:\n{question}",
            },
        ],
    )
    return response.choices[0].message.content or ""


def _extractive_fallback(question: str, contexts: list[dict]) -> str:
    question_terms = _keywords(question)
    sentences: list[tuple[int, int, int, str]] = []

    for context_index, context in enumerate(contexts):
        for sentence_index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", context["text"])):
            sentence = sentence.strip()
            if not sentence:
                continue
            score = len(question_terms.intersection(_keywords(sentence)))
            sentences.append((score, context_index, sentence_index, sentence))

    ranked = sorted(sentences, key=lambda item: (-item[0], item[1], item[2]))
    selected = [sentence for score, _, _, sentence in ranked if score > 0][:4]
    if not selected:
        selected = [contexts[0]["text"][:700]]

    intro = (
        "Answer generated from retrieved document fragments:"
        if _looks_english(question)
        else "Odpowiedz wygenerowana na podstawie znalezionych fragmentow dokumentow:"
    )
    return (
        f"{intro}\n"
        + "\n".join(f"- {sentence}" for sentence in selected)
    )


def _format_context(contexts: list[dict]) -> str:
    blocks = []
    for item in contexts:
        metadata = item["metadata"]
        source = metadata.get("source", "unknown")
        blocks.append(f"[Source: {source}] {item['text']}")
    return "\n\n".join(blocks)


def _keywords(text: str) -> set[str]:
    return {_normalize_token(token) for token in re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", text) if len(token) > 2}


def _normalize_token(token: str) -> str:
    normalized = token.lower()
    for suffix in ("owej", "iego", "ych", "iej", "ami", "ego", "owa", "cki", "ny", "ej", "em", "om", "ow", "ie", "ia", "a", "y", "i"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 3:
            return normalized[: -len(suffix)]
    return normalized


def _looks_english(text: str) -> bool:
    english_markers = {"what", "when", "where", "who", "which", "how", "are", "is", "the", "for"}
    return bool(english_markers.intersection(_keywords(text)))
