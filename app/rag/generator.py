import re

from openai import OpenAI

from app.core.config import get_settings
from app.rag.retriever import detect_subject


SYSTEM_PROMPT = (
    "You are a university study-program assistant using Retrieval-Augmented Generation. "
    "Answer only with facts present in the retrieved context. Do not invent information. "
    "Answer in the same language as the user's question. "
    "Keep the main answer concise and human-readable. Do not paste raw retrieved fragments. "
    "Use bullet points for lists, assessment components, exam rules and subjects. "
    "Do not include source file names in the main answer because sources are displayed separately. "
    "If the context is insufficient, clearly say that the information was not found in the documents."
)

MISSING_PL = "Nie znalazłem tej informacji w dostępnych dokumentach."
MISSING_EN = "I could not find this information in the available documents."


def generate_answer(question: str, contexts: list[dict]) -> str:
    """Generate a grounded answer using OpenAI when configured, otherwise use extractive fallback."""

    settings = get_settings()
    if not contexts:
        return _missing_answer(question)

    if settings.openai_api_key:
        return _generate_with_openai(question, contexts)
    return _extractive_fallback(question, contexts)


def _generate_with_openai(question: str, contexts: list[dict]) -> str:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    language = "English" if detect_language(question) == "en" else "Polish"
    context_text = _format_context(contexts)

    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Required answer language: {language}\n\n"
                    f"Retrieved context:\n{context_text}\n\n"
                    f"Question:\n{question}\n\n"
                    "Use only the retrieved context. If the answer is missing, say so clearly. "
                    "Answer directly without technical introductions."
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""


def _extractive_fallback(question: str, contexts: list[dict]) -> str:
    language = detect_language(question)
    if _is_subject_list_question(question):
        subject_answer = _subject_list_answer(question, contexts, language)
        return subject_answer or _missing_answer(question)
    if _is_assessment_question(question):
        assessment_answer = _assessment_answer(question, contexts, language)
        return assessment_answer or _missing_answer(question)
    if _is_consultation_question(question):
        consultation_answer = _consultation_answer(question, contexts, language)
        return consultation_answer or _missing_answer(question)

    subject = detect_subject(question, [{"metadata": context.get("metadata", {})} for context in contexts])
    if subject:
        contexts = _subject_contexts(contexts, subject)
    question_terms = _keywords(question)
    sentences: list[tuple[float, int, int, str]] = []
    seen_sentences: set[str] = set()

    for context_index, context in enumerate(contexts):
        for sentence_index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", context["text"])):
            sentence = _clean_sentence(sentence)
            sentence_key = sentence.lower()
            if not sentence or sentence_key in seen_sentences:
                continue
            seen_sentences.add(sentence_key)
            sentence_terms = _keywords(sentence)
            overlap = question_terms.intersection(sentence_terms)
            score = len(overlap)
            if score:
                score += _metadata_bonus(question_terms, context.get("metadata", {}))
            sentences.append((score, context_index, sentence_index, sentence))

    ranked = sorted(sentences, key=lambda item: (-item[0], item[1], item[2]))
    selected = [sentence for score, _, _, sentence in ranked if score >= 1.0][:4]
    if not selected:
        return _missing_answer(question)

    if len(selected) == 1:
        return selected[0]
    return "\n".join(f"- {sentence}" for sentence in selected)


def _format_context(contexts: list[dict]) -> str:
    blocks = []
    for item in contexts:
        metadata = item["metadata"]
        source = metadata.get("source", "unknown")
        metadata_text = ", ".join(f"{key}: {value}" for key, value in metadata.items() if value)
        blocks.append(f"[Source: {source}; Metadata: {metadata_text}]\n{item['text']}")
    return "\n\n".join(blocks)


def _keywords(text: str) -> set[str]:
    stopwords = {
        "jak", "jakie", "jest", "dla", "oraz", "the", "are", "what", "when", "where", "who",
        "which", "how", "for", "this", "that", "czy", "sie", "się",
    }
    return {
        _normalize_token(token)
        for token in re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", text)
        if len(token) > 2 and _normalize_token(token) not in stopwords
    }


def _normalize_token(token: str) -> str:
    normalized = token.lower()
    for suffix in ("owej", "iego", "ych", "iej", "ami", "ego", "owa", "cki", "ny", "ej", "em", "om", "ow", "ie", "ia", "a", "y", "i"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 3:
            return normalized[: -len(suffix)]
    return normalized


def detect_language(text: str) -> str:
    english_markers = {"what", "when", "where", "who", "which", "how", "are", "is", "the", "for", "course", "subjects"}
    polish_markers = {"jak", "jakie", "kiedy", "gdzie", "kto", "czy", "dla", "oraz", "semestrze", "przedmioty"}
    lowered = text.lower()
    if any(char in lowered for char in "ąćęłńóśźż"):
        return "pl"
    terms = {_normalize_token(token) for token in re.findall(r"[A-Za-z]+", lowered)}
    if english_markers.intersection(terms) and not polish_markers.intersection(terms):
        return "en"
    return "pl"


def _missing_answer(question: str) -> str:
    return MISSING_EN if detect_language(question) == "en" else MISSING_PL


def _clean_sentence(sentence: str) -> str:
    return " ".join(sentence.split()).strip(" -")


def _metadata_bonus(question_terms: set[str], metadata: dict) -> float:
    metadata_text = " ".join(str(value) for value in metadata.values() if value)
    metadata_terms = _keywords(metadata_text)
    return min(len(question_terms.intersection(metadata_terms)) * 0.5, 2.0)


def _is_subject_list_question(question: str) -> bool:
    lowered = question.lower()
    subject_words = ("przedmiot", "przedmioty", "subjects", "courses", "classes")
    semester_words = ("semestr", "semester")
    list_words = ("jakie", "lista", "wymien", "which", "what", "included", "list")
    return (
        any(word in lowered for word in subject_words)
        and any(word in lowered for word in semester_words)
        and any(word in lowered for word in list_words)
    )


def _subject_list_answer(question: str, contexts: list[dict], language: str) -> str | None:
    subjects: list[dict] = []
    seen: set[tuple[str, str]] = set()
    requested_semester = _extract_semester(question)
    for context in contexts:
        metadata = context.get("metadata", {})
        subject = metadata.get("subject")
        semester = metadata.get("semester")
        if not subject:
            continue
        if requested_semester and str(semester) != requested_semester:
            continue
        key = (str(semester), str(subject))
        if key in seen:
            continue
        seen.add(key)
        subjects.append(metadata)

    if not subjects:
        return None

    subjects.sort(key=lambda item: (str(item.get("semester", "")), str(item.get("subject", ""))))
    if language == "en":
        lines = [f"Semester {requested_semester} includes:" if requested_semester else "Subjects found:"]
        for item in subjects:
            details = [
                f"{item.get('ects')} ECTS" if item.get("ects") else None,
                f"lecturer: {item.get('lecturer')}" if item.get("lecturer") else None,
            ]
            suffix = f" ({', '.join(detail for detail in details if detail)})" if any(details) else ""
            lines.append(f"- {_english_subject_name(str(item.get('subject')))}{suffix}")
        return "\n".join(lines)

    lines = [f"Na semestrze {requested_semester} są:" if requested_semester else "Znalezione przedmioty:"]
    for item in subjects:
        details = [
            f"{item.get('ects')} ECTS" if item.get("ects") else None,
            f"prowadzacy: {item.get('lecturer')}" if item.get("lecturer") else None,
        ]
        suffix = f" ({', '.join(detail for detail in details if detail)})" if any(details) else ""
        lines.append(f"- {item.get('subject')}{suffix}")
    return "\n".join(lines)


def _extract_semester(question: str) -> str | None:
    lowered = question.lower()
    match = re.search(r"(?:semestr|semester|semestrze)\s*(\d+)", lowered)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)\s*(?:semestr|semester|semestrze)", lowered)
    if match:
        return match.group(1)
    return None


def _assessment_answer(question: str, contexts: list[dict], language: str) -> str | None:
    subject = detect_subject(question, [{"metadata": context.get("metadata", {})} for context in contexts])
    relevant = _subject_contexts(contexts, subject) if subject else contexts

    structured = next(
        (
            context.get("metadata", {})
            for context in relevant
            if context.get("metadata", {}).get("assessment_method")
        ),
        None,
    )
    if structured:
        subject_name = str(structured.get("subject") or subject or "")
        components = _assessment_components(str(structured.get("assessment_method", "")), language)
        exam_date = structured.get("exam_date") or _find_exam_date(subject_name, relevant)
        if not components and not exam_date:
            return None
        if language == "en":
            course = _english_subject_name(subject_name)
            lines = [f"The {course} course is assessed through:"]
        else:
            lines = [f"Zaliczenie przedmiotu {subject_name} obejmuje:"]
        lines.extend(f"- {component}" for component in components)
        if exam_date:
            if language == "en":
                lines.append(f"The exam/test is scheduled for {_english_date(str(exam_date))}.")
            else:
                lines.append(f"Egzamin/test odbywa się {exam_date}.")
        return "\n".join(lines)

    sentence_answer = _sentence_answer(question, relevant, language, max_sentences=3)
    return sentence_answer


def _consultation_answer(question: str, contexts: list[dict], language: str) -> str | None:
    relevant = contexts
    subject = detect_subject(question, [{"metadata": context.get("metadata", {})} for context in contexts])
    if subject:
        lecturer = next(
            (
                context.get("metadata", {}).get("lecturer")
                for context in contexts
                if _matches_subject(context, subject) and context.get("metadata", {}).get("lecturer")
            ),
            None,
        )
        if lecturer:
            relevant = [context for context in contexts if lecturer.lower() in _plain(context.get("text", "")).lower()]

    for context in relevant:
        metadata = context.get("metadata", {})
        text = _clean_sentence(context.get("text", ""))
        match = re.search(r"(Konsultacje:\s*[^.]+[.]?)", text, flags=re.IGNORECASE)
        if match:
            lecturer = metadata.get("lecturer")
            consultation = match.group(1)
            if language == "en":
                return f"{lecturer + ': ' if lecturer else ''}{_translate_consultation(consultation)}"
            return f"{lecturer + ': ' if lecturer else ''}{consultation}"
    return _sentence_answer(question, relevant, language, max_sentences=2)


def _sentence_answer(question: str, contexts: list[dict], language: str, max_sentences: int = 4) -> str | None:
    question_terms = _keywords(question)
    ranked: list[tuple[float, int, int, str]] = []
    seen: set[str] = set()
    for context_index, context in enumerate(contexts):
        for sentence_index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", context.get("text", ""))):
            sentence = _clean_sentence(sentence)
            key = sentence.lower()
            if not sentence or key in seen:
                continue
            seen.add(key)
            score = len(question_terms.intersection(_keywords(sentence)))
            if score:
                score += _metadata_bonus(question_terms, context.get("metadata", {}))
                ranked.append((score, context_index, sentence_index, sentence))
    selected = [item[3] for item in sorted(ranked, key=lambda item: (-item[0], item[1], item[2])) if item[0] >= 1][:max_sentences]
    if not selected:
        return None
    if len(selected) == 1:
        return selected[0]
    return "\n".join(f"- {sentence}" for sentence in selected)


def _subject_contexts(contexts: list[dict], subject: str | None) -> list[dict]:
    if not subject:
        return contexts
    matches = [context for context in contexts if _matches_subject(context, subject)]
    return matches or contexts


def _matches_subject(context: dict, subject: str) -> bool:
    aliases = {
        _plain(subject),
        _plain(_english_subject_name(subject)),
    }
    metadata = context.get("metadata", {})
    text = " ".join(str(value) for value in metadata.values() if value) + " " + context.get("text", "")
    plain = _plain(text)
    return any(alias and alias in plain for alias in aliases)


def _assessment_components(text: str, language: str) -> list[str]:
    components: list[str] = []
    for raw_part in re.split(r",|;", text):
        part = raw_part.strip(" .")
        if not part or "%" not in part:
            continue
        if "warunkiem" in part.lower() or "minimum" in part.lower():
            continue
        components.append(_translate_assessment_component(part) if language == "en" else _lower_first(part))
    return _deduplicate(components)


def _find_exam_date(subject: str, contexts: list[dict]) -> str | None:
    if not subject:
        return None
    pattern = re.compile(rf"{re.escape(subject)}:\s*[^.]*?(\d{{1,2}}\s+\w+\s+\d{{4}})", flags=re.IGNORECASE)
    for context in contexts:
        match = pattern.search(context.get("text", ""))
        if match:
            return match.group(1)
    return None


def _is_assessment_question(question: str) -> bool:
    lowered = _plain(question)
    return any(marker in lowered for marker in ("zaliczenie", "zaliczyc", "assessment", "assess", "exam", "egzamin", "passing", "test", "ocen"))


def _is_consultation_question(question: str) -> bool:
    lowered = _plain(question)
    return any(marker in lowered for marker in ("konsultacje", "consultation", "office hours", "dyzur"))


def _english_subject_name(subject: str) -> str:
    names = {
        "Bazy danych": "Databases",
        "Algorytmy i struktury danych": "Algorithms and Data Structures",
        "Systemy operacyjne": "Operating Systems",
        "Sieci komputerowe": "Computer Networks",
        "Wstep do programowania": "Introduction to Programming",
        "Matematyka dyskretna": "Discrete Mathematics",
        "Podstawy sztucznej inteligencji": "Artificial Intelligence Basics",
        "Architektura komputerow": "Computer Architecture",
    }
    return names.get(subject, subject)


def _translate_assessment_component(text: str) -> str:
    translations = {
        "Test SQL": "SQL test",
        "projekt zespolowy": "team project",
        "odpowiedz ustna": "oral answer",
        "Projekt programistyczny": "programming project",
        "laboratoria": "labs",
        "egzamin": "exam",
        "test koncowy": "final test",
        "Kolokwium": "test",
        "egzamin pisemny": "written exam",
    }
    translated = text
    for source, target in translations.items():
        translated = re.sub(source, target, translated, flags=re.IGNORECASE)
    return _lower_first(translated)


def _translate_consultation(text: str) -> str:
    translations = {
        "Konsultacje": "Consultation hours",
        "poniedzialek": "Monday",
        "sroda": "Wednesday",
        "czwartek": "Thursday",
        "piatek": "Friday",
        "pokoj": "room",
        "budynek": "building",
    }
    translated = text
    for source, target in translations.items():
        translated = re.sub(source, target, translated, flags=re.IGNORECASE)
    return translated


def _english_date(text: str) -> str:
    months = {
        "stycznia": "January",
        "lutego": "February",
        "marca": "March",
        "kwietnia": "April",
        "maja": "May",
        "czerwca": "June",
        "lipca": "July",
        "sierpnia": "August",
        "wrzesnia": "September",
        "pazdziernika": "October",
        "listopada": "November",
        "grudnia": "December",
    }
    parts = text.split()
    if len(parts) == 3 and parts[1].lower() in months:
        return f"{int(parts[0])} {months[parts[1].lower()]} {parts[2]}"
    return text


def _plain(text: str) -> str:
    replacements = str.maketrans("ąćęłńóśźż", "acelnoszz")
    return " ".join(text.lower().translate(replacements).split())


def _lower_first(text: str) -> str:
    if len(text) > 1 and text[:2].isupper():
        return text
    return text[:1].lower() + text[1:] if text else text


def _deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
