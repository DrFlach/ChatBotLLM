from functools import lru_cache
import re

from app.core.config import get_settings
from app.rag.vector_store import VectorStore


SUBJECT_ALIASES = {
    "Bazy danych": ("bazy danych", "baz danych", "bazach danych", "databases", "database"),
    "Algorytmy i struktury danych": (
        "algorytmy i struktury danych",
        "algorytmow i struktur danych",
        "algorithms and data structures",
        "algorithms",
    ),
    "Systemy operacyjne": ("systemy operacyjne", "systemow operacyjnych", "systemach operacyjnych", "operating systems"),
    "Sieci komputerowe": ("sieci komputerowe", "computer networks"),
    "Wstep do programowania": ("wstep do programowania", "introduction to programming", "programming"),
    "Matematyka dyskretna": ("matematyka dyskretna", "discrete mathematics"),
    "Podstawy sztucznej inteligencji": (
        "podstawy sztucznej inteligencji",
        "artificial intelligence",
        "ai",
    ),
    "Architektura komputerow": ("architektura komputerow", "computer architecture"),
    "Jezyk angielski B2": ("jezyk angielski b2", "english b2"),
    "Podstawy technologii webowych": ("podstawy technologii webowych", "web technologies basics", "web technologies"),
    "Programowanie obiektowe": ("programowanie obiektowe", "object-oriented programming", "oop"),
    "Inzynieria oprogramowania": ("inzynieria oprogramowania", "software engineering"),
    "Programowanie aplikacji webowych": ("programowanie aplikacji webowych", "web application development", "web applications"),
    "Statystyka dla informatykow": ("statystyka dla informatykow", "statistics for computer scientists", "statistics"),
    "Bezpieczenstwo systemow IT": ("bezpieczenstwo systemow it", "it systems security", "security"),
    "Projekt zespolowy": ("projekt zespolowy", "team software project", "team project"),
    "Chmury obliczeniowe": ("chmury obliczeniowe", "cloud computing"),
    "Aplikacje mobilne": ("aplikacje mobilne", "mobile applications", "mobile apps"),
    "Hurtownie danych": ("hurtownie danych", "data warehouses", "data warehousing"),
    "Metody numeryczne": ("metody numeryczne", "numerical methods"),
}


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore.load(settings.index_dir, settings.embedding_model_name)


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    settings = get_settings()
    store = get_vector_store()
    limit = top_k or settings.top_k
    expanded_question = expand_query(question)
    results = store.search(question, limit)
    if expanded_question != question:
        results = _merge_results(store.search(expanded_question, limit), results)

    if _is_subject_list_question(question):
        results = _merge_results(_structured_subject_results(store, question), results)
    elif _is_regulation_question(question):
        expanded_query = f"{question} {_regulation_query_terms(question)}"
        results = _merge_results(
            _regulation_results(store, question),
            _merge_results(store.search(expanded_query, limit), results),
        )
    else:
        subject = detect_subject(question, store.documents)
        if subject:
            subject_results = _subject_results(store, subject, question)
            if subject_results:
                results = _merge_results(subject_results, results)
                if _is_assessment_question(question):
                    results = _filter_subject_assessment_results(results, subject)
                else:
                    results = _filter_subject_results(results, subject)

    return results[:limit]


def vector_store_status() -> dict:
    settings = get_settings()
    try:
        store = get_vector_store()
    except Exception:
        return {"index_loaded": False, "chunk_count": None}
    return {"index_loaded": True, "chunk_count": store.chunk_count}


def _structured_subject_results(store: VectorStore, question: str) -> list[dict]:
    semester = _extract_semester(question)
    field = _extract_field(question)
    matches: list[dict] = []

    for document in store.documents:
        metadata = document.get("metadata", {})
        if not metadata.get("subject"):
            continue
        if semester and str(metadata.get("semester")) != semester:
            continue
        if field and field not in str(metadata.get("field", "")).lower():
            continue
        matches.append({"text": document["text"], "metadata": metadata, "score": 1.0})

    return sorted(matches, key=lambda item: (str(item["metadata"].get("semester", "")), str(item["metadata"].get("subject", ""))))


def _subject_results(store: VectorStore, subject: str, question: str) -> list[dict]:
    matches: list[dict] = []
    assessment_question = _is_assessment_question(question)
    aliases = _aliases_for_subject(subject, store.documents)

    for document in store.documents:
        metadata = document.get("metadata", {})
        metadata_subject = str(metadata.get("subject", ""))
        text = document.get("text", "")
        text_blob = _normalize_text(" ".join([metadata_subject, text]))

        exact_metadata_match = _normalize_text(metadata_subject) == _normalize_text(subject)
        text_match = any(alias in text_blob for alias in aliases)
        if not exact_metadata_match and not text_match:
            continue
        if assessment_question and metadata.get("document_type") not in {"csv", "txt"}:
            continue

        matches.append({"text": text, "metadata": metadata, "score": 1.0 if exact_metadata_match else 0.85})

    return sorted(matches, key=lambda item: (0 if item["metadata"].get("subject") == subject else 1, -item["score"]))


def _regulation_results(store: VectorStore, question: str) -> list[dict]:
    query_terms = set(_normalize_text(_regulation_query_terms(question)).split())
    matches: list[dict] = []
    for document in store.documents:
        metadata = document.get("metadata", {})
        text = document.get("text", "")
        source = str(metadata.get("source", ""))
        blob = _normalize_text(f"{source} {text}")
        if "regulamin" not in blob and "rules" not in blob:
            continue
        score = 0.9 + min(len(query_terms.intersection(blob.split())) * 0.02, 0.08)
        matches.append({"text": text, "metadata": metadata, "score": score})
    return sorted(matches, key=lambda item: -item["score"])


def _merge_results(primary: list[dict], secondary: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple] = set()
    for item in primary + secondary:
        metadata = item.get("metadata", {})
        key = (metadata.get("source"), metadata.get("row"), metadata.get("page"), metadata.get("section"), metadata.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def detect_subject(question: str, documents: list[dict] | None = None) -> str | None:
    aliases_by_subject = dict(SUBJECT_ALIASES)
    if documents:
        for document in documents:
            subject = document.get("metadata", {}).get("subject")
            if subject and subject not in aliases_by_subject:
                aliases_by_subject[str(subject)] = (str(subject),)

    lowered = _normalize_text(question)
    best: tuple[int, str] | None = None
    for subject, aliases in aliases_by_subject.items():
        for alias in _aliases_for_subject(subject, documents or []):
            if alias and alias in lowered:
                candidate = (len(alias), subject)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    return best[1] if best else None


def _filter_subject_results(results: list[dict], subject: str) -> list[dict]:
    subject_matches = [item for item in results if _item_matches_subject(item, subject)]
    return subject_matches if subject_matches else results


def _filter_subject_assessment_results(results: list[dict], subject: str) -> list[dict]:
    filtered = [
        item for item in results
        if _item_matches_subject(item, subject)
        and item.get("metadata", {}).get("document_type") in {"csv", "txt", None}
    ]
    return filtered if filtered else _filter_subject_results(results, subject)


def _item_matches_subject(item: dict, subject: str) -> bool:
    metadata = item.get("metadata", {})
    text = item.get("text", "")
    aliases = _aliases_for_subject(subject, [])
    blob = _normalize_text(" ".join(str(value) for value in metadata.values() if value) + " " + text)
    return any(alias in blob for alias in aliases)


def _aliases_for_subject(subject: str, documents: list[dict]) -> tuple[str, ...]:
    aliases = set(SUBJECT_ALIASES.get(subject, ()))
    aliases.add(subject)
    for document in documents:
        metadata_subject = document.get("metadata", {}).get("subject")
        if metadata_subject and _normalize_text(str(metadata_subject)) == _normalize_text(subject):
            aliases.add(str(metadata_subject))
    return tuple(sorted((_normalize_text(alias) for alias in aliases if alias), key=len, reverse=True))


def _normalize_text(text: str) -> str:
    replacements = str.maketrans("ąćęłńóśźż", "acelnoszz")
    return " ".join(text.lower().translate(replacements).split())


def _is_assessment_question(question: str) -> bool:
    lowered = _normalize_text(question)
    markers = (
        "zaliczenie",
        "zaliczyc",
        "assessment",
        "assess",
        "exam",
        "egzamin",
        "test",
        "passing",
        "ocena",
        "ocen",
    )
    return any(marker in lowered for marker in markers)


def _is_regulation_question(question: str) -> bool:
    lowered = _normalize_text(question)
    markers = (
        "zasady egzamin",
        "zasady egzaminow",
        "zasady egzaminu",
        "jak wyglada egzamin",
        "zasady zaliczen",
        "zasady zaliczenia semestru",
        "regulamin",
        "poprawic egzamin",
        "poprawa",
        "sesja poprawkowa",
        "plagiat",
        "ects",
        "punktow ects",
        "exam rules",
        "retake rules",
        "passing rules",
        "plagiarism",
    )
    return any(marker in lowered for marker in markers)


def _regulation_query_terms(question: str) -> str:
    lowered = _normalize_text(question)
    base_terms = [
        "regulamin studiow",
        "zasady egzaminow",
        "zasady egzaminu",
        "zasady zaliczen",
        "warunki zaliczenia",
        "sesja egzaminacyjna",
        "assessment rules",
    ]
    if any(term in lowered for term in ("popraw", "retake")):
        base_terms.extend(["sesja poprawkowa", "poprawa egzaminu", "retake rules"])
    if any(term in lowered for term in ("egzamin", "exam")):
        base_terms.extend(["exam rules", "warunki dopuszczenia do egzaminu"])
    if any(term in lowered for term in ("zalic", "passing")):
        base_terms.extend(["passing rules", "zasady zaliczenia semestru"])
    if "plagiat" in lowered or "plagiarism" in lowered:
        base_terms.extend(["plagiat", "plagiarism rules"])
    if "ects" in lowered or "punkt" in lowered:
        base_terms.extend(["punkty ECTS", "ECTS rules"])
    return " ".join(base_terms)


def expand_query(question: str) -> str:
    lowered = _normalize_text(question)
    additions: list[str] = []
    groups = {
        ("prowadzi", "uczy", "wykladowca", "prowadzacy", "teaches", "lecturer", "responsible"): [
            "lecturer", "teacher", "prowadzacy", "wykladowca"
        ],
        ("zaliczenie", "ocena", "oceniany", "wymagania zaliczeniowe", "assessment", "grading"): [
            "assessment_method", "assessment", "grading", "passing", "zaliczenie", "ocena"
        ],
        ("egzamin", "test", "kolokwium", "termin", "exam", "date"): [
            "exam_date", "exam rules", "termin egzaminu", "test", "kolokwium"
        ],
        ("poprawa", "poprawic", "sesja poprawkowa", "retake"): [
            "retake rules", "sesja poprawkowa", "poprawa egzaminu"
        ],
        ("plagiat", "sciaganie", "plagiarism"): [
            "plagiarism", "plagiat", "niesamodzielne wykonanie"
        ],
        ("konsultacje", "dyzur", "office hours", "consultation"): [
            "consultations", "office hours", "konsultacje", "dyzur"
        ],
        ("przedmioty", "kursy", "subjects", "courses"): [
            "subject", "subjects", "courses", "przedmioty"
        ],
        ("semestr", "semester"): [
            "semester", "semestr"
        ],
    }
    for markers, terms in groups.items():
        if any(marker in lowered for marker in markers):
            additions.extend(terms)
    if _is_regulation_question(question):
        additions.append(_regulation_query_terms(question))
    if not additions:
        return question
    return f"{question} {' '.join(additions)}"


def _is_subject_list_question(question: str) -> bool:
    lowered = question.lower()
    subject_words = ("przedmiot", "przedmioty", "subjects", "courses", "classes")
    list_words = ("jakie", "lista", "wymien", "which", "what", "included", "list")
    semester_words = ("semestr", "semester")
    return (
        any(word in lowered for word in subject_words)
        and any(word in lowered for word in semester_words)
        and any(word in lowered for word in list_words)
    )


def _extract_semester(question: str) -> str | None:
    lowered = question.lower()
    match = re.search(r"(?:semestr|semester|semestrze)\s*(\d+)", lowered)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)\s*(?:semestr|semester|semestrze)", lowered)
    if match:
        return match.group(1)
    return None


def _extract_field(question: str) -> str | None:
    lowered = question.lower()
    if "informat" in lowered or "computer science" in lowered:
        return "informat"
    return None
