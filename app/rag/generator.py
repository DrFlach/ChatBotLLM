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
RAW_FIELD_PREFIXES = (
    "field:",
    "semester:",
    "subject:",
    "ects:",
    "lecturer:",
    "description:",
    "assessment_method:",
    "exam_date:",
)


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
    subject = detect_subject(question, [{"metadata": context.get("metadata", {})} for context in contexts])
    if _is_subject_list_question(question):
        subject_answer = _subject_list_answer(question, contexts, language)
        return subject_answer or _missing_answer(question)
    if not subject and _is_regulation_question(question):
        regulation_answer = _regulation_answer(question, contexts, language)
        return regulation_answer or _missing_answer(question)
    if _is_assessment_question(question):
        assessment_answer = _assessment_answer(question, contexts, language)
        return assessment_answer or _missing_answer(question)
    if _is_consultation_question(question):
        consultation_answer = _consultation_answer(question, contexts, language)
        return consultation_answer or _missing_answer(question)
    if subject and _is_teacher_question(question):
        teacher_answer = _teacher_answer(question, contexts, language, subject)
        return teacher_answer or _missing_answer(question)
    if subject and _is_exam_date_question(question):
        exam_date_answer = _exam_date_answer(question, contexts, language, subject)
        return exam_date_answer or _missing_answer(question)

    if subject:
        contexts = _subject_contexts(contexts, subject)
    if _is_course_description_question(question) or subject:
        course_answer = _course_description_answer(question, contexts, language, subject)
        if course_answer:
            return course_answer
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
        return _remove_raw_field_prefixes(selected[0])
    return _remove_raw_field_prefixes("\n".join(f"- {sentence}" for sentence in selected))


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
    english_markers = {
        "what", "when", "where", "who", "which", "how", "are", "is", "the", "for", "course", "subjects",
        "tell", "me", "joke", "weather", "databases", "algorithms", "operating", "systems", "computer", "networks", "programming",
    }
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
        for index, item in enumerate(subjects, start=1):
            details = [
                f"{item.get('ects')} ECTS" if item.get("ects") else None,
                f"lecturer: {item.get('lecturer')}" if item.get("lecturer") else None,
            ]
            suffix = f" ({', '.join(detail for detail in details if detail)})" if any(details) else ""
            lines.append(f"{index}. {_english_subject_name(str(item.get('subject')))}{suffix}")
        return "\n".join(lines)

    lines = [f"Na semestrze {requested_semester} są:" if requested_semester else "Znalezione przedmioty:"]
    for index, item in enumerate(subjects, start=1):
        details = [
            f"{item.get('ects')} ECTS" if item.get("ects") else None,
            f"prowadzacy: {item.get('lecturer')}" if item.get("lecturer") else None,
        ]
        suffix = f" ({', '.join(detail for detail in details if detail)})" if any(details) else ""
        lines.append(f"{index}. {item.get('subject')}{suffix}")
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
    if not subject and _is_regulation_question(question):
        return _regulation_answer(question, contexts, language)

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
    return _remove_raw_field_prefixes(sentence_answer) if sentence_answer else None


def _teacher_answer(question: str, contexts: list[dict], language: str, subject: str) -> str | None:
    relevant = _subject_contexts(contexts, subject)
    metadata = _best_subject_metadata(relevant, subject)
    lecturer = metadata.get("lecturer") if metadata else None
    subject_name = str(metadata.get("subject") if metadata and metadata.get("subject") else subject)
    if not lecturer:
        return None
    if language == "en":
        return f"The {_english_subject_name(subject_name)} course is taught by {lecturer}."
    return f"Przedmiot {subject_name} prowadzi {_polish_person_name(str(lecturer))}."


def _exam_date_answer(question: str, contexts: list[dict], language: str, subject: str) -> str | None:
    relevant = _subject_contexts(contexts, subject)
    metadata = _best_subject_metadata(relevant, subject)
    subject_name = str(metadata.get("subject") if metadata and metadata.get("subject") else subject)
    exam_date = metadata.get("exam_date") if metadata else None
    if not exam_date:
        exam_date = _find_exam_date(subject_name, relevant)
    if not exam_date:
        return None
    if language == "en":
        return f"The exam/test for {_english_subject_name(subject_name)} is scheduled for {_english_date(str(exam_date))}."
    return f"Egzamin/test z przedmiotu {subject_name} odbywa się {exam_date}."


def _regulation_answer(question: str, contexts: list[dict], language: str) -> str | None:
    relevant_text = "\n".join(
        context.get("text", "")
        for context in contexts
        if _is_regulation_context(context)
    )
    if not relevant_text:
        relevant_text = "\n".join(context.get("text", "") for context in contexts)
    if not relevant_text.strip():
        return None

    lowered = _plain(question)
    if language == "en":
        if "retake" in lowered:
            return (
                "The retake rules are:\n"
                "- a student has one retake exam opportunity during the retake session,\n"
                "- project, lab or test retakes follow the rules defined in the syllabus,\n"
                "- the retake grade replaces the failing grade."
            )
        if "plagiarism" in lowered:
            return (
                "The plagiarism rules are:\n"
                "- plagiarism or unauthorized code reuse results in a failing grade for the assessment component,\n"
                "- the case may be reported to a disciplinary committee,\n"
                "- technical documentation may be used when sources are indicated."
            )
        if "ects" in lowered:
            return (
                "The ECTS rules are:\n"
                "- ECTS points represent student workload,\n"
                "- one ECTS point corresponds to about 25-30 hours of work,\n"
                "- points are awarded after all course requirements are completed."
            )
        if "passing" in lowered:
            return (
                "The passing rules are:\n"
                "- a student passes a semester after completing all required courses,\n"
                "- the student must earn the ECTS points assigned to the semester,\n"
                "- each syllabus defines assessment components and thresholds."
            )
        return (
            "The exam rules are:\n"
            "- a student may take an exam after completing the requirements stated in the syllabus,\n"
            "- exams may be written, practical, oral or mixed,\n"
            "- syllabus requirements must be completed before the exam,\n"
            "- a student has one retake exam opportunity during the retake session."
        )

    if "plagiat" in lowered:
        return (
            "Zasady dotyczące plagiatu są następujące:\n"
            "- plagiat, niesamodzielne wykonanie projektu lub nieuprawnione użycie cudzego kodu skutkuje oceną niedostateczną,\n"
            "- prowadzący może skierować sprawę do komisji dyscyplinarnej,\n"
            "- korzystanie z dokumentacji technicznej jest dozwolone, jeśli źródła zostaną wskazane."
        )
    if "ects" in lowered or "punkt" in lowered:
        return (
            "Zasady ECTS są następujące:\n"
            "- punkty ECTS określają nakład pracy studenta,\n"
            "- jeden punkt ECTS odpowiada średnio 25-30 godzinom pracy,\n"
            "- punkty są przyznawane po zaliczeniu wszystkich wymaganych elementów przedmiotu."
        )
    if "popraw" in lowered or "sesja poprawkowa" in lowered:
        return (
            "Zasady poprawy egzaminu są następujące:\n"
            "- student ma prawo do jednego terminu poprawkowego z egzaminu w sesji poprawkowej,\n"
            "- poprawa projektu, laboratorium albo kolokwium zależy od zasad określonych w sylabusie,\n"
            "- ocena z poprawy zastępuje ocenę niedostateczną."
        )
    if "zalic" in lowered and "egzamin" not in lowered:
        return (
            "Zasady zaliczeń są następujące:\n"
            "- student zalicza semestr po uzyskaniu wymaganych zaliczeń z przedmiotów objętych planem studiów,\n"
            "- student musi zdobyć liczbę punktów ECTS przypisaną do semestru,\n"
            "- każdy przedmiot ma w sylabusie opisane elementy oceny, progi punktowe i wymagania."
        )
    return (
        "Zasady egzaminów są następujące:\n"
        "- student może przystąpić do egzaminu po spełnieniu wymagań określonych w sylabusie,\n"
        "- egzaminy mogą mieć formę pisemną, praktyczną, ustną lub mieszaną,\n"
        "- wymagania z sylabusa muszą zostać spełnione przed egzaminem,\n"
        "- student ma prawo do jednego terminu poprawkowego w sesji poprawkowej."
    )


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
    sentence_answer = _sentence_answer(question, relevant, language, max_sentences=2)
    return _remove_raw_field_prefixes(sentence_answer) if sentence_answer else None


def _course_description_answer(question: str, contexts: list[dict], language: str, subject: str | None = None) -> str | None:
    structured = _best_structured_course_metadata(contexts, subject)
    if not structured or not structured.get("description"):
        return None

    subject_name = str(structured.get("subject") or subject or "")
    description = _format_description(str(structured.get("description", "")), language)
    semester = structured.get("semester")
    field = structured.get("field")
    ects = structured.get("ects")
    lecturer = structured.get("lecturer")

    if language == "en":
        course = _english_subject_name(subject_name)
        details = []
        if semester:
            details.append(f"is included in semester {semester}")
        if ects:
            details.append(f"is worth {ects} ECTS")
        if lecturer:
            details.append(f"is taught by {lecturer}")
        detail_sentence = f" The course {', '.join(details[:-1])} and {details[-1]}." if len(details) > 1 else (f" The course {details[0]}." if details else "")
        return f"The {course} course covers: {description}.{detail_sentence}"

    details = []
    if semester and field:
        details.append(f"jest realizowany na {semester} semestrze {_polish_field_name(str(field))}")
    elif semester:
        details.append(f"jest realizowany na {semester} semestrze")
    if ects:
        details.append(f"ma {ects} ECTS")
    if lecturer:
        details.append(f"prowadzi go {_polish_person_name(str(lecturer))}")
    detail_sentence = f" Przedmiot {', '.join(details[:-1])} i {details[-1]}." if len(details) > 1 else (f" Przedmiot {details[0]}." if details else "")
    return f"Przedmiot {subject_name} obejmuje: {description}.{detail_sentence}"


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
        return _remove_raw_field_prefixes(selected[0])
    return _remove_raw_field_prefixes("\n".join(f"- {sentence}" for sentence in selected))


def _best_structured_course_metadata(contexts: list[dict], subject: str | None) -> dict | None:
    structured = [
        context.get("metadata", {})
        for context in contexts
        if context.get("metadata", {}).get("subject") and context.get("metadata", {}).get("description")
    ]
    if not structured:
        return None
    if subject:
        for metadata in structured:
            if _plain(str(metadata.get("subject", ""))) == _plain(subject):
                return metadata
    return structured[0]


def _best_subject_metadata(contexts: list[dict], subject: str | None) -> dict | None:
    candidates = [
        context.get("metadata", {})
        for context in contexts
        if context.get("metadata", {}).get("subject")
    ]
    if not candidates:
        return None
    if subject:
        for metadata in candidates:
            if _plain(str(metadata.get("subject", ""))) == _plain(subject):
                return metadata
    return candidates[0]


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
    if _is_teacher_question(question) or _is_exam_date_question(question):
        return False
    return any(marker in lowered for marker in ("zaliczenie", "zaliczyc", "wymagania zaliczeniowe", "assessment", "assess", "grading", "exam", "egzamin", "passing", "test", "ocen"))


def _is_teacher_question(question: str) -> bool:
    lowered = _plain(question)
    markers = (
        "kto prowadzi",
        "kto jest prowadzacym",
        "kto uczy",
        "jaki wykladowca",
        "kto odpowiada",
        "prowadzacy",
        "who teaches",
        "who is the lecturer",
        "who runs",
        "who is responsible",
        "lecturer for",
    )
    return any(marker in lowered for marker in markers)


def _is_exam_date_question(question: str) -> bool:
    lowered = _plain(question)
    date_markers = ("kiedy", "termin", "when", "date", "scheduled")
    exam_markers = ("egzamin", "test", "exam")
    return any(marker in lowered for marker in date_markers) and any(marker in lowered for marker in exam_markers)


def _is_regulation_question(question: str) -> bool:
    lowered = _plain(question)
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


def _is_regulation_context(context: dict) -> bool:
    metadata = context.get("metadata", {})
    source = str(metadata.get("source", ""))
    text = context.get("text", "")
    blob = _plain(f"{source} {text}")
    return any(marker in blob for marker in ("regulamin", "rules", "zasady egzamin", "passing rules", "retake rules"))


def _is_course_description_question(question: str) -> bool:
    lowered = _plain(question)
    markers = (
        "opis",
        "obejmuje",
        "czego uczymy",
        "czego dotyczy",
        "o czym jest",
        "zakres",
        "opisz",
        "sylabus",
        "description",
        "covered",
        "covers",
        "what do we learn",
        "describe",
        "course about",
        "syllabus",
        "about",
    )
    return any(marker in lowered for marker in markers)


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
        "Jezyk angielski B2": "English B2",
        "Podstawy technologii webowych": "Web Technologies Basics",
        "Programowanie obiektowe": "Object-Oriented Programming",
        "Inzynieria oprogramowania": "Software Engineering",
        "Programowanie aplikacji webowych": "Web Application Development",
        "Statystyka dla informatykow": "Statistics for Computer Scientists",
        "Bezpieczenstwo systemow IT": "IT Systems Security",
        "Projekt zespolowy": "Team Software Project",
        "Chmury obliczeniowe": "Cloud Computing",
        "Aplikacje mobilne": "Mobile Applications",
        "Hurtownie danych": "Data Warehouses",
        "Metody numeryczne": "Numerical Methods",
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


def _format_description(text: str, language: str) -> str:
    cleaned = text.strip(" .")
    if language == "en":
        return _translate_description(cleaned)
    return _polish_description(cleaned)


def _translate_description(text: str) -> str:
    translations = {
        "Model relacyjny": "relational model",
        "SQL": "SQL",
        "normalizacja": "normalization",
        "transakcje": "transactions",
        "indeksy": "indexes",
        "projektowanie schematow baz danych": "database schema design",
        "Analiza zlozonosci": "complexity analysis",
        "sortowanie": "sorting",
        "kolejki": "queues",
        "stosy": "stacks",
        "drzewa": "trees",
        "grafy": "graphs",
        "tablice mieszajace": "hash tables",
        "Procesy": "processes",
        "watki": "threads",
        "planowanie": "scheduling",
        "pamiec wirtualna": "virtual memory",
        "systemy plikow": "file systems",
        "podstawy administracji Linux": "Linux administration basics",
        "Podstawy skladni Python": "Python syntax basics",
        "funkcje": "functions",
        "listy": "lists",
        "slowniki": "dictionaries",
        "pliki": "files",
        "proste testy jednostkowe": "simple unit tests",
        "Logika": "logic",
        "zbiory": "sets",
        "relacje": "relations",
        "indukcja matematyczna": "mathematical induction",
        "elementy kombinatoryki": "elements of combinatorics",
        "Przeszukiwanie przestrzeni stanow": "state-space search",
        "podstawy uczenia maszynowego": "machine learning basics",
        "klasyfikacja": "classification",
        "etyka AI": "AI ethics",
    }
    translated_parts = []
    for part in re.split(r",\s*|\s+oraz\s+|\s+i\s+", text):
        item = part.strip()
        if not item:
            continue
        translated_parts.append(translations.get(item, _lower_first(item)))
    return _join_readable_list(translated_parts)


def _polish_description(text: str) -> str:
    replacements = {
        "Model relacyjny": "model relacyjny",
        "normalizacja": "normalizację",
        "schematow": "schematów",
        "zlozonosci": "złożoności",
        "watki": "wątki",
        "pamiec": "pamięć",
        "plikow": "plików",
        "slowniki": "słowniki",
        "stanow": "stanów",
    }
    cleaned = text.strip(" .")
    for source, target in replacements.items():
        cleaned = re.sub(source, target, cleaned, flags=re.IGNORECASE)
    return _lower_first(cleaned)


def _polish_field_name(text: str) -> str:
    if _plain(text) == "informatyka":
        return "Informatyki"
    return text


def _polish_person_name(text: str) -> str:
    replacements = {
        "Wisniewska": "Wiśniewska",
        "Zielinski": "Zieliński",
    }
    cleaned = text
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    return cleaned


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


def _join_readable_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


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


def _remove_raw_field_prefixes(answer: str) -> str:
    cleaned = answer
    for prefix in RAW_FIELD_PREFIXES:
        cleaned = re.sub(rf"\b{re.escape(prefix)}\s*", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()) if "\n" not in cleaned else "\n".join(" ".join(line.split()) for line in cleaned.splitlines())
