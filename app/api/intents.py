from dataclasses import dataclass
import re

from app.rag.generator import detect_language


@dataclass(frozen=True)
class IntentResult:
    intent: str
    language: str
    answer: str | None = None


GREETING_PL = (
    "Cześć! Jestem chatbotem RAG dla programu studiów. Mogę pomóc w pytaniach "
    "o przedmioty, sylabusy, egzaminy, wykładowców, konsultacje oraz zasady zaliczeń."
)
GREETING_EN = (
    "Hello! I am a RAG chatbot for the study program. I can help with questions "
    "about subjects, syllabuses, exams, lecturers, consultation hours and passing rules."
)

OUT_OF_SCOPE_PL = (
    "Nie znalazłem tej informacji w dokumentach programu studiów. Mogę pomóc głównie "
    "w pytaniach o przedmioty, sylabusy, egzaminy, wykładowców, konsultacje i zasady zaliczeń."
)
OUT_OF_SCOPE_EN = (
    "I could not find this information in the study program documents. I can mainly help "
    "with questions about subjects, syllabuses, exams, lecturers, consultation hours and passing rules."
)


def detect_intent(question: str) -> IntentResult:
    text = " ".join(question.split())
    normalized = _normalize(text)
    language = _detect_intent_language(text, normalized)

    if not normalized:
        return IntentResult("empty", language, _empty_answer(language))
    if _matches(normalized, _GREETINGS_PL, _GREETINGS_EN):
        return IntentResult("greeting", language, GREETING_EN if language == "en" else GREETING_PL)
    if _matches(normalized, _HELP_PL, _HELP_EN):
        return IntentResult("help", language, _help_answer(language))
    if _matches(normalized, _THANKS_PL, _THANKS_EN):
        return IntentResult("thanks", language, _thanks_answer(language))
    if _matches(normalized, _GOODBYE_PL, _GOODBYE_EN):
        return IntentResult("goodbye", language, _goodbye_answer(language))
    ambiguous = _ambiguous_answer(normalized, language)
    if ambiguous:
        return IntentResult("ambiguous_study_question", language, ambiguous)
    category = _study_intent(normalized)
    if category:
        return IntentResult(category, language)
    if _is_study_program_question(normalized):
        return IntentResult("ambiguous_study_question", language, _generic_ambiguous_answer(language))
    return IntentResult("out_of_scope", language, OUT_OF_SCOPE_EN if language == "en" else OUT_OF_SCOPE_PL)


_GREETINGS_PL = {"czesc", "hej", "dzien dobry"}
_GREETINGS_EN = {"hi", "hello", "good morning"}
_HELP_PL = {"co potrafisz", "w czym mozesz pomoc", "pomoc", "jakie pytania moge zadac"}
_HELP_EN = {"what can you do", "help", "what can i ask"}
_THANKS_PL = {"dzieki", "dziekuje", "dziękuję"}
_THANKS_EN = {"thanks", "thank you"}
_GOODBYE_PL = {"do widzenia", "pa"}
_GOODBYE_EN = {"goodbye", "bye"}

_STUDY_TERMS = {
    "przedmiot",
    "przedmioty",
    "subject",
    "subjects",
    "course",
    "courses",
    "semestr",
    "semester",
    "sylabus",
    "sylabusy",
    "syllabus",
    "syllabuses",
    "opis",
    "obejmuje",
    "description",
    "covered",
    "covers",
    "egzamin",
    "egzaminu",
    "egzaminy",
    "egzaminow",
    "exam",
    "exams",
    "zaliczenie",
    "zaliczen",
    "zaliczenia",
    "assessment",
    "assessed",
    "passing",
    "regulamin",
    "regulaminu",
    "studiow",
    "regulations",
    "rules",
    "zasady",
    "ects",
    "punktow",
    "poprawa",
    "poprawic",
    "poprawkowa",
    "sesja poprawkowa",
    "plagiat",
    "obrona",
    "defense",
    "thesis",
    "wykladowca",
    "wykladowcy",
    "wykladowc",
    "prowadzacy",
    "prowadzi",
    "uczy",
    "odpowiada",
    "teaches",
    "taught",
    "lecturer",
    "lecturers",
    "konsultacje",
    "dyzur",
    "consultation",
    "consultations",
    "office hours",
    "informatyka",
    "computer science",
    "bazy danych",
    "databases",
    "algorytmy",
    "algorithms",
    "systemy operacyjne",
    "operating systems",
    "sieci komputerowe",
    "computer networks",
    "programowanie obiektowe",
    "object-oriented programming",
    "inzynieria oprogramowania",
    "software engineering",
    "programowanie aplikacji webowych",
    "web application development",
    "statystyka dla informatykow",
    "statistics for computer scientists",
    "bezpieczenstwo systemow it",
    "it systems security",
    "projekt zespolowy",
    "team software project",
    "chmury obliczeniowe",
    "cloud computing",
    "aplikacje mobilne",
    "mobile applications",
    "hurtownie danych",
    "data warehouses",
    "metody numeryczne",
    "numerical methods",
    "anna kowalska",
    "piotr nowak",
    "maria wisniewska",
    "tomasz zielinski",
    "ewa malinowska",
    "lukasz mazur",
    "pawel kaczmarek",
    "katarzyna wrobel",
}


def _matches(normalized: str, polish: set[str], english: set[str]) -> bool:
    candidates = polish | english
    trimmed = normalized.strip(" ?!.")
    return trimmed in candidates


def _detect_intent_language(text: str, normalized: str) -> str:
    trimmed = normalized.strip(" ?!.")
    english_intents = _GREETINGS_EN | _HELP_EN | _THANKS_EN | _GOODBYE_EN
    polish_intents = _GREETINGS_PL | _HELP_PL | _THANKS_PL | _GOODBYE_PL
    if trimmed in english_intents:
        return "en"
    if trimmed in polish_intents:
        return "pl"
    return detect_language(text)


def _is_study_program_question(normalized: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    if tokens.intersection(_STUDY_TERMS):
        return True
    return any(term in normalized for term in _STUDY_TERMS if " " in term)


def _study_intent(normalized: str) -> str | None:
    if _has_any(normalized, ("plagiat", "sciaganie", "plagiarism")):
        return "plagiarism"
    if _has_any(normalized, ("poprawic", "poprawa", "sesja poprawkowa", "retake")):
        return "retake_rules"
    if _has_any(normalized, ("ects", "punktow", "credits")):
        return "ects"
    if _has_any(normalized, ("zasady egzamin", "zasady egzaminow", "jak wyglada egzamin", "exam rules")):
        return "exam_rules"
    if _has_any(normalized, ("zasady zaliczen", "zasady zaliczenia", "passing rules")):
        return "passing_rules"
    if _has_any(normalized, ("regulamin", "regulations")):
        return "exam_rules"
    if _has_any(normalized, ("konsultacje", "dyzur", "office hours", "consultation")):
        return "consultation"
    if _has_any(normalized, ("kiedy", "termin", "when", "date", "scheduled")) and _has_any(normalized, ("egzamin", "test", "exam", "obrona", "defense")):
        return "exam_date"
    if _has_any(normalized, ("zaliczenie", "zaliczen", "ocena", "oceniany", "wymagania zaliczeniowe", "assessment", "assessed", "grading")):
        return "assessment"
    if _has_any(normalized, ("kto prowadzi", "kto uczy", "wykladowca", "prowadzacy", "kto odpowiada", "who teaches", "who is the lecturer", "who runs", "who is responsible")):
        return "lecturer"
    if _has_any(normalized, ("co obejmuje", "czego uczymy", "o czym jest", "opisz", "zakres", "what is covered", "what do we learn", "describe", "course about")):
        return "course_description"
    if _has_any(normalized, ("jakie przedmioty", "lista przedmiotow", "przedmioty", "subjects", "courses")) and _has_any(normalized, ("semestr", "semester")):
        return "subject_list"
    return None


def _ambiguous_answer(normalized: str, language: str) -> str | None:
    ambiguous_terms = {
        "bazy danych",
        "baz danych",
        "databases",
        "egzamin",
        "exam",
        "konsultacje",
        "consultations",
        "office hours",
        "sylabus",
        "syllabus",
        "semestr 2",
        "semester 2",
    }
    trimmed = normalized.strip(" ?!.")
    if trimmed not in ambiguous_terms:
        return None
    if language == "en":
        return (
            "I found a topic related to the study program. You can ask more specifically, for example:\n"
            "- Who teaches Databases?\n"
            "- What is covered in Databases?\n"
            "- How is Databases assessed?\n"
            "- When is the Databases exam?"
        )
    return (
        "Znalazłem temat związany z programem studiów. Możesz doprecyzować pytanie, np.:\n"
        "- Kto prowadzi Bazy danych?\n"
        "- Co obejmuje przedmiot Bazy danych?\n"
        "- Jak wygląda zaliczenie Bazy danych?\n"
        "- Kiedy jest egzamin z Baz danych?"
    )


def _generic_ambiguous_answer(language: str) -> str:
    if language == "en":
        return "I found a study-program topic, but I need a more specific question about subjects, assessment, exams, lecturers or regulations."
    return "Rozpoznaję temat związany z programem studiów, ale potrzebuję bardziej konkretnego pytania o przedmioty, zaliczenia, egzaminy, prowadzących albo regulamin."


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _help_answer(language: str) -> str:
    if language == "en":
        return "\n".join(
            [
                "I can help with:",
                "- subjects by semester",
                "- course descriptions and syllabuses",
                "- exam and assessment dates",
                "- lecturers and consultation hours",
                "- study regulations and passing rules",
                "- support for Polish and English questions",
            ]
        )
    return "\n".join(
        [
            "Mogę pomóc w pytaniach o:",
            "- lista przedmiotów na danym semestrze",
            "- opisy przedmiotów i sylabusy",
            "- terminy egzaminów i zaliczeń",
            "- informacje o wykładowcach i konsultacjach",
            "- regulamin studiów i zasady zaliczeń",
            "- możliwość zadawania pytań po polsku i angielsku",
        ]
    )


def _empty_answer(language: str) -> str:
    if language == "en":
        return "Please enter a question about the study program."
    return "Wpisz pytanie dotyczące programu studiów."


def _thanks_answer(language: str) -> str:
    if language == "en":
        return "You are welcome. I can help with more study program questions."
    return "Nie ma za co. Mogę pomóc w kolejnych pytaniach o program studiów."


def _goodbye_answer(language: str) -> str:
    if language == "en":
        return "Goodbye! You can come back with more study program questions anytime."
    return "Do widzenia! Możesz wrócić z kolejnymi pytaniami o program studiów."


def _normalize(text: str) -> str:
    replacements = str.maketrans("ąćęłńóśźż", "acelnoszz")
    return " ".join(text.lower().translate(replacements).split())
