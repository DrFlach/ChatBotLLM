import pytest

from app.api import routes
from app.api.routes import ChatRequest, _format_source, chat, health, public_config


def test_source_formatting_includes_useful_metadata() -> None:
    source = _format_source(
        {
            "text": "field: Informatyka semester: 2 subject: Bazy danych",
            "score": 0.9,
            "metadata": {
                "source": "sample_sylabusy.csv",
                "document_type": "csv",
                "chunk_id": 0,
                "subject": "Bazy danych",
                "semester": "2",
                "lecturer": "dr Maria Wisniewska",
            },
        }
    )

    assert source.file_name == "sample_sylabusy.csv"
    assert source.document_type == "csv"
    assert source.subject == "Bazy danych"
    assert source.semester == "2"
    assert source.chunk_number == 0


def test_post_chat_with_normal_question(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "retrieve",
        lambda question: [
            {
                "text": "Semestr 2 zawiera przedmiot Bazy danych.",
                "metadata": {"source": "sample_program_studiow.txt", "document_type": "txt", "chunk_id": 0},
                "score": 0.8,
            }
        ],
    )
    monkeypatch.setattr(routes, "generate_answer", lambda question, contexts: "Semestr 2 zawiera Bazy danych.")

    response = chat(ChatRequest(question="Jakie przedmioty sa na semestrze 2?"))

    assert "Bazy danych" in response.answer
    assert response.sources[0].file_name == "sample_program_studiow.txt"
    assert response.intent == "subject_list"


def test_post_chat_handles_empty_question() -> None:
    response = chat(ChatRequest(question="   "))

    assert response.intent == "empty"
    assert response.sources == []
    assert "Wpisz pytanie" in response.answer


def test_polish_greeting_returns_direct_answer_without_sources(monkeypatch) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: pytest.fail("RAG should not run for greeting"))

    response = chat(ChatRequest(question="cześć"))

    assert response.intent == "greeting"
    assert response.sources == []
    assert "Cześć! Jestem chatbotem RAG" in response.answer


def test_english_greeting_returns_direct_answer_without_sources(monkeypatch) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: pytest.fail("RAG should not run for greeting"))

    response = chat(ChatRequest(question="hello"))

    assert response.intent == "greeting"
    assert response.sources == []
    assert response.answer.startswith("Hello! I am a RAG chatbot")


def test_polish_help_question_returns_capabilities(monkeypatch) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: pytest.fail("RAG should not run for help"))

    response = chat(ChatRequest(question="co potrafisz?"))

    assert response.intent == "help"
    assert response.sources == []
    assert "lista przedmiotów na danym semestrze" in response.answer
    assert "możliwość zadawania pytań po polsku i angielsku" in response.answer


def test_english_help_question_returns_capabilities(monkeypatch) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: pytest.fail("RAG should not run for help"))

    response = chat(ChatRequest(question="what can you do?"))

    assert response.intent == "help"
    assert response.sources == []
    assert "subjects by semester" in response.answer
    assert "support for Polish and English questions" in response.answer


def test_help_and_out_of_scope_return_no_sources(monkeypatch) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: pytest.fail("RAG should not run for direct answers"))

    help_response = chat(ChatRequest(question="pomoc"))
    out_of_scope_response = chat(ChatRequest(question="Jaka jest pogoda?"))

    assert help_response.sources == []
    assert out_of_scope_response.sources == []


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Kto prowadzi Bazy danych?", "lecturer"),
        ("Kto uczy Baz danych?", "lecturer"),
        ("Jak wygląda zaliczenie Bazy danych?", "assessment"),
        ("Z czego jest ocena z Baz danych?", "assessment"),
        ("Kiedy jest egzamin z Baz danych?", "exam_date"),
        ("Czy można poprawić egzamin?", "retake_rules"),
        ("Co grozi za plagiat?", "plagiarism"),
        ("Who teaches Databases?", "lecturer"),
        ("Who is the lecturer for Databases?", "lecturer"),
        ("How is Databases assessed?", "assessment"),
        ("What is the grading for Databases?", "assessment"),
        ("When is the Databases exam?", "exam_date"),
        ("What are the retake rules?", "retake_rules"),
        ("What happens in case of plagiarism?", "plagiarism"),
    ],
)
def test_multiple_formulations_are_in_scope(monkeypatch, question: str, intent: str) -> None:
    monkeypatch.setattr(
        routes,
        "retrieve",
        lambda question: [
            {
                "text": "row",
                "metadata": {"source": "sample_sylabusy.csv", "document_type": "csv", "chunk_id": 0, "subject": "Bazy danych"},
                "score": 1.0,
            }
        ],
    )
    monkeypatch.setattr(routes, "generate_answer", lambda question, contexts: "Odpowiedź z dokumentów.")

    response = chat(ChatRequest(question=question))

    assert response.intent == intent
    assert response.sources


@pytest.mark.parametrize("question", ["Bazy danych", "Databases"])
def test_ambiguous_subject_questions_ask_for_clarification(monkeypatch, question: str) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: pytest.fail("RAG should not run for ambiguous questions"))

    response = chat(ChatRequest(question=question))

    assert response.intent == "ambiguous_study_question"
    assert response.sources == []
    assert "Who teaches Databases?" in response.answer or "Kto prowadzi Bazy danych?" in response.answer


@pytest.mark.parametrize("question", ["Jaka jest pogoda w Warszawie?", "Tell me a joke."])
def test_out_of_scope_examples_do_not_run_rag(monkeypatch, question: str) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: pytest.fail("RAG should not run for out-of-scope questions"))

    response = chat(ChatRequest(question=question))

    assert response.intent == "out_of_scope"
    assert response.sources == []


def test_polish_out_of_scope_question_returns_polite_answer(monkeypatch) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: pytest.fail("RAG should not run for out-of-scope"))

    response = chat(ChatRequest(question="Jaka jest pogoda w Warszawie?"))

    assert response.intent == "out_of_scope"
    assert response.sources == []
    assert "Nie znalazłem tej informacji w dokumentach programu studiów" in response.answer


def test_english_out_of_scope_question_returns_polite_answer(monkeypatch) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: pytest.fail("RAG should not run for out-of-scope"))

    response = chat(ChatRequest(question="Who won the World Cup?"))

    assert response.intent == "out_of_scope"
    assert response.sources == []
    assert "I could not find this information in the study program documents" in response.answer


def test_thanks_and_goodbye_are_direct_answers(monkeypatch) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: pytest.fail("RAG should not run for small talk"))

    thanks = chat(ChatRequest(question="dziękuję"))
    goodbye = chat(ChatRequest(question="bye"))

    assert thanks.intent == "thanks"
    assert thanks.sources == []
    assert goodbye.intent == "goodbye"
    assert goodbye.sources == []


def test_rag_answer_still_returns_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "retrieve",
        lambda question: [
            {
                "text": "Bazy danych: Test SQL 30%, projekt zespolowy 40%, odpowiedz ustna 30%.",
                "metadata": {
                    "source": "sample_sylabusy.csv",
                    "document_type": "csv",
                    "chunk_id": 0,
                    "subject": "Bazy danych",
                },
                "score": 1.0,
            }
        ],
    )
    monkeypatch.setattr(routes, "generate_answer", lambda question, contexts: "Zaliczenie obejmuje test SQL.")

    response = chat(ChatRequest(question="Jak wygląda zaliczenie przedmiotu Bazy danych?"))

    assert response.intent == "assessment"
    assert response.sources
    assert response.sources[0].file_name == "sample_sylabusy.csv"


def test_teacher_question_limits_sources_to_two(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "retrieve",
        lambda question: [
            {
                "text": f"source {index}",
                "metadata": {
                    "source": f"source_{index}.txt",
                    "document_type": "txt",
                    "chunk_id": index,
                    "subject": "Bazy danych",
                    "lecturer": "dr hab. Maria Wisniewska",
                },
                "score": 1.0,
            }
            for index in range(4)
        ],
    )
    monkeypatch.setattr(routes, "generate_answer", lambda question, contexts: "Przedmiot Bazy danych prowadzi dr hab. Maria Wiśniewska.")

    response = chat(ChatRequest(question="Kto prowadzi Bazy danych?"))

    assert response.intent == "lecturer"
    assert len(response.sources) == 2


def test_normal_rag_question_limits_sources_to_three(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "retrieve",
        lambda question: [
            {
                "text": f"source {index}",
                "metadata": {"source": f"source_{index}.txt", "document_type": "txt", "chunk_id": index},
                "score": 1.0,
            }
            for index in range(5)
        ],
    )
    monkeypatch.setattr(routes, "generate_answer", lambda question, contexts: "Odpowiedź z RAG.")

    response = chat(ChatRequest(question="Jakie są zasady egzaminów?"))

    assert response.intent == "exam_rules"
    assert len(response.sources) == 3


def regulation_contexts() -> list[dict]:
    return [
        {
            "text": (
                "Zasady egzaminow: Warunkiem podejscia do egzaminu jest spelnienie wymagan "
                "opisanych w sylabusie. Egzamin moze miec forme pisemna, praktyczna, ustna albo mieszana. "
                "Zasady poprawek: Student ma prawo do jednego terminu poprawkowego z egzaminu w sesji poprawkowej. "
                "Zasady zaliczen: Student zalicza semestr, jezeli uzyska wszystkie wymagane zaliczenia."
            ),
            "metadata": {"source": "sample_regulamin.txt", "document_type": "txt", "chunk_id": 0},
            "score": 0.95,
        }
    ]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Jakie są zasady egzaminów?", "Zasady egzaminów są następujące:"),
        ("Czy można poprawić egzamin?", "Zasady poprawy egzaminu są następujące:"),
        ("Jakie są zasady zaliczeń?", "Zasady zaliczeń są następujące:"),
    ],
)
def test_polish_regulation_questions_use_rag(monkeypatch, question: str, expected: str) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: regulation_contexts())

    response = chat(ChatRequest(question=question))

    assert response.intent in {"exam_rules", "retake_rules", "passing_rules"}
    assert response.sources
    assert response.sources[0].file_name == "sample_regulamin.txt"
    assert expected in response.answer
    assert "Nie znalazłem tej informacji" not in response.answer


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What are the exam rules?", "The exam rules are:"),
        ("What are the retake rules?", "The retake rules are:"),
    ],
)
def test_english_regulation_questions_still_work(monkeypatch, question: str, expected: str) -> None:
    monkeypatch.setattr(routes, "retrieve", lambda question: regulation_contexts())

    response = chat(ChatRequest(question=question))

    assert response.intent in {"exam_rules", "retake_rules"}
    assert response.sources
    assert expected in response.answer
    assert "I could not find this information" not in response.answer


def test_health_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(routes, "index_exists", lambda index_dir: True)

    response = health()

    assert response["status"] == "ok"
    assert response["index_loaded"] is True
    assert response["vector_store"] == "FAISS"


def test_config_endpoint_does_not_expose_openai_key(monkeypatch) -> None:
    monkeypatch.setattr(routes, "indexed_chunk_count", lambda index_dir: 12)

    response = public_config()

    assert response["vector_database"] == "FAISS"
    assert "OPENAI_API_KEY" not in response
    assert response["indexed_chunks"] == 12
