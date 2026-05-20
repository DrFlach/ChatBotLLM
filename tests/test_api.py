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
    assert response.intent == "study_program"


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

    assert response.intent == "study_program"
    assert response.sources
    assert response.sources[0].file_name == "sample_sylabusy.csv"


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
