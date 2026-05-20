from app.rag.generator import generate_answer


def test_fallback_prefers_matching_polish_name_forms(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())

    contexts = [
        {
            "text": (
                "dr Anna Kowalska. Konsultacje: poniedzialek 12:00-13:30, pokoj 214. "
                "dr Piotr Nowak. Konsultacje: sroda 10:00-11:30, pokoj 118."
            ),
            "metadata": {"source": "sample_konsultacje.html"},
            "score": 0.9,
        }
    ]

    answer = generate_answer("Jakie sa konsultacje dr Anny Kowalskiej?", contexts)

    assert "poniedzialek 12:00-13:30" in answer


def test_fallback_answers_in_english(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "Passing rules in English: A student passes a semester after completing all required courses and ECTS points.",
            "metadata": {"source": "sample_regulamin.txt"},
            "score": 0.9,
        }
    ]

    answer = generate_answer("What are the passing rules for a semester?", contexts)

    assert "student passes a semester" in answer.lower()


def test_fallback_returns_polish_missing_answer(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [{"text": "Konsultacje: poniedzialek 12:00.", "metadata": {"source": "x.txt"}, "score": 0.2}]

    answer = generate_answer("Jaki jest numer telefonu do dziekanatu?", contexts)

    assert answer == "Nie znalazłem tej informacji w dostępnych dokumentach."


def test_fallback_returns_english_missing_answer(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [{"text": "Konsultacje: poniedzialek 12:00.", "metadata": {"source": "x.txt"}, "score": 0.2}]

    answer = generate_answer("What is the dean office phone number?", contexts)

    assert answer == "I could not find this information in the available documents."


def test_fallback_subject_list_filters_requested_semester(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {"text": "subject row", "metadata": {"subject": "Matematyka", "semester": "1", "ects": "5"}, "score": 0.6},
        {"text": "subject row", "metadata": {"subject": "Bazy danych", "semester": "2", "ects": "5"}, "score": 1.0},
    ]

    answer = generate_answer("What subjects are included in semester 2?", contexts)

    assert "Databases" in answer
    assert "Matematyka" not in answer


def test_polish_databases_assessment_is_structured(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "subject: Bazy danych assessment_method: Test SQL 30%, projekt zespolowy 40%, odpowiedz ustna 30%.",
            "metadata": {
                "source": "sample_sylabusy.csv",
                "subject": "Bazy danych",
                "exam_date": "5 czerwca 2026",
                "assessment_method": "Test SQL 30%, projekt zespolowy 40%, odpowiedz ustna 30%.",
            },
            "score": 1.0,
        },
        {
            "text": "Systemy operacyjne: Laboratoria 50%, egzamin 50%.",
            "metadata": {"source": "sample_sylabusy.csv", "subject": "Systemy operacyjne"},
            "score": 0.6,
        },
    ]

    answer = generate_answer("Jak wygląda zaliczenie przedmiotu Bazy danych?", contexts)

    assert answer.startswith("Zaliczenie przedmiotu Bazy danych obejmuje:")
    assert "- test SQL 30%" in answer
    assert "- projekt zespolowy 40%" in answer
    assert "- odpowiedz ustna 30%" in answer
    assert "5 czerwca 2026" in answer
    assert "Systemy operacyjne" not in answer


def test_english_databases_assessment_is_structured(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "subject: Bazy danych assessment_method: Test SQL 30%, projekt zespolowy 40%, odpowiedz ustna 30%.",
            "metadata": {
                "source": "sample_sylabusy.csv",
                "subject": "Bazy danych",
                "exam_date": "5 czerwca 2026",
                "assessment_method": "Test SQL 30%, projekt zespolowy 40%, odpowiedz ustna 30%.",
            },
            "score": 1.0,
        }
    ]

    answer = generate_answer("How is the Databases course assessed?", contexts)

    assert answer.startswith("The Databases course is assessed through:")
    assert "- SQL test 30%" in answer
    assert "- team project 40%" in answer
    assert "- oral answer 30%" in answer
    assert "5 June 2026" in answer


def test_polish_semester_two_subject_list(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {"text": "row", "metadata": {"subject": "Wstep do programowania", "semester": "1", "ects": "6"}, "score": 0.6},
        {"text": "row", "metadata": {"subject": "Bazy danych", "semester": "2", "ects": "5", "lecturer": "dr hab. Maria Wisniewska"}, "score": 1.0},
        {"text": "row", "metadata": {"subject": "Systemy operacyjne", "semester": "2", "ects": "5", "lecturer": "dr Piotr Nowak"}, "score": 0.9},
    ]

    answer = generate_answer("Jakie przedmioty są na 2 semestrze?", contexts)

    assert "Na semestrze 2 są:" in answer
    assert "Bazy danych (5 ECTS, prowadzacy: dr hab. Maria Wisniewska)" in answer
    assert "Systemy operacyjne (5 ECTS, prowadzacy: dr Piotr Nowak)" in answer
    assert "Wstep do programowania" not in answer


def test_english_semester_two_subject_list(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {"text": "row", "metadata": {"subject": "Bazy danych", "semester": "2", "ects": "5", "lecturer": "dr hab. Maria Wisniewska"}, "score": 1.0},
        {"text": "row", "metadata": {"subject": "Algorytmy i struktury danych", "semester": "2", "ects": "6", "lecturer": "dr Anna Kowalska"}, "score": 0.9},
        {"text": "row", "metadata": {"subject": "Matematyka dyskretna", "semester": "1", "ects": "5"}, "score": 0.4},
    ]

    answer = generate_answer("What subjects are included in semester 2?", contexts)

    assert "Semester 2 includes:" in answer
    assert "Databases (5 ECTS, lecturer: dr hab. Maria Wisniewska)" in answer
    assert "Algorithms and Data Structures (6 ECTS, lecturer: dr Anna Kowalska)" in answer
    assert "Discrete Mathematics" not in answer


def test_consultation_hours_answer(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "dr hab. Maria Wisniewska Przedmioty: Bazy danych. Konsultacje: czwartek 14:00-15:00, pokoj 301, budynek C.",
            "metadata": {"source": "sample_konsultacje.html", "lecturer": "dr hab. Maria Wisniewska"},
            "score": 0.9,
        }
    ]

    answer = generate_answer("Jakie konsultacje ma dr hab. Maria Wisniewska?", contexts)

    assert "czwartek 14:00-15:00" in answer
    assert "pokoj 301" in answer
