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
    assert "1. Bazy danych (5 ECTS, prowadzacy: dr hab. Maria Wisniewska)" in answer
    assert "2. Systemy operacyjne (5 ECTS, prowadzacy: dr Piotr Nowak)" in answer
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
    assert "2. Databases (5 ECTS, lecturer: dr hab. Maria Wisniewska)" in answer
    assert "1. Algorithms and Data Structures (6 ECTS, lecturer: dr Anna Kowalska)" in answer
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


def test_polish_course_description_answer_is_natural(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": (
                "field: Informatyka semester: 2 subject: Bazy danych ects: 5 "
                "lecturer: dr hab. Maria Wisniewska description: Model relacyjny, SQL, "
                "normalizacja, transakcje, indeksy i projektowanie schematow baz danych."
            ),
            "metadata": {
                "field": "Informatyka",
                "semester": "2",
                "subject": "Bazy danych",
                "ects": "5",
                "lecturer": "dr hab. Maria Wisniewska",
                "description": "Model relacyjny, SQL, normalizacja, transakcje, indeksy i projektowanie schematow baz danych.",
            },
            "score": 1.0,
        }
    ]

    answer = generate_answer("Co obejmuje przedmiot Bazy danych?", contexts)

    assert answer.startswith("Przedmiot Bazy danych obejmuje:")
    assert "model relacyjny, SQL, normalizację, transakcje, indeksy" in answer
    assert "2 semestrze Informatyki" in answer
    assert "5 ECTS" in answer
    assert "dr hab. Maria Wiśniewska" in answer


def test_english_course_description_answer_is_natural(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "raw csv row",
            "metadata": {
                "field": "Informatyka",
                "semester": "2",
                "subject": "Bazy danych",
                "ects": "5",
                "lecturer": "dr hab. Maria Wisniewska",
                "description": "Model relacyjny, SQL, normalizacja, transakcje, indeksy i projektowanie schematow baz danych.",
            },
            "score": 1.0,
        }
    ]

    answer = generate_answer("What is covered in the Databases course?", contexts)

    assert answer.startswith("The Databases course covers:")
    assert "relational model, SQL, normalization, transactions, indexes and database schema design" in answer
    assert "semester 2" in answer
    assert "5 ECTS" in answer
    assert "dr hab. Maria Wisniewska" in answer


def test_answer_does_not_contain_raw_field_prefixes(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": (
                "field: Informatyka semester: 2 subject: Bazy danych ects: 5 "
                "lecturer: dr hab. Maria Wisniewska description: Model relacyjny, SQL."
            ),
            "metadata": {
                "field": "Informatyka",
                "semester": "2",
                "subject": "Bazy danych",
                "ects": "5",
                "lecturer": "dr hab. Maria Wisniewska",
                "description": "Model relacyjny, SQL.",
            },
            "score": 1.0,
        }
    ]

    answer = generate_answer("Opisz przedmiot Bazy danych", contexts)

    raw_prefixes = ("field:", "semester:", "subject:", "ects:", "lecturer:", "description:", "assessment_method:", "exam_date:")
    assert all(prefix not in answer.lower() for prefix in raw_prefixes)


def test_databases_description_includes_lecturer_semester_and_ects(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "structured row",
            "metadata": {
                "field": "Informatyka",
                "semester": "2",
                "subject": "Bazy danych",
                "ects": "5",
                "lecturer": "dr hab. Maria Wisniewska",
                "description": "Model relacyjny, SQL.",
            },
            "score": 1.0,
        }
    ]

    answer = generate_answer("Co obejmuje przedmiot Bazy danych?", contexts)

    assert "2 semestrze" in answer
    assert "5 ECTS" in answer
    assert "dr hab. Maria Wiśniewska" in answer


def test_polish_teacher_question_is_short_and_direct(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "structured row",
            "metadata": {
                "field": "Informatyka",
                "semester": "2",
                "subject": "Bazy danych",
                "ects": "5",
                "lecturer": "dr hab. Maria Wisniewska",
                "description": "Model relacyjny, SQL, normalizacja, transakcje, indeksy i projektowanie schematow baz danych.",
            },
            "score": 1.0,
        }
    ]

    answer = generate_answer("Kto prowadzi Bazy danych?", contexts)

    assert answer == "Przedmiot Bazy danych prowadzi dr hab. Maria Wiśniewska."
    assert "obejmuje" not in answer
    assert "model relacyjny" not in answer


def test_english_teacher_question_is_short_and_direct(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "structured row",
            "metadata": {
                "field": "Informatyka",
                "semester": "2",
                "subject": "Bazy danych",
                "ects": "5",
                "lecturer": "dr hab. Maria Wisniewska",
                "description": "Model relacyjny, SQL, normalizacja, transakcje, indeksy i projektowanie schematow baz danych.",
            },
            "score": 1.0,
        }
    ]

    answer = generate_answer("Who teaches Databases?", contexts)

    assert answer == "The Databases course is taught by dr hab. Maria Wisniewska."
    assert "covers" not in answer
    assert "relational model" not in answer


def test_course_description_question_still_includes_topics(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "structured row",
            "metadata": {
                "field": "Informatyka",
                "semester": "2",
                "subject": "Bazy danych",
                "ects": "5",
                "lecturer": "dr hab. Maria Wisniewska",
                "description": "Model relacyjny, SQL, normalizacja, transakcje, indeksy i projektowanie schematow baz danych.",
            },
            "score": 1.0,
        }
    ]

    answer = generate_answer("Co obejmuje przedmiot Bazy danych?", contexts)

    assert "model relacyjny" in answer
    assert "SQL" in answer
    assert "projektowanie schematów baz danych" in answer


def test_assessment_question_still_includes_assessment_method(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "structured row",
            "metadata": {
                "subject": "Bazy danych",
                "exam_date": "5 czerwca 2026",
                "assessment_method": "Test SQL 30%, projekt zespolowy 40%, odpowiedz ustna 30%.",
            },
            "score": 1.0,
        }
    ]

    answer = generate_answer("Jak wygląda zaliczenie przedmiotu Bazy danych?", contexts)

    assert "- test SQL 30%" in answer
    assert "- projekt zespolowy 40%" in answer


def test_polish_teacher_variant_kto_uczy(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [{"text": "row", "metadata": {"subject": "Bazy danych", "lecturer": "dr hab. Maria Wisniewska"}, "score": 1.0}]

    answer = generate_answer("Kto uczy Baz danych?", contexts)

    assert answer == "Przedmiot Bazy danych prowadzi dr hab. Maria Wiśniewska."


def test_english_teacher_variant_lecturer_for(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [{"text": "row", "metadata": {"subject": "Bazy danych", "lecturer": "dr hab. Maria Wisniewska"}, "score": 1.0}]

    answer = generate_answer("Who is the lecturer for Databases?", contexts)

    assert answer == "The Databases course is taught by dr hab. Maria Wisniewska."


def test_polish_assessment_variant_ocena(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "row",
            "metadata": {
                "subject": "Bazy danych",
                "assessment_method": "Test SQL 30%, projekt zespolowy 40%, odpowiedz ustna 30%.",
            },
            "score": 1.0,
        }
    ]

    answer = generate_answer("Z czego jest ocena z Baz danych?", contexts)

    assert "- test SQL 30%" in answer
    assert "- projekt zespolowy 40%" in answer


def test_english_assessment_variant_grading(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [
        {
            "text": "row",
            "metadata": {
                "subject": "Bazy danych",
                "assessment_method": "Test SQL 30%, projekt zespolowy 40%, odpowiedz ustna 30%.",
            },
            "score": 1.0,
        }
    ]

    answer = generate_answer("What is the grading for Databases?", contexts)

    assert "- SQL test 30%" in answer
    assert "- team project 40%" in answer


def test_polish_exam_date_variant(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [{"text": "row", "metadata": {"subject": "Bazy danych", "exam_date": "5 czerwca 2026"}, "score": 1.0}]

    answer = generate_answer("Kiedy jest egzamin z Baz danych?", contexts)

    assert answer == "Egzamin/test z przedmiotu Bazy danych odbywa się 5 czerwca 2026."


def test_english_exam_date_variant(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.generator.get_settings", lambda: type("Settings", (), {"openai_api_key": None})())
    contexts = [{"text": "row", "metadata": {"subject": "Bazy danych", "exam_date": "5 czerwca 2026"}, "score": 1.0}]

    answer = generate_answer("When is the Databases exam?", contexts)

    assert answer == "The exam/test for Databases is scheduled for 5 June 2026."
