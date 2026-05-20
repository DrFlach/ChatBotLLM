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
