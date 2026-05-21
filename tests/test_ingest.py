from pathlib import Path

from app.rag.ingest import load_document


def test_txt_ingestion(tmp_path: Path) -> None:
    path = tmp_path / "program.txt"
    path.write_text("Program studiow\nSemestr 1: Programowanie", encoding="utf-8")

    records = load_document(path)

    assert len(records) == 1
    assert "Program studiow" in records[0]["text"]
    assert records[0]["metadata"]["document_type"] == "txt"


def test_csv_ingestion_preserves_structured_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sylabus.csv"
    path.write_text(
        "field,semester,subject,ects,lecturer,description,exam_date,assessment_method\n"
        "Informatyka,2,Bazy danych,5,dr Maria Wisniewska,Model relacyjny,5 czerwca 2026,Projekt 40%\n",
        encoding="utf-8",
    )

    records = load_document(path)

    metadata = records[0]["metadata"]
    assert metadata["field"] == "Informatyka"
    assert metadata["semester"] == "2"
    assert metadata["subject"] == "Bazy danych"
    assert metadata["lecturer"] == "dr Maria Wisniewska"
    assert metadata["ects"] == "5"
    assert metadata["description"] == "Model relacyjny"
    assert metadata["exam_date"] == "5 czerwca 2026"
    assert metadata["assessment_method"] == "Projekt 40%"


def test_html_ingestion_splits_sections(tmp_path: Path) -> None:
    path = tmp_path / "konsultacje.html"
    path.write_text(
        """
        <html><body>
          <section><h2>dr Anna Kowalska</h2><p>Konsultacje: poniedzialek 12:00.</p></section>
          <section><h2>dr Piotr Nowak</h2><p>Konsultacje: sroda 10:00.</p></section>
        </body></html>
        """,
        encoding="utf-8",
    )

    records = load_document(path)

    assert len(records) == 2
    assert records[0]["metadata"]["lecturer"] == "dr Anna Kowalska"
    assert records[1]["metadata"]["lecturer"] == "dr Piotr Nowak"


def test_pdf_ingestion_reads_sample_pdf() -> None:
    path = Path("data/raw/sample_regulamin_studiow.pdf")

    records = load_document(path)

    assert records
    assert records[0]["metadata"]["document_type"] == "pdf"
    assert "ECTS" in records[0]["text"]
    assert "Exam rules" in records[0]["text"]
