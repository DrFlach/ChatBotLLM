import csv
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


def load_document(path: Path) -> list[dict]:
    """Load supported file types into text records with source metadata."""

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [_record(path.read_text(encoding="utf-8"), path)]
    if suffix == ".csv":
        return _load_csv(path)
    if suffix in {".html", ".htm"}:
        return _load_html(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    return []


def load_documents(raw_dir: Path) -> list[dict]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    records: list[dict] = []
    for path in sorted(raw_dir.iterdir()):
        if path.is_file():
            records.extend(load_document(path))
    return records


def _record(text: str, path: Path, extra_metadata: dict | None = None) -> dict:
    metadata = {"source": path.name, "document_type": path.suffix.lower().lstrip(".") or "unknown"}
    if extra_metadata:
        metadata.update({key: value for key, value in extra_metadata.items() if value not in (None, "")})
    return {"text": text, "metadata": metadata}


def _load_csv(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=1):
            parts = [f"{key}: {value}" for key, value in row.items() if value]
            records.append(_record("\n".join(parts), path, _csv_metadata(row, row_number)))
    return records


def _load_html(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    records: list[dict] = []
    for section_number, section in enumerate(soup.find_all("section"), start=1):
        title = _clean_text(section.find(["h1", "h2", "h3"]).get_text(" ", strip=True)) if section.find(["h1", "h2", "h3"]) else None
        text = _clean_text(section.get_text("\n", strip=True))
        if text:
            records.append(
                _record(
                    text,
                    path,
                    {
                        "section": section_number,
                        "section_title": title,
                        "lecturer": title if title and _looks_like_lecturer(title) else None,
                    },
                )
            )

    if records:
        return records

    title = _clean_text(soup.find(["h1", "h2"]).get_text(" ", strip=True)) if soup.find(["h1", "h2"]) else None
    text = _clean_text(soup.get_text(separator="\n"))
    return [_record(text, path, {"section_title": title})] if text else []


def _load_pdf(path: Path) -> list[dict]:
    records: list[dict] = []
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            records.append(_record(text, path, {"page": page_number}))
    return records


def _csv_metadata(row: dict, row_number: int) -> dict:
    metadata = {"row": row_number}
    mapping = {
        "field": "field",
        "field_of_study": "field",
        "kierunek": "field",
        "semester": "semester",
        "semestr": "semester",
        "subject": "subject",
        "przedmiot": "subject",
        "lecturer": "lecturer",
        "prowadzacy": "lecturer",
        "ects": "ects",
        "exam_date": "exam_date",
        "termin_egzaminu": "exam_date",
        "assessment": "assessment_method",
        "assessment_method": "assessment_method",
        "zaliczenie": "assessment_method",
    }
    for key, value in row.items():
        canonical = mapping.get((key or "").strip().lower())
        if canonical and value:
            metadata[canonical] = value
    return metadata


def _clean_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _looks_like_lecturer(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith(("dr ", "prof.", "mgr ", "dr hab."))
