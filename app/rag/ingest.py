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
    metadata = {"source": path.name}
    if extra_metadata:
        metadata.update(extra_metadata)
    return {"text": text, "metadata": metadata}


def _load_csv(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=1):
            parts = [f"{key}: {value}" for key, value in row.items() if value]
            records.append(_record("\n".join(parts), path, {"row": row_number}))
    return records


def _load_html(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return [_record(text, path)]


def _load_pdf(path: Path) -> list[dict]:
    records: list[dict] = []
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            records.append(_record(text, path, {"page": page_number}))
    return records
