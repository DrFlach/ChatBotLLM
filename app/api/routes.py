from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.api.intents import detect_intent
from app.core.config import get_settings
from app.rag.generator import generate_answer
from app.rag.retriever import retrieve
from app.rag.vector_store import index_exists, indexed_chunk_count


router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(...)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        return value.strip()


class Source(BaseModel):
    file_name: str
    document_type: str | None = None
    chunk_number: int | None = None
    page: int | None = None
    row: int | None = None
    section: int | None = None
    lecturer: str | None = None
    subject: str | None = None
    semester: str | int | None = None
    field: str | None = None
    ects: str | int | None = None
    exam_date: str | None = None
    assessment_method: str | None = None
    score: float
    metadata: dict
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    intent: str | None = None


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    intent = detect_intent(request.question)
    if intent.intent not in _RAG_INTENTS:
        return ChatResponse(answer=intent.answer or "", sources=[], intent=intent.intent)

    try:
        contexts = retrieve(request.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    answer = generate_answer(request.question, contexts)
    source_limit = _source_limit(request.question)
    sources = [_format_source(item) for item in contexts[:source_limit]]
    return ChatResponse(answer=answer, sources=sources, intent=intent.intent)


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "index_loaded": index_exists(settings.index_dir),
        "embedding_model": settings.embedding_model_name,
        "vector_store": "FAISS",
    }


@router.get("/config")
def public_config() -> dict:
    settings = get_settings()
    return {
        "embedding_model": settings.embedding_model_name,
        "vector_database": "FAISS",
        "openai_configured": bool(settings.openai_api_key),
        "indexed_chunks": indexed_chunk_count(settings.index_dir),
    }


def _format_source(item: dict) -> Source:
    metadata = item.get("metadata", {})
    return Source(
        file_name=metadata.get("source", "unknown"),
        document_type=metadata.get("document_type"),
        chunk_number=metadata.get("chunk_id"),
        page=metadata.get("page"),
        row=metadata.get("row"),
        section=metadata.get("section"),
        lecturer=metadata.get("lecturer"),
        subject=metadata.get("subject"),
        semester=metadata.get("semester"),
        field=metadata.get("field"),
        ects=metadata.get("ects"),
        exam_date=metadata.get("exam_date"),
        assessment_method=metadata.get("assessment_method"),
        score=item["score"],
        metadata=metadata,
        preview=" ".join(item.get("text", "").split())[:350],
    )


def _source_limit(question: str) -> int:
    lowered = question.lower()
    teacher_markers = (
        "kto prowadzi",
        "kto jest prowadzącym",
        "kto jest prowadzacym",
        "kto uczy",
        "jaki wykładowca",
        "jaki wykladowca",
        "kto odpowiada",
        "who teaches",
        "who is the lecturer",
        "who runs",
        "who is responsible",
        "lecturer for",
    )
    if any(marker in lowered for marker in teacher_markers):
        return 2
    return 3


_RAG_INTENTS = {
    "subject_list",
    "course_description",
    "lecturer",
    "consultation",
    "assessment",
    "exam_date",
    "exam_rules",
    "passing_rules",
    "retake_rules",
    "plagiarism",
    "ects",
}
