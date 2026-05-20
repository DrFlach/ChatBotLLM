from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.rag.generator import generate_answer
from app.rag.retriever import retrieve


router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Source(BaseModel):
    source: str
    score: float
    metadata: dict
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        contexts = retrieve(request.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    answer = generate_answer(request.question, contexts)
    sources = [
        Source(
            source=item["metadata"].get("source", "unknown"),
            score=item["score"],
            metadata=item["metadata"],
            preview=item["text"][:350],
        )
        for item in contexts
    ]
    return ChatResponse(answer=answer, sources=sources)
