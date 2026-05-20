from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import BASE_DIR


app = FastAPI(
    title="Chatbot LLM + RAG dla programu studiow",
    description="MVP chatbota odpowiadajacego na pytania o program studiow z uzyciem RAG.",
    version="1.0.0",
)

web_dir = BASE_DIR / "app" / "web"
app.mount("/static", StaticFiles(directory=web_dir), name="static")
app.include_router(router)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
