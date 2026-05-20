from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    index_dir: Path = BASE_DIR / "data" / "index"
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
