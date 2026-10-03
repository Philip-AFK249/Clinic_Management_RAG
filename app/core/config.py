"""Application configuration via Pydantic Settings (.env loading)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
PARSED_MARKDOWN_DIR = DATA_DIR / "parsed_markdown"


class Settings(BaseSettings):
    """Centralized environment configuration for the Clinic RAG system."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # PostgreSQL / pgvector connection
    DB_HOST: str = "localhost"
    DB_PORT: int = 5433
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "mysecretpassword"
    DB_NAME: str = "clinic_rag"

    # External API keys
    GROQ_API_KEY: str = ""
    LLAMA_CLOUD_API_KEY: str = ""

    # Embedding model
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024

    # Chunking
    CHUNK_MAX_CHARS: int = 1200
    CHUNK_OVERLAP: int = 150

    # LLM
    LLM_MODEL_ID: str = "openai/gpt-oss-120b"
    LLM_STT_MODEL_ID: str = "whisper-large-v3"

    # Retrieval
    RETRIEVER_TOP_K: int = 3

    @property
    def db_dsn(self) -> str:
        """Build the psycopg connection URI."""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    def validate_api_keys(self) -> None:
        """Fail fast if required API keys are missing."""
        missing = []
        if not self.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not self.LLAMA_CLOUD_API_KEY:
            missing.append("LLAMA_CLOUD_API_KEY")
        if missing:
            logger.warning("Missing environment variables: %s", ", ".join(missing))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    settings = Settings()
    settings.validate_api_keys()
    return settings