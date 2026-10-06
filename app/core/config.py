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

    # Doctor-schedule service database (Stage 2 of voice triage).
    # Fully decoupled from the pgvector connection above: the Spring Boot
    # DoctorScheduleService runs on its own PostgreSQL instance (port 5432),
    # NOT on the pgvector container (port 5433). Override every value in .env
    # for anything other than local development.
    SCHEDULE_DB_HOST: str = "localhost"
    SCHEDULE_DB_PORT: int = 5432
    SCHEDULE_DB_USER: str = "postgres"
    SCHEDULE_DB_PASSWORD: str = "123"
    SCHEDULE_DB_NAME: str = "clinic_management_doctorschedule_service"
    SCHEDULE_DB_CONNECT_TIMEOUT: int = 5

    # External API keys
    GROQ_API_KEY: str = ""
    LLAMA_CLOUD_API_KEY: str = ""

    # Embedding model
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024

    # Chunking
    CHUNK_MAX_CHARS: int = 1200
    CHUNK_OVERLAP: int = 150

    # LLM & VLM Models
    LLM_MODEL_ID: str = "openai/gpt-oss-120b"
    LLM_STT_MODEL_ID: str = "whisper-large-v3"
    VLM_MODEL_ID: str = "qwen/qwen3.8-27b"

    # Retrieval
    RETRIEVER_TOP_K: int = 3

    @property
    def db_dsn(self) -> str:
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def schedule_db_dsn(self) -> str:
        """DSN for the doctor-schedule service database.

            Built purely from the ``SCHEDULE_DB_*`` settings - deliberately *not*
            derived from the pgvector connection, because the two databases live
            on different servers. Credentials always come from the environment.
        """
        return (
            f"postgresql://{self.SCHEDULE_DB_USER}:{self.SCHEDULE_DB_PASSWORD}"
            f"@{self.SCHEDULE_DB_HOST}:{self.SCHEDULE_DB_PORT}/{self.SCHEDULE_DB_NAME}"
        )

    @property
    def schedule_admin_dsn(self) -> str:
        """Same server, maintenance database - used by scripts/setup_schedule_db.py."""
        return (
            f"postgresql://{self.SCHEDULE_DB_USER}:{self.SCHEDULE_DB_PASSWORD}"
            f"@{self.SCHEDULE_DB_HOST}:{self.SCHEDULE_DB_PORT}/postgres"
        )

    def validate_api_keys(self) -> None:
        missing = []
        if not self.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if missing:
            logger.warning("Missing environment variables: %s", ", ".join(missing))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_api_keys()
    return settings