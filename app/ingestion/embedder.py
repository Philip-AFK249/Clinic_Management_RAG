"""BGE-M3 dense embedding generation (1024 dims)."""
import threading
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_EMBEDDER_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_embedder(settings: Settings | None = None) -> SentenceTransformer:
    settings = settings or get_settings()
    logger.info("Loading embedding model %s ...", settings.EMBEDDING_MODEL)
    try:
        return SentenceTransformer(
            settings.EMBEDDING_MODEL, device="cpu", local_files_only=True
        )
    except OSError:
        logger.info(
            "Local cache miss for %s; falling back to download.",
            settings.EMBEDDING_MODEL,
        )
        return SentenceTransformer(
            settings.EMBEDDING_MODEL, device="cpu", local_files_only=False
        )


def warm_up_embedder(settings: Settings | None = None) -> SentenceTransformer:
    with _EMBEDDER_LOCK:
        return get_embedder(settings)


def embed_texts(texts: List[str], settings: Settings | None = None) -> List[list]:
    model = warm_up_embedder(settings)
    embeddings = model.encode(texts, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def embed_text(text: str, settings: Settings | None = None) -> list:
    return embed_texts([text], settings)[0]
