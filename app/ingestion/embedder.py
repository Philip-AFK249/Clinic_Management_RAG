"""BGE-M3 dense embedding generation (1024 dims)."""
import threading
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Serialises first-time model loads so concurrent requests cannot start two
#: multi-gigabyte bge-m3 loads at the same time.
_EMBEDDER_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_embedder(settings: Settings | None = None) -> SentenceTransformer:
    """Return a singleton SentenceTransformer (BAAI/bge-m3)."""
    settings = settings or get_settings()
    logger.info("Loading embedding model %s ...", settings.EMBEDDING_MODEL)
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def warm_up_embedder(settings: Settings | None = None) -> SentenceTransformer:
    """Load the embedding model once, blocking any concurrent first load.

    Callers queue on the lock; by the time the second one enters, the first has
    already populated the ``lru_cache`` so only a single load ever happens.
    """
    with _EMBEDDER_LOCK:
        return get_embedder(settings)


def embed_texts(texts: List[str], settings: Settings | None = None) -> List[list]:
    """Encode a list of texts into 1024-dim vectors (Python lists)."""
    model = get_embedder(settings)
    embeddings = model.encode(texts, show_progress_bar=True)
    return [emb.tolist() for emb in embeddings]


def embed_text(text: str, settings: Settings | None = None) -> list:
    """Encode a single text into a 1024-dim vector (Python list)."""
    return embed_texts([text], settings)[0]