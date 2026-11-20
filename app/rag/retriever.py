"""Cosine-distance vector search against the pgvector knowledge store."""
from dataclasses import dataclass
from typing import List, Optional

from app.core.config import Settings, get_settings
from app.database.connection import get_connection
from app.core.logging import get_logger
from app.ingestion.embedder import embed_text

logger = get_logger(__name__)


@dataclass
class SearchResult:
    id: int
    doc_id: str
    specialty: str
    content: str
    distance: float

    @property
    def score(self) -> float:
        """Cosine similarity in the conventional direction (higher = better)."""
        return max(0.0, 1.0 - self.distance)


def retrieve_context(
    query: str,
    top_k: int = 3,
    specialty: Optional[str] = None,
    target_audience: Optional[str] = None,
    age_group: Optional[str] = None,
    settings: Settings | None = None,
) -> List[SearchResult]:
    """Search the most relevant chunks by cosine distance.

    ``specialty``, ``target_audience`` and ``age_group`` act as optional metadata
    filters. An age filter keeps matching rows plus rows marked ``tat_ca``.
    """
    settings = settings or get_settings()
    query_vector = embed_text(query, settings)

    where, params = [], []
    if specialty and specialty != "all":
        where.append("specialty = %s")
        params.append(specialty)
    if target_audience and target_audience != "all":
        where.append("target_audience = %s")
        params.append(target_audience)
    if age_group and age_group not in {"all", "tat_ca"}:
        where.append("(age_group = %s OR age_group = 'tat_ca')")
        params.append(age_group)

    sql = """
        SELECT id, doc_id, specialty, content, (dense_embedding <=> %s::vector) AS distance
        FROM clinic_knowledge_nodes
    """
    params = [query_vector] + params
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY distance ASC LIMIT %s;"
    params.append(top_k)

    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    results = [
        SearchResult(id=r[0], doc_id=r[1], specialty=r[2], content=r[3], distance=r[4])
        for r in rows
    ]
    logger.debug("Retrieved %d chunks for query (top_k=%d)", len(results), top_k)
    return results