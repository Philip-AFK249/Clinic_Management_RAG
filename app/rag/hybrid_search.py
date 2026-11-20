"""Hybrid retrieval: full-text (tsvector) + dense vector with reciprocal reranking."""
import math
from dataclasses import dataclass
from typing import List, Optional

from app.core.config import Settings, get_settings
from app.database.connection import get_connection
from app.core.logging import get_logger
from app.rag.retriever import SearchResult, retrieve_context

logger = get_logger(__name__)


@dataclass
class HybridResult:
    result: SearchResult
    dense_rank: int
    fts_rank: int | None
    final_score: float

    @property
    def content(self) -> str:
        return self.result.content

    @property
    def doc_id(self) -> str:
        return self.result.doc_id

    @property
    def specialty(self) -> str:
        return self.result.specialty


def _fts_results(
    query: str,
    top_k: int,
    specialty: Optional[str],
    age_group: Optional[str] = None,
    target_audience: Optional[str] = None,
    settings: Settings | None = None,
) -> List[tuple]:
    sql = """
        SELECT id, doc_id, specialty, content, ts_rank_cd(
            to_tsvector('simple', content),
            plainto_tsquery('simple', %s)
        ) AS rank
        FROM clinic_knowledge_nodes
    """
    params = [query]
    where = []
    if specialty and specialty != "all":
        where.append("specialty = %s")
        params.append(specialty)
    if target_audience and target_audience != "all":
        where.append("target_audience = %s")
        params.append(target_audience)
    if age_group and age_group not in {"all", "tat_ca"}:
        where.append("(age_group = %s OR age_group = 'tat_ca')")
        params.append(age_group)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY rank DESC NULLS LAST LIMIT %s;"
    params.append(top_k)

    with get_connection(settings or get_settings()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [r for r in rows if r[4] is not None]


def _reciprocal_rank(r: int, k: int = 60) -> float:
    return 1.0 / (k + r)


def hybrid_search(
    query: str,
    top_k: int = 3,
    c: float = 0.5,
    specialty: Optional[str] = None,
    target_audience: Optional[str] = None,
    age_group: Optional[str] = None,
    settings: Settings | None = None,
) -> List[HybridResult]:
    """Run dense + full-text search and fuse results via reciprocal rank fusion."""
    settings = settings or get_settings()
    dense_results = retrieve_context(
        query,
        top_k=top_k * 2,
        specialty=specialty,
        target_audience=target_audience,
        age_group=age_group,
        settings=settings,
    )
    fts_results = _fts_results(
        query,
        top_k * 2,
        specialty,
        age_group=age_group,
        target_audience=target_audience,
        settings=settings,
    )

    dense_id_map = {r.id: (i, r) for i, r in enumerate(dense_results, 1)}
    fts_id_map = {r[0]: (i, r) for i, r in enumerate(fts_results, 1)}

    all_ids = set(dense_id_map.keys()) | set(fts_id_map.keys())

    candidates: dict[int, HybridResult] = {}
    for doc_id in all_ids:
        dense_entry = dense_id_map.get(doc_id)
        fts_entry = fts_id_map.get(doc_id)

        dense_rank = dense_entry[0] if dense_entry else None
        fts_rank = fts_entry[0] if fts_entry else None

        if dense_entry:
            base_res = dense_entry[1]
        else:
            raw_fts = fts_entry[1]
            base_res = SearchResult(
                id=raw_fts[0],
                doc_id=raw_fts[1],
                specialty=raw_fts[2],
                content=raw_fts[3],
                distance=1.0,
            )

        final_score = 0.0
        if dense_rank:
            final_score += c * _reciprocal_rank(dense_rank)
        if fts_rank:
            final_score += (1 - c) * _reciprocal_rank(fts_rank)

        candidates[doc_id] = HybridResult(
            result=base_res,
            dense_rank=dense_rank or 999,
            fts_rank=fts_rank,
            final_score=final_score,
        )

    ranked = sorted(candidates.values(), key=lambda h: h.final_score, reverse=True)
    return ranked[:top_k]