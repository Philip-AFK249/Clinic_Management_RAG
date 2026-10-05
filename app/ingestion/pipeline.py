"""Ingestion orchestrator: Parse -> Chunk -> Extract -> Embed -> Insert."""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from psycopg.types.json import Jsonb

from app.core.config import PARSED_MARKDOWN_DIR, RAW_DOCS_DIR, Settings, get_settings
from app.core.logging import get_logger
from app.database.connection import get_connection
from app.database.schema import run_migrations
from app.ingestion.embedder import embed_texts
from app.ingestion.entity_extractor import ExtractedEntities, extract_entities
from app.ingestion.splitter import chunk_markdown

logger = get_logger(__name__)

INSERT_SQL = """
    INSERT INTO clinic_knowledge_nodes
        (doc_id, specialty, target_audience, content, dense_embedding, metadata)
    VALUES (%s, %s, %s, %s, %s, %s::jsonb);
"""

# Department routing for the three clinical departments plus the catch-all.
DEFAULT_SPECIALTY = "chung"
DEFAULT_AUDIENCE = "bac_si"

# Order matters: the first matching department wins, so more specific keyword
# sets are listed before broader ones.
SPECIALTY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tim_mach": (
        "tim_mach", "tim mach", "noi_khoa", "cardio", "cardiology",
        "tuan_hoan", "huyet_ap", "nhoi_mau", "dien_tam", "nong_van",
    ),
    "ho_hap_di_ung": (
        "ho_hap", "di_ung", "tai_mui_hong", "tai_mui", "respiratory",
        "ent", "hen_suyen", "phoi", "sung_mui", "vien_xoang",
    ),
    "da_lieu": ("da_lieu", "derma", "dermat", "dermatology", "vay_nen"),
}

# Documents aimed at reception / triage rather than at clinicians.
AUDIENCE_TRIAGE_KEYWORDS: tuple[str, ...] = (
    "triage", "tiep_don", "sang_loc", "trieu_chung",
)


@dataclass
class IngestResult:
    doc_id: str
    specialty: str
    total_chunks: int
    entities_found: int
    target_audience: str = DEFAULT_AUDIENCE


def purge_document(
    doc_id: str,
    settings: Optional[Settings] = None,
    table: str = "clinic_knowledge_nodes",
) -> int:
    """Delete every existing chunk row for ``doc_id``; return how many were removed.

    Used by the pipeline console's "clean replace" option so re-ingesting a file
    cannot leave the previous version's vectors behind.
    """
    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table} WHERE doc_id = %s;", (doc_id,))
            deleted = cur.rowcount
        conn.commit()
    logger.info("Purged %d existing chunk(s) for doc_id=%s", deleted, doc_id)
    return max(deleted, 0)


def ingest_markdown_file(
    markdown_path: Path,
    doc_id: str,
    specialty: str,
    target_audience: str = DEFAULT_AUDIENCE,
    settings: Optional[Settings] = None,
    migrate: bool = True,
) -> IngestResult:
    """Chunk, embed, and insert a single Markdown file into pgvector.

    ``migrate=False`` lets a batch caller apply the DDL once instead of paying
    for four statements per file.
    """
    settings = settings or get_settings()
    if migrate:
        run_migrations(settings)

    markdown_path = Path(markdown_path)
    chunks = chunk_markdown(
        str(markdown_path),
        max_chars=settings.CHUNK_MAX_CHARS,
        overlap=settings.CHUNK_OVERLAP,
    )
    total = len(chunks)
    if total == 0:
        logger.warning("No usable content in %s - skipping", markdown_path.name)
        return IngestResult(
            doc_id=doc_id,
            specialty=specialty,
            total_chunks=0,
            entities_found=0,
            target_audience=target_audience,
        )

    logger.info("Embedding %d chunks from %s", total, markdown_path.name)
    embeddings = embed_texts(chunks, settings)

    entities_count = 0
    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            for idx, (content, embedding) in enumerate(zip(chunks, embeddings), 1):
                entities: ExtractedEntities = extract_entities(content)
                entities_count += len(entities.icd10_codes) + len(entities.medications)
                metadata = {
                    "entities": entities.as_dict,
                    "chunk_index": idx - 1,
                    "specialty": specialty,
                    "target_audience": target_audience,
                    "source_file": markdown_path.name,
                }
                cur.execute(
                    INSERT_SQL,
                    (doc_id, specialty, target_audience, content, embedding, Jsonb(metadata)),
                )
                if idx % 25 == 0 or idx == total:
                    pct = idx / total * 100
                    logger.info("  -> [%5.1f%%] Ingested %d/%d chunks", pct, idx, total)
        conn.commit()  # Persist all chunks to the database

    logger.info("Pipeline complete for %s (%s/%s: %d chunks, %d entities)",
                doc_id, specialty, target_audience, total, entities_count)
    return IngestResult(
        doc_id=doc_id,
        specialty=specialty,
        total_chunks=total,
        entities_found=entities_count,
        target_audience=target_audience,
    )


def ingest_directory(
    raw_docs_dir: Path = RAW_DOCS_DIR,
    parsed_dir: Path = PARSED_MARKDOWN_DIR,
    settings: Optional[Settings] = None,
) -> list[IngestResult]:
    """Ingest every Markdown file in ``parsed_dir`` using the matching PDF stem.

    Specialty and target audience are inferred from the file stem so the three
    clinical departments stay separated in the vector store and reception staff
    only ever retrieve triage material.
    """
    settings = settings or get_settings()
    results = []
    for md_path in sorted(parsed_dir.glob("*.md")):
        source = next(
            (f for f in raw_docs_dir.glob("*.pdf") if f.stem == md_path.stem),
            None,
        )
        specialty, audience = classify_document(md_path.stem)
        doc_id = md_path.stem
        logger.info(
            "Ingesting %s (specialty=%s, audience=%s, source=%s)",
            md_path.name, specialty, audience, source.name if source else "markdown-only",
        )
        results.append(
            ingest_markdown_file(
                md_path,
                doc_id,
                specialty,
                target_audience=audience,
                settings=settings,
            )
        )
    return results


def classify_document(stem: str) -> Tuple[str, str]:
    """Return ``(specialty, target_audience)`` inferred from a document stem."""
    return _infer_specialty(stem), _infer_target_audience(stem)


def _normalise_stem(stem: str) -> str:
    """Lowercase, strip accents and fold separators for keyword matching."""
    decomposed = unicodedata.normalize("NFD", stem.lower())
    flat = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "_", flat).strip("_")


def _matches_token(haystack: str, keyword: str) -> bool:
    """Whole-token match so ``ent`` never fires inside ``different``/``trent``."""
    token = _normalise_stem(keyword)
    if not token:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", haystack) is not None


def _infer_specialty(stem: str) -> str:
    """Route a document to one of the three departments, else ``chung``."""
    haystack = _normalise_stem(stem)
    if not haystack:
        return DEFAULT_SPECIALTY
    for specialty, keywords in SPECIALTY_KEYWORDS.items():
        if any(_matches_token(haystack, kw) for kw in keywords):
            return specialty
    return DEFAULT_SPECIALTY


def _infer_target_audience(stem: str) -> str:
    """``tiep_don`` for triage/voice-booking material, else ``bac_si``."""
    haystack = _normalise_stem(stem)
    if any(_matches_token(haystack, kw) for kw in AUDIENCE_TRIAGE_KEYWORDS):
        return "tiep_don"
    return DEFAULT_AUDIENCE


if __name__ == "__main__":
    ingest_directory()