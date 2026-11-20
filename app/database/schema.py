"""Table DDL, GIN indexes, and migration scripts for pgvector."""
from typing import Iterable

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.database.connection import get_cursor

logger = get_logger(__name__)

TABLE_NAME = "clinic_knowledge_nodes"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id SERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    specialty TEXT NOT NULL,
    target_audience TEXT NOT NULL DEFAULT 'bac_si',
    age_group TEXT NOT NULL DEFAULT 'tat_ca',
    content TEXT NOT NULL,
    dense_embedding vector(1024) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_HNSW_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_clinic_nodes_hnsw
ON {TABLE_NAME}
USING hnsw (dense_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

CREATE_GIN_CONTENT_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_clinic_nodes_content_gin
ON {TABLE_NAME}
USING GIN (to_tsvector('simple', content));
"""

CREATE_INDEX_ON_SPECIALTY_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_clinic_nodes_specialty
ON {TABLE_NAME} (specialty);
"""

ADD_AGE_GROUP_SQL = f"""
ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS age_group TEXT NOT NULL DEFAULT 'tat_ca';
"""

CREATE_INDEX_ON_AGE_GROUP_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_clinic_nodes_age_group
ON {TABLE_NAME} (age_group);
"""

# UNIQUE: a chunk's entity markers (used by optional dedupe)
_STATEMENTS: list[str] = [
    CREATE_TABLE_SQL,
    ADD_AGE_GROUP_SQL,
    CREATE_INDEX_ON_AGE_GROUP_SQL,
    CREATE_INDEX_ON_SPECIALTY_SQL,
    CREATE_HNSW_INDEX_SQL,
    CREATE_GIN_CONTENT_INDEX_SQL,
]


def run_migrations(settings: Settings, statements: Iterable[str] | None = None):
    """Execute schema + index DDL statements against the database."""
    statements = list(statements or _STATEMENTS)
    for stmt in statements:
        logger.info("Executing DDL: %s", _summarize(stmt))
        with get_cursor(settings) as cur:
            cur.execute(stmt)
    logger.info("Migrations applied (%d statements)", len(statements))


def truncate_table(settings: Settings) -> None:
    """Delete all rows from the knowledge table (used before re-ingestion)."""
    with get_cursor(settings) as cur:
        cur.execute(f"TRUNCATE {TABLE_NAME} RESTART IDENTITY;")
        logger.info("Truncated table %s", TABLE_NAME)


def _summarize(stmt: str) -> str:
    first_line = stmt.strip().splitlines()[0]
    return first_line[:80]


if __name__ == "__main__":
    run_migrations(get_settings())