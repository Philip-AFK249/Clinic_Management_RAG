"""Table DDL, GIN indexes, and migration scripts for pgvector."""
import json
import secrets
import threading
from datetime import date, datetime
from typing import Any, Iterable, Optional

from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.logging import get_logger
from app.database.connection import get_connection, get_cursor

logger = get_logger(__name__)

TABLE_NAME = "clinic_knowledge_nodes"
FLAG_JSONL_PATH = PROJECT_ROOT / "data" / "flagged_symptoms" / "unmatched_symptoms.jsonl"

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

CREATE_UNMATCHED_LOGS_SQL = """
CREATE TABLE IF NOT EXISTS unmatched_symptom_logs (
    id SERIAL PRIMARY KEY,
    flag_id TEXT UNIQUE NOT NULL,
    transcription TEXT NOT NULL,
    age_group TEXT NOT NULL DEFAULT 'tat_ca',
    ai_disease_guess TEXT,
    ai_icd10_code TEXT,
    ai_department_id INTEGER,
    ai_department_name TEXT,
    ai_priority_level TEXT,
    flag_reason TEXT NOT NULL DEFAULT 'LOW_CONFIDENCE_MATCH',
    similarity_score DOUBLE PRECISION,
    ticket_number TEXT,
    patient_name TEXT,
    confirmed_icd10_code TEXT,
    confirmed_disease_name TEXT,
    confirmed_by_doctor TEXT,
    medical_symptom_standard TEXT,
    red_flags TEXT,
    reception_action TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING_DOCTOR',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_UNMATCHED_LOGS_STATUS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_unmatched_logs_status
ON unmatched_symptom_logs (status, created_at DESC);
"""

CREATE_UNMATCHED_LOGS_TICKET_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_unmatched_logs_ticket
ON unmatched_symptom_logs (ticket_number);
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

UNMATCHED_LOG_COLUMNS = (
    "id",
    "flag_id",
    "transcription",
    "age_group",
    "ai_disease_guess",
    "ai_icd10_code",
    "ai_department_id",
    "ai_department_name",
    "ai_priority_level",
    "flag_reason",
    "similarity_score",
    "ticket_number",
    "patient_name",
    "confirmed_icd10_code",
    "confirmed_disease_name",
    "confirmed_by_doctor",
    "medical_symptom_standard",
    "red_flags",
    "reception_action",
    "status",
    "created_at",
    "updated_at",
)

UNMATCHED_LOG_WRITABLE_COLUMNS = (
    "transcription",
    "age_group",
    "ai_disease_guess",
    "ai_icd10_code",
    "ai_department_id",
    "ai_department_name",
    "ai_priority_level",
    "flag_reason",
    "similarity_score",
    "ticket_number",
    "patient_name",
    "confirmed_icd10_code",
    "confirmed_disease_name",
    "confirmed_by_doctor",
    "medical_symptom_standard",
    "red_flags",
    "reception_action",
    "status",
)

FLAG_STATUSES = ("PENDING_DOCTOR", "DOCTOR_CONFIRMED", "TRAINED")

_JSONL_LOCK = threading.Lock()
_UNMATCHED_TABLE_READY = False

# UNIQUE: a chunk's entity markers (used by optional dedupe)
_STATEMENTS: list[str] = [
    CREATE_TABLE_SQL,
    CREATE_UNMATCHED_LOGS_SQL,
    CREATE_UNMATCHED_LOGS_STATUS_INDEX_SQL,
    CREATE_UNMATCHED_LOGS_TICKET_INDEX_SQL,
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


def _ensure_unmatched_table(settings: Settings) -> None:
    global _UNMATCHED_TABLE_READY
    if _UNMATCHED_TABLE_READY:
        return
    try:
        run_migrations(
            settings,
            [
                CREATE_UNMATCHED_LOGS_SQL,
                CREATE_UNMATCHED_LOGS_STATUS_INDEX_SQL,
                CREATE_UNMATCHED_LOGS_TICKET_INDEX_SQL,
            ],
        )
        _UNMATCHED_TABLE_READY = True
    except Exception as exc:
        logger.warning("Không tạo được bảng unmatched_symptom_logs: %s", exc)


def _summarize(stmt: str) -> str:
    first_line = stmt.strip().splitlines()[0]
    return first_line[:80]


def _serialise_log(row: dict[str, Any]) -> dict[str, Any]:
    serialised: dict[str, Any] = {}
    for column in UNMATCHED_LOG_COLUMNS:
        value = row.get(column)
        if isinstance(value, (date, datetime)):
            value = value.isoformat()
        serialised[column] = value
    return serialised


def _row_from_tuple(row: tuple) -> dict[str, Any]:
    return _serialise_log(dict(zip(UNMATCHED_LOG_COLUMNS, row)))


def _read_jsonl_unlocked() -> list[dict[str, Any]]:
    if not FLAG_JSONL_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with FLAG_JSONL_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(_serialise_log(payload))
    except OSError as exc:
        logger.warning("Không đọc được %s: %s", FLAG_JSONL_PATH, exc)
    return rows


def _write_jsonl_unlocked(rows: list[dict[str, Any]]) -> None:
    FLAG_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FLAG_JSONL_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_serialise_log(row), ensure_ascii=False) + "\n")


def list_unmatched_logs(settings: Optional[Settings] = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    _ensure_unmatched_table(settings)
    columns = ", ".join(UNMATCHED_LOG_COLUMNS)
    try:
        with get_connection(settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {columns} FROM unmatched_symptom_logs "
                    "ORDER BY created_at DESC, id DESC"
                )
                rows = cur.fetchall()
            conn.commit()
        return [_row_from_tuple(row) for row in rows]
    except Exception as exc:
        logger.warning("Falling back to JSONL for flagged symptoms: %s", exc)
        with _JSONL_LOCK:
            return _read_jsonl_unlocked()


def new_flag_id(settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    existing = {row.get("flag_id") for row in list_unmatched_logs(settings)}
    while True:
        candidate = f"FLG-{datetime.now().strftime('%Y%m%d')}-{secrets.randbelow(10000):04d}"
        if candidate not in existing:
            return candidate


def insert_unmatched_log(
    values: dict[str, Any], settings: Optional[Settings] = None
) -> dict[str, Any]:
    settings = settings or get_settings()
    _ensure_unmatched_table(settings)
    columns = [
        column
        for column in UNMATCHED_LOG_COLUMNS
        if column not in ("id", "created_at", "updated_at")
    ]
    payload = {column: values.get(column) for column in columns}
    payload["flag_id"] = values.get("flag_id") or new_flag_id(settings)
    payload["transcription"] = str(payload.get("transcription") or "")
    payload["age_group"] = payload.get("age_group") or "tat_ca"
    payload["flag_reason"] = payload.get("flag_reason") or "LOW_CONFIDENCE_MATCH"
    payload["status"] = payload.get("status") or "PENDING_DOCTOR"

    try:
        placeholders = ", ".join(["%s"] * len(columns))
        returning = ", ".join(UNMATCHED_LOG_COLUMNS)
        with get_cursor(settings) as cur:
            cur.execute(
                f"INSERT INTO unmatched_symptom_logs ({', '.join(columns)}) "
                f"VALUES ({placeholders}) RETURNING {returning}",
                tuple(payload[column] for column in columns),
            )
            row = cur.fetchone()
        return _row_from_tuple(row)
    except Exception as exc:
        logger.warning("Không lưu được flagged symptom vào DB, dùng JSONL: %s", exc)

    with _JSONL_LOCK:
        rows = _read_jsonl_unlocked()
        now = datetime.now().isoformat(timespec="seconds")
        next_id = max(
            (int(row.get("id") or 0) for row in rows if str(row.get("id") or "").isdigit()),
            default=0,
        ) + 1
        row = {column: payload.get(column) for column in UNMATCHED_LOG_COLUMNS}
        row["id"] = next_id
        row["created_at"] = now
        row["updated_at"] = now
        rows.insert(0, row)
        _write_jsonl_unlocked(rows)
    return _serialise_log(row)


def update_unmatched_log(
    flag_id: str,
    changes: dict[str, Any],
    settings: Optional[Settings] = None,
) -> Optional[dict[str, Any]]:
    settings = settings or get_settings()
    _ensure_unmatched_table(settings)
    updates = {
        column: value
        for column, value in changes.items()
        if column in UNMATCHED_LOG_WRITABLE_COLUMNS
    }
    if not updates:
        return None

    returning = ", ".join(UNMATCHED_LOG_COLUMNS)
    set_clause = ", ".join(f"{column} = %s" for column in updates)
    values = list(updates.values()) + [flag_id]
    try:
        with get_cursor(settings) as cur:
            cur.execute(
                f"UPDATE unmatched_symptom_logs SET {set_clause}, updated_at = NOW() "
                f"WHERE flag_id = %s RETURNING {returning}",
                tuple(values),
            )
            row = cur.fetchone()
        if row:
            return _row_from_tuple(row)
    except Exception as exc:
        logger.warning("Không cập nhật được flagged symptom trong DB, dùng JSONL: %s", exc)

    with _JSONL_LOCK:
        rows = _read_jsonl_unlocked()
        updated: Optional[dict[str, Any]] = None
        for row in rows:
            if str(row.get("flag_id")) == flag_id:
                row.update(updates)
                row["updated_at"] = datetime.now().isoformat(timespec="seconds")
                updated = _serialise_log(row)
                break
        if updated:
            _write_jsonl_unlocked(rows)
    return updated


def delete_unmatched_log(
    flag_id: str, settings: Optional[Settings] = None
) -> bool:
    settings = settings or get_settings()
    _ensure_unmatched_table(settings)
    try:
        with get_cursor(settings) as cur:
            cur.execute("DELETE FROM unmatched_symptom_logs WHERE flag_id = %s", (flag_id,))
            deleted = cur.rowcount
        if deleted:
            return True
    except Exception as exc:
        logger.warning("Không xoá được flagged symptom trong DB, dùng JSONL: %s", exc)

    with _JSONL_LOCK:
        rows = _read_jsonl_unlocked()
        remaining = [row for row in rows if str(row.get("flag_id")) != flag_id]
        if len(remaining) == len(rows):
            return False
        _write_jsonl_unlocked(remaining)
    return True


if __name__ == "__main__":
    run_migrations(get_settings())