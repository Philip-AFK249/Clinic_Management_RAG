"""Append-only JSONL store for BHYT records confirmed by the patient.

Deliberately separate from the pgvector knowledge base: `clinic_knowledge_nodes`
holds embedding vectors for clinical protocols, so patient identity documents
must never be written there.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import DATA_DIR
from app.core.logging import get_logger

logger = get_logger(__name__)

RECORDS_DIR = DATA_DIR / "bhyt_records"
RECORDS_FILE = RECORDS_DIR / "confirmed_bhyt.jsonl"

_write_lock = threading.Lock()


def save_confirmed_record(payload: dict) -> dict:
    """Persist one patient-confirmed payload and return the stored record."""
    record = {
        "id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "groq_vlm_human_in_the_loop",
        **payload,
    }

    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    with _write_lock, RECORDS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Đã lưu hồ sơ BHYT %s (%s)", record["id"], record.get("ma_so_bhyt"))
    return record


def count_records() -> int:
    """Number of confirmed records currently on disk."""
    if not RECORDS_FILE.exists():
        return 0
    with RECORDS_FILE.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def records_path() -> Path:
    return RECORDS_FILE