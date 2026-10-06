"""Provision and seed the doctor-schedule database.

Run with::

    python scripts/setup_schedule_db.py

Safe to re-run: the database, tables, departments and doctors are created only
when missing, and outpatient shifts are inserted only for
(doctor, department, date, session) combinations that have no OUTPATIENT row
yet. Nothing that already exists is overwritten or deleted.

Credentials come from ``SCHEDULE_DB_*`` in ``app.core.config.Settings`` (i.e.
.env), defaulting to the local Spring Boot PostgreSQL instance on port 5432.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple
from zoneinfo import ZoneInfo

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402

logger = get_logger("setup_schedule_db")

CLINIC_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Codes/names mirror the departments already used by the clinic services.
SEED_DEPARTMENTS: Tuple[Tuple[int, str, str], ...] = (
    (1, "INT-CARD", "Khoa Nội Tổng quát & Tim mạch"),
    (2, "RESP-ALLERGY", "Khoa Hô hấp & Dị ứng - Miễn dịch lâm sàng"),
    (3, "DERM", "Khoa Da liễu"),
)

# name, title, room, department_id, max_patients_per_slot
SEED_DOCTORS: Tuple[Tuple[int, str, str, str, int, int], ...] = (
    (1, "PGS. TS. BS. Trần Minh Tuấn", "PGS.TS", "Phòng 101", 1, 4),
    (2, "BS. CKI. Nguyễn Văn Dũng", "BS.CKI", "Phòng 102", 1, 4),
    (3, "ThS. BS. Phạm Quốc Bảo", "ThS.BS", "Phòng 101", 1, 4),
    (4, "BS. CKI. Lê Thị Hoàng Yến", "BS.CKI", "Phòng 201", 2, 4),
    (5, "BS. CKII. Phạm Thị Hoa", "BS.CKII", "Phòng 202", 2, 4),
    (6, "ThS. BS. Hoàng Hoài Nam", "ThS.BS", "Phòng 201", 2, 4),
    (7, "BS. CKI. Nguyễn Thị Lan", "BS.CKI", "Phòng 205", 3, 4),
    (8, "ThS. BS. Vũ Minh Đức", "ThS.BS", "Phòng 206", 3, 4),
    (9, "BS. Mai Thu Hương", "BS", "Phòng 205", 3, 4),
)

SESSIONS = ("MORNING", "AFTERNOON")

SCHEMA_STATEMENTS: Tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS departments (
        id   BIGINT PRIMARY KEY,
        code VARCHAR(32),
        name VARCHAR(255)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS doctors (
        id            BIGINT PRIMARY KEY,
        full_name     VARCHAR(255),
        title         VARCHAR(64),
        room_number   VARCHAR(32),
        department_id BIGINT,
        active        BOOLEAN DEFAULT TRUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS doctor_shifts (
        id                     BIGSERIAL PRIMARY KEY,
        doctor_id              BIGINT,
        department_id          BIGINT,
        shift_date             DATE,
        session                VARCHAR(16),
        duty_type              VARCHAR(16),
        max_patients_per_slot  INTEGER,
        room_number            VARCHAR(32)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS slot_assignments (
        id                BIGSERIAL PRIMARY KEY,
        ticket_number     VARCHAR(32),
        doctor_id         BIGINT,
        department_id     BIGINT,
        appointment_date  DATE,
        slot_start_time   TIME,
        status            VARCHAR(16) DEFAULT 'BOOKED'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_shifts_dept_date ON doctor_shifts (department_id, shift_date)",
    "CREATE INDEX IF NOT EXISTS idx_shifts_doctor_date ON doctor_shifts (doctor_id, shift_date)",
    "CREATE INDEX IF NOT EXISTS idx_slots_date ON slot_assignments (appointment_date)",
)


def _admin_dsn(settings) -> str:
    return settings.schedule_admin_dsn


def ensure_database(settings) -> bool:
    """Create the database if missing. Returns True when it had to create it."""
    target = settings.SCHEDULE_DB_NAME
    with psycopg.connect(_admin_dsn(settings), connect_timeout=10, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (target,))
            if cur.fetchone():
                logger.info("Database '%s' already exists - reusing it.", target)
                return False
            # CREATE DATABASE has no IF NOT EXISTS, hence the explicit check.
            cur.execute(f'CREATE DATABASE "{target}";')
            logger.info("Created database '%s'.", target)
            return True


def ensure_tables(settings) -> List[str]:
    """Create any missing tables/indexes. Returns the statements that ran."""
    applied: List[str] = []
    with psycopg.connect(
        settings.schedule_db_dsn, connect_timeout=10, autocommit=True
    ) as conn:
        with conn.cursor() as cur:
            for statement in SCHEMA_STATEMENTS:
                cur.execute(statement)
                applied.append(" ".join(statement.split())[:70])
    return applied


def _table_exists(cur, name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name=%s;",
        (name,),
    )
    return cur.fetchone() is not None


def _columns(cur, table: str) -> List[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position;",
        (table,),
    )
    return [r[0] for r in cur.fetchall()]


def seed_departments(settings) -> int:
    """Insert any missing department. Never updates an existing row."""
    inserted = 0
    with psycopg.connect(
        settings.schedule_db_dsn, connect_timeout=10, autocommit=True
    ) as conn:
        with conn.cursor() as cur:
            if not _table_exists(cur, "departments"):
                logger.warning("departments table missing - run ensure_tables first.")
                return 0
            for dep_id, code, name in SEED_DEPARTMENTS:
                cur.execute(
                    "SELECT 1 FROM departments WHERE id = %s;", (dep_id,)
                )
                if cur.fetchone():
                    continue
                cur.execute(
                    "INSERT INTO departments (id, code, name) VALUES (%s, %s, %s);",
                    (dep_id, code, name),
                )
                inserted += 1
    return inserted


def seed_doctors(settings) -> int:
    """Insert doctors that are absent; leave existing rows untouched.

    Adapts to pre-existing tables that carry extra columns (``user_id``) or
    omit optional ones.
    """
    inserted = 0
    with psycopg.connect(
        settings.schedule_db_dsn, connect_timeout=10, autocommit=True
    ) as conn:
        with conn.cursor() as cur:
            if not _table_exists(cur, "doctors"):
                logger.warning("doctors table missing - run ensure_tables first.")
                return 0
            cols = _columns(cur, "doctors")
            wanted = ["id", "full_name", "title", "room_number", "department_id", "active"]
            usable = [c for c in wanted if c in cols]
            if "id" not in usable or "department_id" not in usable:
                logger.warning("doctors table lacks id/department_id - skipping seed.")
                return 0
            for doc_id, name, title, room, dept_id, _cap in SEED_DOCTORS:
                cur.execute("SELECT 1 FROM doctors WHERE id = %s;", (doc_id,))
                if cur.fetchone():
                    continue
                values = {
                    "id": doc_id,
                    "full_name": name,
                    "title": title,
                    "room_number": room,
                    "department_id": dept_id,
                    "active": True,
                }
                cur.execute(
                    f"INSERT INTO doctors ({', '.join(usable)}) "
                    f"VALUES ({', '.join(['%s'] * len(usable))});",
                    [values[c] for c in usable],
                )
                inserted += 1
    return inserted


def seed_shifts(settings, days: int, duty_type: str = "OUTPATIENT") -> Tuple[int, int, List[str]]:
    """Ensure an outpatient duty row per (department, date, session).

    The live table carries ``UNIQUE (doctor_id, shift_date, session)``, so a
    doctor who already has *any* duty in that slot cannot also be given an
    outpatient one. When that happens the seeder falls through to another active
    doctor in the same department and, if none is left, records a coverage gap
    for the operator instead of silently doing nothing.

    Returns ``(inserted, skipped, gaps)``.
    """
    inserted = 0
    skipped = 0
    gaps: List[str] = []
    today = datetime_now_in_clinic_tz()
    dates = [today + timedelta(days=offset) for offset in range(days)]

    with psycopg.connect(
        settings.schedule_db_dsn, connect_timeout=10, autocommit=True
    ) as conn:
        with conn.cursor() as cur:
            if not (_table_exists(cur, "doctor_shifts") and _table_exists(cur, "doctors")):
                logger.warning("doctor_shifts/doctors missing - run ensure_tables first.")
                return 0, 0, gaps
            shift_cols = _columns(cur, "doctor_shifts")
            cur.execute(
                "SELECT id, department_id, room_number FROM doctors "
                "WHERE active IS TRUE ORDER BY id;"
            )
            doctors = cur.fetchall()
            by_dept: dict[int, List[Tuple[int, str]]] = {}
            for doc_id, dept_id, room in doctors:
                by_dept.setdefault(int(dept_id), []).append((doc_id, room))

            for dept_id, _code, dept_name in SEED_DEPARTMENTS:
                team = by_dept.get(dept_id, [])
                for day in dates:
                    for session in SESSIONS:
                        made_progress = False
                        for doc_id, room in team:
                            # Any existing duty (INPATIENT included) occupies the slot.
                            cur.execute(
                                "SELECT duty_type FROM doctor_shifts "
                                "WHERE doctor_id=%s AND shift_date=%s AND session=%s;",
                                (doc_id, day, session),
                            )
                            existing = cur.fetchone()
                            if existing:
                                skipped += 1
                                if existing[0] == duty_type:
                                    made_progress = True
                                continue
                            cols = ["doctor_id", "department_id", "shift_date",
                                    "session", "duty_type"]
                            vals: List[object] = [doc_id, dept_id, day, session, duty_type]
                            if "max_patients_per_slot" in shift_cols:
                                cols.append("max_patients_per_slot")
                                vals.append(4)
                            if "room_number" in shift_cols:
                                cols.append("room_number")
                                vals.append(room)
                            placeholders = ", ".join(["%s"] * len(cols))
                            cur.execute(
                                f"INSERT INTO doctor_shifts ({', '.join(cols)}) "
                                f"VALUES ({placeholders}) ON CONFLICT DO NOTHING;",
                                vals,
                            )
                            inserted += cur.rowcount
                            if cur.rowcount:
                                made_progress = True
                        if not made_progress:
                            gaps.append(
                                f"{dept_name} | {day} {session}: "
                                f"không còn bác sĩ nào trống (tất cả đã có ca khác)"
                            )
    return inserted, skipped, gaps


def datetime_now_in_clinic_tz() -> date:
    from datetime import datetime

    return datetime.now(CLINIC_TZ).date()


def summarise(settings, days: int) -> None:
    """Print what the triage endpoint will see for each department."""
    from app.database.schedule_queries import query_department_schedule

    today = datetime_now_in_clinic_tz()
    print("\n--- Preview for the triage endpoint ---")
    for dept_id in (1, 2, 3):
        for offset in range(days):
            day = today + timedelta(days=offset)
            result = query_department_schedule(dept_id, day.isoformat(), settings)
            if not result.get("is_available"):
                print(f"  dept {dept_id} {day}: DEGRADED -> {result.get('detail', '')[:60]}")
                continue
            morning = result.get("morning") or {}
            afternoon = result.get("afternoon") or {}
            print(
                f"  dept {dept_id} {day} [{result['department_name'][:34]:34}] "
                f"-> {result['recommended_shift']:10} "
                f"M: {len(morning.get('doctors', []))} bác sĩ, "
                f"{morning.get('total_available_slots', 0)} suất trống | "
                f"A: {len(afternoon.get('doctors', []))} bác sĩ, "
                f"{afternoon.get('total_available_slots', 0)} suất trống"
            )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="How many consecutive days (from today) to cover with shifts. Default 2.",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Only create the database/tables; do not insert departments, doctors or shifts.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = get_settings()
    print("=" * 68)
    print(" SMART CLINIC - DOCTOR SCHEDULE DATABASE PROVISIONING")
    print("=" * 68)
    print(f"  host          : {settings.SCHEDULE_DB_HOST}:{settings.SCHEDULE_DB_PORT}")
    print(f"  database      : {settings.SCHEDULE_DB_NAME}")
    print(f"  user          : {settings.SCHEDULE_DB_USER}")
    print(f"  clinic dates  : today + {max(args.days - 1, 0)} more day(s) "
          f"(timezone {CLINIC_TZ})")
    print("=" * 68)

    try:
        created_db = ensure_database(settings)
    except psycopg.Error as exc:
        print(f"\n[FATAL] Cannot reach PostgreSQL at "
              f"{settings.SCHEDULE_DB_HOST}:{settings.SCHEDULE_DB_PORT}\n"
              f"        {type(exc).__name__}: {' '.join(str(exc).split())[:200]}")
        return 1

    try:
        applied = ensure_tables(settings)
    except psycopg.Error as exc:
        print(f"\n[FATAL] Cannot create tables: {type(exc).__name__}: "
              f"{' '.join(str(exc).split())[:200]}")
        return 1
    print(f"[OK] Schema ready ({len(applied)} statements verified/created).")

    if args.skip_seed:
        print("[--] Seeding skipped (--skip-seed).")
        return 0

    try:
        depts = seed_departments(settings)
        docs = seed_doctors(settings)
        inserted, skipped, gaps = seed_shifts(settings, args.days)
    except psycopg.Error as exc:
        print(f"\n[FATAL] Seeding failed: {type(exc).__name__}: "
              f"{' '.join(str(exc).split())[:200]}")
        return 1

    print(f"[OK] Departments: {depts} inserted, {len(SEED_DEPARTMENTS) - depts} already present.")
    print(f"[OK] Doctors    : {docs} inserted, {len(SEED_DOCTORS) - docs} already present.")
    print(f"[OK] Shifts     : {inserted} outpatient rows inserted, {skipped} already existed.")
    if gaps:
        print(f"[!!] {len(gaps)} slot(s) have no outpatient doctor available:")
        for gap in gaps:
            print(f"       - {gap}")
    else:
        print("[OK] Every department/session has outpatient coverage.")

    summarise(settings, args.days)
    print("\n[DONE] Doctor-schedule database is ready.")
    print(f"       Created database: {created_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())