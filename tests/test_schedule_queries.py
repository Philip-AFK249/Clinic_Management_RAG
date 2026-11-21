"""Unit tests for :mod:`app.database.schedule_queries`.

These tests run the real ``_SCHEDULE_SQL`` against an in-memory SQLite database
through a tiny psycopg-compatible adapter. That keeps the suite dependency-free
(no live PostgreSQL or pytest needed) while still exercising the SQL filters,
the aggregation and the ``recommended_shift`` decision logic.

Only the PostgreSQL ``TIME '...'`` literal and ``%(name)s`` placeholders are
translated for SQLite; everything else is the production statement.
"""

from __future__ import annotations

import re
import sqlite3
import unittest
from datetime import date
from datetime import time as time_cls
from datetime import datetime
from unittest.mock import patch

from app.database import schedule_queries as sq

TARGET_DATE = date(2099, 1, 5)
TARGET_DATE_STR = TARGET_DATE.isoformat()
TODAY = TARGET_DATE


def _frozen_datetime(frozen_at: datetime):
    """Return a datetime subclass whose ``now()`` always yields ``frozen_at``."""

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return frozen_at.replace(tzinfo=None)
            return frozen_at.astimezone(tz)

    return _FrozenDateTime

_PLACEHOLDER_RE = re.compile(r"%\((\w+)\)s")
_TIME_LITERAL_RE = re.compile(r"\bTIME\s+'([^']+)'", re.IGNORECASE)


def _sqlite_sql(sql: str) -> str:
    """Translate the small PostgreSQL-specific surface used by _SCHEDULE_SQL."""
    sql = _PLACEHOLDER_RE.sub(r":\1", sql)
    sql = _TIME_LITERAL_RE.sub(r"'\1'", sql)
    sql = re.sub(r"\bNOW\s*\(\s*\)", "datetime('now')", sql, flags=re.IGNORECASE)
    return sql


def _sqlite_params(params: dict | None) -> dict:
    adapted: dict = {}
    for key, value in (params or {}).items():
        if isinstance(value, datetime):
            adapted[key] = value.isoformat(sep=" ")
        elif isinstance(value, (date, time_cls)):
            adapted[key] = value.isoformat()
        else:
            adapted[key] = value
    return adapted


class _SQLiteCursor:
    """Minimal psycopg cursor shim backed by sqlite3."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._cursor = connection.cursor()

    def execute(self, sql: str, params: dict | None = None) -> None:
        self._cursor.execute(_sqlite_sql(sql), _sqlite_params(params))

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def close(self) -> None:
        self._cursor.close()

    def __enter__(self) -> "_SQLiteCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


class _SQLiteConnection:
    """Minimal psycopg connection shim; exits without closing the shared DB."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def cursor(self) -> _SQLiteCursor:
        return _SQLiteCursor(self._connection)

    def __enter__(self) -> "_SQLiteConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _ScheduleHarness:
    """In-memory doctor-schedule database for one test case."""

    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.executescript(
            """
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT
            );
            CREATE TABLE doctors (
                id INTEGER PRIMARY KEY,
                full_name TEXT,
                title TEXT,
                room_number TEXT,
                department_id INTEGER,
                active INTEGER DEFAULT 1
            );
            CREATE TABLE doctor_shifts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id INTEGER,
                department_id INTEGER,
                shift_date TEXT,
                session TEXT,
                duty_type TEXT,
                max_patients_per_slot INTEGER,
                room_number TEXT
            );
            CREATE TABLE slot_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_number TEXT,
                doctor_id INTEGER,
                department_id INTEGER,
                appointment_date TEXT,
                slot_start_time TEXT,
                status TEXT DEFAULT 'BOOKED',
                expires_at TEXT
            );
            """
        )

    def connect(self, _settings=None) -> _SQLiteConnection:
        return _SQLiteConnection(self.connection)

    def close(self) -> None:
        self.connection.close()

    def add_department(self, department_id: int, name: str) -> None:
        self.connection.execute(
            "INSERT INTO departments (id, name) VALUES (?, ?)",
            (department_id, name),
        )

    def add_doctor(
        self,
        doctor_id: int,
        full_name: str,
        department_id: int,
        active: bool = True,
    ) -> None:
        self.connection.execute(
            "INSERT INTO doctors (id, full_name, title, room_number, department_id, active) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doctor_id, full_name, "BS", f"P{doctor_id}", department_id, int(active)),
        )

    def add_shift(
        self,
        doctor_id: int,
        department_id: int,
        session: str,
        duty_type: str = "OUTPATIENT",
        max_patients_per_slot: int = 4,
        shift_date: date = TARGET_DATE,
    ) -> int:
        cursor = self.connection.execute(
            "INSERT INTO doctor_shifts "
            "(doctor_id, department_id, shift_date, session, duty_type, "
            "max_patients_per_slot, room_number) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                doctor_id,
                department_id,
                shift_date.isoformat(),
                session,
                duty_type,
                max_patients_per_slot,
                f"P{doctor_id}",
            ),
        )
        return int(cursor.lastrowid)

    def delete_shift(self, shift_id: int) -> None:
        self.connection.execute(
            "DELETE FROM doctor_shifts WHERE id = ?", (shift_id,)
        )

    def add_assignment(
        self,
        doctor_id: int,
        department_id: int,
        slot_start_time: str,
        status: str = "BOOKED",
        appointment_date: date = TARGET_DATE,
        expires_at=None,
    ) -> None:
        if isinstance(expires_at, datetime):
            expires_at = expires_at.isoformat(sep=" ")
        self.connection.execute(
            "INSERT INTO slot_assignments "
            "(ticket_number, doctor_id, department_id, appointment_date, "
            "slot_start_time, status, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"T{doctor_id}-{appointment_date.isoformat()}-{slot_start_time}",
                doctor_id,
                department_id,
                appointment_date.isoformat(),
                slot_start_time,
                status,
                expires_at,
            ),
        )


class QueryDepartmentScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _ScheduleHarness()
        self.db.add_department(1, "Khoa Nội Tổng quát & Tim mạch")
        self.db.add_department(2, "Khoa Hô hấp & Dị ứng - Miễn dịch lâm sàng")
        self.db.add_doctor(1, "BS. Nguyễn Văn A", department_id=1)
        self.db.add_doctor(2, "BS. Trần Thị B", department_id=2)

        patcher = patch.object(sq, "_connect", new=self.db.connect)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.db.close)

    def _query(self, department_id: int = 1) -> dict:
        return sq.query_department_schedule(department_id, TARGET_DATE_STR)

    def test_shift_deleted_or_moved_to_inpatient_leaves_no_outpatient_duty(self) -> None:
        shift_id = self.db.add_shift(
            doctor_id=1, department_id=1, session="MORNING"
        )
        before = self._query()
        self.assertEqual(before["recommended_shift"], "MORNING")
        self.assertTrue(before["has_outpatient_duty"])

        self.db.delete_shift(shift_id)
        after_delete = self._query()

        self.assertTrue(after_delete["is_available"])
        self.assertFalse(after_delete["has_outpatient_duty"])
        self.assertEqual(after_delete["recommended_shift"], "NO_DUTY")
        self.assertEqual(after_delete["morning"]["doctors"], [])
        self.assertEqual(after_delete["morning"]["total_available_slots"], 0)
        self.assertEqual(after_delete["afternoon"]["doctors"], [])
        self.assertEqual(after_delete["afternoon"]["total_available_slots"], 0)

        # A shift changed from OUTPATIENT to INPATIENT must behave the same.
        self.db.add_shift(
            doctor_id=1,
            department_id=1,
            session="MORNING",
            duty_type="INPATIENT",
        )
        after_change = self._query()
        self.assertFalse(after_change["has_outpatient_duty"])
        self.assertEqual(after_change["recommended_shift"], "NO_DUTY")

    def test_readded_shift_restores_capacity(self) -> None:
        shift_id = self.db.add_shift(
            doctor_id=1, department_id=1, session="MORNING"
        )
        self.db.add_assignment(doctor_id=1, department_id=1, slot_start_time="08:00:00")

        first = self._query()
        self.assertEqual(first["recommended_shift"], "MORNING")
        first_doctor = first["morning"]["doctors"][0]
        self.assertEqual(first_doctor["total_capacity"], 16)
        self.assertEqual(first_doctor["booked_count"], 1)
        self.assertEqual(first_doctor["available_capacity"], 15)

        self.db.delete_shift(shift_id)
        self.assertEqual(self._query()["recommended_shift"], "NO_DUTY")

        self.db.add_shift(doctor_id=1, department_id=1, session="MORNING")
        restored = self._query()

        self.assertEqual(restored["recommended_shift"], "MORNING")
        self.assertTrue(restored["has_outpatient_duty"])
        restored_doctor = restored["morning"]["doctors"][0]
        self.assertEqual(restored_doctor["total_capacity"], 16)
        self.assertEqual(restored_doctor["booked_count"], 1)
        self.assertEqual(restored_doctor["available_capacity"], 15)
        self.assertEqual(restored["morning"]["total_available_slots"], 15)

    def test_bookings_from_another_department_do_not_inflate_booked_count(self) -> None:
        self.db.add_shift(doctor_id=1, department_id=1, session="MORNING")
        self.db.add_assignment(doctor_id=1, department_id=1, slot_start_time="08:00:00")
        for slot in ("08:30:00", "09:00:00", "09:30:00"):
            self.db.add_assignment(
                doctor_id=1, department_id=2, slot_start_time=slot
            )

        result = self._query(department_id=1)

        doctor = result["morning"]["doctors"][0]
        self.assertEqual(
            doctor["booked_count"],
            1,
            "bookings belonging to department 2 must not count against dept 1's shift",
        )
        self.assertEqual(doctor["available_capacity"], 15)

    def _load_today_at_3pm(self) -> dict:
        """Run _load_schedule as if it is 15:00 Asia/Ho_Chi_Minh today."""
        now = datetime(2099, 1, 5, 15, 0, tzinfo=sq.CLINIC_TIMEZONE)
        with patch.object(sq, "datetime", _frozen_datetime(now)):
            return sq._load_schedule(1, now.date(), object())

    def test_afternoon_recommended_at_3pm_instead_of_morning(self) -> None:
        self.db.add_shift(
            doctor_id=1, department_id=1, session="MORNING", shift_date=TODAY
        )
        self.db.add_shift(
            doctor_id=1, department_id=1, session="AFTERNOON", shift_date=TODAY
        )

        result = self._load_today_at_3pm()

        self.assertTrue(result["morning_elapsed"])
        self.assertEqual(result["clinic_time"], "15:00")
        self.assertEqual(result["recommended_shift"], "AFTERNOON")
        self.assertNotEqual(result["recommended_shift"], "MORNING")

    def test_next_day_when_afternoon_is_unscheduled_after_morning_elapsed(self) -> None:
        self.db.add_shift(
            doctor_id=1, department_id=1, session="MORNING", shift_date=TODAY
        )

        result = self._load_today_at_3pm()

        self.assertTrue(result["morning_elapsed"])
        self.assertEqual(result["clinic_time"], "15:00")
        self.assertEqual(result["recommended_shift"], "NEXT_DAY")

    def test_next_day_when_afternoon_is_full_after_morning_elapsed(self) -> None:
        self.db.add_shift(
            doctor_id=1, department_id=1, session="MORNING", shift_date=TODAY
        )
        self.db.add_shift(
            doctor_id=1, department_id=1, session="AFTERNOON", shift_date=TODAY
        )
        for _ in range(16):
            self.db.add_assignment(
                doctor_id=1,
                department_id=1,
                slot_start_time="13:00:00",
                appointment_date=TODAY,
            )

        result = self._load_today_at_3pm()

        self.assertTrue(result["morning_elapsed"])
        self.assertTrue(result["afternoon"]["is_full"])
        self.assertEqual(result["recommended_shift"], "NEXT_DAY")



class TemporaryHoldCapacityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _ScheduleHarness()
        self.db.add_department(1, "Khoa Nội Tổng quát & Tim mạch")
        self.db.add_doctor(1, "BS. Nguyễn Văn A", department_id=1)
        patcher = patch.object(sq, "_connect", new=self.db.connect)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.db.close)

    def test_temporary_and_booked_count_toward_capacity(self) -> None:
        self.db.add_shift(doctor_id=1, department_id=1, session="MORNING")
        self.db.add_assignment(
            doctor_id=1, department_id=1, slot_start_time="08:00:00", status="BOOKED"
        )
        self.db.add_assignment(
            doctor_id=1,
            department_id=1,
            slot_start_time="08:30:00",
            status="TEMPORARY",
        )
        # Ignored rows.
        self.db.add_assignment(
            doctor_id=1, department_id=1, slot_start_time="09:30:00", status="CANCELLED"
        )

        result = sq.query_department_schedule(1, TARGET_DATE_STR)
        doctor = result["morning"]["doctors"][0]
        self.assertEqual(doctor["booked_count"], 2)
        self.assertFalse(result["morning"]["is_full"])

    def test_session_becomes_full_with_holds(self) -> None:
        self.db.add_shift(doctor_id=1, department_id=1, session="MORNING")
        for i in range(16):
            self.db.add_assignment(
                doctor_id=1,
                department_id=1,
                slot_start_time="08:00:00",
                status="TEMPORARY",
            )
        result = sq.query_department_schedule(1, TARGET_DATE_STR)
        self.assertTrue(result["morning"]["is_full"])



if __name__ == "__main__":
    unittest.main()
