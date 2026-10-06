"""Stage 2 of voice triage: deterministic doctor-schedule lookups.

This module is the *only* place that reads the doctor-schedule service database
(``clinic_management_doctorschedule_service``). Rosters, duty shifts and booked
capacity always come from here - never from an LLM.

Business rules implemented (per the clinic's outpatient capacity policy):

* only shifts with ``duty_type = 'OUTPATIENT'`` and ``doctors.active`` are used;
* a session is MORNING 07:30-11:30 or AFTERNOON 13:00-17:00 and always holds
  four 60-minute slots;
* one doctor can take ``max_patients_per_slot * 4`` patients per session;
* a session is full when the patients booked across *all* duty doctors in that
  session reach the combined capacity of the session.

Every value that originates outside this module (department id, date) is bound
as a query parameter - no SQL string is ever built by concatenation.
"""

from __future__ import annotations

from datetime import date as date_cls
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import psycopg
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

CLINIC_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")

#: Session label -> human readable window. Fixed by clinic policy.
SESSION_WINDOWS: Dict[str, str] = {
    "MORNING": "07:30 - 11:30",
    "AFTERNOON": "13:00 - 17:00",
}

#: A session always spans four 60-minute slots.
SLOTS_PER_SESSION = 4

#: Used when a shift row does not state a per-slot patient limit.
DEFAULT_MAX_PATIENTS_PER_SLOT = 4

#: Canonical department catalogue (departments table is the source of truth; this
#: is the fallback used only when that lookup row is missing).
FALLBACK_DEPARTMENTS: Dict[int, str] = {
    1: "Khoa Nội Tổng quát & Tim mạch",
    2: "Khoa Hô hấp & Dị ứng - Miễn dịch lâm sàng",
    3: "Khoa Da liễu",
}

# Sessions are separated by the gap between 11:30 and 13:00, so 12:00 is a
# boundary that no real slot_start_time can land on. The CASE below is the only
# derived expression in the statement; it contains no user input.
_SCHEDULE_SQL = """
WITH duty AS (
    SELECT
        s.doctor_id,
        s.session,
        COALESCE(s.max_patients_per_slot, %(default_max_per_slot)s) AS max_per_slot,
        d.full_name,
        COALESCE(d.title, '') AS title,
        COALESCE(d.room_number, '') AS room_number
    FROM doctor_shifts s
    JOIN doctors d ON d.id = s.doctor_id
    WHERE s.department_id = %(department_id)s
      AND s.shift_date = %(target_date)s
      AND s.duty_type = 'OUTPATIENT'
      AND COALESCE(d.active, TRUE) = TRUE
),
booked AS (
    SELECT
        sa.doctor_id,
        CASE WHEN sa.slot_start_time < TIME '12:00' THEN 'MORNING' ELSE 'AFTERNOON' END
            AS session,
        COUNT(*) AS booked_count
    FROM slot_assignments sa
    WHERE sa.appointment_date = %(target_date)s
      AND COALESCE(sa.status, 'BOOKED') = 'BOOKED'
    GROUP BY sa.doctor_id, 2
)
SELECT
    duty.session,
    duty.doctor_id,
    duty.full_name,
    duty.title,
    duty.room_number,
    duty.max_per_slot * %(slots_per_session)s AS total_capacity,
    COALESCE(booked.booked_count, 0) AS booked_count
FROM duty
LEFT JOIN booked
       ON booked.doctor_id = duty.doctor_id
      AND booked.session = duty.session
ORDER BY
    CASE duty.session WHEN 'MORNING' THEN 0 ELSE 1 END,
    duty.full_name;
"""

_DEPARTMENT_SQL = "SELECT id, name FROM departments WHERE id = %(department_id)s;"


class ScheduleDatabaseUnavailable(RuntimeError):
    """Kept for callers that want strict behaviour; never raised by the query.

    ``query_department_schedule`` degrades gracefully instead - see
    :func:`unavailable_schedule`.
    """


#: Shape returned whenever the schedule service cannot be reached or read.
_UNAVAILABLE_MESSAGE = "Chưa thể kết nối cơ sở dữ liệu lịch trực phòng khám."


def unavailable_schedule(
    department_id: int,
    target_date: date_cls,
    reason: str = "",
    department_name: str = "",
) -> Dict[str, Any]:
    """Fallback payload so Stage 1 can still reach the client with HTTP 200."""
    return {
        "is_available": False,
        "error_message": _UNAVAILABLE_MESSAGE,
        "recommended_shift": "MANUAL_PICK",
        "morning": None,
        "afternoon": None,
        "department_id": department_id,
        "department_name": department_name
        or FALLBACK_DEPARTMENTS.get(department_id, ""),
        "target_date": target_date.isoformat(),
        "timezone": str(CLINIC_TIMEZONE),
        "has_outpatient_duty": False,
        "detail": reason,
    }


class DoctorShiftInfo(BaseModel):
    """One doctor's duty shift inside a session, with live capacity."""

    doctor_id: int
    doctor_name: str
    title: str
    room_number: str
    booked_count: int
    total_capacity: int
    available_capacity: int
    is_full: bool


class SessionSchedule(BaseModel):
    """Aggregated capacity for one session across every duty doctor."""

    session: str = Field(description="MORNING or AFTERNOON")
    time_window: str = Field(description="Human readable duty window, e.g. 07:30 - 11:30")
    is_full: bool
    total_available_slots: int = Field(
        description="Remaining patient capacity across all duty doctors in the session"
    )
    doctors: List[DoctorShiftInfo] = Field(default_factory=list)


def today_in_clinic_timezone() -> date_cls:
    """Current date in the clinic's timezone (never the server's local date)."""
    return datetime.now(CLINIC_TIMEZONE).date()


def normalise_target_date(target_date: Optional[str]) -> date_cls:
    """Accept ``YYYY-MM-DD`` or None; default to today in Asia/Ho_Chi_Minh."""
    if not target_date:
        return today_in_clinic_timezone()
    try:
        return datetime.strptime(target_date.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"target_date phải có định dạng YYYY-MM-DD, nhận được: {target_date!r}"
        ) from exc


def _connect(settings: Settings) -> psycopg.Connection:
    """Open the schedule DB with a bounded connect timeout.

    Raises whatever psycopg raises; :func:`query_department_schedule` decides
    how to degrade.
    """
    return psycopg.connect(
        settings.schedule_db_dsn,
        connect_timeout=settings.SCHEDULE_DB_CONNECT_TIMEOUT,
        autocommit=True,  # read-only workload; no transaction state to leak
    )


def _fetch_department_name(cur: psycopg.Cursor, department_id: int, fallback: str) -> str:
    try:
        cur.execute(_DEPARTMENT_SQL, {"department_id": department_id})
        row = cur.fetchone()
    except psycopg.Error:
        # A missing `departments` table must not break the schedule query.
        logger.warning("Could not read the departments table", exc_info=True)
        return fallback
    return str(row[1]) if row and row[1] else fallback


def query_department_schedule(
    department_id: int,
    target_date: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Return real-time outpatient duty shifts and capacity for one department.

        ``department_id`` must be one of the three clinical departments (1, 2, 3).
        ``target_date`` defaults to today in Asia/Ho_Chi_Minh.

        This function never raises for infrastructure problems. If the schedule
        database is unreachable, missing, or its tables are absent, it returns
        :func:`unavailable_schedule` with ``is_available=False`` so the caller can
        still answer HTTP 200 with the Stage 1 clinical result.
    """
    settings = settings or get_settings()
    try:
        department_id = int(department_id)
    except (TypeError, ValueError):
        # A malformed id cannot be routed anywhere; degrade rather than 500.
        return unavailable_schedule(
            1, today_in_clinic_timezone(), reason=f"department_id không hợp lệ: {department_id!r}"
        )
    if department_id not in FALLBACK_DEPARTMENTS:
        return unavailable_schedule(
            1,
            today_in_clinic_timezone(),
            reason=f"department_id phải là 1, 2 hoặc 3; nhận được: {department_id}",
        )

    try:
        day = normalise_target_date(target_date)
    except ValueError as exc:
        return unavailable_schedule(department_id, today_in_clinic_timezone(), reason=str(exc))

    try:
        return _load_schedule(department_id, day, settings)
    except Exception as exc:  # noqa: BLE001 - graceful degradation is the contract
        reason = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
        logger.warning(
            "Schedule lookup unavailable (dept=%s, dsn=%s@%s:%s/%s): %s",
            department_id,
            settings.SCHEDULE_DB_USER,
            settings.SCHEDULE_DB_HOST,
            settings.SCHEDULE_DB_PORT,
            settings.SCHEDULE_DB_NAME,
            reason,
        )
        return unavailable_schedule(department_id, day, reason=reason)


def _load_schedule(
    department_id: int, day: date_cls, settings: Settings
) -> Dict[str, Any]:
    """Query + aggregate the schedule. Raises on infrastructure failure."""
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            department_name = _fetch_department_name(
                cur, department_id, FALLBACK_DEPARTMENTS[department_id]
            )
            cur.execute(
                _SCHEDULE_SQL,
                {
                    "department_id": department_id,
                    "target_date": day,
                    "slots_per_session": SLOTS_PER_SESSION,
                    "default_max_per_slot": DEFAULT_MAX_PATIENTS_PER_SLOT,
                },
            )
            rows = cur.fetchall()

    # Aggregate per session: session capacity is the sum over its duty doctors.
    sessions: Dict[str, SessionSchedule] = {
        label: SessionSchedule(
            session=label,
            time_window=window,
            is_full=False,
            total_available_slots=0,
            doctors=[],
        )
        for label, window in SESSION_WINDOWS.items()
    }

    for session_label, doctor_id, full_name, title, room, capacity, booked in rows:
        if session_label not in sessions:
            # Defensive: a rogue session value must not corrupt the response.
            logger.warning("Ignoring unknown session value %r", session_label)
            continue
        capacity = int(capacity or 0)
        booked = int(booked or 0)
        available = max(capacity - booked, 0)
        sessions[session_label].doctors.append(
            DoctorShiftInfo(
                doctor_id=int(doctor_id),
                doctor_name=str(full_name or ""),
                title=str(title or ""),
                room_number=str(room or ""),
                booked_count=booked,
                total_capacity=capacity,
                available_capacity=available,
                is_full=available == 0,
            )
        )
        sessions[session_label].total_available_slots += available

    for schedule in sessions.values():
        capacity_total = sum(d.total_capacity for d in schedule.doctors)
        booked_total = sum(d.booked_count for d in schedule.doctors)
        # Per business rule: full when bookings reach the session's total capacity.
        schedule.is_full = bool(schedule.doctors) and booked_total >= capacity_total

    morning, afternoon = sessions["MORNING"], sessions["AFTERNOON"]

    if morning.doctors and morning.is_full:
        recommended = "AFTERNOON" if afternoon.doctors and not afternoon.is_full else "NEXT_DAY"
    elif morning.doctors:
        # Both open -> prefer the morning; morning only -> morning.
        recommended = "MORNING"
    elif afternoon.doctors and not afternoon.is_full:
        recommended = "AFTERNOON"
    elif afternoon.doctors:
        recommended = "NEXT_DAY"
    else:
        # No outpatient duty at all for this department on this date.
        recommended = "NO_DUTY"

    result: Dict[str, Any] = {
        "is_available": True,
        "error_message": "",
        "department_id": department_id,
        "department_name": department_name,
        "target_date": day.isoformat(),
        "timezone": str(CLINIC_TIMEZONE),
        "recommended_shift": recommended,
        "is_fully_booked": morning.is_full and afternoon.is_full,
        "has_outpatient_duty": bool(morning.doctors or afternoon.doctors),
        "morning": morning.model_dump(),
        "afternoon": afternoon.model_dump(),
        "detail": "",
    }
    logger.info(
        "Schedule dept=%s date=%s -> %s (M %d/%d full, A %d/%d full)",
        department_id,
        day.isoformat(),
        recommended,
        sum(d.booked_count for d in morning.doctors),
        sum(d.total_capacity for d in morning.doctors),
        sum(d.booked_count for d in afternoon.doctors),
        sum(d.total_capacity for d in afternoon.doctors),
    )
    return result