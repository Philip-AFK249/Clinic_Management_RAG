"""Short-lived, in-process cache for triage context captured after Whisper STT.

The RAG gateway is a single FastAPI process (uvicorn on port 8008), so no Redis
or extra database round-trip is needed to remember what was extracted from a
patient's recording.  A booking hold lasts three minutes; within that window
the reception flow may be rejected/reset by an admin, and the patient retries
with the same ``flag_id`` (or ``patient_id`` / ``ticket_number``).  This cache
lets the gateway return the already extracted chief complaint instantly instead
of asking the patient to repeat symptoms into the microphone.

Entries expire on a monotonic clock, are always accessed under a lock, and can
be released explicitly when an admin cancels a booking or a hold times out.
The cache is deliberately *only* a speed optimisation: the durable record in
``unmatched_symptom_logs`` (and the LLM/Whisper pipeline) remains the source of
truth when an entry has already expired.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

#: Default lifetime of a temporary booking/triage hold.
TRIAGE_HOLD_SECONDS = 180


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TriageSessionContext:
    """Everything needed to rebuild a triage response without re-running STT."""

    flag_id: str
    transcription: str = ""
    normalized_transcription: str = ""
    normalized_symptoms: list[str] = field(default_factory=list)
    chief_complaint_summary: str = ""
    department_id: Optional[int] = None
    department_name: str = ""
    priority_level: str = ""
    disease_guess: str = ""
    icd10_code: str = ""
    patient_id: Optional[int] = None
    ticket_number: str = ""
    cancelled: bool = False
    age_group: str = ""
    recommended_shift: str = ""
    created_at: str = field(default_factory=_utc_now)
    #: Monotonic deadline. Kept separate from ``created_at`` so wall-clock
    #: adjustments cannot extend or shorten a hold.
    expires_at: float = field(
        default_factory=lambda: time.monotonic() + TRIAGE_HOLD_SECONDS
    )

    @property
    def remaining_seconds(self) -> int:
        """Whole seconds left before this hold expires (never negative)."""
        return max(0, int(self.expires_at - time.monotonic()))

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view used by the restore endpoint and diagnostics."""
        return {
            "flag_id": self.flag_id,
            "transcription": self.transcription,
            "normalized_transcription": self.normalized_transcription,
            "normalized_symptoms": list(self.normalized_symptoms),
            "chief_complaint_summary": self.chief_complaint_summary,
            "department_id": self.department_id,
            "department_name": self.department_name,
            "priority_level": self.priority_level,
            "disease_guess": self.disease_guess,
            "icd10_code": self.icd10_code,
            "patient_id": self.patient_id,
            "ticket_number": self.ticket_number,
            "cancelled": self.cancelled,
            "age_group": self.age_group,
            "recommended_shift": self.recommended_shift,
            "created_at": self.created_at,
            "expires_in_seconds": self.remaining_seconds,
        }


class TriageSessionCache:
    """Thread-safe TTL store indexed by flag id, patient id and ticket number."""

    def __init__(self, ttl_seconds: int = TRIAGE_HOLD_SECONDS) -> None:
        self._ttl = max(1, int(ttl_seconds))
        self._lock = threading.RLock()
        self._sessions: dict[str, TriageSessionContext] = {}
        self._by_patient: dict[int, str] = {}
        self._by_ticket: dict[str, str] = {}

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    def put(self, context: TriageSessionContext) -> TriageSessionContext:
        """Store (or replace) a triage context and refresh its three-minute TTL."""
        flag_id = (context.flag_id or "").strip()
        if not flag_id:
            raise ValueError("A flag_id is required to cache triage context.")
        context.flag_id = flag_id
        context.expires_at = time.monotonic() + self._ttl
        with self._lock:
            self._prune_locked()
            self._forget_locked(flag_id)
            self._sessions[flag_id] = context
            if context.patient_id is not None:
                self._by_patient[int(context.patient_id)] = flag_id
            if context.ticket_number:
                self._by_ticket[context.ticket_number.strip()] = flag_id
        return context

    def get(
        self,
        *,
        flag_id: Optional[str] = None,
        patient_id: Optional[int] = None,
        ticket_number: Optional[str] = None,
    ) -> Optional[TriageSessionContext]:
        """Return a live context for any of the supplied identifiers."""
        with self._lock:
            self._prune_locked()
            resolved = self._resolve_locked(flag_id, patient_id, ticket_number)
            context = self._sessions.get(resolved) if resolved else None
            if context and context.expires_at <= time.monotonic():
                self._forget_locked(resolved)
                return None
            return context

    def release(
        self,
        *,
        flag_id: Optional[str] = None,
        patient_id: Optional[int] = None,
        ticket_number: Optional[str] = None,
    ) -> Optional[TriageSessionContext]:
        """Drop a hold after admin rejection / expiry; return what was removed."""
        with self._lock:
            resolved = self._resolve_locked(flag_id, patient_id, ticket_number)
            removed = self._sessions.pop(resolved, None) if resolved else None
            if removed is not None:
                self._forget_indexes_locked(removed)
            # A caller may release by ticket/patient whose index points at an
            # already-expired context; clear those stale keys as well.
            if patient_id is not None:
                self._by_patient.pop(int(patient_id), None)
            if ticket_number:
                self._by_ticket.pop(ticket_number.strip(), None)
            return removed

    def cancel_hold(
        self,
        *,
        flag_id: Optional[str] = None,
        patient_id: Optional[int] = None,
        ticket_number: Optional[str] = None,
    ) -> Optional[TriageSessionContext]:
        """Release the tied ticket but keep the extracted context for a retry.

        An admin rejection (or an expired hold) invalidates the *booking*, not
        the clinical extraction. The context stays addressable by ``flag_id``
        and ``patient_id`` for another full hold window so the patient can retry
        without repeating the recording. The old ticket index is removed so a
        stale ticket can never be restored into a new booking.
        """
        with self._lock:
            self._prune_locked()
            resolved = self._resolve_locked(flag_id, patient_id, ticket_number)
            context = self._sessions.get(resolved) if resolved else None
            if context is None:
                if patient_id is not None:
                    self._by_patient.pop(int(patient_id), None)
                if ticket_number:
                    self._by_ticket.pop(ticket_number.strip(), None)
                return None
            if context.ticket_number:
                self._by_ticket.pop(context.ticket_number.strip(), None)
            context.ticket_number = ""
            context.cancelled = True
            context.expires_at = time.monotonic() + self._ttl
            return context

    def clear(self) -> None:
        """Primarily for tests and graceful shutdown."""
        with self._lock:
            self._sessions.clear()
            self._by_patient.clear()
            self._by_ticket.clear()

    # -- internals ---------------------------------------------------------

    def _resolve_locked(
        self,
        flag_id: Optional[str],
        patient_id: Optional[int],
        ticket_number: Optional[str],
    ) -> Optional[str]:
        candidate = (flag_id or "").strip()
        if candidate:
            return candidate
        if patient_id is not None:
            return self._by_patient.get(int(patient_id))
        if ticket_number:
            return self._by_ticket.get(ticket_number.strip())
        return None

    def _forget_locked(self, flag_id: str) -> None:
        context = self._sessions.pop(flag_id, None)
        if context is not None:
            self._forget_indexes_locked(context)

    def _forget_indexes_locked(self, context: TriageSessionContext) -> None:
        if context.patient_id is not None:
            self._by_patient.pop(int(context.patient_id), None)
        if context.ticket_number:
            self._by_ticket.pop(context.ticket_number.strip(), None)

    def _prune_locked(self) -> None:
        now = time.monotonic()
        expired = [
            flag_id
            for flag_id, context in self._sessions.items()
            if context.expires_at <= now
        ]
        for flag_id in expired:
            self._forget_locked(flag_id)


#: Process-wide holder shared by the triage and admin endpoints.
triage_session_cache = TriageSessionCache()
