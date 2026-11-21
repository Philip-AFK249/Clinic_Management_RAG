"""Contracts for the 3-minute triage hold, structured analyze payload and
admin ticket-cancellation handling.

Dependency-free (unittest + mocked seams): no PostgreSQL, Groq or audio is
needed. Run with::

    PYTHONPATH=. python tests/test_triage_session_cache.py
"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.api as api
from app.services import triage_session_cache as cache_module
from app.services.triage_session_cache import (
    TRIAGE_HOLD_SECONDS,
    TriageSessionCache,
    TriageSessionContext,
)


def _context(**overrides) -> TriageSessionContext:
    values = {
        "flag_id": "FLG-TEST-001",
        "transcription": "tim đập thình thịch, đau họng",
        "normalized_transcription": "tim đập thình thịch, đau họng",
        "normalized_symptoms": ["tim đập thình thịch", "đau họng"],
        "chief_complaint_summary": "Hồi hộp và đau họng",
        "department_id": 1,
        "department_name": "Khoa Nội Tổng quát & Tim mạch",
        "priority_level": "P2",
        "disease_guess": "Hồi hộp",
        "icd10_code": "R00.2",
        "patient_id": 42,
        "ticket_number": "#A-001",
    }
    values.update(overrides)
    return TriageSessionContext(**values)


def _triage(**overrides) -> api.VoiceScheduleTriageResponse:
    values = {
        "transcription": "tim đập thành thịt và đau họng",
        "normalized_transcription": "tim đập thình thịch và đau họng",
        "disease_guess": "Hồi hộp",
        "icd10_code": "R00.2",
        "department_id": 1,
        "department_name": "Khoa Nội Tổng quát & Tim mạch",
        "priority_level": "P2",
        "chief_complaint_summary": "Hồi hộp, đau họng",
        "recommended_shift": "MORNING",
        "advice": "Đi khám sớm",
    }
    values.update(overrides)
    return api.VoiceScheduleTriageResponse(**values)


class TriageSessionCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = TriageSessionCache()

    def test_default_hold_is_three_minutes(self):
        self.assertEqual(TRIAGE_HOLD_SECONDS, 180)
        context = _context()
        self.cache.put(context)
        self.assertLessEqual(context.remaining_seconds, 180)
        self.assertGreater(context.remaining_seconds, 178)

    def test_restore_by_flag_patient_and_ticket(self):
        context = _context()
        self.cache.put(context)
        self.assertIs(self.cache.get(flag_id="FLG-TEST-001"), context)
        self.assertIs(self.cache.get(patient_id=42), context)
        self.assertIs(self.cache.get(ticket_number="#A-001"), context)

    def test_entry_expires_at_end_of_hold_window(self):
        context = _context()
        with patch.object(cache_module.time, "monotonic", return_value=1000.0):
            self.cache.put(context)
            self.assertIs(self.cache.get(flag_id="FLG-TEST-001"), context)
        with patch.object(cache_module.time, "monotonic", return_value=1180.0):
            self.assertIsNone(self.cache.get(flag_id="FLG-TEST-001"))
            self.assertIsNone(self.cache.get(patient_id=42))
            self.assertIsNone(self.cache.get(ticket_number="#A-001"))

    def test_release_clears_every_identifier(self):
        self.cache.put(_context())
        removed = self.cache.release(flag_id="FLG-TEST-001")
        self.assertIsNotNone(removed)
        self.assertIsNone(self.cache.get(flag_id="FLG-TEST-001"))
        self.assertIsNone(self.cache.get(patient_id=42))
        self.assertIsNone(self.cache.get(ticket_number="#A-001"))

    def test_cancel_hold_keeps_context_but_detaches_ticket(self):
        self.cache.put(_context())
        cancelled = self.cache.cancel_hold(
            flag_id="FLG-TEST-001", patient_id=42, ticket_number="#A-001"
        )
        self.assertIsNotNone(cancelled)
        self.assertTrue(cancelled.cancelled)
        self.assertEqual(cancelled.ticket_number, "")
        self.assertIs(self.cache.get(flag_id="FLG-TEST-001"), cancelled)
        self.assertIs(self.cache.get(patient_id=42), cancelled)
        self.assertIsNone(self.cache.get(ticket_number="#A-001"))


class AnalyzeSymptomPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        api.triage_session_cache.clear()

    def test_structured_payload_has_flag_id_camel_case_and_normalized_symptoms(self):
        triage = _triage()
        with patch.object(api, "new_flag_id", return_value="FLG-NEW"):
            flag, context = api._remember_triage(triage, settings=None)
        response = api._analyze_response_from_triage(triage, context)

        self.assertEqual(flag, "FLG-NEW")
        self.assertEqual(response.flag_id, "FLG-NEW")
        self.assertEqual(response.departmentId, 1)
        self.assertEqual(response.priorityLevel, "P2")
        # snake_case twins stay intact for the existing kiosk client.
        self.assertEqual(response.department_id, 1)
        self.assertEqual(response.priority_level, "P2")
        self.assertIn("tim đập thình thịch", response.normalizedSymptoms)
        self.assertEqual(response.normalizedSymptoms, response.normalized_symptoms)
        self.assertEqual(response.expires_in_seconds, TRIAGE_HOLD_SECONDS)

    def test_restore_returns_cached_complaint_without_a_pipeline_call(self):
        context = _context()
        api.triage_session_cache.put(context)
        with patch.object(
            api, "_retrieve_triage_context", side_effect=AssertionError("RAG must not run")
        ), patch.object(
            api, "_extract_triage_facts", side_effect=AssertionError("LLM must not run")
        ):
            response = api._restore_triage_context(flag_id="FLG-TEST-001", settings=None)

        self.assertIsNotNone(response)
        self.assertTrue(response.restored_from_cache)
        self.assertEqual(response.flag_id, "FLG-TEST-001")
        self.assertEqual(response.chief_complaint_summary, "Hồi hộp và đau họng")
        self.assertEqual(response.departmentId, 1)
        self.assertEqual(response.priorityLevel, "P2")

    def test_restore_falls_back_to_persisted_flag_row(self):
        row = {
            "flag_id": "FLG-DB",
            "transcription": "sốt và ho",
            "normalized_transcription": "sốt và ho",
            "patient_id": 77,
            "ai_department_id": 2,
            "ai_department_name": "Khoa Hô hấp",
            "ai_priority_level": "P3",
            "ai_disease_guess": "Viêm họng",
            "ai_icd10_code": "J02.9",
            "status": "CANCELLED",
        }
        with patch.object(api, "list_unmatched_logs", return_value=[row]):
            response = api._restore_triage_context(patient_id=77, settings=None)
        self.assertIsNotNone(response)
        self.assertEqual(response.flag_id, "FLG-DB")
        self.assertIn("sốt", response.normalizedSymptoms)
        self.assertEqual(response.departmentId, 2)
        self.assertEqual(response.priorityLevel, "P3")

    def test_analyze_symptom_route_is_exposed_for_post_and_get(self):
        paths = TestClient(api.app).get("/openapi.json").json()["paths"]
        route = paths["/api/v1/triage/analyze-symptom"]
        self.assertIn("post", route)
        self.assertIn("get", route)

    def test_post_json_with_only_flag_id_restores_without_audio(self):
        context = _context()
        api.triage_session_cache.put(context)
        with patch.object(
            api, "_retrieve_triage_context", side_effect=AssertionError("RAG must not run")
        ):
            response = TestClient(api.app).post(
                "/api/v1/triage/analyze-symptom", json={"flag_id": "FLG-TEST-001"}
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["flag_id"], "FLG-TEST-001")
        self.assertTrue(payload["restored_from_cache"])


class TicketCancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        api.triage_session_cache.clear()

    def test_hold_expired_event_releases_ticket_but_keeps_extracted_context(self):
        context = _context()
        api.triage_session_cache.put(context)
        captured = {}

        def fake_update(flag_id, changes, settings):
            captured.update(changes)
            return {"flag_id": flag_id, **changes}

        payload = api.FlaggedSymptomUpdate(
            event="HOLD_EXPIRED", patient_id=42, ticket_number="#A-001"
        )
        with patch.object(api, "get_settings", return_value=None), patch.object(
            api, "update_unmatched_log", side_effect=fake_update
        ):
            response = asyncio.run(api.update_flagged_symptom("FLG-TEST-001", payload))

        self.assertTrue(response.success)
        self.assertEqual(captured["status"], "CANCELLED")
        self.assertEqual(captured["ticket_number"], "")
        self.assertNotIn("normalized_transcription", captured)
        self.assertIn("triệu chứng", captured["reception_action"])
        remaining = api.triage_session_cache.get(flag_id="FLG-TEST-001")
        self.assertIsNotNone(remaining)
        self.assertTrue(remaining.cancelled)
        self.assertEqual(remaining.ticket_number, "")
        # The patient can still restore the chief complaint after rejection.
        restored = api._restore_triage_context(flag_id="FLG-TEST-001", settings=None)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.chief_complaint_summary, "Hồi hộp và đau họng")

    def test_duplicate_cancellation_is_graceful(self):
        payload = api.FlaggedSymptomUpdate(event="BOOKING_REJECTED")
        with patch.object(api, "get_settings", return_value=None), patch.object(
            api, "update_unmatched_log", return_value=None
        ):
            response = asyncio.run(api.update_flagged_symptom("FLG-GONE", payload))
        self.assertTrue(response.success)
        self.assertIsNone(response.item)
        self.assertIn("giải phóng", response.message)

    def test_plain_update_still_works(self):
        captured = {}

        def fake_update(flag_id, changes, settings):
            captured.update(changes)
            return {"flag_id": flag_id, "status": "PENDING_DOCTOR", **changes}

        payload = api.FlaggedSymptomUpdate(ticket_number="#A-002", patient_id=99)
        with patch.object(api, "get_settings", return_value=None), patch.object(
            api, "update_unmatched_log", side_effect=fake_update
        ):
            response = asyncio.run(api.update_flagged_symptom("FLG-TEST-001", payload))
        self.assertTrue(response.success)
        self.assertEqual(captured["ticket_number"], "#A-002")
        self.assertEqual(captured["patient_id"], 99)
        self.assertEqual(response.item.status, "PENDING_DOCTOR")


if __name__ == "__main__":
    unittest.main()
