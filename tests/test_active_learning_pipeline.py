"""End-to-end contract tests for the Voice Triage -> Active Learning chain.

These tests are dependency-free (unittest only) and never touch PostgreSQL, the
Groq API or the filesystem: the storage/ingestion seams are monkeypatched so the
contract between FastAPI, the flag report writer and pgvector can be asserted.

Run with::

    PYTHONPATH=. python tests/test_active_learning_pipeline.py
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import app.api as api
from app.ingestion.pipeline import extract_age_group


class VoiceNormalisationTests(unittest.TestCase):
    def test_known_whisper_homophones_are_corrected(self):
        raw = (
            "tiền đập thành thịt, chóng ván, điên tục, sâu bắn, củ tay, chữ chứng"
        )
        cleaned = api._normalise_voice_typos(raw)
        for expected in (
            "tim đập thình thịch",
            "chóng mặt",
            "liên tục",
            "sâu cắn",
            "cổ họng",
            "triệu chứng",
        ):
            self.assertIn(expected, cleaned)

    def test_repetition_filler_is_removed(self):
        cleaned = api._normalise_voice_typos("lực lực lực lực đau họng")
        self.assertNotIn("lực", cleaned)
        self.assertIn("đau họng", cleaned)

    def test_hallucination_loop_rejected_but_clinical_text_kept(self):
        self.assertTrue(api._looks_like_whisper_hallucination("lực lực lực lực"))
        self.assertTrue(api._looks_like_whisper_hallucination("ừ ừ ừ ừ ừ ừ"))
        self.assertFalse(
            api._looks_like_whisper_hallucination(
                "đau họng và sốt hai ngày nay, nuốt đau"
            )
        )

    def test_long_rambling_query_keeps_only_clinical_clauses(self):
        raw = (
            "Hôm qua tôi đi ăn cơm hộp với bạn rồi uống trà sữa, tối về tắm khuya "
            "và xem phim rất khuya. Nhưng mà từ sáng nay tim đập thình thịch, "
            "chóng mặt, đau họng và nuốt đau, người ớn lạnh. Tôi cũng ăn sáng "
            "bằng bánh mì và uống cà phê sữa đá trước khi đến phòng khám."
        )
        query = api._extract_clinical_search_query(raw)
        self.assertIn("tim đập thình thịch", query)
        self.assertIn("đau họng", query)
        self.assertNotIn("trà sữa", query.lower())


def _final_payload(**overrides):
    payload = {
        "disease_guess": "Viêm họng cấp",
        "icd10_code": "J02.9",
        "department_id": 2,
        "priority_level": "P2",
        "chief_complaint_summary": "Đau họng, sốt",
        "needs_clarification": False,
        "is_out_of_dataset": False,
        "flag_reason": "",
    }
    payload.update(overrides)
    return payload


class OutOfDatasetTests(unittest.TestCase):
    def test_low_similarity_marks_out_of_dataset(self):
        facts, _ = api._sanitise_triage_payload(
            _final_payload(), best_score=0.30, raw_text="đau bụng dưới nhiều ngày"
        )
        self.assertTrue(facts["is_out_of_dataset"])
        self.assertEqual(facts["flag_reason"], "LOW_SIMILARITY_MATCH")

    def test_low_similarity_overrides_free_form_reason(self):
        facts, _ = api._sanitise_triage_payload(
            _final_payload(is_out_of_dataset=True, flag_reason="OUT_OF_SPECIALTY"),
            best_score=0.40,
            raw_text="đau bụng dưới nhiều ngày",
        )
        self.assertTrue(facts["is_out_of_dataset"])
        self.assertEqual(facts["flag_reason"], "LOW_SIMILARITY_MATCH")

    def test_greeting_is_not_flagged(self):
        facts, _ = api._sanitise_triage_payload(
            _final_payload(), best_score=0.30, raw_text="xin chào"
        )
        self.assertFalse(facts["is_out_of_dataset"])
        self.assertEqual(facts["flag_reason"], "")

    def test_zero_score_is_not_flagged(self):
        facts, _ = api._sanitise_triage_payload(
            _final_payload(), best_score=0.0, raw_text="đau bụng dưới"
        )
        self.assertFalse(facts["is_out_of_dataset"])

    def test_high_score_is_not_flagged(self):
        facts, _ = api._sanitise_triage_payload(
            _final_payload(), best_score=0.81, raw_text="đau bụng dưới"
        )
        self.assertFalse(facts["is_out_of_dataset"])

    def test_clarification_round_keeps_low_similarity_flag(self):
        facts, _ = api._sanitise_triage_payload(
            _final_payload(needs_clarification=True),
            best_score=0.30,
            raw_text="đau bụng dưới",
        )
        # Rule 8 keeps disease/ICD empty while asking, but Active Learning must
        # still receive the case as soon as the similarity floor is breached.
        self.assertTrue(facts["needs_clarification"])
        self.assertEqual(facts["disease_guess"], "")
        self.assertEqual(facts["icd10_code"], "")
        self.assertTrue(facts["is_out_of_dataset"])
        self.assertEqual(facts["flag_reason"], "LOW_SIMILARITY_MATCH")

    def test_threshold_boundary_is_not_flagged(self):
        facts, _ = api._sanitise_triage_payload(
            _final_payload(),
            best_score=api.OOD_SCORE_THRESHOLD,
            raw_text="đau bụng dưới",
        )
        self.assertFalse(facts["is_out_of_dataset"])


def _flag_row(**overrides):
    row = {
        "flag_id": "FLG-20260101-0001",
        "transcription": "tiền đập thành thịt",
        "age_group": "nguoi_truong_thanh",
        "ai_disease_guess": "Hồi hộp (R00.2)",
        "ai_icd10_code": "R00.2",
        "ai_department_id": 1,
        "ai_department_name": "Khoa Nội Tổng quát & Tim mạch",
        "ai_priority_level": "P2",
        "flag_reason": "NEW_SYMPTOM_PATTERN",
        "status": "PENDING_DOCTOR",
        "ticket_number": "",
        "confirmed_icd10_code": "",
        "confirmed_disease_name": "",
        "medical_symptom_standard": "",
    }
    row.update(overrides)
    return row


class ActiveLearningStorageTests(unittest.TestCase):
    def test_stats_merge_duplicate_cluster_labels(self):
        rows = [
            _flag_row(ai_disease_guess="Hồi hộp (R00.2)"),
            _flag_row(flag_id="FLG-2", ai_disease_guess="hoi hop (R00.2)"),
            _flag_row(
                flag_id="FLG-3",
                ai_disease_guess="",
                status="DOCTOR_CONFIRMED",
                confirmed_disease_name="Hồi hộp",
                confirmed_icd10_code="R00.2",
            ),
        ]
        stats = api._flag_stats(rows)
        self.assertEqual(len(stats.top_clusters), 1)
        self.assertEqual(stats.top_clusters[0].count, 3)

    def test_report_prefers_confirmed_and_canonical_age_token(self):
        rows = [
            _flag_row(
                status="DOCTOR_CONFIRMED",
                confirmed_disease_name="Viêm họng cấp",
                confirmed_icd10_code="J02.9",
            )
        ]
        section = api._flag_report_markdown(rows, sanitize_raw=True)
        self.assertIn("## KHOA 1: Nội Tổng quát & Tim mạch", section)
        self.assertIn("[nguoi_truong_thanh]", section)
        self.assertIn("Viêm họng cấp (J02.9)", section)
        self.assertNotIn("Hồi hộp (R00.2) (R00.2)", section)
        # sanitize_raw=True must never embed the raw garbled STT text.
        self.assertNotIn("tiền đập thành thịt", section)
        self.assertIn("tim đập thình thịch", section)

    def test_log_reuses_identical_pending_transcript(self):
        existing = [_flag_row(flag_id="FLG-OLD")]
        updates = {}

        def fake_update(flag_id, changes, settings):
            updates["flag_id"] = flag_id
            updates["changes"] = changes
            return {"flag_id": flag_id, **changes}

        with patch.object(api, "list_unmatched_logs", return_value=existing), patch.object(
            api, "update_unmatched_log", side_effect=fake_update
        ), patch.object(api, "insert_unmatched_log") as insert_mock:
            facts = {
                "is_out_of_dataset": True,
                "department_id": 1,
                "department_name": "Khoa Nội Tổng quát & Tim mạch",
                "priority_level": "P2",
                "disease_guess": "Hồi hộp",
                "icd10_code": "R00.2",
                "detected_age_group": "nguoi_truong_thanh",
                "flag_reason": "NEW_SYMPTOM_PATTERN",
                "similarity_score": 0.4,
            }
            api._log_out_of_dataset_facts(facts, "tiền đập thành thịt", settings=None)

        insert_mock.assert_not_called()
        self.assertEqual(updates["flag_id"], "FLG-OLD")
        self.assertEqual(facts["flag_id"], "FLG-OLD")
        self.assertIn("tim đập thình thịch", updates["changes"]["medical_symptom_standard"])

    def test_log_updates_flag_id_carried_from_earlier_round(self):
        updates = {}

        def fake_update(flag_id, changes, settings):
            updates["flag_id"] = flag_id
            return {"flag_id": flag_id, **changes}

        with patch.object(api, "list_unmatched_logs", return_value=[]), patch.object(
            api, "update_unmatched_log", side_effect=fake_update
        ), patch.object(api, "insert_unmatched_log") as insert_mock:
            facts = {
                "is_out_of_dataset": False,
                "department_id": 2,
                "department_name": "Khoa Hô hấp",
                "priority_level": "P3",
                "disease_guess": "Viêm họng",
                "icd10_code": "J02.9",
                "flag_reason": "",
            }
            api._log_out_of_dataset_facts(
                facts, "đau họng", settings=None, existing_flag_id="FLG-ROUND1"
            )

        insert_mock.assert_not_called()
        self.assertEqual(updates["flag_id"], "FLG-ROUND1")
        self.assertEqual(facts["flag_id"], "FLG-ROUND1")

    def test_non_ood_without_existing_flag_writes_nothing(self):
        with patch.object(api, "insert_unmatched_log") as insert_mock, patch.object(
            api, "update_unmatched_log"
        ) as update_mock:
            api._log_out_of_dataset_facts(
                {"is_out_of_dataset": False, "department_id": 1},
                "đau họng",
                settings=None,
            )
        insert_mock.assert_not_called()
        update_mock.assert_not_called()


class TicketConfirmationTests(unittest.TestCase):
    def test_confirm_by_ticket_sets_doctor_confirmed(self):
        rows = [_flag_row(ticket_number="#A-001", flag_id="FLG-9")]
        captured = {}

        def fake_update(flag_id, changes, settings):
            captured["flag_id"] = flag_id
            captured["status"] = changes.get("status")
            return {**rows[0], **changes}

        request = api.FlaggedSymptomConfirmRequest(
            confirmed_icd10_code="J02.9",
            confirmed_disease_name="Viêm họng cấp",
            confirmed_by_doctor="BS. An",
        )
        with patch.object(api, "list_unmatched_logs", return_value=rows), patch.object(
            api, "update_unmatched_log", side_effect=fake_update
        ):
            response = asyncio.run(
                api.confirm_flagged_symptom_by_ticket("#A-001", request)
            )

        self.assertTrue(response.success)
        self.assertEqual(captured["flag_id"], "FLG-9")
        self.assertEqual(captured["status"], "DOCTOR_CONFIRMED")

    def test_confirm_falls_back_to_flag_id_when_ticket_not_linked(self):
        rows = [_flag_row(ticket_number="", flag_id="FLG-ROUND1")]
        request = api.FlaggedSymptomConfirmRequest(
            confirmed_icd10_code="R00.2",
            confirmed_disease_name="Hồi hộp",
            confirmed_by_doctor="BS. Bình",
            flag_id="FLG-ROUND1",
        )
        with patch.object(api, "list_unmatched_logs", return_value=rows), patch.object(
            api, "update_unmatched_log", side_effect=lambda fid, ch, s: {**rows[0], **ch}
        ) as update_mock:
            response = asyncio.run(
                api.confirm_flagged_symptom_by_ticket("#A-999", request)
            )

        self.assertTrue(response.success)
        update_mock.assert_called_once()


class TrainingPipelineTests(unittest.TestCase):
    def test_train_ingests_sanitized_report_as_triage_audience(self):
        rows = [
            _flag_row(
                status="DOCTOR_CONFIRMED",
                confirmed_disease_name="Hồi hộp",
                confirmed_icd10_code="R00.2",
            )
        ]
        captured = {}

        def fake_write(section):
            captured["section"] = section

        def fake_ingest(path, doc_id, specialty, **kwargs):
            captured["ingest"] = {
                "doc_id": doc_id,
                "target_audience": kwargs.get("target_audience"),
            }

            class _Result:
                total_chunks = 2

            return _Result()

        with patch.object(api, "list_unmatched_logs", return_value=rows), patch.object(
            api, "_write_flag_reports", side_effect=fake_write
        ), patch.object(api, "update_unmatched_log"), patch(
            "app.ingestion.pipeline.purge_document"
        ) as purge_mock, patch(
            "app.ingestion.pipeline.ingest_markdown_file", side_effect=fake_ingest
        ):
            response = asyncio.run(
                api.train_flagged_symptoms(api.FlaggedTrainRequest(flag_ids=["FLG-20260101-0001"]))
            )

        purge_mock.assert_called_once()
        self.assertEqual(captured["ingest"]["doc_id"], api.FLAG_REPORT_DOC_ID)
        self.assertEqual(captured["ingest"]["target_audience"], "tiep_don")
        self.assertIn("[nguoi_truong_thanh]", captured["section"])
        self.assertNotIn("tiền đập thành thịt", captured["section"])
        self.assertEqual(response.chunks_created, 2)
        self.assertEqual(response.chunks, 2)


class EndToEndPipelineTests(unittest.TestCase):
    """The three dry-run scenarios requested for the next module hand-off."""

    RAMBLING_STT = (
        "hom qua toi an com hop va uong tra sua, toi ve xem phim rat khuya. "
        "nhung tu sang nay tien dap thanh thit, chong van, dau cua hong, "
        "noi man do ngua lien tuc. toi cung uong ca phe sua da truoc khi den "
        "phong kham."
    )

    def test_rambling_stt_cleaned_ood_and_one_flag(self):
        cleaned = api._clean_clinical_transcript(self.RAMBLING_STT)
        for expected in (
            "tim đập thình thịch",
            "chóng mặt",
            "đau cổ họng",
            "nổi mẩn đỏ, ngứa liên tục",
        ):
            self.assertIn(expected, cleaned)

        summary = api._clinical_symptom_summary(self.RAMBLING_STT)
        self.assertNotIn("tra sua", summary)
        self.assertNotIn("com hop", summary)

        llm_payload = {
            "department_id": 3,
            "priority_level": "P2",
            "disease_guess": "Viêm da do côn trùng cắn",
            # The LLM's known hallucination: L27.0 is a drug eruption, not a bite.
            "icd10_code": "L27.0",
            "chief_complaint_summary": "Hồi hộp, đau họng và nổi mẩn ngứa.",
            "needs_clarification": False,
            "is_out_of_dataset": False,
        }
        facts, warnings = api._sanitise_triage_payload(
            llm_payload, best_score=0.31, raw_text=self.RAMBLING_STT
        )
        self.assertTrue(facts["is_out_of_dataset"])
        self.assertEqual(facts["flag_reason"], "LOW_SIMILARITY_MATCH")
        self.assertEqual(facts["icd10_code"], "L23.8")
        self.assertEqual(
            facts["disease_guess"], "Viêm da tiếp xúc dị ứng do côn trùng/sâu"
        )
        # Cardiac risk wins over the skin/ENT complaint; the other two are kept.
        self.assertEqual(facts["department_id"], 1)
        self.assertEqual(facts["secondary_department_id"], 2)
        self.assertIn("Khoa 3", facts["multi_specialty_note"])
        self.assertTrue(any("L27.0" in warning for warning in warnings))

        stored: list[dict[str, object]] = []

        def fake_insert(values, settings):
            row = {
                "id": len(stored) + 1,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                **values,
            }
            stored.append(row)
            return row

        with patch.object(
            api, "list_unmatched_logs", side_effect=lambda settings=None: list(stored)
        ), patch.object(
            api,
            "update_unmatched_log",
            side_effect=lambda fid, changes, s: next(
                (row | changes for row in stored if row["flag_id"] == fid), None
            ),
        ), patch.object(api, "new_flag_id", return_value="FLG-TEST"), patch.object(
            api, "insert_unmatched_log", side_effect=fake_insert
        ) as insert:
            api._log_out_of_dataset_facts(dict(facts), self.RAMBLING_STT, None)
            # A clarify round reuses the same flag id instead of creating another.
            api._log_out_of_dataset_facts(
                dict(facts),
                self.RAMBLING_STT,
                None,
                existing_flag_id=facts.get("flag_id"),
            )

        insert.assert_called_once()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["transcription"], self.RAMBLING_STT)
        self.assertIn("tim đập thình thịch", stored[0]["normalized_transcription"])
        self.assertEqual(stored[0]["ai_icd10_code"], "L23.8")
        self.assertEqual(stored[0]["ai_department_id"], 1)

    def test_clarification_response_keeps_clean_text_and_flag(self):
        facts, _ = api._sanitise_triage_payload(
            _final_payload(needs_clarification=True),
            best_score=0.30,
            raw_text=self.RAMBLING_STT,
        )
        facts["flag_id"] = "FLG-CLARIFY"
        with patch.object(
            api, "_retrieve_triage_context", return_value=("", [], [], 0.30)
        ), patch.object(
            api, "_extract_triage_facts", return_value=(dict(facts), [])
        ), patch.object(api, "_log_out_of_dataset_facts") as log:
            response = api._run_triage_from_transcription(
                self.RAMBLING_STT, "2099-01-05", None, None, 0.0, None
            )

        self.assertTrue(response.needs_clarification)
        self.assertIn("tim đập thình thịch", response.normalized_transcription)
        self.assertTrue(response.is_out_of_dataset)
        self.assertEqual(response.flag_reason, "LOW_SIMILARITY_MATCH")
        self.assertEqual(response.flag_id, "FLG-CLARIFY")
        log.assert_called_once()

    def test_flag_ticket_link_then_doctor_confirmation(self):
        rows = [_flag_row(flag_id="FLG-LINK", ticket_number="")]
        captured: dict[str, object] = {}

        def fake_update(flag_id, changes, settings):
            captured.setdefault("flag_id", flag_id)
            captured.update(changes)
            rows[0].update(changes)
            return dict(rows[0])

        link_payload = api.FlaggedSymptomUpdate(
            ticket_number="#A-001",
            patient_name="Nguyễn Văn A",
            flag_reason="LOW_SIMILARITY_MATCH",
            ai_icd10_code="R00.2",
            ai_department_id=1,
            ai_priority_level="P2",
        )
        with patch.object(api, "get_settings", return_value=None), patch.object(
            api, "update_unmatched_log", side_effect=fake_update
        ):
            link_response = asyncio.run(
                api.update_flagged_symptom("FLG-LINK", link_payload)
            )
        self.assertTrue(link_response.success)
        self.assertEqual(captured["ticket_number"], "#A-001")
        self.assertEqual(captured["flag_reason"], "LOW_SIMILARITY_MATCH")

        confirm_payload = api.FlaggedSymptomConfirmRequest(
            confirmed_icd10_code="R00.2",
            confirmed_disease_name="Hồi hộp/đánh trống ngực",
            confirmed_by_doctor="BS. An",
            flag_id="FLG-LINK",
        )
        with patch.object(api, "list_unmatched_logs", return_value=rows), patch.object(
            api, "update_unmatched_log", side_effect=fake_update
        ):
            confirm_response = asyncio.run(
                api.confirm_flagged_symptom_by_ticket("#A-001", confirm_payload)
            )

        self.assertTrue(confirm_response.success)
        self.assertEqual(captured["status"], "DOCTOR_CONFIRMED")
        self.assertEqual(captured["confirmed_icd10_code"], "R00.2")
        self.assertEqual(captured["confirmed_by_doctor"], "BS. An")
        self.assertEqual(confirm_response.items[0].status, "DOCTOR_CONFIRMED")

    def test_training_ingests_tiep_don_age_group_and_retrieves_it(self):
        from app.rag.retriever import SearchResult

        rows = [
            _flag_row(
                status="DOCTOR_CONFIRMED",
                confirmed_disease_name="Hồi hộp",
                confirmed_icd10_code="R00.2",
            )
        ]
        captured: dict[str, object] = {}

        def fake_write(section):
            captured["section"] = section

        def fake_ingest(path, doc_id, specialty, **kwargs):
            captured["target_audience"] = kwargs.get("target_audience")

            class _Result:
                total_chunks = 1

            return _Result()

        with patch.object(api, "list_unmatched_logs", return_value=rows), patch.object(
            api, "_write_flag_reports", side_effect=fake_write
        ), patch.object(api, "update_unmatched_log"), patch(
            "app.ingestion.pipeline.purge_document"
        ), patch(
            "app.ingestion.pipeline.ingest_markdown_file", side_effect=fake_ingest
        ):
            response = asyncio.run(
                api.train_flagged_symptoms(api.FlaggedTrainRequest(flag_ids=["FLG-20260101-0001"]))
            )

        self.assertEqual(response.chunks_created, 1)
        self.assertEqual(captured["target_audience"], "tiep_don")
        self.assertEqual(
            extract_age_group(str(captured["section"])), "nguoi_truong_thanh"
        )

        def fake_retrieve(query, **kwargs):
            captured["retrieval"] = kwargs
            return [
                SearchResult(
                    id=1,
                    doc_id=api.FLAG_REPORT_DOC_ID,
                    specialty="Tổng hợp tiếp đón",
                    content="Hồi hộp [nguoi_truong_thanh]",
                    distance=0.2,
                )
            ]

        with patch("app.rag.retriever.retrieve_context", side_effect=fake_retrieve), patch.object(
            api.ambient, "pgvector_reachable", return_value=True
        ):
            context, _sources, _warnings, score = api._retrieve_triage_context(
                "tim đập thình thịch", None, age_group="nguoi_truong_thanh"
            )

        self.assertIn("[nguoi_truong_thanh]", context)
        self.assertGreater(score, 0.5)
        self.assertEqual(captured["retrieval"]["target_audience"], "tiep_don")
        self.assertEqual(captured["retrieval"]["age_group"], "nguoi_truong_thanh")


if __name__ == "__main__":
    unittest.main()
