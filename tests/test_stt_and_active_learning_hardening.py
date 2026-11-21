"""Regression contracts for STT payloads and active-learning ingestion.

No PostgreSQL / Groq network access: all external seams are mocked.
Run with::

    PYTHONPATH=. python tests/test_stt_and_active_learning_hardening.py
"""

from __future__ import annotations

import asyncio
import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import app.api as api


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        GROQ_API_KEY="test-key",
        LLM_STT_MODEL_ID="whisper-large-v3-turbo",
    )


def _flag_row(**overrides):
    row = {
        "flag_id": "FLG-A",
        "status": "DOCTOR_CONFIRMED",
        "age_group": "nguoi_truong_thanh",
        "ai_department_id": 1,
        "ai_priority_level": "P2",
        "ai_disease_guess": "Hồi hộp",
        "ai_icd10_code": "R00.2",
        "medical_symptom_standard": "tim đập thình thịch",
        "normalized_transcription": "tim đập thình thịch",
        "transcription": "tiền đập thành thịt",
        "red_flags": "",
        "reception_action": "",
        "confirmed_disease_name": "",
        "confirmed_icd10_code": "",
    }
    row.update(overrides)
    return row


class SttPayloadTests(unittest.TestCase):
    def test_audio_file_tuple_has_bytes_stream_and_mime(self):
        for filename, expected_mime in (
            ("recording.webm", "audio/webm"),
            ("recording.wav", "audio/wav"),
            ("recording.mp3", "audio/mpeg"),
            ("recording.m4a", "audio/mp4"),
            ("recording.ogg", "audio/ogg"),
        ):
            name, stream, mime = api._groq_audio_file(b"audio-bytes", filename)
            self.assertIsInstance(stream, io.BytesIO)
            self.assertEqual(stream.read(), b"audio-bytes")
            self.assertEqual(mime, expected_mime)
            self.assertTrue(name.endswith(filename.split(".")[-1]))

    def test_model_candidates_respect_runtime_configuration_then_fallback(self):
        self.assertEqual(
            api._stt_model_candidates("whisper-large-v3"),
            ["whisper-large-v3", "whisper-large-v3-turbo"],
        )
        self.assertEqual(
            api._stt_model_candidates("custom-model"),
            ["custom-model", "whisper-large-v3-turbo", "whisper-large-v3"],
        )

    def test_voice_transcription_passes_buffered_audio_with_mime(self):
        create = MagicMock(return_value="Tê chân")
        client = MagicMock()
        client.audio.transcriptions.create = create
        with patch("groq.Groq", return_value=client), patch.object(
            api, "_runtime_model", return_value="whisper-large-v3-turbo"
        ):
            text = api._transcribe_voice(b"RIFFdata", "blob", _settings())

        self.assertEqual(text, "Tê chân")
        kwargs = create.call_args.kwargs
        name, stream, mime = kwargs["file"]
        self.assertEqual(name, "blob")
        self.assertIsInstance(stream, io.BytesIO)
        self.assertEqual(stream.read(), b"RIFFdata")
        self.assertEqual(mime, "audio/webm")  # no suffix -> browser blob default
        self.assertEqual(kwargs["model"], "whisper-large-v3-turbo")
        self.assertEqual(kwargs["language"], "vi")
        self.assertEqual(kwargs["response_format"], "text")

    def test_ambient_transcription_retries_fallback_model_with_fresh_stream(self):
        class FakeBadRequest(Exception):
            pass

        create = MagicMock(side_effect=[FakeBadRequest("model unavailable"), "Sốt cao"])
        client = MagicMock()
        client.audio.transcriptions.create = create
        with patch("groq.Groq", return_value=client), patch(
            "groq.BadRequestError", FakeBadRequest
        ), patch.object(api, "_runtime_model", return_value="whisper-large-v3-turbo"):
            text = api._transcribe_ambient(b"OggS", "clinic.ogg", _settings())

        self.assertEqual(text, "Sốt cao")
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args_list[0].kwargs["model"], "whisper-large-v3-turbo")
        self.assertEqual(create.call_args_list[1].kwargs["model"], "whisper-large-v3")
        for call in create.call_args_list:
            _name, stream, mime = call.kwargs["file"]
            self.assertEqual(stream.read(), b"OggS")
            self.assertEqual(mime, "audio/ogg")


class ShortSymptomDetectionTests(unittest.TestCase):
    def test_valid_short_clinical_utterances_are_not_noise(self):
        for utterance in (
            "Tê chân",
            "Đau rát",
            "Sốt cao",
            "Ho khan",
            "Nổi mẩn",
            "Nhức đầu",
            "Mờ mắt",
            "Ngứa họng",
            "Buốt chân",
            "Sốt",
            "Nổi",
        ):
            with self.subTest(utterance=utterance):
                self.assertFalse(api._looks_like_whisper_hallucination(utterance))

    def test_known_noise_is_still_rejected(self):
        self.assertTrue(api._looks_like_whisper_hallucination("subscribe channel"))
        self.assertTrue(api._looks_like_whisper_hallucination("ừ ừ ừ ừ ừ ừ"))


class ActiveLearningHardeningTests(unittest.TestCase):
    def test_markdown_cells_escape_pipes_and_keep_line_breaks(self):
        row = _flag_row(
            medical_symptom_standard="đau ngực | tức\nkhó thở",
            red_flags="tức ngực\nméo miệng | liệt nửa người",
            reception_action="gọi cấp cứu | nằm yên\nđợi xe",
        )
        section = api._flag_report_markdown([row], sanitize_raw=True)
        self.assertIn("đau ngực \\| tức<br>khó thở", section)
        self.assertIn("tức ngực<br>méo miệng \\| liệt nửa người", section)
        self.assertIn("gọi cấp cứu \\| nằm yên<br>đợi xe", section)
        self.assertIn("FLG-A", section)

    def test_training_purges_before_ingest_deduplicates_and_marks_trained(self):
        calls: list[str] = []
        rows = [
            _flag_row(flag_id="FLG-A", status="DOCTOR_CONFIRMED"),
            _flag_row(flag_id="FLG-A", status="TRAINED"),  # duplicate log row
            _flag_row(flag_id="FLG-B", status="TRAINED"),  # previously trained
        ]
        captured: dict[str, object] = {}

        def fake_write(section):
            captured["section"] = section

        def fake_purge(doc_id, settings):
            calls.append("purge")
            self.assertEqual(doc_id, api.FLAG_REPORT_DOC_ID)
            return 2

        def fake_ingest(path, doc_id, specialty, **kwargs):
            calls.append("ingest")
            self.assertEqual(doc_id, api.FLAG_REPORT_DOC_ID)
            self.assertEqual(kwargs.get("target_audience"), "tiep_don")

            class _Result:
                total_chunks = 3

            return _Result()

        def fake_update(flag_id, changes, settings):
            calls.append(f"update:{flag_id}")
            self.assertEqual(changes, {"status": "TRAINED"})
            return {"flag_id": flag_id, **changes}

        with patch.object(api, "list_unmatched_logs", return_value=rows), patch.object(
            api, "_flag_ids_in_report_files", return_value={"FLG-B"}
        ), patch.object(api, "_write_flag_reports", side_effect=fake_write), patch.object(
            api, "update_unmatched_log", side_effect=fake_update
        ), patch(
            "app.ingestion.pipeline.purge_document", side_effect=fake_purge
        ), patch(
            "app.ingestion.pipeline.ingest_markdown_file", side_effect=fake_ingest
        ):
            response = asyncio.run(
                api.train_flagged_symptoms(
                    api.FlaggedTrainRequest(flag_ids=["FLG-A", "FLG-B"])
                )
            )

        self.assertEqual(calls[:2], ["purge", "ingest"])
        self.assertEqual(calls[2:], ["update:FLG-A", "update:FLG-B"])
        section = str(captured["section"])
        self.assertEqual(section.count("FLG-A"), 1)
        self.assertEqual(section.count("FLG-B"), 1)
        self.assertEqual(response.trained_count, 2)
        self.assertEqual(response.chunks_created, 3)


if __name__ == "__main__":
    unittest.main()
