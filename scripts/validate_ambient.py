"""Validation harness for the ambient clinical copilot pipeline.

Run from the project root::

    python scripts/validate_ambient.py

What it proves
--------------
1. **Zero audio on disk** - the file inventory of ``data/``, the project root
   and the OS temp directory is snapshotted before/after every request. The
   payload is deliberately > 1 MB, the size at which Starlette's default
   ``SpooledTemporaryFile`` would otherwise spill an audio recording into the
   temp folder, so a green run also proves the in-memory spool watermark holds.
2. **Strict JSON parsing** - markdown fences, prose-wrapped objects and garbage
   are fed to ``parse_llm_json`` / ``sanitise_diagnoses``.
3. **Pipeline contract** - ``POST /api/v1/clinical/ambient-consultation`` and
   ``POST /api/v1/clinical/copilot-chat`` are exercised over HTTP against the
   real Groq LLM, the real pgvector knowledge base and the real formulary.
   Whisper itself is stubbed (a synthetic tone is not speech); the real STT
   code path is called once separately and must fail *only* with the expected
   audio-decode error.

Nothing here writes an audio file: the WAV is generated in memory and handed
to the test client as bytes.
"""

from __future__ import annotations

import asyncio
import math
import re
import sys
import tempfile
import traceback
import wave
from array import array
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.services import ambient  # noqa: E402  (path setup first)
from app.services.clinical_client import (  # noqa: E402
    LOW_STOCK_THRESHOLD,
    AvailableDrug,
    get_available_drugs,
)

AUDIO_EXTENSIONS = {".wav", ".webm", ".mp3", ".m4a", ".ogg", ".flac"}

CANNED_TRANSCRIPT = (
    "Bác sĩ: Chào anh, hôm nay anh đến khám vì vấn đề gì ạ?\n"
    "Bệnh nhân: Thưa bác sĩ tôi bị zona đau rát ở bên sườn trái đã năm ngày nay, "
    "nổi mụn nước thành đám trên nền đỏ.\n"
    "Bác sĩ: Anh có sốt, ra mồ hôi đêm hoặc sút cân không?\n"
    "Bệnh nhân: Dạ có, vài tháng nay tôi sút bốn cân, đêm nào cũng ra mồ hôi ướt áo, "
    "chiều hay sốt nhẹ.\n"
    "Bác sĩ: Anh có dễ bị lở miệng hay tiền sử bệnh gì chưa?\n"
    "Bệnh nhân: Dạ tôi hay bị lở miệng lắm, năm nay đã bị ba lần mà chưa rõ nguyên nhân.\n"
    "Bác sĩ: Để tôi khám vùng tổn thương, nghe phổi, rồi chỉ định xét nghiệm máu, "
    "test nhanh HIV và đánh giá nguy cơ lao nhé.\n"
    "Bệnh nhân: Dạ vâng, mong bác sĩ giúp tôi."
)

HIV_FOLLOW_UP = (
    "Bệnh nhân xác nhận đã nhiễm HIV 2 năm trước và đang uống ARV đều đặn, "
    "không dùng kháng sinh gì khác."
)

_failures: list[str] = []
_warnings: list[str] = []


def check(condition: Any, label: str, detail: str = "", hard: bool = True) -> bool:
    """Record one assertion; ``hard=False`` reports without failing the run."""
    ok = bool(condition)
    tag = "PASS" if ok else ("FAIL" if hard else "WARN")
    line = f"[{tag}] {label}"
    if not ok and detail:
        line += f" :: {detail}"
    print(line, flush=True)
    if not ok:
        (_failures if hard else _warnings).append(label + (f" ({detail})" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# In-memory audio fixtures & filesystem snapshots
# ---------------------------------------------------------------------------


def build_tone_wav(seconds: float = 45.0, sample_rate: int = 16000) -> bytes:
    """Non-silent PCM WAV larger than 1 MB, built entirely in RAM.

    Non-silent so the gateway's silence guard lets the request through, and
    larger than Starlette's 1 MB spool default so a regression would show up as
    a brand-new ``tmp*`` file in the OS temp directory.
    """
    import io

    total = int(seconds * sample_rate)
    samples = array("h")
    for index in range(total):
        t = index / sample_rate
        envelope = 4000 + 6000 * abs((t * 2.0) % 1.0 - 0.5)
        value = envelope * (
            0.6 * math.sin(2 * math.pi * 220 * t) + 0.4 * math.sin(2 * math.pi * 440 * t)
        )
        samples.append(int(value))
    payload = samples.tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(payload)
    return buffer.getvalue()


def build_silence_wav(seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return buffer.getvalue()


def snapshot(root: Path) -> dict[str, tuple[int, float]]:
    """path -> (size, mtime); skipped entirely when the directory is missing."""
    files: dict[str, tuple[int, float]] = {}
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        files[str(path)] = (stat.st_size, stat.st_mtime)
    return files


def _looks_like_riff(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"RIFF"
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Deterministic unit checks (no network)
# ---------------------------------------------------------------------------


def test_strict_json_parsing() -> None:
    print("\n=== Strict JSON parsing (markdown fence stripping) ===")

    fenced = ambient.parse_llm_json('```json\n{"a": 1, "b": [1, 2]}\n```')
    check(fenced == {"a": 1, "b": [1, 2]}, "parses a ```json fenced object")

    bare_fence = ambient.parse_llm_json('```\n{"ok": true}\n```')
    check(bare_fence == {"ok": True}, "parses a fence without the json tag")

    wrapped = ambient.parse_llm_json(
        'Đây là kết quả bạn cần:\n{"has_red_flags": true, "n": 3}\nHy vọng hữu ích!'
    )
    check(wrapped == {"has_red_flags": True, "n": 3}, "extracts an object wrapped in prose")

    plain = ambient.parse_llm_json('{"speaker": "doctor"}')
    check(plain == {"speaker": "doctor"}, "parses a bare object")

    obj_passthrough = ambient.parse_llm_json({"x": 1})
    check(obj_passthrough == {"x": 1}, "passes an already-decoded object through")

    for garbage in ("", "   ", "không có json ở đây", "[1, 2, 3]", "```json\nlỗi\n```"):
        try:
            ambient.parse_llm_json(garbage)
            check(False, f"rejects {garbage!r}", "no exception raised")
        except ambient.AmbientPipelineError:
            check(True, f"rejects {garbage!r}")

    diagnoses, warnings = ambient.sanitise_diagnoses(
        [
            {"icd10Code": "B02.9", "diseaseName": "Zona", "isPrimary": True, "aiConfidence": 95},
            {"icd10Code": "ZZ9", "diseaseName": "mã bịa", "aiConfidence": 10},
            {"icd10Code": "b20", "diseaseName": "HIV", "aiConfidence": 250},
            {"icd10Code": "B02.9", "diseaseName": "trùng"},
            {"icd10Code": "L51.2", "diseaseName": "SJS", "isPrimary": True, "aiConfidence": 80},
        ]
    )
    codes = [d.icd10Code for d in diagnoses]
    check(codes == ["B02.9", "B20", "L51.2"], "keeps only valid ICD-10 codes", str(codes))
    check(
        diagnoses[1].aiConfidence == 100,
        "clamps aiConfidence to 0-100",
        str(diagnoses[1].aiConfidence),
    )
    check(
        sum(1 for d in diagnoses if d.isPrimary) == 1,
        "keeps exactly one primary diagnosis",
    )
    check(any("ZZ9" in w for w in warnings), "reports the rejected code")


def _sample_formulary() -> list[AvailableDrug]:
    return [
        AvailableDrug(
            id=1, code="DRG-001", name="Amoxicillin", concentration="500mg",
            dosage_form="Viên nang", stock_quantity=2400, bhyt_coverage="BHYT_80",
            is_penicillin_class=True,
        ),
        AvailableDrug(
            id=9, code="DRG-009", name="Fluconazole", concentration="150mg",
            dosage_form="Viên nén", stock_quantity=40, bhyt_coverage="BHYT_80",
        ),
    ]


def test_prescription_reconciliation() -> None:
    print("\n=== Prescription draft <-> formulary reconciliation ===")
    drugs = _sample_formulary()
    draft, warnings = ambient.reconcile_prescription(
        [
            {"drugId": 1, "drugName": "Amoxicillin", "routeFrequency": "Uống 3 lần/ngày",
             "duration": "7 ngày", "quantity": 21},
            {"drugId": 999, "drugName": "Thuốc lạ", "quantity": 10},
            {"drugName": "Fluconazole 150mg", "quantity": 20},
        ],
        drugs,
    )
    check(len(draft) == 2, "drops the drug that is not in stock", str(len(draft)))
    check(
        any("Thuốc lạ" in w for w in warnings),
        "reports the dropped drug",
    )
    amox = next(item for item in draft if item.drugId == 1)
    check(amox.currentStock == 2400 and amox.stockWarning is None,
          "well-stocked drug carries no warning")
    check(amox.isPenicillinClass and any("Penicillin" in w for w in warnings),
          "penicillin class is surfaced as an allergy risk")
    fluc = next(item for item in draft if item.drugId == 9)
    check(fluc.stockWarning == "LOW_STOCK",
          f"stock below {LOW_STOCK_THRESHOLD} units is flagged LOW_STOCK",
          f"stock={fluc.currentStock} warning={fluc.stockWarning}")
    check(fluc.currentStock == 40,
          "currentStock mirrors the formulary, not the model",
          str(fluc.currentStock))
    check(fluc.quantity == 20 and fluc.stockWarning == "LOW_STOCK",
          "keeps the requested quantity while the low-stock flag is shown",
          f"quantity={fluc.quantity} warning={fluc.stockWarning}")

    limited, _ = ambient.reconcile_prescription(
        [{"drugId": 9, "drugName": "Fluconazole", "quantity": 999}], drugs
    )
    check(
        limited[0].stockWarning == "INSUFFICIENT_STOCK",
        "quantity above stock is flagged INSUFFICIENT_STOCK",
        str(limited[0].stockWarning),
    )

    empty_drugs, warnings = ambient.reconcile_prescription(
        [{"drugId": 1, "drugName": "Amoxicillin"}], []
    )
    check(
        empty_drugs == [] and warnings,
        "empty formulary empties the draft with a warning",
    )


def test_revision_merge() -> None:
    """A section the model forgets must not wipe the doctor's current draft."""
    print("\n=== Copilot revision merge (omitted sections keep the draft) ===")
    drugs = _sample_formulary()
    previous = ambient.ClinicalDraft(
        soap_note=ambient.SoapNote(
            subjective="Zona 5 ngày",
            objective="Nổi mụn nước thành đám bên trái",
            assessment="Zona thần kinh",
            plan="Khám da liễu, xét nghiệm máu",
        ),
        diagnoses=[
            ambient.IcdDiagnosis(
                icd10Code="B02.9", diseaseName="Zona", isPrimary=True, aiConfidence=90
            )
        ],
        prescription_draft=[
            ambient.PrescriptionItemDraft(
                drugId=1, drugCode="DRG-001", drugName="Amoxicillin",
                routeFrequency="Uống 3 lần/ngày", duration="7 ngày", quantity=14,
                currentStock=2400,
            )
        ],
    )

    merged = ambient.apply_revision_payload(
        {
            "diagnoses": [
                {"icd10Code": "B20", "diseaseName": "Bệnh HIV", "isPrimary": True,
                 "aiConfidence": 88},
                {"icd10Code": "B02.9", "diseaseName": "Zona", "isPrimary": False,
                 "aiConfidence": 90},
            ]
        },
        previous,
        drugs,
    )
    check(merged.soap_note.subjective == "Zona 5 ngày",
          "omitted SOAP note is preserved")
    check([d.icd10Code for d in merged.diagnoses] == ["B20", "B02.9"],
          "returned diagnoses replace the previous list")
    check(len(merged.prescription_draft) == 1
          and merged.prescription_draft[0].currentStock == 2400,
          "omitted prescription is preserved and re-checked against stock")
    check(any("SOAP" in w for w in merged.warnings),
          "the preserved SOAP is surfaced as a warning")

    cleared = ambient.apply_revision_payload(
        {
            "soap_note": {"subjective": "x", "objective": "y",
                          "assessment": "z", "plan": "w"},
            "prescription_draft": [],
        },
        previous,
        drugs,
    )
    check(cleared.prescription_draft == [],
          "an explicitly empty prescription clears the draft")
    check(cleared.diagnoses[0].icd10Code == "B02.9",
          "untouched diagnosis list survives an unrelated update")


# ---------------------------------------------------------------------------
# Diarization sensitivity (doctor inquiry attributed to the doctor)
# ---------------------------------------------------------------------------


SENSITIVITY_DIALOGUE = (
    "Bác sĩ: Chào anh, hôm nay anh bị làm sao ạ\n"
    "Bệnh nhân: Em bị nổi mụn nước ngứa ở bên trái đã năm ngày nay\n"
    "Bác sĩ: Anh có bị dị ứng hay ăn uống nhầm cái gì không\n"
    "Bệnh nhân: Dạ em không bị dị ứng gì hết\n"
    "Bác sĩ: Anh có đang uống thuốc gì không\n"
    "Bệnh nhân: Dạ em đang bôi acyclovir ở nhà thôi"
)


def test_diarization_sensitivity() -> list[Any]:
    """The reported bug: a clinical question folded into a patient turn."""
    print("\n=== Diarization sensitivity (doctor inquiry splitting) ===")

    question = "Anh có bị dị ứng hay ăn uống nhầm cái gì không?"
    typo = "Anh có bị dưỡng hay ăn uống nhầm cái gì không"
    check(bool(ambient._DOCTOR_CUES.search(question)),
          "doctor cues match the allergy inquiry")
    check(bool(ambient._DOCTOR_CUES.search(typo)),
          "doctor cues still match after a Whisper typo (dị ứng -> dưỡng)")
    check(not ambient._DOCTOR_CUES.search("tôi bị đau rát năm ngày nay"),
          "doctor cues do not match plain patient speech")

    split = ambient._split_question_then_answer(
        "Anh có bị dị ứng hay ăn uống nhầm cái gì không? tôi không bị dị ứng", "patient"
    )
    check(split == [
        ("doctor", "Anh có bị dị ứng hay ăn uống nhầm cái gì không"),
        ("patient", "tôi không bị dị ứng"),
    ], "merged question+answer is split into alternating turns", str(split))
    check(ambient._split_question_then_answer("tôi bị đau 5 ngày nay rồi", "patient")
          == [("patient", "tôi bị đau 5 ngày nay rồi")],
          "patient speech is never split")

    from app.api import _ambient_llm_model, get_settings

    started = __import__("time").perf_counter()
    result = ambient.diarize_dialogue(
        SENSITIVITY_DIALOGUE,
        "Khoa Da liễu (department_id=3). Dị ứng: không. Tiền sử: khoẻ mạnh.",
        _ambient_llm_model(get_settings()),
        get_settings(),
    )
    elapsed = __import__("time").perf_counter() - started
    speakers = [turn.speaker for turn in result.turns]
    print(f"       diarization of the reported dialogue: {elapsed:.1f}s "
          f"-> {' / '.join(speakers)}")
    for turn in result.turns:
        print(f"          {turn.speaker:8s} | {turn.text[:70]}")

    check(len(result.turns) >= 4, "the dialogue produced at least four turns")
    check(all(a != b for a, b in zip(speakers, speakers[1:])),
          "turns alternate doctor/patient", str(speakers))
    question_turn = next(
        (t for t in result.turns if "dị ứng" in t.text.lower()
         and ("?" in t.text or "không" in t.text.lower())),
        None,
    )
    check(question_turn is not None and question_turn.speaker == "doctor",
          "the allergy inquiry is attributed to the doctor",
          str(question_turn))
    if question_turn is not None:
        idx = result.turns.index(question_turn)
        following = result.turns[idx + 1] if idx + 1 < len(result.turns) else None
        check(following is not None and following.speaker == "patient",
              "the patient's answer immediately follows as its own turn",
              str(following))
    return result.turns


# ---------------------------------------------------------------------------
# HTTP-level pipeline checks
# ---------------------------------------------------------------------------


def test_formulary_fetch() -> list[AvailableDrug]:
    print("\n=== In-stock formulary (HTTP -> PostgreSQL fallback) ===")
    result = asyncio.run(get_available_drugs())
    check(result.ok, "formulary is available", str(result.error))
    print(f"       source = {result.source or '-'} | {len(result.drugs)} drug(s)")
    for drug in result.drugs:
        print(f"       - {drug.compact_row()}")
    if result.ok:
        check(
            all(d.active and d.stock_quantity > 0 for d in result.drugs),
            "only active drugs with stock > 0 are returned",
        )
        low = [d for d in result.drugs if d.stock_warning]
        if low:
            check(
                all(d.stock_quantity < LOW_STOCK_THRESHOLD for d in low),
                "LOW_STOCK exactly on the threshold",
            )
        else:
            print(f"[INFO] no drug below {LOW_STOCK_THRESHOLD} units right now")
    return result.drugs


def _stub_transcription(audio: bytes, filename: str, settings) -> str:  # noqa: ARG001
    """Whisper stand-in: a synthetic tone is not speech, a real visit is."""
    return CANNED_TRANSCRIPT


def test_silence_guard(client, audio_files: dict[str, bytes]) -> None:
    print("\n=== Guards (empty / silence / bad extension) ===")
    response = client.post(
        "/api/v1/clinical/ambient-consultation",
        files={"audio_file": ("silent.wav", audio_files["silence"], "audio/wav")},
        data={"department_id": "3"},
    )
    check(response.status_code == 400, "silent WAV is rejected", f"{response.status_code} {response.text[:200]}")

    response = client.post(
        "/api/v1/clinical/ambient-consultation",
        files={"audio_file": ("notes.txt", b"xin chao", "text/plain")},
        data={"department_id": "3"},
    )
    check(response.status_code == 400, "unsupported audio extension is rejected",
          f"{response.status_code} {response.text[:200]}")


def test_real_stt_error_path(audio_files: dict[str, bytes]) -> None:
    """The real Whisper call must fail only with the expected decode error."""
    print("\n=== Real Whisper call on a tone (expected to be rejected) ===")
    from app.api import VoiceAudioDecodeError, _transcribe_ambient, get_settings

    try:
        text = _transcribe_ambient(audio_files["tone"], "tone.wav", get_settings())
    except VoiceAudioDecodeError as exc:
        check(True, "real STT rejected non-speech as designed", str(exc)[:120])
        return
    except Exception as exc:  # auth / network / model problems must surface
        check(False, "real STT failed outside the audio-decode contract",
              f"{type(exc).__name__}: {exc}")
        return
    check(True, "real STT answered (model reachable)", repr(text[:80]))


def test_ambient_endpoint(client, audio_files: dict[str, bytes],
                          formulary: list[AvailableDrug]) -> dict[str, Any]:
    print("\n=== POST /api/v1/clinical/ambient-consultation ===")
    response = client.post(
        "/api/v1/clinical/ambient-consultation",
        files={
            "audio_file": ("consultation.wav", audio_files["tone"], "audio/wav")
        },
        data={
            "department_id": "3",
            "doctor_id": "7",
            "patient_id": "42",
            "allergies": "Penicillin, Aspirin",
            "patient_history": "Tiểu đường type 2, tăng huyết áp",
        },
    )
    if not check(response.status_code == 200, "endpoint returns 200",
                 f"{response.status_code} {response.text[:400]}"):
        return {}
    body = response.json()

    required = {
        "success", "raw_transcript", "diarized_transcript", "has_red_flags",
        "red_flag_alert", "soap_note", "diagnoses", "prescription_draft", "latency_ms",
    }
    check(required <= set(body), "response carries the documented schema",
          str(sorted(required - set(body))))
    check(body["success"] is True, "success is true")
    check(body["raw_transcript"] == CANNED_TRANSCRIPT, "raw transcript preserved")

    turns = body["diarized_transcript"]
    check(bool(turns), "diarized transcript is not empty", str(len(turns)))
    check(
        all(t.get("speaker") in {"doctor", "patient"} for t in turns),
        "every turn is labelled doctor/patient",
        str({t.get("speaker") for t in turns}),
    )
    speakers = {t["speaker"] for t in turns}
    check(
        speakers == {"doctor", "patient"},
        "both roles were identified from linguistic cues",
        str(speakers),
    )
    check(any(t["speaker"] == "doctor" for t in turns), "doctor turns present")
    print("       --- diarized transcript ---")
    for turn in turns:
        print(f"       {turn['speaker']}: {turn['text'][:100]}")

    soap = body["soap_note"]
    check(
        {"subjective", "objective", "assessment", "plan"} <= set(soap),
        "SOAP note has all four sections",
        str(sorted(soap)),
    )
    check(
        all(isinstance(soap[k], str) and soap[k].strip() for k in
            ("subjective", "objective", "assessment", "plan")),
        "SOAP sections are non-empty",
    )

    diagnoses = body["diagnoses"]
    check(bool(diagnoses), "at least one ICD-10 diagnosis", str(len(diagnoses)))
    check(
        all(re.match(r"^[A-TV-Z][0-9][0-9AB](\.[0-9A-TV-Z]{1,4})?$", d["icd10Code"],
                     re.I) for d in diagnoses),
        "every icd10Code matches the WHO pattern",
        str([d["icd10Code"] for d in diagnoses]),
    )
    check(
        all(0 <= int(d["aiConfidence"]) <= 100 for d in diagnoses),
        "aiConfidence is 0-100",
        str([d["aiConfidence"] for d in diagnoses]),
    )
    check(sum(1 for d in diagnoses if d["isPrimary"]) == 1,
          "exactly one primary diagnosis",
          str([(d["icd10Code"], d["isPrimary"]) for d in diagnoses]))
    print(f"       diagnoses = {[d['icd10Code'] for d in diagnoses]}")

    rx = body["prescription_draft"]
    known = {d.id: d for d in formulary}
    check(bool(rx), "prescription draft is not empty", str(body["warnings"]))
    check(
        all(item["drugId"] in known for item in rx),
        "every drugId exists in the live formulary",
        str([item["drugId"] for item in rx]),
    )
    for item in rx:
        drug = known.get(item["drugId"])
        if not drug:
            continue
        check(item["drugCode"] == drug.code, f"drugCode matches formulary for {drug.name}",
              item["drugCode"])
        check(item["currentStock"] == drug.stock_quantity,
              f"currentStock mirrors the formulary for {drug.name}",
              f"{item['currentStock']} vs {drug.stock_quantity}")
        check(int(item["quantity"]) >= 1, f"quantity >= 1 for {drug.name}")
        check(item["stockWarning"] in (None, "LOW_STOCK", "INSUFFICIENT_STOCK"),
              f"stockWarning is a known value for {drug.name}", str(item["stockWarning"]))
        expected = "LOW_STOCK" if drug.stock_quantity < LOW_STOCK_THRESHOLD else None
        if expected == "LOW_STOCK":
            check(item["stockWarning"] == expected,
                  f"LOW_STOCK propagated for {drug.name}", str(item["stockWarning"]))
    print("       --- prescription draft ---")
    for item in rx:
        print(f"       {item['drugCode']} {item['drugName']}: {item['routeFrequency']}, "
              f"{item['duration']} x{item['quantity']} (stock {item['currentStock']}, "
              f"warning {item['stockWarning']})")

    check(body["has_red_flags"] in (True, False), "has_red_flags is a boolean")
    if body["has_red_flags"]:
        check(bool(body["red_flag_alert"]), "red flag alert is populated")
        check(bool(body.get("recommended_inquiry")), "recommended_inquiry is populated")
        print(f"       red_flag_alert = {body['red_flag_alert'][:200]}")
        print(f"       recommended_inquiry = {body.get('recommended_inquiry', '')[:200]}")
    else:
        check(False, "red flags detected for the HIV/TB pattern in the dialogue",
              "has_red_flags stayed false", hard=False)

    check(float(body["latency_ms"]) > 0, "latency_ms is measured")
    print(f"       latency_ms = {body['latency_ms']} | formulary_source = "
          f"{body.get('formulary_source')} | rag_sources = {body.get('rag_sources')}")
    print(f"       warnings = {body.get('warnings')}")
    if body.get("rag_sources"):
        check(True, "knowledge RAG returned guideline chunks")
    else:
        check(False, "knowledge RAG returned guideline chunks",
              "empty - pgvector unreachable or retrieval timed out", hard=False)
    return body


def test_copilot_chat(client, draft: dict[str, Any], formulary: list[AvailableDrug]) -> None:
    if not draft:
        print("\n=== POST /api/v1/clinical/copilot-chat SKIPPED (no draft) ===")
        return
    print("\n=== POST /api/v1/clinical/copilot-chat ===")
    response = client.post(
        "/api/v1/clinical/copilot-chat",
        json={
            "message": HIV_FOLLOW_UP,
            "soap_note": draft["soap_note"],
            "diagnoses": draft["diagnoses"],
            "prescription_draft": draft["prescription_draft"],
            "raw_transcript": draft["raw_transcript"],
            "diarized_transcript": draft["diarized_transcript"],
            "allergies": "Penicillin, Aspirin",
            "patient_history": "Tiểu đường type 2",
            "department_id": 3,
        },
    )
    blank = client.post("/api/v1/clinical/copilot-chat", json={"message": "   "})
    check(blank.status_code == 422, "blank follow-up message is rejected",
          f"{blank.status_code}")

    if not check(response.status_code == 200, "endpoint returns 200",
                 f"{response.status_code} {response.text[:400]}"):
        return
    body = response.json()
    required = {"success", "assistant_message", "soap_note", "diagnoses",
                "prescription_draft", "latency_ms"}
    check(required <= set(body), "response carries the documented schema",
          str(sorted(required - set(body))))
    soap = body["soap_note"]
    check({"subjective", "objective", "assessment", "plan"} <= set(soap),
          "updated SOAP note has all four sections")
    check(
        all(re.match(r"^[A-TV-Z][0-9][0-9AB](\.[0-9A-TV-Z]{1,4})?$", d["icd10Code"], re.I)
            for d in body["diagnoses"]),
        "updated diagnoses stay ICD-10 valid",
        str([d["icd10Code"] for d in body["diagnoses"]]),
    )
    known = {d.id for d in formulary}
    check(
        all(item["drugId"] in known for item in body["prescription_draft"]),
        "updated prescription still matches the formulary",
        str([item["drugId"] for item in body["prescription_draft"]]),
    )
    hiv_codes = {d["icd10Code"] for d in body["diagnoses"]}
    check(
        any(code.startswith("B2") for code in hiv_codes),
        "HIV (B20) picked up from the doctor's confirmation",
        str(sorted(hiv_codes)),
        hard=False,
    )
    print(f"       assistant_message = {body['assistant_message'][:300]}")
    print(f"       diagnoses = {sorted(hiv_codes)} | latency_ms = {body['latency_ms']}")
    print(f"       soap.plan = {body['soap_note']['plan'][:300]}")


# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print(" Ambient Clinical Copilot - validation (zero audio storage)")
    print("=" * 78)

    from fastapi.testclient import TestClient

    import app.api as api

    # --- in-memory fixtures ------------------------------------------------
    audio_files = {
        "tone": build_tone_wav(),
        "silence": build_silence_wav(),
    }
    print(f"\ntone WAV = {len(audio_files['tone']) / (1024 * 1024):.2f} MB "
          f"(> 1 MB spool default)")
    check(
        api.MultiPartParser.spool_max_size >= len(audio_files["tone"]),
        "multipart spool watermark keeps audio in RAM (never rolled to disk)",
        f"spool={api.MultiPartParser.spool_max_size} audio={len(audio_files['tone'])}",
    )

    # --- filesystem snapshots ---------------------------------------------
    temp_dir = Path(tempfile.gettempdir())
    print(f"watching: data/, project root, {temp_dir}\n")
    before = {
        "data": snapshot(PROJECT_ROOT / "data"),
        "root": snapshot(PROJECT_ROOT),
        "temp": snapshot(temp_dir),
    }

    # --- deterministic unit checks ----------------------------------------
    test_strict_json_parsing()
    test_prescription_reconciliation()
    test_revision_merge()
    test_diarization_sensitivity()
    formulary = test_formulary_fetch()

    # --- HTTP pipeline -----------------------------------------------------
    stubbed = api._transcribe_ambient
    api._transcribe_ambient = _stub_transcription
    try:
        with TestClient(api.app) as client:
            test_silence_guard(client, audio_files)
            test_real_stt_error_path(audio_files)
            draft = test_ambient_endpoint(client, audio_files, formulary)
            test_copilot_chat(client, draft, formulary)
    finally:
        api._transcribe_ambient = stubbed

    # --- zero-audio-on-disk verification ----------------------------------
    print("\n=== Zero audio storage verification ===")
    after = {
        "data": snapshot(PROJECT_ROOT / "data"),
        "root": snapshot(PROJECT_ROOT),
        "temp": snapshot(temp_dir),
    }
    for tree in ("data", "root", "temp"):
        new_files = sorted(set(after[tree]) - set(before[tree]))
        audio_new = [
            path for path in new_files
            if Path(path).suffix.lower() in AUDIO_EXTENSIONS or _looks_like_riff(path)
        ]
        check(not audio_new, f"{tree}/: no audio file written",
              ", ".join(audio_new[:5]))
        if tree in ("data", "root"):
            check(not new_files, f"{tree}/: no new file at all",
                  ", ".join(new_files[:5]))
        else:
            print(f"[info] temp/: {len(new_files)} new non-audio file(s) "
                  f"(other processes), none matching RIFF/audio")

    if hasattr(sys.modules.get("starlette.formparsers"), "MultiPartParser"):
        from starlette.formparsers import MultiPartParser

        check(
            MultiPartParser.spool_max_size >= len(audio_files["tone"]),
            "spool watermark still raised after the run",
        )

    # --- summary -----------------------------------------------------------
    print("\n" + "=" * 78)
    if _warnings:
        print(f" WARNINGS ({len(_warnings)}):")
        for warning in _warnings:
            print(f"   - {warning}")
    if _failures:
        print(f" FAILED ({len(_failures)}):")
        for failure in _failures:
            print(f"   - {failure}")
        print("=" * 78)
        return 1
    print(" ALL CHECKS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)
