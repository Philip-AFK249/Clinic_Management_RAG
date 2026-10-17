"""FastAPI gateway: BHYT card upload -> Groq VLM extraction -> human verification."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.formparsers import MultiPartParser

from app.core.config import (
    EXTRACTED_IMAGES_DIR,
    PARSED_MARKDOWN_DIR,
    PROJECT_ROOT,
    RAW_DOCS_DIR,
    get_settings,
)
from app.core.logging import get_logger
from app.database.connection import get_connection
from app.database.schedule_queries import (
    CLINIC_CLOSING_TIME,
    CLINIC_TIMEZONE,
    DoctorShiftInfo,
    SessionSchedule,
    query_department_schedule,
)
from app.parsers.bhyt_vlm import (
    BhytData,
    alias_camel_keys,
    extract_bhyt_from_bytes,
    normalise_bhyt_payload,
    normalise_insurance_code,
    optimize_image,
    validate_bhyt,
)
from app.services import ambient, bhyt_store
from app.services.ambient import (
    ClinicalDraft,
    DiarizedTurn,
    IcdDiagnosis,
    PrescriptionItemDraft,
    SoapNote,
)
from app.services.clinical_client import get_available_drugs

logger = get_logger(__name__)

INDEX_HTML = PROJECT_ROOT / "index.html"
PIPELINE_HTML = PROJECT_ROOT / "app" / "static" / "pipeline.html"

# Document types the pipeline console is allowed to ingest.
RAW_DOC_EXTENSIONS = (".pdf", ".docx")

# Hard ceiling for a single document uploaded through the pipeline console.
PIPELINE_UPLOAD_MAX_BYTES = 50 * 1024 * 1024

# Figure/diagram formats LlamaParse may drop into data/extracted_images/.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")

# Voice triage (Stage 1) + doctor dispatch (Stage 2).
TRIAGE_KB_DOC_ID = "Triệu chứng chẩn đoán sớm"
VOICE_ALLOWED_EXTENSIONS = (".wav", ".webm", ".m4a", ".mp3", ".ogg", ".flac")
VOICE_MAX_BYTES = 25 * 1024 * 1024  # Groq's per-file STT ceiling
VALID_DEPARTMENT_IDS = (1, 2, 3)
VALID_PRIORITIES = ("P1", "P2", "P3")

# Privacy rule for the ambient consultation: audio is NEVER written to disk.
# Starlette spools every multipart file part above 1 MB into a SpooledTemporaryFile
# that rolls over into the OS temp directory, which would put a consultation
# recording on disk for as long as the upload is being parsed. Raising the spool
# watermark above the accepted audio ceiling keeps uploads in RAM for their whole
# lifetime, so the only bytes that ever exist are the ones returned by
# ``UploadFile.read()`` - and those are dropped again right after Whisper runs.
MultiPartParser.spool_max_size = VOICE_MAX_BYTES

#: Speech-to-text model mandated for the ambient consultation pipeline.
AMBIENT_STT_MODEL = "whisper-large-v3-turbo"

#: Vocabulary bias for Whisper so it keeps Vietnamese clinical wording intact.
#: The clinical terms lead the prompt so phonetically close words ("dị ứng" vs
#: "dương", "phát ban" vs "pháp bản") survive the first decode pass. The prompt
#: is strictly comma-separated medical vocabulary: any scripted dialogue marker
#: (e.g. "Bác sĩ hỏi:... Bệnh nhân trả lời:...") primes Whisper's repetition
#: hallucination loops such as "Bệnh nhân trả lời, chàm, bệnh nhân trả lời...".
AMBIENT_STT_PROMPT = (
    "Khám lâm sàng, chẩn đoán, phác đồ điều trị, triệu chứng, tiền sử dị ứng, "
    "dị ứng thức ăn, phát ban, mẩn ngứa, bọng nước, mụn nước, chàm, ghẻ, vảy nến, "
    "viêm da cơ địa, hen phế quản, khó thở, ho có đờm, đau ngực, huyết áp, sốt."
)

VOICE_STT_PROMPT = (
    "Khám bệnh, đau thắt ngực, khó thở, huyết áp, hồi hộp, tim mạch, sốt cao, "
    "ho khạc đờm, ho khan, đau rát họng, khò khè, hen suyễn, sổ mũi, viêm xoang, "
    "dị ứng, nổi mẩn đỏ, ngứa ngáy, mụn nước, dát sẩn, vảy nến, zona, sốc phản vệ, "
    "mề đay, tê bàn chân, tiêu chảy, nôn ói."
)

VOICE_TRIAGE_SYSTEM_PROMPT = """Bạn là trợ lý phân loại triệu chứng lâm sàng tại phòng tiếp đón.
Nhiệm vụ: đọc lời kể của bệnh nhân và NGỮ CẢNH TRIỆU CHỨNG được cung cấp, rồi trả về đúng MỘT đối tượng JSON.

Quy tắc bắt buộc:
1. CHỈ dùng thông tin trong [NGỮ CẢNH TRIỆU CHỨNG] để suy ra bệnh và mã ICD-10. Nếu ngữ cảnh không nói, đặt icd10_code là chuỗi rỗng "" - TUYỆT ĐỐI không tự bịa mã bệnh.
2. department_id chỉ được là 1, 2 hoặc 3:
   - 1 = Khoa Nội Tổng quát & Tim mạch (triệu chứng tim mạch, hô hấp, nội khoa chung, tiêu hóa)
   - 2 = Khoa Hô hấp & Dị ứng - Miễn dịch lâm sàng (triệu chứng hô hấp, dị ứng, tai mũi họng, da liễu dị ứng)
   - 3 = Khoa Da liễu (triệu chứng da, tóc, móng)
3. department_name phải khớp đúng tên khoa ứng với department_id.
4. priority_level: "P1" = cấp cứu (đau ngực dữ dội, khó thở nặng, lơ mơ, sốc phản vệ, loét da diện rộng cấp), "P2" = ưu tiên khám trong ngày, "P3" = khám thường.
5. KHÔNG BAO GIỜ suy đoán danh sách bác sĩ, ca trực, phòng khám hay số suất còn trống. Lịch làm việc sẽ được tra cứu từ cơ sở dữ liệu riêng.
6. chief_complaint_summary: tóm tắt triệu chứng chính trong 1-2 câu tiếng Việt để bệnh nhân trình bày với bác sĩ.

Định dạng JSON bắt buộc:
{"disease_guess": "...", "icd10_code": "...", "department_id": 1, "department_name": "...", "priority_level": "P2", "chief_complaint_summary": "..."}"""

# ---------------------------------------------------------------------------
# Runtime model configuration (admin dashboard - hot swappable, no restart)
# ---------------------------------------------------------------------------

#: Whitelists exposed to the admin dashboard. Anything outside these lists is
#: rejected by POST /api/v1/admin/models so a typo cannot poison the gateway.
AVAILABLE_MODELS: dict[str, list[str]] = {
    "llm_models": [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ],
    "stt_models": [
        "whisper-large-v3",
        "whisper-large-v3-turbo",
    ],
    "vlm_models": [
        "qwen/qwen3.8-27b",
    ],
}

_DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_STT_MODEL = "whisper-large-v3"
_DEFAULT_VLM_MODEL = "qwen/qwen3.8-27b"

_runtime_lock = threading.Lock()


def _default_runtime_models() -> dict[str, str]:
    """Seed the runtime holder from .env/settings, falling back to the defaults."""
    settings = get_settings()
    return {
        "llm_model": settings.LLM_MODEL_ID or _DEFAULT_LLM_MODEL,
        "stt_model": settings.LLM_STT_MODEL_ID or _DEFAULT_STT_MODEL,
        "vlm_model": settings.VLM_MODEL_ID or _DEFAULT_VLM_MODEL,
    }


#: Live model selection shared by every Groq invocation (LLM / STT / VLM).
_RUNTIME_MODELS: dict[str, str] = _default_runtime_models()


def _runtime_model(key: str, fallback: str) -> str:
    """Thread-safe read of the active model for ``key``."""
    with _runtime_lock:
        return _RUNTIME_MODELS.get(key) or fallback


class ModelConfigPayload(BaseModel):
    """Requested LLM / STT / VLM selection from the admin dashboard."""

    llm_model: str = Field(min_length=1)
    stt_model: str = Field(min_length=1)
    vlm_model: str = Field(min_length=1)


def _warm_up_knowledge_embedder() -> None:
    """Pre-load BAAI/bge-m3 in the background (skipped when pgvector is down).

    The ambient copilot's Step 4 searches ``clinic_knowledge_nodes`` with bge-m3;
    loading the model inside the first consultation would stall the visit for
    roughly a minute, so it is paid once at boot instead.
    """
    settings = get_settings()
    try:
        if not ambient.pgvector_reachable(settings):
            logger.info("Embedder warm-up skipped: pgvector is unreachable.")
            return
        from app.ingestion.embedder import warm_up_embedder

        warm_up_embedder(settings)
        logger.info("Knowledge embedder warmed up (%s).", settings.EMBEDDING_MODEL)
    except Exception as exc:  # never let a warm-up crash the gateway
        logger.warning("Knowledge embedder warm-up failed: %s", exc)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Loaded during startup, not on a side thread: the RAG step is now capped
    # at ambient.RAG_TIMEOUT_SECONDS (2.5s), and a request that arrives while
    # bge-m3 is still loading would wait on the embedder lock, trip that cap
    # and silently lose the guideline context. A warm retrieval costs ~0.6s.
    await asyncio.to_thread(_warm_up_knowledge_embedder)
    yield


app = FastAPI(
    title="Smart Clinic - AI Gateway API",
    version="3.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8008",
        "http://127.0.0.1:8008",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the figures LlamaParse pulled out of the ingested documents so the
# admin console can render them next to their chunks.
EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/images",
    StaticFiles(directory=str(EXTRACTED_IMAGES_DIR)),
    name="extracted_images",
)


class BhytOcrResponse(BaseModel):
    success: bool = True
    data: BhytData
    warnings: list[str] = Field(default_factory=list)
    model: str
    latency_ms: float
    image_preview: str
    image_size_kb: float
    raw_text: str = ""


class BhytSaveRequest(BaseModel):
    """Patient-confirmed payload coming back from the verification table.

    Accepts either the Vietnamese snake_case keys or the camelCase keys used by
    the React frontend / Spring Boot services; both are folded onto snake_case
    before validation so a mixed payload from either client works unchanged.
    """

    ho_ten: str = Field(min_length=1, description="Họ tên bệnh nhân đã xác nhận")
    ma_so_bhyt: str = Field(min_length=1, description="Mã số thẻ BHYT đã xác nhận")
    ngay_sinh: Optional[str] = None
    gioi_tinh: Optional[str] = None
    ma_noi_dkkcb_ban_dau: Optional[str] = None
    noi_kham_chua_benh_ban_dau: Optional[str] = None
    gia_tri_su_dung_tu: Optional[str] = None
    gia_tri_su_dung_den: Optional[str] = None
    con_han: Optional[bool] = None
    ghi_chu: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _accept_camel_case(cls, value):
        if isinstance(value, dict):
            return alias_camel_keys(value)
        return value

    @field_validator("ma_so_bhyt")
    @classmethod
    def _clean_code(cls, value: str) -> str:
        return normalise_insurance_code(value) or ""

    @field_validator("ho_ten")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        return " ".join(value.split())


class BhytSaveResponse(BaseModel):
    success: bool = True
    message: str
    warnings: list[str] = Field(default_factory=list)
    record: dict


class PipelineFileInfo(BaseModel):
    filename: str
    size_kb: float
    ext: str
    has_parsed_md: bool = False


class DatabaseStatus(BaseModel):
    ok: bool
    detail: str


class PipelineFilesResponse(BaseModel):
    raw_files: list[PipelineFileInfo] = Field(default_factory=list)
    parsed_files: list[PipelineFileInfo] = Field(default_factory=list)
    raw_docs_dir: str
    parsed_markdown_dir: str
    database: DatabaseStatus
    embedding_model: str


class PipelineRunRequest(BaseModel):
    """Ask the ingestion pipeline to process exactly one file from raw_docs."""

    filename: str = Field(min_length=1, description="Tên file trong data/raw_docs/")
    parser_type: Literal["llama", "docling"] = "docling"
    force_reparse: bool = False
    clean_doc_first: bool = True
    target_audience_override: Literal["auto", "tiep_don", "bac_si"] = "auto"
    specialty_override: Optional[Literal["tim_mach", "ho_hap_di_ung", "da_lieu", "chung"]] = None

    @field_validator("filename")
    @classmethod
    def _reject_path_traversal(cls, value: str) -> str:
        """Only ever accept a bare filename, never a path."""
        cleaned = value.strip()
        if not cleaned or cleaned in {".", ".."}:
            raise ValueError("Tên file không hợp lệ.")
        if Path(cleaned).name != cleaned or "/" in cleaned or "\\" in cleaned:
            raise ValueError("Chỉ được truyền tên file, không dùng đường dẫn.")
        return cleaned


class PipelineRunResponse(BaseModel):
    success: bool
    doc_id: str = ""
    filename: str = ""
    specialty: str = ""
    target_audience: str = ""
    parser_used: str = ""
    markdown_path: str = ""
    chunks_created: int = 0
    entities_found: int = 0
    rows_deleted: int = 0
    reparsed: bool = False
    duration_ms: float = 0.0
    message: str = ""
    logs: list[str] = Field(default_factory=list)


class VoiceAudioDecodeError(ValueError):
    """The uploaded audio could not be decoded by the speech-to-text service."""


#: Peak amplitude below this (16-bit PCM ≈ -42 dBFS) counts as silence.
_SILENCE_PEAK_THRESHOLD = 500


def _wav_has_no_speech(audio: bytes) -> Optional[bool]:
    """Cheap stdlib silence check for PCM WAV files.

    Whisper *hallucinates* fluent text from silence or noise (it happily
    returns "subscribe to my channel" for a pure tone), which would otherwise
    feed a fabricated transcript into clinical triage. Returns True when the
    file is decodable and silent, False when it carries signal, and None when
    the format is outside what :mod:`wave` can read (letting Groq decide).
    """
    import wave
    from array import array

    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if wav.getcomptype() != "NONE":
                return None
            width = wav.getsampwidth()
            # 8-bit WAV is unsigned and biased around 128 - skip rather than
            # misread it as signed samples.
            if width not in (2, 4):
                return None
            frames = wav.readframes(min(wav.getnframes(), 16000 * 30))
    except (wave.Error, EOFError, ValueError):
        return None

    if not frames:
        return True

    samples = array("h" if width == 2 else "i")
    samples.frombytes(frames[: len(frames) - (len(frames) % samples.itemsize)])
    if not samples:
        return True
    peak = max(max(samples), -min(samples))
    return peak < _SILENCE_PEAK_THRESHOLD


#: WHO ICD-10 morphology: letter, 2 digits, optional sub-code. 'U' is excluded
#: because it is reserved for special-purpose codes, and the LLM is told to
#: return "" rather than guess, so anything malformed is dropped. Shared with
#: the ambient copilot so both pipelines enforce one identical contract.
_ICD10_RE = ambient.ICD10_RE

_DEPARTMENT_NAMES = {
    1: "Khoa Nội Tổng quát & Tim mạch",
    2: "Khoa Hô hấp & Dị ứng - Miễn dịch lâm sàng",
    3: "Khoa Da liễu",
}


#: Whisper reliably hallucinates these stock phrases when fed silence, room tone
#: or noise. Left unchecked they look like plausible Vietnamese and get fed
#: straight into clinical triage, so they are rejected as non-clinical audio.
#: Examples observed in this project: "Hãy subscribe cho kênh Ghiền Mì Gõ",
#: "Hãy subscribe cho kênh La La La School", "Hẹn gặp lại các bạn".
_WHISPER_HALLUCINATION_PATTERNS = (
    re.compile(r"subscribe", re.IGNORECASE),
    re.compile(r"ghi[eề]n m[iì] g[oõ]", re.IGNORECASE),
    re.compile(r"la\s*la\s*school", re.IGNORECASE),
    re.compile(r"h[eẹ]n g[ặa]p l[ạa]i", re.IGNORECASE),
    re.compile(r"c[ảa]m\s*[ơo]n\s*b[ạa]n", re.IGNORECASE),
    re.compile(r"^\W*$"),
)

#: Anything shorter than this carries no usable clinical content.
_MIN_TRANSCRIPTION_CHARS = 5

#: Terms that make a short transcript worth trusting despite its brevity.
_CLINICAL_HINTS = (
    "đau", "sốt", "ho", "khó thở", "khó thở", "ngứa", "nổi", "mẩn", "chảy máu",
    "sau khi", "nhiều ngày", "buồn nôn", "nôn", "tiêu chảy", "mệt", "chóng mặt",
)

_NO_SPEECH_MESSAGE = (
    "Không phát hiện âm thanh triệu chứng rõ ràng. "
    "Vui lòng thử lại và nói to hơn vào micro."
)


def _looks_like_whisper_hallucination(text: str) -> bool:
    """True when a transcript is a known Whisper phantom or clinically empty."""
    stripped = text.strip()
    for pattern in _WHISPER_HALLUCINATION_PATTERNS:
        if pattern.search(stripped):
            return True
    lowered = stripped.lower()
    if len(stripped) < _MIN_TRANSCRIPTION_CHARS and not any(
        hint in lowered for hint in _CLINICAL_HINTS
    ):
        return True
    return False


def _transcribe_voice(audio: bytes, filename: str, settings) -> str:
    """Stage 1a: Groq Whisper STT, Vietnamese, with the clinic vocabulary prompt."""
    from groq import BadRequestError
    from groq import Groq

    if not settings.GROQ_API_KEY:
        raise VoiceAudioDecodeError("Thiếu GROQ_API_KEY trong .env.")

    client = Groq(api_key=settings.GROQ_API_KEY)
    stt_model = _runtime_model("stt_model", settings.LLM_STT_MODEL_ID)
    logger.info("STT request using model '%s'", stt_model)
    try:
        completion = client.audio.transcriptions.create(
            file=(filename, audio),
            model=stt_model,
            language="vi",
            prompt=VOICE_STT_PROMPT,
            response_format="text",
        )
    except BadRequestError as exc:
        raise VoiceAudioDecodeError(
            f"Không giải mã được tệp audio ({filename}): {exc}"
        ) from exc

    text = (completion or "").strip()
    if not text or _looks_like_whisper_hallucination(text):
        logger.warning(
            "Rejected transcription as hallucination/noise: %r", (text or "")[:120]
        )
        raise VoiceAudioDecodeError(_NO_SPEECH_MESSAGE)
    return text


def _transcribe_ambient(audio: bytes, filename: str, settings) -> str:
    """Ambient Step 1: Groq Whisper on the in-memory buffer, nothing persisted.

    ``audio`` is the raw bytes object handed over by ``UploadFile.read()``; it is
    passed straight to Groq and the caller drops its reference the moment this
    returns, so no audio ever reaches ``data/`` or the OS temp directory.
    """
    from groq import BadRequestError
    from groq import Groq

    if not settings.GROQ_API_KEY:
        raise VoiceAudioDecodeError("Thiếu GROQ_API_KEY trong .env.")

    client = Groq(api_key=settings.GROQ_API_KEY)
    logger.info("Ambient STT request using model '%s'", AMBIENT_STT_MODEL)
    try:
        completion = client.audio.transcriptions.create(
            file=(filename, audio),
            model=AMBIENT_STT_MODEL,
            language="vi",
            prompt=AMBIENT_STT_PROMPT,
            response_format="text",
        )
    except BadRequestError as exc:
        raise VoiceAudioDecodeError(
            f"Không giải mã được tệp audio ({filename}): {exc}"
        ) from exc

    text = (completion or "").strip()
    if not text or _looks_like_whisper_hallucination(text):
        logger.warning(
            "Rejected ambient transcription as hallucination/noise: %r",
            (text or "")[:120],
        )
        raise VoiceAudioDecodeError(_NO_SPEECH_MESSAGE)
    return text


def _resolve_ambient_filename(file: UploadFile) -> str:
    """Audio filename for Whisper; browser blobs often arrive without a suffix."""
    filename = (file.filename or "").strip()
    extension = Path(filename).suffix.lower()

    if extension:
        if extension not in VOICE_ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Định dạng audio không hỗ trợ: {extension}. "
                f"Chấp nhận: {', '.join(VOICE_ALLOWED_EXTENSIONS)}",
            )
        return filename

    content_type = (file.content_type or "").lower()
    if "webm" in content_type:
        extension = ".webm"
    elif "wav" in content_type:
        extension = ".wav"
    elif "mp4" in content_type or "m4a" in content_type:
        extension = ".m4a"
    elif "ogg" in content_type:
        extension = ".ogg"
    elif "mp3" in content_type or "mpeg" in content_type:
        extension = ".mp3"
    else:
        extension = ".webm"  # Mặc định của Web MediaRecorder
    return f"consultation{extension}"


def _ambient_llm_model(settings) -> str:
    """Chat model for the ambient pipeline: gpt-oss-120b or llama-3.3-70b."""
    active = _runtime_model("llm_model", settings.LLM_MODEL_ID)
    if active in ambient.PREFERRED_LLM_MODELS:
        return active
    return ambient.PREFERRED_LLM_MODELS[0]


def _clinical_profile(
    department_id: int,
    allergies: str,
    patient_history: str,
    doctor_id: Optional[int] = None,
    patient_id: Optional[int] = None,
    vitals: Optional[ambient.VitalSigns] = None,
) -> str:
    """Compact text block fed to every ambient prompt."""
    lines = []
    department_name = _DEPARTMENT_NAMES.get(department_id, "")
    lines.append(
        f"department_id={department_id}" + (f" ({department_name})" if department_name else "")
    )
    if doctor_id is not None:
        lines.append(f"doctor_id={doctor_id}")
    if patient_id is not None:
        lines.append(f"patient_id={patient_id}")
    lines.append(f"allergies={(allergies or '').strip() or '(chưa rõ)'}")
    lines.append(f"patient_history={(patient_history or '').strip() or '(chưa có)'}")
    if vitals is not None:
        lines.append("[DẤU HIỆU SINH TỒN]")
        lines.append(vitals.format_block())
    return "\n".join(lines)


def _ambient_rag_query(diarization, raw_transcript: str) -> str:
    """Guideline lookup query: complaints/symptoms beat the raw transcript."""
    parts = diarization.chief_complaints + diarization.symptoms
    if not parts:
        return raw_transcript[:600]
    return ("Phác đồ điều trị " + "; ".join(parts))[:800]


async def _retrieve_knowledge_safely(
    query: str, settings
) -> tuple[str, list[str], list[str]]:
    """RAG retrieval with a hard ceiling; a slow/down cluster degrades to a warning.

    ``asyncio.wait_for`` cannot kill the worker thread, but it does stop the
    request from hanging while a cold bge-m3 load finishes in the background.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(ambient.retrieve_knowledge_context, query, settings),
            timeout=ambient.RAG_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning(
            "Knowledge retrieval exceeded %gs; continuing without RAG",
            ambient.RAG_TIMEOUT_SECONDS,
        )
        return (
            "",
            [],
            [
                f"Truy xuất phác đồ quá {ambient.RAG_TIMEOUT_SECONDS:g} giây; "
                "bỏ qua RAG để không làm chậm buổi khám."
            ],
        )
    except Exception as exc:  # pragma: no cover - defensive
        detail = f"{type(exc).__name__}: {exc}".strip().splitlines()[0][:200]
        logger.warning("Knowledge retrieval failed: %s", detail)
        return "", [], [f"Không truy xuất được phác đồ Bộ Y tế ({detail})."]


def _retrieve_triage_context(transcription: str, settings) -> tuple[str, list[str], list[str]]:
    """Stage 1b: Fast clinical triage context (Bypasses local 2.2GB bge-m3 download/CPU inference)."""
    context = (
        "B\u1ea2NG PH\u00c2N LU\u1ed2NG CHUY\u00caN KHOA PH\u00d2NG KH\u00c1M:\n"
        "- Khoa 1: N\u1ed9i T\u1ed5ng qu\u00e1t & Tim m\u1ea1ch (departmentId = 1). C\u00e1c b\u1ec7nh: \u0110au th\u1eaft ng\u1ef1c (I21), \u0110\u1ed9t qu\u1ef5 (I63), "
        "T\u0103ng huy\u1ebft \u00e1p k\u1ecbch ph\u00e1t (I10), Suy tim (I50), R\u1ed1i lo\u1ea1n nh\u1ecbp tim (I49), Tr\u00e0o ng\u01b0\u1ee3c d\u1ea1 d\u00e0y (K21), "
        "\u0110\u00e1i th\u00e1o \u0111\u01b0\u1eddng (E11), C\u01a1n g\u00fat c\u1ea5p (M10), B\u1ec7nh th\u1eadn m\u1ea1n (N18).\n"
        "- Khoa 2: H\u00f4 h\u1ea5p & D\u1ecb \u1ee9ng - Mi\u1ec5n d\u1ecbch l\u00e2m s\u00e0ng / Tai M\u0169i H\u1ecdng (departmentId = 2). C\u00e1c b\u1ec7nh: S\u1ed1c ph\u1ea3n v\u1ec7 (T78.2), "
        "Hen ph\u1ebf qu\u1ea3n c\u1ea5p (J45), \u0110\u1ee3t c\u1ea5p COPD (J44.0), Vi\u00eam ph\u1ed5i (J18), \u00c1p xe amidan (J36), Vi\u00eam tai gi\u1eefa (H66), "
        "Vi\u00eam h\u1ecdng c\u1ea5p (J02), Vi\u00eam m\u0169i xoang (J01), Vi\u00eam m\u0169i d\u1ecb \u1ee9ng (J30), Vi\u00eam ph\u1ebf qu\u1ea3n (J20).\n"
        "- Khoa 3: Da li\u1ec5u (departmentId = 3). C\u00e1c b\u1ec7nh: Stevens-Johnson (L51.2), \u0110\u1ecf da to\u00e0n th\u00e2n (L53.9), "
        "Zona th\u1ea7n kinh (B02), \u00c1p xe da / Nh\u1ecdt (L02), Ch\u1ed1c l\u1edf (L01), Vi\u00eam da ti\u1ebfp x\u00fac (L23), V\u1ea3y n\u1ebfn (L40.0), "
        "Gh\u1ebb (B86), N\u1ea5m da (B35), Tr\u1ee9ng c\u00e1 (L70.0)."
    )
    sources = ["Tri th\u1ee9c l\u00e2m s\u00e0ng chu\u1ea9n B\u1ed9 Y t\u1ebf (Fast-Triage)"]
    warnings: list[str] = []
    return (context, sources, warnings)


def _sanitise_triage_payload(payload: dict) -> tuple[dict, list[str]]:
    """Validate the LLM's JSON against the clinical contract before using it.

    The LLM is untrusted: an out-of-range department, an unknown priority or a
    malformed ICD-10 code is corrected (and reported) rather than passed on.
    """
    warnings: list[str] = []

    def _text(key: str, limit: int = 500) -> str:
        value = payload.get(key)
        if not isinstance(value, str):
            value = "" if value is None else str(value)
        return " ".join(value.split())[:limit].strip()

    try:
        department_id = int(payload.get("department_id", 0))
    except (TypeError, ValueError):
        department_id = 0
    if department_id not in VALID_DEPARTMENT_IDS:
        warnings.append(
            f"department_id không hợp lệ ({payload.get('department_id')!r}); mặc định về khoa 1."
        )
        department_id = 1

    priority = _text("priority_level", 8).upper().replace(" ", "")
    if priority not in VALID_PRIORITIES:
        # Bias towards urgency rather than down-triage an unparseable answer.
        warnings.append(
            f"priority_level không hợp lệ ({payload.get('priority_level')!r}); mặc định P2."
        )
        priority = "P2"

    icd10 = _text("icd10_code", 16).upper().replace(" ", "")
    if icd10 and not _ICD10_RE.match(icd10):
        warnings.append(f"Mã ICD-10 không đúng định dạng ({icd10!r}); đã bỏ qua.")
        icd10 = ""

    # The departments table is authoritative; the LLM only proposes a name.
    department_name = _DEPARTMENT_NAMES[department_id]

    return (
        {
            "disease_guess": _text("disease_guess", 300) or "Chưa xác định",
            "icd10_code": icd10,
            "department_id": department_id,
            "department_name": department_name,
            "priority_level": priority,
            "chief_complaint_summary": _text("chief_complaint_summary", 400),
        },
        warnings,
    )


def _extract_triage_facts(transcription: str, context: str, settings) -> tuple[dict, list[str]]:
    """Stage 1c: force the LLM into a single JSON object and validate it."""
    from app.rag.generator import get_groq_client

    user_prompt = (
        f"[NGỮ CẠNH TRIỆU CHỨNG]\n{context}\n\n"
        f"[LỜI KỂ BỆNH NHÂN]\n{transcription}"
    )
    llm_model = _runtime_model("llm_model", settings.LLM_MODEL_ID)
    raw = get_groq_client().complete(
        VOICE_TRIAGE_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.1,
        response_format={"type": "json_object"},
        model=llm_model,
    )
    logger.info("Triage LLM request using model '%s'", llm_model)

    text = (raw or "").strip()
    if text.startswith("```"):  # strip an accidental markdown fence
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned non-JSON triage output: %s", text[:200])
        warnings = [f"Không đọc được JSON từ mô hình ({exc.msg}); dùng giá trị mặc định."]
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return _sanitise_triage_payload(payload)


#: Prepended to the advice whenever the clinic has already closed (17:00) and
#: the schedule lookup was rolled over to tomorrow.
ROLLOVER_ADVICE_PREFIX = (
    "Đã qua giờ tiếp nhận hôm nay (sau 17:00). "
    "Hệ thống đã tự động chuyển lịch khám sang ngày mai. "
)

#: Surfaced in the triage response warnings when that automatic rollover fired.
ROLLOVER_WARNING = "Lịch khám đã tự động chuyển sang ngày hôm sau do đã quá 17:00."


def _schedule_rolled_to_tomorrow(schedule: dict[str, Any]) -> bool:
    """True when reception has closed (>= 17:00) and the roster moved to tomorrow.

    ``schedule`` is the payload returned by ``query_department_schedule``; its
    ``target_date`` is already normalized, so comparing it against tomorrow in
    the clinic timezone is enough to detect the automatic rollover.
    """
    if not isinstance(schedule, dict):
        return False
    target_date = str(schedule.get("target_date") or "")
    if not target_date:
        return False
    now_in_clinic = datetime.now(CLINIC_TIMEZONE)
    if now_in_clinic.time() < CLINIC_CLOSING_TIME:
        return False
    tomorrow = (now_in_clinic.date() + timedelta(days=1)).isoformat()
    return target_date == tomorrow


def _build_advice(
    priority: str,
    recommended_shift: str,
    schedule: dict[str, Any],
    department_name: str,
    connected: bool = True,
) -> str:
    """Deterministic advice - never generated by the LLM, never invents a roster."""
    # Post-17:00 requests were rolled to tomorrow: say so up front and make the
    # recommended shift explicitly refer to tomorrow's roster.
    rolled_over = _schedule_rolled_to_tomorrow(schedule)
    notice = ROLLOVER_ADVICE_PREFIX if rolled_over else ""
    day_label = " ngày mai" if rolled_over else ""

    if not connected:
        return notice + (
            "Hệ thống đã nhận diện triệu chứng và đề xuất chuyên khoa thành công. "
            "Lịch trực bác sĩ đang được cập nhật, vui lòng chọn ca khám bên dưới "
            "hoặc liên hệ quầy tiếp đón."
        )

    shift_labels = {
        "MORNING": f"ca sáng (07:30 - 11:30){day_label}",
        "AFTERNOON": f"ca chiều (13:00 - 17:00){day_label}",
        "NEXT_DAY": "ngày hôm sau",
        "NO_DUTY": "chưa có lịch khám",
        "MANUAL_PICK": "thủ công",
    }
    slot = shift_labels.get(recommended_shift, recommended_shift)

    if priority == "P1":
        return notice + (
            f"Triệu chứng thuộc nhóm CẤP CỨU (P1). Đề nghị gọi cấp cứu 115 hoặc đưa người bệnh "
            f"vào khoa cấp cứu ngay, không chờ khám định kỳ. Nhánh tiếp đón sẽ ưu tiên "
            f"hỗ trợ tại {department_name}. Lịch khám chỉ có ý nghĩa sau khi bệnh nhân "
            f"đã được đánh giá đủ an toàn để khám ngoại trú."
        )

    if recommended_shift == "NO_DUTY":
        return notice + (
            f"{department_name} không có lịch khám ngoại trú vào ngày cần xếp. "
            "Đề nghị liên hệ phòng tiếp đón hoặc chọn ngày làm việc kế tiếp. "
            "Mức độ ưu tiên: " + priority + "."
        )

    if recommended_shift == "NEXT_DAY":
        return notice + (
            f"Cả ca sáng và ca chiều tại {department_name} đã đủ bệnh trong ngày cần xếp. "
            "Đề nghị đặt lịch vào ngày làm việc kế tiếp. Mức độ ưu tiên: " + priority + "."
        )

    # Lấy thông tin ca sáng an toàn (chấp nhận cả key "morning" và "MORNING", tránh KeyError)
    morning_data = schedule.get("morning") or schedule.get("MORNING") or {}
    morning_open = False
    if isinstance(morning_data, dict):
        morning_open = not morning_data.get("is_full", True)

    if priority == "P2":
        return notice + (
            f"Nên khám trong ngày tại {department_name}. Đề nghị ưu tiên {slot} "
            f"(còn trống). Nếu triệu chứng nặng lên như khó thở dữ dội, đau ngực "
            "dữ dội hoặc lơ mơ thì chuyển sang cấp cứu ngay."
        )
    return notice + (
        f"Có thể khám theo lịch thường tại {department_name}, ưu tiên {slot}"
        + (" nếu bệnh nhân thuận lợi hơn." if morning_open else ".")
        + " Đề nghị tái khám nếu triệu chứng không cải thiện sau 3-5 ngày hoặc nặng thêm."
    )


def _run_voice_triage(
    audio: bytes, filename: str, target_date: Optional[str]
) -> VoiceScheduleTriageResponse:
    """Blocking two-stage body, executed off the event loop."""
    started = time.perf_counter()
    settings = get_settings()

    transcription = _transcribe_voice(audio, filename, settings)
    logger.info("STT (%d chars): %s", len(transcription), transcription[:120])

    context, sources, warnings = _retrieve_triage_context(transcription, settings)
    facts, fact_warnings = _extract_triage_facts(transcription, context, settings)
    warnings.extend(fact_warnings)

    # Stage 2: deterministic. The inferred department id is the only input.
    # query_department_schedule degrades instead of raising, so this call always
    # returns and the client always gets HTTP 200. The target date is passed
    # straight through so normalise_target_date can apply the post-17:00
    # rollover to tomorrow; its normalized value comes back in the payload.
    schedule_payload = query_department_schedule(
        facts["department_id"], target_date, settings
    )
    connected = bool(schedule_payload.get("is_available"))
    recommended_shift = schedule_payload["recommended_shift"]
    schedule: dict[str, Optional[SessionSchedule]] = {}
    if connected:
        for label in ("morning", "afternoon"):
            data = schedule_payload.get(label)
            if data:
                schedule[label] = SessionSchedule(**data)
    else:
        schedule["morning"] = None
        schedule["afternoon"] = None
        warnings.append(schedule_payload.get("error_message") or "")
        if schedule_payload.get("detail"):
            warnings.append(f"Chi tiết lỗi lịch trực: {schedule_payload['detail']}")

    if _schedule_rolled_to_tomorrow(schedule_payload):
        # schedule_payload["target_date"] already carries the next-day date.
        warnings.append(ROLLOVER_WARNING)

    advice = _build_advice(
        facts["priority_level"],
        recommended_shift,
        schedule_payload,
        schedule_payload["department_name"],
        connected=connected,
    )
    facts["department_name"] = schedule_payload["department_name"]

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "Voice triage: dept=%s priority=%s icd=%s shift=%s target_date=%s "
        "schedule_connected=%s (%.0f ms)",
        facts["department_id"],
        facts["priority_level"],
        facts["icd10_code"] or "-",
        recommended_shift,
        schedule_payload.get("target_date", "-"),
        connected,
        latency_ms,
    )
    return VoiceScheduleTriageResponse(
        success=True,
        transcription=transcription,
        recommended_shift=recommended_shift,
        advice=advice,
        schedule=schedule,
        schedule_connected=connected,
        schedule_error=schedule_payload.get("error_message", "") if not connected else "",
        target_date=schedule_payload["target_date"],
        rag_sources=sources,
        warnings=warnings,
        latency_ms=latency_ms,
        **facts,
    )


class VoiceScheduleTriageResponse(BaseModel):

    success: bool = True
    transcription: str
    disease_guess: str
    icd10_code: str
    department_id: int
    department_name: str
    priority_level: str
    chief_complaint_summary: str
    recommended_shift: str
    advice: str
    schedule: dict[str, Optional[SessionSchedule]] = Field(
        default_factory=dict,
        description="morning/afternoon are null when the schedule service is unreachable",
    )
    schedule_connected: bool = Field(
        default=True,
        description="False = Stage 2 degraded; clinical triage is still valid",
    )
    schedule_error: str = ""
    target_date: str = ""
    rag_sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


@app.get("/", include_in_schema=False)
async def serve_playground():
    """Serve the BHYT upload & human-verification playground."""
    if not INDEX_HTML.exists():
        logger.error("Không tìm thấy %s", INDEX_HTML)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy giao diện index.html.",
        )
    return FileResponse(INDEX_HTML)


@app.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "service": "Smart Clinic AI Gateway",
        "vlm_model": _runtime_model("vlm_model", settings.VLM_MODEL_ID),
        "confirmed_records": bhyt_store.count_records(),
    }


async def _run_ocr(file: UploadFile) -> BhytOcrResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vui lòng tải lên tệp ảnh hợp lệ (.jpg, .jpeg, .png, .heic, .webp)",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Tệp ảnh rỗng."
        )

    start = time.perf_counter()
    settings = get_settings()
    vlm_model = _runtime_model("vlm_model", settings.VLM_MODEL_ID)
    try:
        # Optimize once so the preview the patient checks against is exactly the
        # pixels the VLM saw; then run the (blocking) Groq call off the event loop.
        optimized = await asyncio.to_thread(optimize_image, image_bytes)
        data, raw_text, _ = await asyncio.to_thread(
            extract_bhyt_from_bytes, image_bytes, optimized, vlm_model
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:  # Groq auth/ratelimit/network failures
        logger.exception("Lỗi khi bóc tách thẻ BHYT bằng VLM: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Không thể gọi dịch vụ Groq AI: {exc}",
        ) from exc

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info("Hoàn tất bóc tách BHYT sau %s ms", elapsed_ms)

    return BhytOcrResponse(
        data=data,
        warnings=validate_bhyt(data),
        model=vlm_model,
        latency_ms=elapsed_ms,
        image_preview="data:image/jpeg;base64,"
        + base64.b64encode(optimized).decode("utf-8"),
        image_size_kb=round(len(optimized) / 1024, 1),
        raw_text=raw_text,
    )


@app.post("/api/v1/ocr/bhyt", response_model=BhytOcrResponse)
async def process_bhyt_upload(file: UploadFile = File(...)):
    """Extract structured fields from an uploaded BHYT card via the Groq VLM."""
    try:
        return await _run_ocr(file)
    finally:
        await file.close()


@app.post("/api/v1/chat", response_model=BhytOcrResponse)
async def chat_alias(file: UploadFile = File(...)):
    """Backwards-compatible alias for clients built against /api/v1/chat."""
    try:
        return await _run_ocr(file)
    finally:
        await file.close()


@app.post("/api/v1/bhyt/save", response_model=BhytSaveResponse)
async def save_bhyt_record(payload: BhytSaveRequest):
    """Persist the payload the patient confirmed in the verification table."""
    data = normalise_bhyt_payload(payload.model_dump())
    if not data.ho_ten:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Họ và tên không được để trống.",
        )
    if not data.ma_so_bhyt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Mã số BHYT không được để trống.",
        )

    warnings = validate_bhyt(data)
    record = bhyt_store.save_confirmed_record(data.model_dump())

    return BhytSaveResponse(
        message="Đã xác nhận và lưu thông tin thẻ BHYT thành công.",
        warnings=warnings,
        record=record,
    )


class _LogCollector(logging.Handler):
    """Buffers `app.*` log records so the UI can show the run's log afterwards."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("app"):
            return
        try:
            self.records.append(self.format(record))
        except Exception:  # never let logging break the request
            pass


@contextmanager
def _capture_logs():
    """Tee application logs into a list for the duration of the block."""
    collector = _LogCollector()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(collector)
    root.setLevel(logging.INFO)
    try:
        yield collector
    finally:
        root.removeHandler(collector)
        root.setLevel(previous_level)


def _file_size_kb(path: Path) -> float:
    return round(path.stat().st_size / 1024, 1)


def _check_database(settings) -> DatabaseStatus:
    """Liveness probe with a hard 3s timeout so the UI cannot hang on page load.

    libpq has no default connect timeout, so the probe sets one explicitly
    instead of going through ``get_connection`` (which leaves config untouched).
    """
    import psycopg

    try:
        with psycopg.connect(settings.db_dsn, connect_timeout=3, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return DatabaseStatus(ok=True, detail="Đã kết nối")
    except Exception as exc:
        return DatabaseStatus(ok=False, detail=str(exc).strip().splitlines()[0][:200])


def _resolve_raw_file(filename: str) -> Path:
    """Map a bare filename onto data/raw_docs/, refusing anything that escapes it."""
    candidate = (RAW_DOCS_DIR / filename).resolve()
    try:
        candidate.relative_to(RAW_DOCS_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Đường dẫn không hợp lệ.",
        )
    if candidate.suffix.lower() not in RAW_DOC_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chỉ hỗ trợ {' / '.join(RAW_DOC_EXTENSIONS)}.",
        )
    if not candidate.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy file '{filename}' trong data/raw_docs/.",
        )
    return candidate


def _parse_to_markdown(source: Path, parser_type: str) -> tuple[Path, str]:
    """Parse ``source`` into data/parsed_markdown/; returns (md_path, parser_used)."""
    # Heavy imports stay inside the worker thread: pulling in sentence-transformers
    # and docling at module scope would add ~30s to API startup.
    target_dir = PARSED_MARKDOWN_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if parser_type == "llama":
        from app.parsers.llamaparse_parser import LlamaParseParser

        try:
            parser = LlamaParseParser(output_dir=target_dir, settings=get_settings())
        except Exception as exc:
            logger.warning("LlamaParse unavailable (%s) - falling back to Docling", exc)
            return _parse_to_markdown(source, "docling")
        parser.parse(source)
        return target_dir / f"{source.stem}.md", "llama"

    from app.parsers.docling_parser import DoclingParser

    DoclingParser(output_dir=target_dir).parse(source)
    return target_dir / f"{source.stem}.md", "docling"


def _run_single_pipeline(payload: PipelineRunRequest) -> PipelineRunResponse:
    """Blocking pipeline body; always executed off the event loop."""
    from app.ingestion.pipeline import (
        classify_document,
        ingest_markdown_file,
        purge_document,
    )

    started = time.perf_counter()
    settings = get_settings()
    source = None
    doc_id = Path(payload.filename).stem
    markdown_path = PARSED_MARKDOWN_DIR / f"{doc_id}.md"
    parser_used = payload.parser_type

    with _capture_logs() as collector:
      try:
        source = _resolve_raw_file(payload.filename)
        doc_id = source.stem
        markdown_path = PARSED_MARKDOWN_DIR / f"{doc_id}.md"
        logger.info("=== Pipeline run: %s ===", source.name)
        logger.info("Source: %s (%.1f KB)", source, _file_size_kb(source))

        if markdown_path.exists() and not payload.force_reparse:
            logger.info("Reusing existing Markdown %s (force_reparse=false)", markdown_path.name)
        else:
            logger.info("Parsing with %s -> data/parsed_markdown/%s.md", payload.parser_type, doc_id)
            markdown_path, parser_used = _parse_to_markdown(source, payload.parser_type)
            if not markdown_path.is_file():
                raise RuntimeError(f"Bộ phân tích không tạo ra file {markdown_path.name}.")
            logger.info("Parsed Markdown ready (%d chars)", len(markdown_path.read_text(encoding="utf-8")))

        auto_specialty, auto_audience = classify_document(doc_id)
        specialty = payload.specialty_override or auto_specialty
        target_audience = (
            auto_audience
            if payload.target_audience_override == "auto"
            else payload.target_audience_override
        )
        logger.info("Routing -> specialty=%s | target_audience=%s", specialty, target_audience)

        rows_deleted = 0
        if payload.clean_doc_first:
            rows_deleted = purge_document(doc_id, settings)
        else:
            logger.warning("clean_doc_first=false: existing chunks of this doc are kept.")

        result = ingest_markdown_file(
            markdown_path,
            doc_id,
            specialty,
            target_audience,
            settings,
            migrate=True,
        )

        message = (
            f"Đã nạp {result.total_chunks} chunk cho '{doc_id}' "
            f"[{specialty} / {target_audience}]."
        )
        logger.info(message)
        return PipelineRunResponse(
            success=True,
            doc_id=doc_id,
            filename=source.name,
            specialty=specialty,
            target_audience=target_audience,
            parser_used=parser_used,
            markdown_path=str(markdown_path.relative_to(PROJECT_ROOT)),
            chunks_created=result.total_chunks,
            entities_found=result.entities_found,
            rows_deleted=rows_deleted,
            reparsed=bool(payload.force_reparse),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            message=message,
            logs=collector.records,
        )
      except HTTPException:
        raise
      except Exception as exc:
        # Surface a correct status code while still shipping the partial log.
        logger.exception("Pipeline run failed for %s: %s", payload.filename, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=PipelineRunResponse(
                success=False,
                doc_id=doc_id,
                filename=payload.filename,
                parser_used=parser_used,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                message=f"{type(exc).__name__}: {exc}",
                logs=collector.records,
            ).model_dump(),
        )


@app.get("/pipeline", include_in_schema=False)
async def serve_pipeline_console():
    """Serve the RAG pipeline controller (vanilla HTML, no Streamlit)."""
    if not PIPELINE_HTML.exists():
        logger.error("Không tìm thấy %s", PIPELINE_HTML)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy giao diện pipeline.html.",
        )
    return FileResponse(PIPELINE_HTML)


@app.get("/api/v1/pipeline/files", response_model=PipelineFilesResponse)
async def list_pipeline_files():
    """List ingestible documents in raw_docs plus already-parsed Markdown."""
    settings = get_settings()
    raw_files: list[PipelineFileInfo] = []
    if RAW_DOCS_DIR.is_dir():
        for path in sorted(RAW_DOCS_DIR.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in RAW_DOC_EXTENSIONS:
                continue
            raw_files.append(
                PipelineFileInfo(
                    filename=path.name,
                    size_kb=_file_size_kb(path),
                    ext=path.suffix.lower(),
                    has_parsed_md=(PARSED_MARKDOWN_DIR / f"{path.stem}.md").is_file(),
                )
            )

    parsed_files: list[PipelineFileInfo] = []
    if PARSED_MARKDOWN_DIR.is_dir():
        for path in sorted(PARSED_MARKDOWN_DIR.glob("*.md")):
            if path.is_file():
                parsed_files.append(
                    PipelineFileInfo(
                        filename=path.name,
                        size_kb=_file_size_kb(path),
                        ext=path.suffix.lower(),
                    )
                )

    return PipelineFilesResponse(
        raw_files=raw_files,
        parsed_files=parsed_files,
        raw_docs_dir=str(RAW_DOCS_DIR.relative_to(PROJECT_ROOT)),
        parsed_markdown_dir=str(PARSED_MARKDOWN_DIR.relative_to(PROJECT_ROOT)),
        database=_check_database(settings),
        embedding_model=settings.EMBEDDING_MODEL,
    )


def _sanitise_upload_filename(filename: str) -> str:
    """Reduce an uploaded name to a safe basename that cannot escape raw_docs."""
    cleaned = Path((filename or "").strip().replace("\\", "/")).name.strip()
    # Drop control characters plus characters that are illegal on Windows.
    cleaned = re.sub(r'[\x00-\x1f<>:"|?*]', "", cleaned).strip()
    return cleaned


@app.post("/api/v1/pipeline/upload")
async def upload_pipeline_file(file: UploadFile = File(...)):
    """Store an uploaded .pdf/.docx into data/raw_docs/ for the ingestion pipeline."""
    filename = _sanitise_upload_filename(file.filename or "")
    if not filename or filename in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên file không hợp lệ.",
        )
    if Path(filename).suffix.lower() not in RAW_DOC_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chỉ hỗ trợ {' / '.join(RAW_DOC_EXTENSIONS)}.",
        )

    destination = RAW_DOCS_DIR / filename
    try:
        destination.resolve().relative_to(RAW_DOCS_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Đường dẫn không hợp lệ.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File rỗng."
        )
    if len(content) > PIPELINE_UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File vượt quá {PIPELINE_UPLOAD_MAX_BYTES // (1024 * 1024)} MB.",
        )

    try:
        RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(destination.write_bytes, content)
    except OSError as exc:
        logger.exception("Không lưu được file upload '%s': %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không lưu được file: {exc}",
        ) from exc
    finally:
        await file.close()

    size_kb = round(len(content) / 1024, 1)
    logger.info("Đã tải lên %s (%.1f KB) -> %s", filename, size_kb, destination)
    return {
        "success": True,
        "filename": filename,
        "size_kb": size_kb,
        "message": "File uploaded successfully to data/raw_docs/.",
    }


@app.post("/api/v1/pipeline/run-single", response_model=PipelineRunResponse)
async def run_single_file(payload: PipelineRunRequest):
    """Parse -> chunk -> embed -> insert one document from raw_docs into pgvector."""
    try:
        return await asyncio.to_thread(_run_single_pipeline, payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline run failed for %s: %s", payload.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/v1/triage/voice-schedule", response_model=VoiceScheduleTriageResponse)
async def voice_schedule_triage(
    file: UploadFile = File(..., description="Ghi âm: .wav, .webm, .m4a, .mp3"),
    target_date: Optional[str] = Form(
        None, description="Ngày cần xếp khám (YYYY-MM-DD). Mặc định: hôm nay Asia/Ho_Chi_Minh."
    ),
):
    """Two-stage voice triage: Whisper + RAG + LLM, then a real roster query."""
    try:
        filename = (file.filename or "").strip()
        extension = Path(filename).suffix.lower()

        # 1. Tự động nhận diện định dạng nếu trình duyệt gửi file blob không có đuôi mở rộng
        if not extension:
            content_type = (file.content_type or "").lower()
            if "webm" in content_type:
                extension = ".webm"
            elif "wav" in content_type:
                extension = ".wav"
            elif "mp4" in content_type or "m4a" in content_type:
                extension = ".m4a"
            elif "ogg" in content_type:
                extension = ".ogg"
            elif "mp3" in content_type or "mpeg" in content_type:
                extension = ".mp3"
            else:
                extension = ".webm"  # Mặc định của Web MediaRecorder
            filename = f"voice_recording{extension}"

        # 2. Kiểm tra định dạng audio hợp lệ
        if extension not in VOICE_ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Định dạng audio không hỗ trợ: {extension or '(không có)'}. "
                f"Chấp nhận: {', '.join(VOICE_ALLOWED_EXTENSIONS)}",
            )

        # 3. Đọc dữ liệu audio
        audio = await file.read()
        if not audio:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Tệp audio rỗng."
            )
        if len(audio) > VOICE_MAX_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File audio vượt quá {VOICE_MAX_BYTES // (1024 * 1024)} MB.",
            )

        # 4. Kiểm tra khoảng lặng (áp dụng cho PCM WAV)
        if _wav_has_no_speech(audio) is True:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Không phát hiện giọng nói trong tệp WAV (âm thanh im lặng hoặc chỉ có nhiễu). "
                "Vui lòng ghi âm lại gần micro hơn.",
            )

        # 5. Thực thi luồng xử lý Voice Triage & Tra cứu lịch trực
        return await asyncio.to_thread(
            _run_voice_triage, audio, filename, target_date
        )

    except HTTPException:
        # Giữ nguyên các mã lỗi HTTP có chủ đích (400, 413, v.v.)
        raise
    except ValueError as exc:
        logger.warning("Voice triage validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        # Bắt toàn bộ lỗi ngoại lệ khác (Groq API, kết nối DB, timeout) để tránh sập ngầm 500
        logger.exception("Lỗi hệ thống khi xử lý voice_schedule_triage: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý Voice Triage: {str(exc)}",
        ) from exc
    finally:
        await file.close()


# ---------------------------------------------------------------------------
# Ambient AI Clinical Copilot for the Doctor EHR portal
# Privacy: consultation audio lives only in RAM for the Whisper call - it is
# never written to data/, the OS temp directory or the database. Only the
# structured textual insights below (transcript, symptoms, SOAP, ICD-10,
# prescription draft) leave this process.
# ---------------------------------------------------------------------------


class AmbientConsultationResponse(BaseModel):
    """Structured outcome of one in-memory ambient consultation."""

    success: bool = True
    raw_transcript: str
    diarized_transcript: list[DiarizedTurn] = Field(
        default_factory=list, description="Câu thoại đã gán vai doctor/patient"
    )
    has_red_flags: bool = False
    red_flag_alert: str = ""
    recommended_inquiry: str = ""
    chief_complaints: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    soap_note: SoapNote
    diagnoses: list[IcdDiagnosis] = Field(default_factory=list)
    prescription_draft: list[PrescriptionItemDraft] = Field(default_factory=list)
    formulary_source: str = Field(
        default="", description="Endpoint hoặc PostgreSQL lấy danh mục thuốc"
    )
    rag_sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class CopilotChatRequest(BaseModel):
    """Doctor follow-up text plus the draft it should update."""

    message: str = Field(
        min_length=1,
        max_length=4000,
        description="Phản hồi/xác nhận của bác sĩ, ví dụ 'Bệnh nhân xác nhận đã nhiễm HIV 2 năm'",
    )

    @field_validator("message")
    @classmethod
    def _clean_message(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Tin nhắn của bác sĩ không được để trống.")
        return cleaned
    soap_note: Optional[SoapNote] = None
    diagnoses: list[IcdDiagnosis] = Field(default_factory=list)
    prescription_draft: list[PrescriptionItemDraft] = Field(default_factory=list)
    raw_transcript: str = ""
    diarized_transcript: list[DiarizedTurn] = Field(default_factory=list)
    allergies: str = ""
    patient_history: str = ""
    department_id: int = 1
    doctor_id: Optional[int] = None
    patient_id: Optional[int] = None


class CopilotChatResponse(BaseModel):
    """Updated SOAP note / ICD-10 list / prescription draft after one message."""

    success: bool = True
    assistant_message: str = ""
    soap_note: SoapNote
    diagnoses: list[IcdDiagnosis] = Field(default_factory=list)
    prescription_draft: list[PrescriptionItemDraft] = Field(default_factory=list)
    formulary_source: str = ""
    rag_sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


def _formulary_warnings(formulary) -> tuple[list, list[str]]:
    """Drugs plus the degradation warning when a formulary source was down."""
    warnings: list[str] = []
    if formulary.error:
        warnings.append(formulary.error)
    return formulary.drugs, warnings


@app.post(
    "/api/v1/clinical/ambient-consultation",
    response_model=AmbientConsultationResponse,
)
async def ambient_consultation(
    audio_file: UploadFile = File(
        ..., description="Ghi âm buổi khám (.wav, .webm, .m4a, .mp3, .ogg, .flac)"
    ),
    department_id: int = Form(1),
    doctor_id: Optional[int] = Form(None),
    patient_id: Optional[int] = Form(None),
    allergies: str = Form("", description='Dị ứng, ví dụ "Penicillin, Aspirin"'),
    patient_history: str = Form(""),
    blood_pressure: str = Form("", description='Huyết áp đo tại phòng, ví dụ "120/80"'),
    heart_rate: Optional[int] = Form(None, description="Mạch (lần/phút)"),
    spo2: Optional[int] = Form(None, description="SpO2 (%)"),
    temperature: str = Form("", description="Nhiệt độ (°C), ví dụ 38.5"),
):
    """Ambient voice consultation, zero audio storage.

    Step 1 transcribes the upload straight out of RAM and drops the bytes;
    Steps 2-5 turn that transcript into a diarized dialogue, a red-flag scan
    (dialogue patterns + deterministic vital-sign thresholds), a SOAP note,
    ICD-10 diagnoses and a prescription draft matched against the live
    in-stock formulary (pgvector guidelines + ClinicalConsultationService
    are retrieved in parallel).
    """
    started = time.perf_counter()
    settings = get_settings()

    try:
        # Rejecting an unsupported upload early still goes through the same
        # close path, so the in-memory buffer is dropped on every branch.
        filename = _resolve_ambient_filename(audio_file)

        # --- Step 1: in-memory speech-to-text ------------------------------
        audio_bytes = await audio_file.read()
        try:
            if not audio_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Tệp audio rỗng."
                )
            if len(audio_bytes) > VOICE_MAX_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File audio vượt quá {VOICE_MAX_BYTES // (1024 * 1024)} MB.",
                )
            if _wav_has_no_speech(audio_bytes) is True:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Không phát hiện giọng nói trong tệp WAV (âm thanh im lặng "
                    "hoặc chỉ có nhiễu). Vui lòng ghi âm lại gần micro hơn.",
                )

            raw_transcript = await asyncio.to_thread(
                _transcribe_ambient, audio_bytes, filename, settings
            )
        finally:
            # Privacy rule: the buffer exists only while Whisper is running.
            del audio_bytes

        logger.info(
            "Ambient STT (%d chars): %s", len(raw_transcript), raw_transcript[:120]
        )

        llm_model = _ambient_llm_model(settings)
        vitals = ambient.VitalSigns(
            blood_pressure=" ".join((blood_pressure or "").split()),
            heart_rate=heart_rate,
            spo2=spo2,
            temperature=" ".join((temperature or "").split()),
        )
        clinical_profile = _clinical_profile(
            department_id, allergies, patient_history, doctor_id, patient_id,
            vitals=vitals,
        )

        # --- Step 2: semantic diarization & role labelling -----------------
        diarization = await asyncio.to_thread(
            ambient.diarize_dialogue,
            raw_transcript,
            clinical_profile,
            llm_model,
            settings,
        )
        logger.info(
            "Ambient step 2: %d turn(s), %d complaint(s), %d symptom(s)",
            len(diarization.turns),
            len(diarization.chief_complaints),
            len(diarization.symptoms),
        )

        # --- Steps 3 + 4 run concurrently: the red-flag scan is an LLM call,
        # the guideline RAG is CPU-bound embedding and the formulary is a DB
        # lookup - none of them depend on each other, so wall-clock time is
        # whichever of the three is slowest rather than their sum.
        red_flags, knowledge_result, formulary = await asyncio.gather(
            asyncio.to_thread(
                ambient.scan_red_flags,
                diarization,
                clinical_profile,
                llm_model,
                settings,
                vitals=vitals,
            ),
            _retrieve_knowledge_safely(
                _ambient_rag_query(diarization, raw_transcript), settings
            ),
            get_available_drugs(settings),
        )
        logger.info("Ambient step 3: has_red_flags=%s", red_flags.has_red_flags)

        knowledge_context, rag_sources, warnings = knowledge_result
        drugs, formulary_warnings = _formulary_warnings(formulary)
        warnings.extend(formulary_warnings)
        logger.info(
            "Ambient step 4: %d guideline chunk(s) [%s] + %d in-stock drug(s) [%s]",
            len(rag_sources),
            ", ".join(rag_sources) or "-",
            len(drugs),
            formulary.source or "unavailable",
        )

        # --- Step 5: SOAP + ICD-10 + prescription draft ---------------------
        draft = await asyncio.to_thread(
            ambient.synthesise_clinical_note,
            diarization,
            red_flags,
            clinical_profile,
            knowledge_context,
            drugs,
            llm_model,
            settings,
        )
        warnings.extend(draft.warnings)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Ambient consultation done: turns=%d red_flags=%s diagnoses=%d rx=%d (%.0f ms)",
            len(diarization.turns),
            red_flags.has_red_flags,
            len(draft.diagnoses),
            len(draft.prescription_draft),
            latency_ms,
        )
        return AmbientConsultationResponse(
            success=True,
            raw_transcript=raw_transcript,
            diarized_transcript=diarization.turns,
            has_red_flags=red_flags.has_red_flags,
            red_flag_alert=red_flags.red_flag_alert,
            recommended_inquiry=red_flags.recommended_inquiry,
            chief_complaints=diarization.chief_complaints,
            symptoms=diarization.symptoms,
            soap_note=draft.soap_note,
            diagnoses=draft.diagnoses,
            prescription_draft=draft.prescription_draft,
            formulary_source=formulary.source,
            rag_sources=rag_sources,
            warnings=warnings,
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except VoiceAudioDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except ambient.AmbientPipelineError as exc:
        logger.error("Ambient pipeline rejected model output: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Không xử lý được phiên khám: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Lỗi hệ thống khi xử lý ambient consultation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý Ambient Consultation: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        # Closing the upload drops the spooled buffer; nothing was ever on disk.
        await audio_file.close()


@app.post("/api/v1/clinical/copilot-chat", response_model=CopilotChatResponse)
async def copilot_chat(payload: CopilotChatRequest):
    """Apply one doctor follow-up message to the current clinical draft.

    Example: *"Bệnh nhân xác nhận đã nhiễm HIV 2 năm trước và đang uống ARV"* -
    the SOAP note, the secondary diagnoses (e.g. ``B20``) and the medication
    contraindications are revised in place, still strictly matched against the
    live in-stock formulary.
    """
    started = time.perf_counter()
    settings = get_settings()
    llm_model = _ambient_llm_model(settings)
    clinical_profile = _clinical_profile(
        payload.department_id,
        payload.allergies,
        payload.patient_history,
        payload.doctor_id,
        payload.patient_id,
    )
    conversation_history = "\n".join(
        f"{turn.speaker}: {turn.text}" for turn in payload.diarized_transcript
    ) or payload.raw_transcript[:1500]

    try:
        knowledge_result, formulary = await asyncio.gather(
            _retrieve_knowledge_safely(payload.message, settings),
            get_available_drugs(settings),
        )
        knowledge_context, rag_sources, warnings = knowledge_result
        drugs, formulary_warnings = _formulary_warnings(formulary)
        warnings.extend(formulary_warnings)

        previous = ClinicalDraft(
            soap_note=payload.soap_note or SoapNote(),
            diagnoses=list(payload.diagnoses),
            prescription_draft=list(payload.prescription_draft),
        )
        draft, assistant_message = await asyncio.to_thread(
            ambient.revise_clinical_note,
            payload.message,
            previous,
            clinical_profile,
            knowledge_context,
            drugs,
            llm_model,
            settings,
            conversation_history,
        )
        warnings.extend(draft.warnings)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Copilot chat: diagnoses=%d rx=%d warnings=%d (%.0f ms)",
            len(draft.diagnoses),
            len(draft.prescription_draft),
            len(warnings),
            latency_ms,
        )
        return CopilotChatResponse(
            success=True,
            assistant_message=assistant_message,
            soap_note=draft.soap_note,
            diagnoses=draft.diagnoses,
            prescription_draft=draft.prescription_draft,
            formulary_source=formulary.source,
            rag_sources=rag_sources,
            warnings=warnings,
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except ambient.AmbientPipelineError as exc:
        logger.error("Copilot chat rejected model output: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Không cập nhật được hồ sơ: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Lỗi hệ thống khi xử lý copilot-chat: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý Copilot Chat: {type(exc).__name__}: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Admin: AI Gateway & RAG Knowledge management
# ---------------------------------------------------------------------------


class ModelConfigResponse(BaseModel):
    """Active models + whitelists + infrastructure health for the dashboard.

    The dashboard reads both key conventions: ``active`` / ``active_models``
    and ``available`` / ``available_options`` (id+name option lists).
    """

    llm_model: str
    stt_model: str
    vlm_model: str
    embedding_model: str
    active: dict[str, str] = Field(default_factory=dict)
    active_models: dict[str, str] = Field(default_factory=dict)
    available: dict[str, list[str]] = Field(default_factory=dict)
    available_options: dict[str, list[dict[str, str]]] = Field(default_factory=dict)
    infrastructure: dict[str, dict[str, Any]] = Field(default_factory=dict)
    infrastructure_status: dict[str, str] = Field(default_factory=dict)


class ModelConfigUpdateResponse(ModelConfigResponse):
    success: bool = True
    message: str = ""


class KnowledgeDocItem(BaseModel):
    id: str
    title: str
    specialty: str
    target_audience: str = ""
    chunkCount: int = 0
    embeddingModel: str = ""
    createdAt: str = ""
    status: str = "INDEXED"


class KnowledgeDocsResponse(BaseModel):
    documents: list[KnowledgeDocItem] = Field(default_factory=list)
    total_docs: int = 0


class KnowledgeChunk(BaseModel):
    # Serialised as text: the deployed clinic_knowledge_nodes uses UUID ids
    # while the local DDL declares SERIAL, so str() covers both.
    id: str
    content: str
    specialty: str = ""
    target_audience: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class KnowledgeChunksResponse(BaseModel):
    doc_id: str
    total_chunks: int = 0
    chunks: list[KnowledgeChunk] = Field(default_factory=list)
    extracted_images: list[str] = Field(default_factory=list)


class KnowledgeDocDeleteResponse(BaseModel):
    success: bool = True
    message: str


class OcrLogItem(BaseModel):
    id: str
    documentType: str = "BHYT_CARD"
    patientName: str = ""
    extractedId: str = ""
    confidenceScore: int = 0
    timestamp: str = ""
    status: str = "SUCCESS"


class OcrLogsResponse(BaseModel):
    logs: list[OcrLogItem] = Field(default_factory=list)
    total: int = 0


KNOWLEDGE_DOCS_SQL = """
SELECT doc_id, specialty, target_audience, COUNT(*) AS chunk_count, MIN(created_at) AS created_at
FROM clinic_knowledge_nodes
GROUP BY doc_id, specialty, target_audience
ORDER BY MIN(created_at) DESC;
"""

CHUNKS_SQL = """
SELECT id, content, specialty, target_audience, metadata, created_at
FROM clinic_knowledge_nodes
WHERE doc_id = %s
ORDER BY id ASC;
"""

#: OCR audit rows kept for the dashboard (newest first).
OCR_LOG_LIMIT = 50


def _infrastructure_health(settings) -> dict[str, dict[str, Any]]:
    """Probe ai_gateway / pgvector / embedding_engine without ever raising."""
    health: dict[str, dict[str, Any]] = {}

    gateway_ok = bool(settings.GROQ_API_KEY)
    health["ai_gateway"] = {
        "status": "healthy" if gateway_ok else "degraded",
        "detail": (
            "Groq API key đã cấu hình" if gateway_ok else "Thiếu GROQ_API_KEY trong .env"
        ),
    }

    database = _check_database(settings)
    health["pgvector"] = {
        "status": "healthy" if database.ok else "down",
        "detail": (
            f"PostgreSQL {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME} - "
            f"{database.detail}"
        ),
    }

    if not settings.EMBEDDING_MODEL:
        embedding_status, embedding_detail = "degraded", "Chưa cấu hình EMBEDDING_MODEL."
    elif not database.ok:
        embedding_status = "degraded"
        embedding_detail = f"Không xác minh được index pgvector ({database.detail})"
    else:
        embedding_status = "healthy"
        embedding_detail = f"{settings.EMBEDDING_MODEL} (dim={settings.EMBEDDING_DIM})"

    health["embedding_engine"] = {"status": embedding_status, "detail": embedding_detail}
    return health


def _models_payload(settings) -> dict[str, Any]:
    """Shape shared by GET/POST /api/v1/admin/models.

    Exposes every key the admin dashboard may read (``llm_model`` flat keys,
    ``active`` / ``active_models`` dicts, ``available`` lists,
    ``available_options`` id+name lists, ``infrastructure`` and the flattened
    ``infrastructure_status``) so the frontend never sees ``undefined``.
    """
    active = {
        "llm_model": _runtime_model("llm_model", settings.LLM_MODEL_ID),
        "stt_model": _runtime_model("stt_model", settings.LLM_STT_MODEL_ID),
        "vlm_model": _runtime_model("vlm_model", settings.VLM_MODEL_ID),
        "embedding_model": settings.EMBEDDING_MODEL,
    }
    infra = _infrastructure_health(settings)
    return {
        **active,
        "active": dict(active),
        "active_models": dict(active),
        "available": {key: list(value) for key, value in AVAILABLE_MODELS.items()},
        "available_options": {
            "llm_models": [{"id": m, "name": m} for m in AVAILABLE_MODELS["llm_models"]],
            "stt_models": [{"id": m, "name": m} for m in AVAILABLE_MODELS["stt_models"]],
            "vlm_models": [{"id": m, "name": m} for m in AVAILABLE_MODELS["vlm_models"]],
        },
        "infrastructure": infra,
        "infrastructure_status": {
            "ai_gateway": infra.get("ai_gateway", {}).get("status", "healthy"),
            "pgvector": infra.get("pgvector", {}).get("status", "healthy"),
            "embedding_device": infra.get("embedding_engine", {}).get("detail", "CPU/CUDA"),
        },
    }


def _db_error_detail(exc: Exception, settings) -> str:
    """Human-readable DB error that never leaks the password back to a client."""
    message = f"{type(exc).__name__}: {exc}".strip()
    if settings.DB_PASSWORD:
        message = message.replace(settings.DB_PASSWORD, "***")
    return message[:300]


@app.get("/api/v1/admin/models", response_model=ModelConfigResponse)
async def get_admin_models():
    """Active LLM/STT/VLM selection, whitelists and infrastructure health."""
    # The pgvector probe inside _infrastructure_health blocks for up to 3s.
    return await asyncio.to_thread(_models_payload, get_settings())


@app.post("/api/v1/admin/models", response_model=ModelConfigUpdateResponse)
async def set_admin_models(payload: ModelConfigPayload):
    """Hot-swap the LLM / STT / VLM models without restarting Uvicorn."""
    settings = get_settings()

    allowed = {
        "llm_model": AVAILABLE_MODELS["llm_models"],
        "stt_model": AVAILABLE_MODELS["stt_models"],
        "vlm_model": AVAILABLE_MODELS["vlm_models"],
    }
    rejected = {
        field: getattr(payload, field)
        for field, options in allowed.items()
        if getattr(payload, field) not in options
    }
    if rejected:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Mô hình không hợp lệ: {rejected}. "
                f"Các giá trị cho phép: {allowed}"
            ),
        )

    with _runtime_lock:
        for field in allowed:
            _RUNTIME_MODELS[field] = getattr(payload, field)
        updated = dict(_RUNTIME_MODELS)

    message = (
        f"Đã chuyển model: LLM={updated['llm_model']}, "
        f"STT={updated['stt_model']}, VLM={updated['vlm_model']}."
    )
    logger.info(message)

    body = await asyncio.to_thread(_models_payload, settings)
    return ModelConfigUpdateResponse(success=True, message=message, **body)


@app.get("/api/v1/admin/knowledge-docs", response_model=KnowledgeDocsResponse)
async def list_knowledge_docs():
    """Grouped view of every document indexed in clinic_knowledge_nodes."""
    settings = get_settings()

    def _fetch() -> list[tuple]:
        with get_connection(settings) as conn:
            with conn.cursor() as cur:
                cur.execute(KNOWLEDGE_DOCS_SQL)
                rows = cur.fetchall()
            conn.commit()
        return rows

    try:
        rows = await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.exception("Không truy vấn được clinic_knowledge_nodes: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Không đọc được kho tri thức từ pgvector: {_db_error_detail(exc, settings)}",
        ) from exc

    documents: list[KnowledgeDocItem] = []
    for doc_id, specialty, target_audience, chunk_count, created_at in rows:
        created = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        documents.append(
            KnowledgeDocItem(
                id=str(doc_id),
                title=str(doc_id),
                specialty=str(specialty or ""),
                target_audience=str(target_audience or ""),
                chunkCount=int(chunk_count or 0),
                embeddingModel=settings.EMBEDDING_MODEL,
                createdAt=created,
                status="INDEXED",
            )
        )

    logger.info("Danh sách tri thức: %d tài liệu", len(documents))
    return KnowledgeDocsResponse(documents=documents, total_docs=len(documents))


def _extracted_image_urls(doc_id: str) -> list[str]:
    """Static URLs for the figures LlamaParse extracted into ``data/extracted_images/{doc_id}``."""
    safe_doc_id = Path((doc_id or "").strip().replace("\\", "/")).name.strip()
    if not safe_doc_id or safe_doc_id in {".", ".."}:
        return []
    doc_dir = (EXTRACTED_IMAGES_DIR / safe_doc_id).resolve()
    try:
        doc_dir.relative_to(EXTRACTED_IMAGES_DIR.resolve())
    except ValueError:
        return []
    if not doc_dir.is_dir():
        return []
    image_files = sorted(
        (
            path
            for path in doc_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    return [
        f"/static/images/{quote(safe_doc_id)}/{quote(path.name)}"
        for path in image_files
    ]


@app.get(
    "/api/v1/admin/knowledge-docs/{doc_id}/chunks",
    response_model=KnowledgeChunksResponse,
)
async def get_knowledge_doc_chunks(doc_id: str):
    """Inspect every ingested chunk (plus extracted images) of one document."""
    settings = get_settings()

    def _fetch() -> list[tuple]:
        with get_connection(settings) as conn:
            with conn.cursor() as cur:
                cur.execute(CHUNKS_SQL, (doc_id,))
                rows = cur.fetchall()
            conn.commit()
        return rows

    try:
        rows = await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.exception("Không truy vấn được chunk của '%s': %s", doc_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Không đọc được chunk từ pgvector: {_db_error_detail(exc, settings)}",
        ) from exc

    chunks: list[KnowledgeChunk] = []
    for row_id, content, specialty, target_audience, metadata, created_at in rows:
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        chunks.append(
            KnowledgeChunk(
                id=str(row_id),
                content=str(content or ""),
                specialty=str(specialty or ""),
                target_audience=str(target_audience or ""),
                metadata=metadata if isinstance(metadata, dict) else {},
                created_at=(
                    created_at.isoformat()
                    if hasattr(created_at, "isoformat")
                    else str(created_at or "")
                ),
            )
        )

    images = _extracted_image_urls(doc_id)
    logger.info(
        "Soát phê chunk '%s': %d chunk(s), %d image(s)", doc_id, len(chunks), len(images)
    )
    return KnowledgeChunksResponse(
        doc_id=doc_id,
        total_chunks=len(chunks),
        chunks=chunks,
        extracted_images=images,
    )


def _purge_knowledge_doc(doc_id: str, settings) -> int:
    """Heavy import kept inside the worker thread so the event loop never blocks."""
    from app.ingestion.pipeline import purge_document

    return purge_document(doc_id, settings)


@app.delete(
    "/api/v1/admin/knowledge-docs/{doc_id}",
    response_model=KnowledgeDocDeleteResponse,
)
async def delete_knowledge_doc(doc_id: str):
    """Remove every pgvector chunk belonging to ``doc_id``."""
    settings = get_settings()
    try:
        deleted = await asyncio.to_thread(_purge_knowledge_doc, doc_id, settings)
    except Exception as exc:
        logger.exception("Không xoá được tài liệu '%s': %s", doc_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Không xoá được tài liệu: {_db_error_detail(exc, settings)}",
        ) from exc

    message = f"Purged {deleted} chunks for document '{doc_id}'"
    logger.info(message)
    return KnowledgeDocDeleteResponse(success=True, message=message)


def _to_ocr_log(record: dict, index: int) -> dict[str, Any]:
    """Map one confirmed_bhyt.jsonl record onto the dashboard audit row."""
    verified = bool(record.get("isOcrVerified"))
    in_validity = bool(record.get("con_han"))
    return {
        "id": str(record.get("id") or f"record-{index}"),
        "documentType": "BHYT_CARD",
        "patientName": str(record.get("ho_ten") or ""),
        "extractedId": str(
            record.get("ma_so_bhyt_formatted") or record.get("ma_so_bhyt") or ""
        ),
        "confidenceScore": 99 if verified else 80,
        "timestamp": str(record.get("created_at") or ""),
        "status": "SUCCESS" if in_validity else "FLAGGED_FOR_REVIEW",
    }


def _read_ocr_logs(path: Path) -> list[dict[str, Any]]:
    """Parse confirmed_bhyt.jsonl newest-first, capped at OCR_LOG_LIMIT rows."""
    parsed: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line in %s:%d", path.name, line_no)
                continue
            if isinstance(record, dict):
                parsed.append(record)

    parsed.reverse()
    return [_to_ocr_log(record, index) for index, record in enumerate(parsed[:OCR_LOG_LIMIT])]


@app.get("/api/v1/admin/ocr-logs", response_model=OcrLogsResponse)
async def list_ocr_logs():
    """Real OCR audit telemetry read from data/bhyt_records/confirmed_bhyt.jsonl."""
    path = bhyt_store.records_path()
    if not path.exists():
        logger.info("Chưa có file audit OCR tại %s", path)
        return OcrLogsResponse(logs=[], total=0)

    try:
        logs = await asyncio.to_thread(_read_ocr_logs, path)
    except OSError as exc:
        logger.exception("Không đọc được file audit OCR: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Không đọc được confirmed_bhyt.jsonl: {exc}",
        ) from exc

    return OcrLogsResponse(logs=logs, total=len(logs))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8008, reload=True)