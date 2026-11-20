"""FastAPI gateway: BHYT card upload -> Groq VLM extraction -> human verification."""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import logging
import re
import threading
import time
import unicodedata
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
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
from app.database.schema import (
    FLAG_STATUSES,
    delete_unmatched_log,
    insert_unmatched_log,
    list_unmatched_logs,
    new_flag_id,
    update_unmatched_log,
)
from app.database.schedule_queries import (
    CLINIC_CLOSING_TIME,
    CLINIC_TIMEZONE,
    DoctorShiftInfo,
    SessionSchedule,
    query_department_schedule,
)
from app.parsers.bhyt_vlm import (
    BhytData,
    extract_bhyt_from_bytes,
    optimize_image,
    validate_bhyt,
)
from app.parsers.invoice_vlm import (
    INVOICE_MAX_DIMENSION,
    INVOICE_MAX_IMAGE_BYTES,
    InvoiceOcrData,
    extract_invoice_from_bytes,
    validate_invoice,
)
from app.services import ambient
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
RAW_DOC_EXTENSIONS = (
    ".pdf", ".docx", ".md", ".markdown", ".txt", ".csv", ".tsv",
    ".xlsx", ".xls", ".json", ".html", ".htm",
)

# Formats converted straight to Markdown without Docling/LlamaParse or OCR.
DIRECT_MARKDOWN_EXTENSIONS = (
    ".md", ".markdown", ".txt", ".csv", ".tsv",
    ".xlsx", ".xls", ".json", ".html", ".htm",
)

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
TRIAGE_AGE_GROUPS = ("tre_em", "nguoi_truong_thanh", "nguoi_cao_tuoi", "tat_ca")
MAX_CLARIFICATION_ROUNDS = 4
OOD_SCORE_THRESHOLD = 0.52
FLAG_REPORT_DOC_ID = "du_lieu_hoc_tang_cuong"
FLAG_REPORT_RAW_PATH = RAW_DOCS_DIR / f"{FLAG_REPORT_DOC_ID}.md"
FLAG_REPORT_PARSED_PATH = PARSED_MARKDOWN_DIR / f"{FLAG_REPORT_DOC_ID}.md"
FLAG_REPORT_SECTION_START = "# DỮ LIỆU HUẤN LUYỆN TỪ CA NGOÀI TẬP DỮ LIỆU"
FLAG_REPORT_SECTION_END = "# KẾT THÚC DỮ LIỆU HUẤN LUYỆN TỪ CA NGOÀI TẬP DỮ LIỆU"
FLAG_REPORT_COLUMNS = (
    "STT",
    "Nhóm tuổi (age_group)",
    "Triệu chứng khẩu ngữ (Bệnh nhân nói vào Mic)",
    "Triệu chứng Y khoa chuẩn",
    "Bệnh nghi ngờ & ICD-10",
    "Mức độ (priority) MD",
    "Dấu hiệu đỏ (red_flags) MD",
    "Hướng xử lý tiếp đón MD",
)
AGE_GROUP_LABELS = {
    "tre_em": "Trẻ em (< 16 tuổi)",
    "nguoi_truong_thanh": "Người trưởng thành (16–59 tuổi)",
    "nguoi_cao_tuoi": "Người cao tuổi (≥ 60 tuổi)",
    "tat_ca": "Mọi lứa tuổi",
}
_P1_RED_FLAG_RE = re.compile(
    r"(tức ngực|đau thắt ngực|đau[^.]{0,20}ngực[^.]{0,30}(dữ dội|nặng|liên tục|kéo dài|không giảm)"
    r"|méo miệng|liệt nửa người|yếu nửa người|nói đớ|thất ngôn|đột quỵ"
    r"|khó thở (nặng|dữ dội|nghẹt)|ngộp thở|không thở được|sốc phản vệ"
    r"|(nôn|đi cầu|đại tiện|ho) ra máu|phân đen|chảy máu ồ ạt|hôn mê|lơ mơ|co giật|ngất xỉu)",
    re.IGNORECASE,
)

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
    "mề đay, tê bàn chân, tiêu chảy, nôn ói, "
    "tiểu buốt, tiểu rắt, nước tiểu đục, đau hông lưng, leo cầu thang, hồi hộp, "
    "tim đập nhanh, thở không ra hơi, khàn tiếng, mất tiếng, khô họng, rát họng, "
    "ho khan, nổi mẩn đỏ, ngứa ngáy."
)

VOICE_TRIAGE_SYSTEM_PROMPT = """Bạn là trợ lý phân loại triệu chứng lâm sàng tại phòng tiếp đón.
Nhiệm vụ: đọc lời kể của bệnh nhân và NGỮ CẢNH TRIỆU CHỨNG được cung cấp, rồi trả về đúng MỘT đối tượng JSON.

Quy tắc bắt buộc:
1. Với triệu chứng ĐÃ CÓ trong [NGỮ CẢNH TRIỆU CHỨNG], chỉ dùng ngữ cảnh để suy ra bệnh và mã ICD-10; nếu ngữ cảnh không nói thì đặt icd10_code="" thay vì tự bịa. Với triệu chứng NGOÀI ngữ cảnh hoặc thuộc chuyên khoa chưa cấu hình, hãy dùng kiến thức y khoa tổng quát để dự đoán icd10_code WHO hợp lý nhất và disease_guess tiếng Việt, KHÔNG được để trống icd10_code chỉ vì thiếu ngữ cảnh. Dù triệu chứng lạ hay thuộc chuyên khoa khác, vẫn PHẢI chọn department_id 1/2/3 gần nhất và priority_level để tiếp đón không bị chặn.
2. department_id chỉ được là 1, 2 hoặc 3:
   - 1 = Khoa Nội Tổng quát & Tim mạch (tim mạch, tình trạng nội khoa nặng cấp tính, cấp cứu chuyển hóa)
   - 2 = Khoa Hô hấp & Dị ứng - Miễn dịch lâm sàng / Tai Mũi Họng (sốc phản vệ, hen phế quản, suy hô hấp cấp, tai mũi họng)
   - 3 = Khoa Da liễu (phát ban, ngứa, tổn thương da do nấm/vi khuẩn, ghẻ)
3. department_name phải khớp đúng tên khoa ứng với department_id.
4. priority_level: "P1" = cấp cứu (đau ngực dữ dội, khó thở nặng, lơ mơ, sốc phản vệ, loét da diện rộng cấp), "P2" = ưu tiên khám trong ngày, "P3" = khám thường.
5. KHÔNG BAO GIỜ suy đoán danh sách bác sĩ, ca trực, phòng khám hay số suất còn trống. Lịch làm việc sẽ được tra cứu từ cơ sở dữ liệu riêng.
6. chief_complaint_summary: tóm tắt triệu chứng chính trong 1-2 câu tiếng Việt để bệnh nhân trình bày với bác sĩ.
7. TRIỆU CHỨNG ĐA CƠ QUAN / ĐA CHUYÊN KHOA (bắt buộc):
   - Khi lời kể bao phủ nhiều chuyên khoa (ví dụ: đau ngực dữ dội kèm ngứa/nổi mụn nước ngoài da), KHÔNG được chọn bừa một khoa và KHÔNG được bỏ sót khoa còn lại.
   - Thang ưu tiên lâm sàng: P1 (Cấp cứu) > P2 (Ưu tiên) > P3 (Thường quy).
   - department_id = khoa có mức nguy cơ/độ khẩn cao NHẤT trong các khoa liên quan; priority_level theo chính khoa đó.
     Ví dụ: đau ngực dữ dội lan cánh tay + ngứa mụn nước bắp chân → department_id 1, priority_level "P1", secondary_department_id 3.
   - secondary_department_id = khoa phụ (khoa còn lại) và secondary_department_name đúng tên khoa phụ đó.
    - Nếu triệu chứng chỉ nằm ở MỘT chuyên khoa: secondary_department_id = null, secondary_department_name = null, multi_specialty_note = "".
    - Nếu hai triệu chứng cùng mức ưu tiên, chọn triệu chứng nguy hiểm/gây tàn phế hơn làm khoa chính.
    - TRƯỜNG HỢP 3 CHUYÊN KHOA (bệnh nhân kể lan man): ví dụ khó thở gắng sức/hồi hộp/tim đập nhanh kèm tiểu buốt/nước tiểu đục (Khoa 1) + sốt/viêm họng/khàn tiếng (Khoa 2) + nổi mẩn đỏ/ngứa ngáy (Khoa 3). Bắt buộc: department_id = 1 (chuyên khoa nguy cơ cao nhất), priority_level = "P2" nếu KHÔNG có dấu hiệu P1; secondary_department_id = 2 (chuyên khoa khẩn cấp kế tiếp; nếu không có cụm hô hấp/TMH mà chỉ có da liễu thì chọn 3).
    - multi_specialty_note PHẢI liệt kê TẤT CẢ các cụm cơ quan đã phát hiện (kể cả cụm phụ) VÀ thứ tự hội chẩn. Không được bỏ sót cụm nào chỉ vì đang hỏi làm rõ.
8. LÀM RÕ THÍCH ỨNG 1-4 LƯỢT (Claude-style progressive narrowing):
   - KHÔNG ép số lượt hỏi. Nếu lời kể ban đầu hoặc sau 1-2 câu trả lời đã làm rõ kiểu triệu chứng, mức nguy cơ và nhóm tuổi, hãy đặt needs_clarification=false và trả kết quả cuối ngay.
   - Chỉ đặt needs_clarification=true khi round_index <= 4, KHÔNG có dấu hiệu P1 và còn bị thiếu thông tin lâm sàng quan trọng, cụ thể: (a) chưa có [NHÓM TUỔI ĐÃ XÁC NHẬN] trong khi tuổi làm đổi mức nguy cơ hoặc chọn đúng dòng trong [NGỮ CẢNH TRIỆU CHỨNG]; (b) cần chi tiết phân biệt giữa các bệnh đang được gợi ý trong ngữ cảnh (tính chất đau, khởi phát/thời gian, sốt, phát ban, khó thở, tiền sử liên quan) hoặc cần làm rõ triệu chứng lạ ngoài tập dữ liệu.
   - Mỗi lượt chỉ hỏi MỘT câu clarifying_question ngắn gọn, đồng cảm bằng tiếng Việt, kèm 2-4 clarification_options cụ thể, dễ chạm, được thiết kế riêng từ [NGỮ CẢNH TRIỆU CHỨNG] và [LỊCH SỬ HỎI ĐÁP LÀM RÕ]. TUYỆT ĐỐI không lặp lại câu hỏi đã có trong lịch sử.
   - Mỗi option phải có dạng {"id": "opt_...", "label": "...", "category": "age_group" | "medical_history" | "symptom_detail", "age_group_value": "tre_em" | "nguoi_truong_thanh" | "nguoi_cao_tuoi" | "tat_ca" | null}; label dùng tiếng đời thường, không dùng thuật ngữ y khoa khô khan.
   - Khi round_index >= 4, luôn needs_clarification=false và chốt kết quả, dù thông tin vẫn chưa đầy đủ.
9. ƯU TIÊN P1 GHI ĐÈ LÀM RÕ: nếu có dấu hiệu đe dọa tính mạng (đau ngực dữ dội/kéo dài, méo miệng/yếu nửa người/nói đớ, khó thở nặng, sốc phản vệ, nôn hoặc đi cầu ra máu), luôn đặt needs_clarification=false và priority_level="P1" ngay, KHÔNG bắt bệnh nhân chờ trả lời.
10. LƯỢT TINH CHỈNH (khi prompt có [CÂU TRẢ LỜI LÀM RÕ]): nếu round_index < 4 và còn một chi tiết phân biệt thực sự quan trọng, được phép đặt needs_clarification=true để hỏi tiếp một câu MỚI; ngược lại hãy tổng hợp lời kể gốc, lịch sử hỏi đáp, các câu đã chọn và ngữ cảnh lọc theo nhóm tuổi để trả về chẩn đoán cuối cùng.
11. [NHÓM TUỔI ĐÃ XÁC NHẬN], [TIỀN SỬ], [LỊCH SỬ HỎI ĐÁP LÀM RÕ] và [CÂU TRẢ LỜI LÀM RÕ] là thông tin đã xác nhận, được phép dùng để suy luận.
12. PHÁT HIỆN TRIỆU CHỨNG NGOÀI TẬP DỮ LIỆU:
    - Đặt is_out_of_dataset=true và flag_reason="OUT_OF_SPECIALTY" khi lời kể rõ ràng thuộc chuyên khoa ngoài 3 khoa hiện có (Nha khoa, Mắt, Sản phụ khoa, Chỉnh hình, Tâm thần, Nhi khoa chuyên sâu...).
    - Đặt is_out_of_dataset=true và flag_reason="NEW_SYMPTOM_PATTERN" khi lời kể không được [NGỮ CẢNH TRIỆU CHỨNG] bao phủ rõ ràng dù vẫn có thể gán gần đúng vào một trong 3 khoa.
    - Ví dụ: "nhìn mờ và thấy đốm đen trước mắt" → is_out_of_dataset=true, flag_reason="OUT_OF_SPECIALTY", icd10_code="H35.3" hoặc "H43.9", disease_guess="Đục dịch kính / Tổn thương võng mạc nghi ngờ (Chuyên khoa Mắt)", department_id vẫn là 1/2/3 gần nhất.
    - Nếu triệu chứng đã có trong ngữ cảnh và nằm đúng chuyên khoa 1/2/3, đặt is_out_of_dataset=false và flag_reason="".
13. KHÔNG BAO GIỜ TRẢ VỀ GỢI Ý TRỐNG, kể cả khi needs_clarification=true:
    - Dù đang hỏi làm rõ, BẮT BUỘC điền có nghĩa các trường: disease_guess, icd10_code, department_id, department_name, priority_level, chief_complaint_summary, secondary_department_id, secondary_department_name, multi_specialty_note. Đây là "preliminary triage banner" được hiển thị NGAY cho bệnh nhân trong lúc chờ trả lời 1-4 câu hỏi.
    - Không được trả về chuỗi rỗng/null cho các trường trên (ngoại trừ secondary_department_id/secondary_department_name=null và multi_specialty_note="" khi chỉ có đúng một chuyên khoa). Hãy dùng phán đoán lâm sàng tốt nhất từ lời kể hiện có, KHÔNG chờ câu trả lời để điền.

Định dạng JSON bắt buộc:
{"disease_guess": "...", "icd10_code": "...", "department_id": 1, "department_name": "...", "priority_level": "P1", "chief_complaint_summary": "...", "secondary_department_id": 3, "secondary_department_name": "...", "multi_specialty_note": "...", "needs_clarification": false, "clarifying_question": "", "clarification_options": [], "detected_age_group": null, "is_out_of_dataset": false, "flag_reason": ""}
Khi needs_clarification=true, clarification_options phải có 2-4 phần tử đúng dạng {"id": "opt_...", "label": "...", "category": "...", "age_group_value": ...}. 9 trường phán đoán lâm sàng ở trên vẫn BẮT BUỘC có giá trị, không được để trống.

Ví dụ đầu ra khi đa chuyên khoa:

{"disease_guess": "Hội chứng vành cấp nghi ngờ / Kèm Viêm da dị ứng", "icd10_code": "I21", "department_id": 1, "department_name": "Khoa Nội Tổng quát & Tim mạch", "priority_level": "P1", "chief_complaint_summary": "Đau tức ngực trái lan vai kèm ngứa rát ngoài da", "secondary_department_id": 3, "secondary_department_name": "Khoa Da liễu", "multi_specialty_note": "Cần ưu tiên xử trí tim mạch khẩn cấp; sau khi ổn định đề xuất hội chẩn chuyên khoa Da liễu.", "needs_clarification": false, "clarifying_question": "", "clarification_options": [], "detected_age_group": null, "is_out_of_dataset": false, "flag_reason": ""}

Ví dụ khi cần làm rõ nhưng vẫn có preliminary banner (3 chuyên khoa):

{"disease_guess": "Khó thở gắng sức/Nhiễm đường tiết niệu nghi ngờ / Kèm viêm họng và viêm da dị ứng", "icd10_code": "R07.9", "department_id": 1, "department_name": "Khoa Nội Tổng quát & Tim mạch", "priority_level": "P2", "chief_complaint_summary": "Khó thở khi leo cầu thang, hồi hộp tim đập nhanh, tiểu buốt nước tiểu đục, kèm khàn tiếng và nổi mẩn ngứa", "secondary_department_id": 2, "secondary_department_name": "Khoa Hô hấp & Dị ứng - Miễn dịch lâm sàng", "multi_specialty_note": "Các cụm triệu chứng: (1) Tim mạch/Tiết niệu: khó thở gắng sức, hồi hộp, tiểu buốt, nước tiểu đục; (2) Hô hấp/TMH: sốt, khàn tiếng, khô họng; (3) Da liễu: nổi mẩn đỏ, ngứa ngáy. Thứ tự xử trí: Khoa 1 trước, sau đó hội chẩn Khoa 2, cuối cùng Khoa 3.", "needs_clarification": true, "clarifying_question": "Bạn bao nhiêu tuổi và khó thở xuất hiện từ lúc nào ạ?", "clarification_options": [{"id": "opt_age_adult", "label": "Tôi 16-59 tuổi", "category": "age_group", "age_group_value": "nguoi_truong_thanh"}, {"id": "opt_age_senior", "label": "Tôi từ 60 tuổi trở lên", "category": "age_group", "age_group_value": "nguoi_cao_tuoi"}], "detected_age_group": null, "is_out_of_dataset": false, "flag_reason": ""}"""

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


class InvoiceOcrResponse(BaseModel):
    success: bool = True
    data: InvoiceOcrData
    warnings: list[str] = Field(default_factory=list)
    model: str
    latency_ms: float
    image_preview: str
    image_size_kb: float
    raw_text: str = ""


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
    parser_type: Literal["llama", "docling", "auto"] = "docling"
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


_FALLBACK_TRIAGE_CONTEXT = (
    "BẢNG PHÂN LUỒNG CHUYÊN KHOA PHÒNG KHÁM:\n"
    "- Khoa 1: Nội Tổng quát & Tim mạch (departmentId = 1). Các bệnh: Đau thắt ngực (I21), Đột quỵ (I63), "
    "Tăng huyết áp kịch phát (I10), Suy tim (I50), Rối loạn nhịp tim (I49), Trào ngược dạ dày (K21), "
    "Đái tháo đường (E11), Cơn gút cấp (M10), Bệnh thận mạn (N18).\n"
    "- Khoa 2: Hô hấp & Dị ứng - Miễn dịch lâm sàng / Tai Mũi Họng (departmentId = 2). Các bệnh: Sốc phản vệ (T78.2), "
    "Hen phế quản cấp (J45), Đợt cấp COPD (J44.0), Viêm phổi (J18), Áp xe amidan (J36), Viêm tai giữa (H66), "
    "Viêm họng cấp (J02), Viêm mũi xoang (J01), Viêm mũi dị ứng (J30), Viêm phế quản (J20).\n"
    "- Khoa 3: Da liễu (departmentId = 3). Các bệnh: Stevens-Johnson (L51.2), Đỏ da toàn thân (L53.9), "
    "Zona thần kinh (B02), Áp xe da / Nhọt (L02), Chốc lở (L01), Viêm da tiếp xúc (L23), Vảy nến (L40.0), "
    "Ghẻ (B86), Nấm da (B35), Trứng cá (L70.0)."
)


_LONG_TRANSCRIPT_CHARS = 180
_CLINICAL_QUERY_MAX_CHARS = 1000

#: Whisper's most frequent Vietnamese homophone slips in clinical narratives.
#: Both the accented form and its tone-mark-less form are mapped so a transcript
#: with or without diacritics still yields the same high-signal search terms.
_VOICE_TYPO_CORRECTIONS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in (
        (r"\bkhô\s+hộp\b", "khô họng"),
        (r"\bkho\s+hop\b", "khô họng"),
        (r"\bho\s+khang\b", "ho khan"),
        (r"\bđeo\s+khẩu\s+thang\b", "leo cầu thang"),
        (r"\bdeo\s+khau\s+thang\b", "leo cầu thang"),
        (r"\bcảm\s+thấy\s+bút\b", "tiểu buốt"),
        (r"\bcam\s+thay\s+but\b", "tiểu buốt"),
        (r"\btiểu\s+bút\b", "tiểu buốt"),
        (r"\btieu\s+but\b", "tiểu buốt"),
        (r"\bchữ\s+chứng\b", "triệu chứng"),
        (r"\bchu\s+chung\b", "triệu chứng"),
    )
)

_CLINICAL_QUERY_KEYWORDS = (
    "đau", "sốt", "ho", "khó thở", "thở", "khò khè", "hen", "sổ mũi", "viêm",
    "dị ứng", "nổi mẩn", "mẩn đỏ", "mẩn ngứa", "nổi mụn", "ngứa", "mụn",
    "dát sẩn", "vảy nến", "zona", "sốc phản vệ", "mề đay", "tê", "tiêu chảy",
    "nôn", "ói", "buồn nôn", "mệt", "chóng mặt", "tức ngực", "hồi hộp",
    "tim", "huyết áp", "tiểu", "nước tiểu", "đục", "buốt", "rắt", "hông",
    "lưng", "leo cầu thang", "khàn", "mất tiếng", "khô họng", "rát họng",
    "đờm", "chảy máu", "đi cầu", "phân đen", "lơ mơ", "ngất", "co giật",
    "méo miệng", "liệt", "nói đớ", "phát ban", "bọng nước", "ghẻ", "nấm",
    "khó chịu", "réo tai", "chảy nước mũi", "đau họng", "nuốt", "khạc",
    "run", "đánh trống ngực",
)
_CLINICAL_QUERY_CLAUSE_SPLIT_RE = re.compile(
    r"[,;.!?\n]+|\b(?:và|rồi|xong|mà|còn|nữa|cũng|với|cùng|nên|thì|vì|do)\b",
    re.IGNORECASE,
)


def _fold_vietnamese_for_match(value: str) -> str:
    """Drop tone marks (and fold đ->d) so keyword checks survive STT slips."""
    lowered = (value or "").lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


_CLINICAL_SIGNAL_RE = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(_fold_vietnamese_for_match(keyword))
        for keyword in sorted(_CLINICAL_QUERY_KEYWORDS, key=len, reverse=True)
    )
    + r")\b"
)


def _normalise_voice_typos(text: str) -> str:
    """Apply the known Whisper homophone fixes without dropping any clause."""
    normalised = " ".join((text or "").split())
    for pattern, replacement in _VOICE_TYPO_CORRECTIONS:
        normalised = pattern.sub(replacement, normalised)
    return normalised


def _extract_clinical_search_query(raw_text: str) -> str:
    """Condense a rambling transcript into high-signal symptom terms for pgvector.

    Whisper mishears clinical words as everyday homophones ("khô họng" -> "khô
    hộp", "leo cầu thang" -> "đeo khẩu thang"), and long narratives bury those
    symptoms between lifestyle chatter. The known confusions are corrected first;
    for transcripts longer than ``_LONG_TRANSCRIPT_CHARS`` only clauses carrying
    a clinical signal are kept, so "cơm hộp, trà sữa, tắm khuya" cannot dilute
    the embedding that drives RAG retrieval.
    """
    text = _normalise_voice_typos(raw_text)
    if not text:
        return ""
    if len(text) <= _LONG_TRANSCRIPT_CHARS:
        return text[:_CLINICAL_QUERY_MAX_CHARS]

    clauses = [
        " ".join(part.split())
        for part in _CLINICAL_QUERY_CLAUSE_SPLIT_RE.split(text)
    ]
    clinical = [
        clause
        for clause in clauses
        if clause and _CLINICAL_SIGNAL_RE.search(_fold_vietnamese_for_match(clause))
    ]
    query = ", ".join(dict.fromkeys(clinical or clauses))
    return (
        query[:_CLINICAL_QUERY_MAX_CHARS].strip(" ,;")
        or text[:_CLINICAL_QUERY_MAX_CHARS]
    )


#: (department_id, acuity rank, display label, folded-text pattern). Higher
#: acuity wins the primary department; the runner-up in a different department
#: becomes the secondary consult. Used as the deterministic backstop whenever
#: the LLM omits the preliminary assessment while asking for clarification.
_ORGAN_CLUSTER_PATTERNS: tuple[tuple[int, int, str, re.Pattern[str]], ...] = (
    (
        1, 4,
        "Tim mạch / Nội khoa: khó thở gắng sức, hồi hộp, tim đập nhanh, đau tức ngực",
        re.compile(
            r"\b(?:leo cau thang|hoi hop|tim dap nhanh|danh trong nguc|tuc nguc"
            r"|dau nguc|kho tho|tho khong ra hoi)\b"
        ),
    ),
    (
        1, 3,
        "Tiết niệu: tiểu buốt, tiểu rắt, nước tiểu đục, đau hông lưng",
        re.compile(
            r"\b(?:tieu buot|tieu rat|nuoc tieu duc|dau hong lung|dau lung"
            r"|cam thay but)\b"
        ),
    ),
    (
        2, 2,
        "Hô hấp / Tai Mũi Họng: sốt, viêm họng, khàn tiếng, khô họng, ho khan",
        re.compile(
            r"\b(?:vien hong|sot|khan tieng|mat tieng|kho hong|rat hong"
            r"|ho khan|dau hong|kho khe)\b"
        ),
    ),
    (
        3, 1,
        "Da liễu: nổi mẩn đỏ, ngứa ngáy, phát ban",
        re.compile(
            r"\b(?:noi man do|ngua ngay|phat ban|man ngua|me day|mun nuoc"
            r"|bong nuoc|dat san|ngua)\b"
        ),
    ),
)

_CLUSTER_DISEASE_FALLBACK = {
    4: "Hội chứng đau ngực / Rối loạn nhịp nghi ngờ",
    3: "Nhiễm đường tiết niệu nghi ngờ",
    2: "Viêm họng / Viêm thanh quản nghi ngờ",
    1: "Phát ban ngứa / Viêm da dị ứng nghi ngờ",
}
_CLUSTER_ICD10_FALLBACK = {4: "R07.9", 3: "N39.0", 2: "J02.9", 1: "R21"}


def _detect_organ_clusters(raw_text: Optional[str]) -> list[tuple[int, int, str]]:
    """List detected organ clusters, highest acuity first."""
    folded = _fold_vietnamese_for_match(_normalise_voice_typos(raw_text or ""))
    clusters: list[tuple[int, int, str]] = []
    seen: set[tuple[int, str]] = set()
    for department_id, acuity, label, pattern in _ORGAN_CLUSTER_PATTERNS:
        if not pattern.search(folded):
            continue
        key = (department_id, label)
        if key in seen:
            continue
        seen.add(key)
        clusters.append((department_id, acuity, label))
    return sorted(clusters, key=lambda item: item[1], reverse=True)


def _build_multi_specialty_note(clusters: list[tuple[int, int, str]]) -> str:
    """Explicitly name every cluster and the consult sequence, dropping none."""
    labels = " ; ".join(
        f"{index}. {label}"
        for index, (_, _, label) in enumerate(clusters, 1)
    )
    sequence: list[int] = []
    for department_id, _, _ in clusters:
        if department_id not in sequence:
            sequence.append(department_id)
    names = [
        f"Khoa {department_id} ({_DEPARTMENT_NAMES[department_id]})"
        for department_id in sequence
    ]
    return (
        f"Đã phát hiện {len(clusters)} cụm triệu chứng: {labels}. "
        f"Thứ tự xử trí/hội chẩn đề xuất: {' -> '.join(names)}."
    )


def _retrieve_triage_context(
    transcription: str, settings, age_group: Optional[str] = None
) -> tuple[str, list[str], list[str], float]:
    """Stage 1b: age-filtered RAG chunks, or the built-in department summary.

    Only the cleaned clinical query is embedded; the untouched ``transcription``
    is still handed to the LLM in ``_extract_triage_facts``. The fallback is used
    only when pgvector is unreachable, retrieval throws, or no row matched -
    triage must still work on an offline/clinic-local cluster.
    """
    warnings: list[str] = []
    search_query = _extract_clinical_search_query(transcription) or transcription
    if ambient.pgvector_reachable(settings):
        try:
            from app.rag.retriever import retrieve_context

            results = retrieve_context(
                search_query,
                top_k=5,
                target_audience="tiep_don",
                age_group=age_group,
                settings=settings,
            )
            if results:
                best_score = max((result.score for result in results), default=0.0)
                context = "\n\n---\n\n".join(result.content for result in results)
                sources: list[str] = []
                for result in results:
                    label = (
                        f"{result.doc_id} ({result.specialty})"
                        if result.specialty
                        else str(result.doc_id)
                    )
                    if label not in sources:
                        sources.append(label)
                logger.info(
                    "Triage RAG: %d chunk(s), age_group=%s",
                    len(results),
                    age_group or "all",
                )
                return context, sources, warnings, float(best_score)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}".strip().splitlines()[0][:200]
            logger.warning("Triage RAG retrieval failed: %s", detail)
            warnings.append(f"Không truy xuất được triệu chứng theo nhóm tuổi ({detail}).")

    return (
        _FALLBACK_TRIAGE_CONTEXT,
        ["Tri thức lâm sàng chuẩn Bộ Y tế (Fast-Triage)"],
        warnings,
        0.0,
    )



def _normalise_age_group(value: object) -> Optional[str]:
    """Fold a supplied age stratum onto one of the four canonical keys."""
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    return key if key in TRIAGE_AGE_GROUPS else None


def _fallback_clarification_options(age_group: Optional[str]) -> list[dict]:
    """Concrete chips used when the model flags ambiguity without usable options."""
    options: list[dict] = []
    if not age_group:
        options.extend(
            [
                {
                    "id": "age_tre_em",
                    "label": "Dưới 16 tuổi (Trẻ em)",
                    "category": "age_group",
                    "age_group_value": "tre_em",
                },
                {
                    "id": "age_nguoi_truong_thanh",
                    "label": "Từ 16–59 tuổi (Người trưởng thành)",
                    "category": "age_group",
                    "age_group_value": "nguoi_truong_thanh",
                },
                {
                    "id": "age_nguoi_cao_tuoi",
                    "label": "Từ 60 tuổi trở lên (Người cao tuổi)",
                    "category": "age_group",
                    "age_group_value": "nguoi_cao_tuoi",
                },
                {
                    "id": "history_chronic",
                    "label": "Có tiền sử tăng huyết áp / tim mạch / tiểu đường",
                    "category": "medical_history",
                    "age_group_value": None,
                },
            ]
        )
    else:
        options.extend(
            [
                {
                    "id": "history_chronic",
                    "label": "Có tiền sử tăng huyết áp / tim mạch / tiểu đường",
                    "category": "medical_history",
                    "age_group_value": None,
                },
                {
                    "id": "history_none",
                    "label": "Không có bệnh mạn tính nào đã biết",
                    "category": "medical_history",
                    "age_group_value": None,
                },
                {
                    "id": "detail_chest",
                    "label": "Đau tức nặng vùng ngực trái khi đi lại",
                    "category": "symptom_detail",
                    "age_group_value": None,
                },
                {
                    "id": "detail_fatigue",
                    "label": "Chỉ mệt mỏi, ăn uống kém vài ngày nay",
                    "category": "symptom_detail",
                    "age_group_value": None,
                },
            ]
        )
    return options[:4]


def _sanitise_clarification_options(raw: object, age_group: Optional[str]) -> list[dict]:
    """Keep only well-formed chips; pad short lists so the patient always has taps."""
    options: list[dict] = []
    seen: set[str] = set()
    categories = {"age_group", "medical_history", "symptom_detail"}
    if isinstance(raw, list):
        for idx, entry in enumerate(raw, 1):
            if not isinstance(entry, dict):
                continue
            label = " ".join(str(entry.get("label") or "").split())[:120]
            category = str(entry.get("category") or "").strip()
            if not label or category not in categories:
                continue
            ident = " ".join(str(entry.get("id") or f"opt_{idx}").split())[:60] or f"opt_{idx}"
            if ident in seen:
                continue
            seen.add(ident)
            options.append(
                {
                    "id": ident,
                    "label": label,
                    "category": category,
                    "age_group_value": (
                        _normalise_age_group(entry.get("age_group_value"))
                        if category == "age_group"
                        else None
                    ),
                }
            )
            if len(options) == 4:
                break

    if len(options) < 2:
        for fallback in _fallback_clarification_options(age_group):
            if len(options) >= 4:
                break
            if fallback["id"] not in seen:
                options.append(fallback)
                seen.add(fallback["id"])
    return options


def _clarification_turns(
    history: Optional[list[dict[str, str]]],
) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            question = " ".join(str(item.get("question") or "").split())[:300]
            answer = " ".join(str(item.get("answer") or "").split())[:500]
        elif isinstance(item, str):
            question = ""
            answer = " ".join(item.split())[:500]
        else:
            continue
        if question or answer:
            turns.append((question, answer))
    return turns


def _sanitise_triage_payload(
    payload: dict,
    is_refinement: bool = False,
    age_group: Optional[str] = None,
    best_score: float = 1.0,
    round_index: int = 1,
    force_final: bool = False,
    raw_text: Optional[str] = None,
) -> tuple[dict, list[str]]:
    """Validate the LLM's JSON against the clinical contract before using it.

    The LLM is untrusted: an out-of-range department, an unknown priority or a
    malformed ICD-10 code is corrected (and reported) rather than passed on.
    ``raw_text`` lets the clinical backstop fill a preliminary banner whenever
    the LLM asks for clarification but forgets to populate it.
    """
    warnings: list[str] = []
    clusters = _detect_organ_clusters(raw_text)

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
        fallback_department = clusters[0][0] if clusters else 1
        warnings.append(
            f"department_id không hợp lệ ({payload.get('department_id')!r}); "
            f"mặc định về khoa {fallback_department}."
        )
        department_id = fallback_department

    priority = _text("priority_level", 8).upper().replace(" ", "")
    priority_was_missing = priority not in VALID_PRIORITIES
    if priority_was_missing:
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

    # Multi-organ complaints: the secondary department must be a different,
    # valid khoa; anything else collapses to None instead of leaking a bogus id.
    secondary_department_id: Optional[int] = None
    raw_secondary = payload.get("secondary_department_id")
    if raw_secondary is not None and str(raw_secondary).strip().lower() not in (
        "",
        "null",
        "none",
    ):
        try:
            parsed_secondary = int(raw_secondary)
        except (TypeError, ValueError):
            parsed_secondary = 0
        if parsed_secondary not in VALID_DEPARTMENT_IDS:
            warnings.append(
                f"secondary_department_id không hợp lệ ({raw_secondary!r}); bỏ qua khoa phụ."
            )
        elif parsed_secondary == department_id:
            warnings.append(
                "secondary_department_id trùng department_id; bỏ qua khoa phụ."
            )
        else:
            secondary_department_id = parsed_secondary

    secondary_department_name = (
        _DEPARTMENT_NAMES[secondary_department_id]
        if secondary_department_id is not None
        else None
    )
    multi_specialty_note = _text("multi_specialty_note", 500)

    detected_age_group = _normalise_age_group(payload.get("detected_age_group")) or age_group
    raw_needs = payload.get("needs_clarification")
    needs_clarification = raw_needs is True or (
        isinstance(raw_needs, str) and raw_needs.strip().lower() in {"true", "1", "yes"}
    )
    if (
        priority == "P1"
        or force_final
        or (is_refinement and round_index >= MAX_CLARIFICATION_ROUNDS)
    ):
        needs_clarification = False

    clarifying_question = _text("clarifying_question", 240)
    clarification_options: list[dict] = []
    if needs_clarification:
        clarification_options = _sanitise_clarification_options(
            payload.get("clarification_options"), detected_age_group
        )
        if not clarifying_question:
            clarifying_question = (
                "Để chúng tôi phân luồng đúng, bạn vui lòng chọn thêm thông tin bên dưới nhé."
            )

    # A clarification round must never blank the preliminary triage banner.
    # The prompt asks the LLM to fill these fields; this is the deterministic
    # backstop for omissions, malformed JSON and 3-specialty rambling cases.
    disease_guess = _text("disease_guess", 300)
    chief_complaint_summary = _text("chief_complaint_summary", 400)
    assessment_missing = not disease_guess or disease_guess == "Chưa xác định"
    if clusters:
        primary_department, primary_acuity, _ = clusters[0]
        if needs_clarification and assessment_missing:
            department_id = primary_department
            department_name = _DEPARTMENT_NAMES[department_id]
        if not disease_guess or disease_guess == "Chưa xác định":
            disease_guess = " / ".join(
                _CLUSTER_DISEASE_FALLBACK[acuity] for _, acuity, _ in clusters
            )[:300]
        if not chief_complaint_summary:
            chief_complaint_summary = "; ".join(
                label.split(": ", 1)[-1] for _, _, label in clusters
            )[:400]
        if not icd10:
            icd10 = _CLUSTER_ICD10_FALLBACK.get(primary_acuity, "")
        if priority_was_missing:
            priority = "P2" if primary_acuity >= 2 else "P3"
        if secondary_department_id is None:
            secondary_department_id = next(
                (
                    department
                    for department, _, _ in clusters[1:]
                    if department != primary_department
                ),
                None,
            )
            secondary_department_name = (
                _DEPARTMENT_NAMES[secondary_department_id]
                if secondary_department_id is not None
                else None
            )
        distinct_departments = list(dict.fromkeys(dept for dept, _, _ in clusters))
        if len(distinct_departments) >= 2 and (needs_clarification or not multi_specialty_note):
            deterministic_note = _build_multi_specialty_note(clusters)
            multi_specialty_note = (
                f"{multi_specialty_note} {deterministic_note}".strip()
                if multi_specialty_note
                else deterministic_note
            )[:500]
    if not chief_complaint_summary and raw_text:
        chief_complaint_summary = _normalise_voice_typos(raw_text)[:400]
    disease_guess = disease_guess or "Chưa xác định"

    raw_out = payload.get("is_out_of_dataset")
    is_out_of_dataset = raw_out is True or (
        isinstance(raw_out, str) and raw_out.strip().lower() in {"true", "1", "yes"}
    )
    flag_reason = _text("flag_reason", 64).upper().replace(" ", "_")
    valid_flag_reasons = {"OUT_OF_SPECIALTY", "NEW_SYMPTOM_PATTERN", "LOW_SIMILARITY_MATCH"}
    if flag_reason not in valid_flag_reasons:
        flag_reason = ""
    explicit_reason = flag_reason
    if is_out_of_dataset and not flag_reason:
        flag_reason = "NEW_SYMPTOM_PATTERN"
    if not needs_clarification and (
        best_score < OOD_SCORE_THRESHOLD or (not icd10 and not is_out_of_dataset)
    ):
        is_out_of_dataset = True
        if not explicit_reason:
            flag_reason = "LOW_SIMILARITY_MATCH"
    if not is_out_of_dataset:
        flag_reason = ""

    return (
        {
            "disease_guess": disease_guess,
            "icd10_code": icd10,
            "department_id": department_id,
            "department_name": department_name,
            "priority_level": priority,
            "chief_complaint_summary": chief_complaint_summary,
            "secondary_department_id": secondary_department_id,
            "secondary_department_name": secondary_department_name,
            "multi_specialty_note": multi_specialty_note,
            "needs_clarification": needs_clarification,
            "clarifying_question": clarifying_question,
            "clarification_options": clarification_options,
            "detected_age_group": detected_age_group,
            "is_out_of_dataset": is_out_of_dataset,
            "flag_reason": flag_reason,
            "similarity_score": float(best_score),
        },
        warnings,
    )


def _has_p1_red_flag(text: str) -> bool:
    """Conservative keyword guard so a red flag is never delayed by clarification."""
    return bool(_P1_RED_FLAG_RE.search(text or ""))


def _extract_triage_facts(
    transcription: str,
    context: str,
    settings,
    age_group: Optional[str] = None,
    patient_history: Optional[str] = None,
    selected_clarifications: Optional[list[str]] = None,
    is_refinement: bool = False,
    best_score: float = 1.0,
    clarification_history: Optional[list[dict[str, str]]] = None,
    round_index: int = 1,
) -> tuple[dict, list[str]]:
    """Stage 1c: force the LLM into a single JSON object and validate it."""
    from app.rag.generator import get_groq_client

    history_turns = _clarification_turns(clarification_history)
    answers = [item for item in (selected_clarifications or []) if item]
    has_p1 = _has_p1_red_flag(
        " ".join([transcription, patient_history or "", *answers, *(a for _, a in history_turns)])
    )
    prompt_parts = [
        f"[NGỮ CẢNH TRIỆU CHỨNG]\n{context}",
        f"[LỜI KỂ BỆNH NHÂN]\n{transcription}",
    ]
    if age_group:
        prompt_parts.append(f"[NHÓM TUỔI ĐÃ XÁC NHẬN]\n{AGE_GROUP_LABELS.get(age_group, age_group)}")
    if patient_history:
        prompt_parts.append(f"[TIỀN SỬ BỆNH NHÂN]\n{patient_history.strip()[:500]}")
    if is_refinement or history_turns:
        history_lines = [
            f"- Q: {question or '(không rõ)'} -> A: {answer or '(không có)'}"
            for question, answer in history_turns
        ]
        prompt_parts.append(
            "[LỊCH SỬ HỎI ĐÁP LÀM RÕ]\n"
            + ("\n".join(history_lines) or "- (không có)")
        )
    if is_refinement:
        prompt_parts.append(
            "[CÂU TRẢ LỜI LÀM RÕ]\n"
            + ("\n".join(f"- {answer[:300]}" for answer in answers) or "- (không có)")
        )
        if has_p1 or round_index >= MAX_CLARIFICATION_ROUNDS:
            mode = (
                f"[CHẾ ĐỘ LƯỢT LÀM RÕ]\nround_index={round_index}/{MAX_CLARIFICATION_ROUNDS}. "
                "Đây là lượt chốt hoặc có dấu hiệu P1: bắt buộc needs_clarification=false, "
                "tổng hợp mọi thông tin đã có để trả kết quả cuối cùng."
            )
        else:
            mode = (
                f"[CHẾ ĐỘ LƯỢT LÀM RÕ]\nround_index={round_index}/{MAX_CLARIFICATION_ROUNDS}. "
                "Nếu còn đúng một chi tiết lâm sàng quan trọng chưa biết, được phép đặt "
                "needs_clarification=true và hỏi tiếp MỘT câu MỚI chưa từng hỏi. "
                "Nếu đã đủ rõ, đặt needs_clarification=false ngay."
            )
        prompt_parts.append(mode)
    else:
        prompt_parts.append(
            f"[CHẾ ĐỘ]\nLượt đầu round_index=1/{MAX_CLARIFICATION_ROUNDS}: áp dụng quy tắc "
            "làm rõ thích ứng; nếu có P1 thì chốt ngay không hỏi thêm."
        )
    user_prompt = "\n\n".join(prompt_parts)

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

    facts, fact_warnings = _sanitise_triage_payload(
        payload,
        is_refinement=is_refinement,
        age_group=age_group,
        best_score=best_score,
        round_index=round_index,
        force_final=has_p1,
        raw_text=transcription,
    )
    if has_p1:
        if facts["priority_level"] != "P1":
            fact_warnings.append("Phát hiện dấu hiệu đe dọa tính mạng; đã ghi đè mức ưu tiên P1.")
        facts["priority_level"] = "P1"
        facts["needs_clarification"] = False
        facts["clarifying_question"] = ""
        facts["clarification_options"] = []
    return facts, fact_warnings


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
    secondary_department_name: Optional[str] = None,
) -> str:
    """Deterministic advice - never generated by the LLM, never invents a roster.

    ``secondary_department_name`` is the khoa phụ of a multi-organ complaint; when
    present the advice gains an explicit "primary first, then consult" sentence.
    """
    advice = _build_base_advice(
        priority, recommended_shift, schedule, department_name, connected
    )
    if secondary_department_name:
        advice += (
            f" Lưu ý: Bệnh nhân có triệu chứng phối hợp liên quan đến "
            f"{secondary_department_name}. Đề nghị khám và xử trí ưu tiên tại "
            f"{department_name} trước, sau đó phối hợp hội chẩn chuyên khoa liên quan."
        )
    return advice


def _build_base_advice(
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

    morning_elapsed = schedule.get("morning_elapsed") is True
    afternoon_data = schedule.get("afternoon") or schedule.get("AFTERNOON") or {}
    afternoon_open = False
    if isinstance(afternoon_data, dict):
        afternoon_open = bool(afternoon_data.get("doctors")) and not afternoon_data.get(
            "is_full", True
        )

    if morning_elapsed and recommended_shift == "MORNING":
        # Defensive: never point at a session that has already ended today.
        recommended_shift = "AFTERNOON" if afternoon_open else "NEXT_DAY"

    shift_labels = {
        "MORNING": f"ca sáng (07:30 - 11:30){day_label}",
        "AFTERNOON": f"ca chiều (13:00 - 17:00){day_label}",
        "NEXT_DAY": "ngày hôm sau",
        "NO_DUTY": "chưa có lịch khám",
        "MANUAL_PICK": "thủ công",
    }
    slot = shift_labels.get(recommended_shift, recommended_shift)

    # A deleted/replaced outpatient shift leaves the roster empty while the
    # schedule service itself is still reachable. Treat that exactly like
    # NO_DUTY so the patient gets an explanation instead of a phantom session.
    no_outpatient_duty = recommended_shift == "NO_DUTY" or (
        schedule.get("has_outpatient_duty") is False
    )

    if priority == "P1":
        return notice + (
            f"Triệu chứng thuộc nhóm CẤP CỨU (P1). Đề nghị gọi cấp cứu 115 hoặc đưa người bệnh "
            f"vào khoa cấp cứu ngay, không chờ khám định kỳ. Nhánh tiếp đón sẽ ưu tiên "
            f"hỗ trợ tại {department_name}. Lịch khám chỉ có ý nghĩa sau khi bệnh nhân "
            f"đã được đánh giá đủ an toàn để khám ngoại trú."
        )

    if no_outpatient_duty:
        return notice + (
            f"{department_name} hiện không có lịch khám ngoại trú vào ngày cần xếp "
            "(ca trực đã bị xóa, đổi sang nội trú hoặc chưa được xếp lại). "
            "Đề nghị liên hệ phòng tiếp đón hoặc chọn ngày làm việc kế tiếp. "
            "Mức độ ưu tiên: " + priority + "."
        )

    if recommended_shift == "NEXT_DAY":
        if morning_elapsed:
            return notice + (
                f"Ca sáng hôm nay tại {department_name} đã kết thúc (sau 11:30) và ca chiều "
                "hôm nay đã đủ bệnh hoặc không còn ca trực. Đề nghị đặt lịch khám vào ngày mai "
                "hoặc ngày làm việc kế tiếp. Mức độ ưu tiên: " + priority + "."
            )
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
        if morning_elapsed and recommended_shift == "AFTERNOON":
            return notice + (
                f"Nên khám trong ngày tại {department_name}. Ca sáng hôm nay đã kết thúc, "
                "đề nghị ưu tiên ca chiều (13:00 - 17:00) (còn trống). Nếu bệnh nhân muốn "
                "khám buổi sáng, vui lòng đặt lịch sáng ngày mai. Nếu triệu chứng nặng lên "
                "như khó thở dữ dội, đau ngực dữ dội hoặc lơ mơ thì chuyển sang cấp cứu ngay."
            )
        return notice + (
            f"Nên khám trong ngày tại {department_name}. Đề nghị ưu tiên {slot} "
            f"(còn trống). Nếu triệu chứng nặng lên như khó thở dữ dội, đau ngực "
            "dữ dội hoặc lơ mơ thì chuyển sang cấp cứu ngay."
        )
    if morning_elapsed and recommended_shift == "AFTERNOON":
        return notice + (
            f"Có thể khám theo lịch thường tại {department_name}, ưu tiên ca chiều "
            "(13:00 - 17:00) hôm nay. Ca sáng hôm nay đã kết thúc; nếu bệnh nhân muốn "
            "khám buổi sáng, vui lòng đặt lịch sáng ngày mai. Đề nghị tái khám nếu "
            "triệu chứng không cải thiện sau 3-5 ngày hoặc nặng thêm."
        )
    return notice + (
        f"Có thể khám theo lịch thường tại {department_name}, ưu tiên {slot}"
        + (" nếu bệnh nhân thuận lợi hơn." if morning_open else ".")
        + " Đề nghị tái khám nếu triệu chứng không cải thiện sau 3-5 ngày hoặc nặng thêm."
    )


def _run_triage_from_transcription(
    transcription: str,
    target_date: Optional[str],
    age_group: Optional[str],
    patient_history: Optional[str],
    started: float,
    settings,
) -> VoiceScheduleTriageResponse:
    """Stage 1 + Stage 2 shared verbatim by the voice and text triage routes."""
    age_group = _normalise_age_group(age_group)

    context, sources, warnings, best_score = _retrieve_triage_context(
        transcription, settings, age_group=age_group
    )
    facts, fact_warnings = _extract_triage_facts(
        transcription,
        context,
        settings,
        age_group=age_group,
        patient_history=patient_history,
        best_score=best_score,
        round_index=1,
    )
    warnings.extend(fact_warnings)

    if facts.get("needs_clarification"):
        logger.info(
            "Triage clarification: age_group=%s question=%s",
            age_group or "all",
            facts.get("clarifying_question", "")[:120],
        )
        return VoiceScheduleTriageResponse(
            success=True,
            transcription=transcription,
            recommended_shift="",
            advice="",
            schedule={},
            schedule_connected=True,
            target_date=target_date or "",
            rag_sources=sources,
            warnings=warnings,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            **facts,
        )

    return _complete_triage_with_schedule(
        facts, transcription, target_date, sources, warnings, started, settings
    )


def _run_voice_triage(
    audio: bytes,
    filename: str,
    target_date: Optional[str],
    age_group: Optional[str] = None,
    patient_history: Optional[str] = None,
) -> VoiceScheduleTriageResponse:
    """Blocking two-stage body: Whisper STT, then the shared triage pipeline."""
    started = time.perf_counter()
    settings = get_settings()

    transcription = _transcribe_voice(audio, filename, settings)
    logger.info("STT (%d chars): %s", len(transcription), transcription[:120])

    return _run_triage_from_transcription(
        transcription, target_date, age_group, patient_history, started, settings
    )


class ClarificationOption(BaseModel):
    """One tappable chip offered when the spoken symptoms are ambiguous."""

    id: str
    label: str
    category: Literal["age_group", "medical_history", "symptom_detail"]
    age_group_value: Optional[
        Literal["tre_em", "nguoi_truong_thanh", "nguoi_cao_tuoi", "tat_ca"]
    ] = None


class RefineTriageRequest(BaseModel):
    """Answers selected by the patient in one adaptive clarification round."""

    transcription: str
    selected_options: list[str] = Field(default_factory=list)
    extra_note: Optional[str] = ""
    age_group: Optional[str] = None
    patient_history: Optional[str] = None
    target_date: Optional[str] = None
    clarification_history: list[dict[str, str]] = Field(default_factory=list)
    round_index: int = Field(default=1, ge=1, le=5)


class TextTriageRequest(BaseModel):
    """Typed/edited transcript submitted instead of a Whisper recording."""

    transcription: str = Field(min_length=1, description="Lời kể triệu chứng đã gõ/chỉnh sửa.")
    target_date: Optional[str] = None
    age_group: Optional[str] = None
    patient_history: Optional[str] = None


class VoiceScheduleTriageResponse(BaseModel):

    success: bool = True
    transcription: str
    disease_guess: str
    icd10_code: str
    department_id: int
    department_name: str
    priority_level: str
    chief_complaint_summary: str
    secondary_department_id: Optional[int] = Field(
        default=None,
        description="Khoa phụ cần khám kèm nếu có triệu chứng đa cơ quan",
    )
    secondary_department_name: Optional[str] = Field(default=None)
    multi_specialty_note: Optional[str] = Field(default="")
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
    needs_clarification: bool = False
    clarifying_question: str = ""
    clarification_options: list[ClarificationOption] = Field(default_factory=list)
    detected_age_group: Optional[str] = None
    is_out_of_dataset: bool = False
    flag_reason: str = ""
    flag_id: Optional[str] = None
    similarity_score: Optional[float] = None


def _log_out_of_dataset_facts(facts: dict, transcription: str, settings) -> None:
    if not facts.get("is_out_of_dataset"):
        return
    try:
        stored = insert_unmatched_log(
            {
                "flag_id": new_flag_id(settings),
                "transcription": transcription[:4000],
                "age_group": facts.get("detected_age_group") or "tat_ca",
                "ai_disease_guess": facts.get("disease_guess") or "",
                "ai_icd10_code": facts.get("icd10_code") or "",
                "ai_department_id": facts.get("department_id"),
                "ai_department_name": facts.get("department_name") or "",
                "ai_priority_level": facts.get("priority_level") or "",
                "flag_reason": facts.get("flag_reason") or "NEW_SYMPTOM_PATTERN",
                "similarity_score": facts.get("similarity_score"),
                "status": "PENDING_DOCTOR",
            },
            settings,
        )
        facts["flag_id"] = stored.get("flag_id")
    except Exception as exc:
        logger.warning("Không ghi nhận ca triệu chứng ngoài tập dữ liệu: %s", exc)


def _complete_triage_with_schedule(
    facts: dict,
    transcription: str,
    target_date: Optional[str],
    sources: list[str],
    warnings: list[str],
    started: float,
    settings,
) -> VoiceScheduleTriageResponse:
    """Stage 2: deterministic schedule lookup shared by first pass and refinement."""
    schedule_payload = query_department_schedule(
        facts["department_id"], target_date, settings
    )
    connected = bool(schedule_payload.get("is_available"))
    recommended_shift = schedule_payload.get("recommended_shift") or "MANUAL_PICK"
    has_outpatient_duty = bool(
        schedule_payload.get("has_outpatient_duty", connected)
    )
    schedule: dict[str, Optional[SessionSchedule]] = {}
    if connected and has_outpatient_duty and recommended_shift != "NO_DUTY":
        for label in ("morning", "afternoon"):
            data = schedule_payload.get(label)
            if data:
                schedule[label] = SessionSchedule(**data)
    else:
        schedule["morning"] = None
        schedule["afternoon"] = None
        if not connected:
            warnings.append(schedule_payload.get("error_message") or "")
            if schedule_payload.get("detail"):
                warnings.append(f"Chi tiết lỗi lịch trực: {schedule_payload['detail']}")

    if _schedule_rolled_to_tomorrow(schedule_payload):
        warnings.append(ROLLOVER_WARNING)

    advice = _build_advice(
        facts["priority_level"],
        recommended_shift,
        schedule_payload,
        schedule_payload["department_name"],
        connected=connected,
        secondary_department_name=facts.get("secondary_department_name"),
    )
    facts["department_name"] = schedule_payload["department_name"]

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "Voice triage: dept=%s priority=%s icd=%s secondary=%s shift=%s target_date=%s "
        "schedule_connected=%s (%.0f ms)",
        facts["department_id"],
        facts["priority_level"],
        facts["icd10_code"] or "-",
        facts.get("secondary_department_id") or "-",
        recommended_shift,
        schedule_payload.get("target_date", "-"),
        connected,
        latency_ms,
    )
    _log_out_of_dataset_facts(facts, transcription, settings)

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


def _run_text_triage(payload: TextTriageRequest) -> VoiceScheduleTriageResponse:
    """Blocking text body: skip Whisper, then run the same Stage 1 + Stage 2."""
    started = time.perf_counter()
    settings = get_settings()
    transcription = " ".join(payload.transcription.split())[:4000]
    if not transcription:
        raise ValueError("Thiếu nội dung lời kể cần phân tích.")
    logger.info("Text triage (%d chars): %s", len(transcription), transcription[:120])

    return _run_triage_from_transcription(
        transcription,
        payload.target_date,
        payload.age_group,
        payload.patient_history,
        started,
        settings,
    )


def _run_refined_triage(payload: RefineTriageRequest) -> VoiceScheduleTriageResponse:
    """Re-run triage with patient answers; may request one more adaptive round."""
    started = time.perf_counter()
    settings = get_settings()
    age_group = _normalise_age_group(payload.age_group)
    transcription = " ".join(payload.transcription.split())[:2000]

    selected = [item for item in payload.selected_options if item]
    if payload.extra_note:
        selected.append(payload.extra_note)
    history_turns = _clarification_turns(payload.clarification_history)

    if not age_group:
        age_sources = [*selected, payload.extra_note or ""]
        for question, answer in history_turns:
            age_sources.extend([question, answer])
        for source in age_sources:
            lowered = source.lower()
            age_group = next((group for group in TRIAGE_AGE_GROUPS if group in lowered), None)
            if age_group:
                break

    rag_parts: list[str] = [transcription]
    rag_parts.extend(answer for _, answer in history_turns)
    rag_parts.extend(selected)
    rag_query = " ".join(
        dict.fromkeys(part.strip() for part in rag_parts if part and part.strip())
    )[:4000]

    context, sources, warnings, best_score = _retrieve_triage_context(
        rag_query, settings, age_group=age_group
    )
    facts, fact_warnings = _extract_triage_facts(
        rag_query,
        context,
        settings,
        age_group=age_group,
        patient_history=payload.patient_history,
        selected_clarifications=selected,
        is_refinement=True,
        best_score=best_score,
        clarification_history=payload.clarification_history,
        round_index=payload.round_index,
    )
    warnings.extend(fact_warnings)

    if facts.get("needs_clarification"):
        logger.info(
            "Triage refinement round %d/%d: age_group=%s question=%s",
            payload.round_index,
            MAX_CLARIFICATION_ROUNDS,
            age_group or "all",
            facts.get("clarifying_question", "")[:120],
        )
        return VoiceScheduleTriageResponse(
            success=True,
            transcription=rag_query,
            recommended_shift="",
            advice="",
            schedule={},
            schedule_connected=True,
            target_date=payload.target_date or "",
            rag_sources=sources,
            warnings=warnings,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            **facts,
        )

    return _complete_triage_with_schedule(
        facts,
        rag_query,
        payload.target_date,
        sources,
        warnings,
        started,
        settings,
    )


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


async def _run_invoice_ocr(
    file: UploadFile,
    invoice_type: Literal[
        "SUPPLIER_DRUG", "PATIENT_MEDICAL", "RECEPTION_RECEIPT"
    ],
) -> InvoiceOcrResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vui lòng tải lên tệp ảnh hóa đơn hợp lệ (.jpg, .jpeg, .png, .heic, .webp)",
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
        # Invoice tables need a higher resolution budget than a BHYT card so
        # batch numbers, expiry dates and unit prices survive compression.
        optimized = await asyncio.to_thread(
            optimize_image,
            image_bytes,
            max_dimension=INVOICE_MAX_DIMENSION,
            max_bytes=INVOICE_MAX_IMAGE_BYTES,
        )
        data, raw_text, _ = await asyncio.to_thread(
            extract_invoice_from_bytes, image_bytes, invoice_type, optimized, vlm_model
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:  # Groq auth/ratelimit/network failures
        logger.exception("Lỗi khi bóc tách hóa đơn bằng VLM: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Không thể gọi dịch vụ Groq AI: {exc}",
        ) from exc

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info("Hoàn tất bóc tách hóa đơn sau %s ms", elapsed_ms)

    return InvoiceOcrResponse(
        data=data,
        warnings=validate_invoice(data),
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


@app.post("/api/v1/ocr/invoice", response_model=InvoiceOcrResponse)
async def process_invoice_upload(
    file: UploadFile = File(...),
    invoice_type: Literal[
        "SUPPLIER_DRUG", "PATIENT_MEDICAL", "RECEPTION_RECEIPT"
    ] = Form("SUPPLIER_DRUG"),
):
    """Extract a supplier drug invoice, patient medical bill or reception receipt."""
    try:
        return await _run_invoice_ocr(file, invoice_type)
    finally:
        await file.close()


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
    """Map a filename onto data/raw_docs/, refusing anything that escapes it.

    Administrators often type the bare stem in the pipeline modal ("chẩn đoán
    sớm 2" instead of "chẩn đoán sớm 2.docx"), so a name with a missing or
    unsupported suffix first probes ``data/raw_docs/`` for the same stem plus a
    supported extension before it is rejected.
    """
    clean_name = (filename or "").strip()
    raw_root = RAW_DOCS_DIR.resolve()

    def _inside(path: Path) -> bool:
        try:
            path.relative_to(raw_root)
        except ValueError:
            return False
        return True

    candidate = (RAW_DOCS_DIR / clean_name).resolve()
    if not _inside(candidate):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Đường dẫn không hợp lệ.",
        )

    # Smart extension auto-probe: bare stem (or e.g. ".txt") -> <stem>.pdf/.docx
    if not candidate.suffix or candidate.suffix.lower() not in RAW_DOC_EXTENSIONS:
        for ext in RAW_DOC_EXTENSIONS:
            probe = (RAW_DOCS_DIR / f"{clean_name}{ext}").resolve()
            if _inside(probe) and probe.is_file():
                candidate = probe
                break

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
    target_dir = PARSED_MARKDOWN_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() in DIRECT_MARKDOWN_EXTENSIONS:
        from app.parsers.multi_format_parser import MultiFormatParser

        MultiFormatParser(output_dir=target_dir).parse(source)
        return target_dir / f"{source.stem}.md", "structured_text"

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
        if source.suffix.lower() in DIRECT_MARKDOWN_EXTENSIONS:
            parser_used = "structured_text"
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
        for path in sorted(RAW_DOCS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
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
    age_group: Optional[str] = Form(
        None,
        description="Nhóm tuổi đã biết: tre_em | nguoi_truong_thanh | nguoi_cao_tuoi | tat_ca.",
    ),
    patient_history: Optional[str] = Form(
        None, description="Tiền sử bệnh đã biết (tăng huyết áp, tim mạch, tiểu đường...)."
    ),
):
    """Two-stage voice triage: Whisper + age-filtered RAG + LLM, then roster query."""
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
            _run_voice_triage, audio, filename, target_date, age_group, patient_history
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


@app.post("/api/v1/triage/text-schedule", response_model=VoiceScheduleTriageResponse)
async def text_schedule_triage(payload: TextTriageRequest):
    """Fully typed/edited transcript -> same RAG + LLM + schedule pipeline as voice."""
    try:
        return await asyncio.to_thread(_run_text_triage, payload)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Text triage validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Lỗi hệ thống khi xử lý text_schedule_triage: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý Text Triage: {str(exc)}",
        ) from exc


@app.post("/api/v1/triage/refine", response_model=VoiceScheduleTriageResponse)
async def refine_triage(payload: RefineTriageRequest):
    """Second turn: fold the patient's clarification chips into a final triage."""
    try:
        if not payload.transcription.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Thiếu nội dung lời kể ban đầu cần làm rõ.",
            )
        return await asyncio.to_thread(_run_refined_triage, payload)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Refine triage validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Lỗi hệ thống khi xử lý refine_triage: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý làm rõ triệu chứng: {str(exc)}",
        ) from exc


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
        default_factory=list,
        description=(
            "Deprecated, kept for client compatibility. Always empty: raw_transcript "
            "is the canonical unified script; no heuristic speaker splitting is done."
        ),
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
    """Ambient voice consultation, zero audio storage, single-pass extraction.

    Step 1 transcribes the upload straight out of RAM and drops the bytes. The
    transcript is kept as one unified clinical script - no heuristic
    doctor/patient diarization is attempted. The guideline RAG and the live
    in-stock formulary are then fetched concurrently; ONE LLM call produces
    chief complaints, red flags, the SOAP note, ICD-10 diagnoses and a
    prescription draft; vital-sign emergencies are merged in deterministically
    before the response is assembled.
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

        # --- Step 2: keep the unified script, no heuristic speaker split ---
        stt_done = time.perf_counter()

        # --- Steps 3 + 4 run concurrently: the guideline RAG is CPU-bound
        # embedding and the formulary is a DB lookup - neither depends on the
        # other, so wall-clock time is whichever is slowest rather than the sum.
        knowledge_result, formulary = await asyncio.gather(
            _retrieve_knowledge_safely(raw_transcript[:600], settings),
            get_available_drugs(settings),
        )
        prep_done = time.perf_counter()

        knowledge_context, rag_sources, warnings = knowledge_result
        drugs, formulary_warnings = _formulary_warnings(formulary)
        warnings.extend(formulary_warnings)
        logger.info(
            "Ambient prep: %d guideline chunk(s) [%s] + %d in-stock drug(s) [%s]",
            len(rag_sources),
            ", ".join(rag_sources) or "-",
            len(drugs),
            formulary.source or "unavailable",
        )

        # --- Step 5: ONE LLM call - complaints + red flags + SOAP/ICD/Rx ----
        draft, red_flags, chief_complaints, symptoms = await asyncio.to_thread(
            ambient.single_pass_ambient_synthesis,
            raw_transcript,
            clinical_profile,
            knowledge_context,
            drugs,
            llm_model,
            settings,
        )
        synth_done = time.perf_counter()
        warnings.extend(draft.warnings)

        # --- Step 6: deterministic vital-sign emergencies -------------------
        # Hypertensive crisis / hypoxia outrank whatever the model answered.
        if ambient.merge_vital_red_flags(red_flags, vitals):
            draft.soap_note.plan = ambient.append_inquiry_to_plan(
                draft.soap_note.plan, red_flags.recommended_inquiry
            )
        logger.info("Ambient red flags: has_red_flags=%s", red_flags.has_red_flags)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Ambient single-pass done: complaints=%d red_flags=%s "
            "diagnoses=%d rx=%d | stt=%.0fms prep=%.0fms llm=%.0fms (%.0f ms)",
            len(chief_complaints),
            red_flags.has_red_flags,
            len(draft.diagnoses),
            len(draft.prescription_draft),
            (stt_done - started) * 1000,
            (prep_done - stt_done) * 1000,
            (synth_done - prep_done) * 1000,
            latency_ms,
        )
        return AmbientConsultationResponse(
            success=True,
            raw_transcript=raw_transcript,
            diarized_transcript=[],
            has_red_flags=red_flags.has_red_flags,
            red_flag_alert=red_flags.red_flag_alert,
            recommended_inquiry=red_flags.recommended_inquiry,
            chief_complaints=chief_complaints,
            symptoms=symptoms,
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
    conversation_history = (payload.raw_transcript or "").strip()[:1500]
    if not conversation_history:
        conversation_history = "\n".join(
            turn.text for turn in payload.diarized_transcript if turn.text
        )[:1500]

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


@app.post("/api/v1/clinical/copilot-chat/voice", response_model=CopilotChatResponse)
async def copilot_chat_voice(
    audio_file: UploadFile = File(..., description="Ghi âm bổ sung của bác sĩ"),
    draft_payload: str = Form(..., description="JSON stringified of current draft context"),
):
    """Apply one doctor follow-up *audio* clip to the current clinical draft.

    The extra clip (e.g. the patient mentions a new allergy mid-consultation)
    is transcribed in RAM, then fed to the same revision pass as the text
    copilot so the existing SOAP note, ICD-10 list and prescription draft are
    updated in place instead of being regenerated from scratch.
    """
    started = time.perf_counter()
    settings = get_settings()

    try:
        filename = _resolve_ambient_filename(audio_file)
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

            doctor_voice_text = await asyncio.to_thread(
                _transcribe_ambient, audio_bytes, filename, settings
            )
        finally:
            del audio_bytes

        try:
            data = json.loads(draft_payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid draft_payload JSON: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="draft_payload phải là một đối tượng JSON.",
            )

        llm_model = _ambient_llm_model(settings)
        clinical_profile = _clinical_profile(
            data.get("department_id", 1),
            data.get("allergies", ""),
            data.get("patient_history", ""),
            data.get("doctor_id"),
            data.get("patient_id"),
        )
        conversation_history = str(data.get("raw_transcript") or "").strip()[:1500]
        if not conversation_history:
            conversation_history = "\n".join(
                str(turn.get("text") or "").strip()
                for turn in data.get("diarized_transcript") or []
                if isinstance(turn, dict) and str(turn.get("text") or "").strip()
            )[:1500]

        try:
            previous = ClinicalDraft(
                soap_note=(
                    SoapNote(**data["soap_note"])
                    if isinstance(data.get("soap_note"), dict)
                    else SoapNote()
                ),
                diagnoses=[IcdDiagnosis(**d) for d in data.get("diagnoses") or []],
                prescription_draft=[
                    PrescriptionItemDraft(**p)
                    for p in data.get("prescription_draft") or []
                ],
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"draft_payload không hợp lệ: {exc}",
            ) from exc

        knowledge_result, formulary = await asyncio.gather(
            _retrieve_knowledge_safely(doctor_voice_text, settings),
            get_available_drugs(settings),
        )
        knowledge_context, rag_sources, warnings = knowledge_result
        drugs, formulary_warnings = _formulary_warnings(formulary)
        warnings.extend(formulary_warnings)

        draft, assistant_message = await asyncio.to_thread(
            ambient.revise_clinical_note,
            doctor_voice_text,
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
            "Copilot chat voice: diagnoses=%d rx=%d warnings=%d (%.0f ms)",
            len(draft.diagnoses),
            len(draft.prescription_draft),
            len(warnings),
            latency_ms,
        )
        return CopilotChatResponse(
            success=True,
            assistant_message=f'Đã nghe: "{doctor_voice_text}". {assistant_message}',
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
        logger.error("Copilot chat voice rejected model output: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Không cập nhật được hồ sơ: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Lỗi hệ thống khi xử lý copilot-chat/voice: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý Copilot Chat Voice: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        await audio_file.close()


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


class FlaggedSymptomItem(BaseModel):
    id: Optional[int] = None
    flag_id: str
    transcription: str = ""
    patient_description: str = ""
    age_group: str = "tat_ca"
    age_group_label: str = ""
    ai_disease_guess: str = ""
    ai_icd10_code: str = ""
    ai_department_id: Optional[int] = None
    ai_department_name: str = ""
    ai_priority_level: str = ""
    flag_reason: str = ""
    similarity_score: Optional[float] = None
    ticket_number: str = ""
    patient_name: str = ""
    confirmed_icd10_code: str = ""
    confirmed_disease_name: str = ""
    confirmed_by_doctor: str = ""
    medical_symptom_standard: str = ""
    red_flags: str = ""
    reception_action: str = ""
    status: str = "PENDING_DOCTOR"
    created_at: str = ""
    updated_at: str = ""


class FlaggedSymptomCluster(BaseModel):
    label: str
    count: int


class FlaggedSymptomsStats(BaseModel):
    total: int = 0
    pending_doctor: int = 0
    doctor_confirmed: int = 0
    trained: int = 0
    top_clusters: list[FlaggedSymptomCluster] = Field(default_factory=list)


class FlaggedSymptomsResponse(BaseModel):
    items: list[FlaggedSymptomItem] = Field(default_factory=list)
    stats: FlaggedSymptomsStats = Field(default_factory=FlaggedSymptomsStats)


class FlaggedSymptomUpdate(BaseModel):
    ticket_number: Optional[str] = None
    patient_name: Optional[str] = None
    age_group: Optional[str] = None
    ai_department_id: Optional[int] = None
    ai_priority_level: Optional[str] = None
    confirmed_icd10_code: Optional[str] = None
    confirmed_disease_name: Optional[str] = None
    medical_symptom_standard: Optional[str] = None
    red_flags: Optional[str] = None
    reception_action: Optional[str] = None
    status: Optional[Literal["PENDING_DOCTOR", "DOCTOR_CONFIRMED", "TRAINED"]] = None


class FlaggedSymptomConfirmRequest(BaseModel):
    confirmed_icd10_code: str = ""
    confirmed_disease_name: str = ""
    confirmed_by_doctor: str = ""
    medical_symptom_standard: str = ""
    red_flags: str = ""
    reception_action: str = ""


class FlaggedTrainRequest(BaseModel):
    flag_ids: list[str] = Field(default_factory=list)


class FlaggedSymptomMutationResponse(BaseModel):
    success: bool = True
    message: str
    item: Optional[FlaggedSymptomItem] = None


class FlaggedSymptomConfirmResponse(BaseModel):
    success: bool = True
    message: str
    items: list[FlaggedSymptomItem] = Field(default_factory=list)


class FlaggedSymptomDeleteResponse(BaseModel):
    success: bool = True
    message: str


class FlaggedTrainResponse(BaseModel):
    success: bool = True
    message: str
    trained_count: int = 0
    doc_id: str = FLAG_REPORT_DOC_ID
    chunks: int = 0


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


def _flag_text(value: object) -> str:
    return "" if value is None else str(value)


def _flag_item(row: dict[str, Any]) -> FlaggedSymptomItem:
    raw_id = row.get("id")
    raw_department = row.get("ai_department_id")
    raw_score = row.get("similarity_score")
    try:
        parsed_id = int(raw_id) if raw_id not in (None, "") else None
    except (TypeError, ValueError):
        parsed_id = None
    try:
        parsed_department = int(raw_department) if raw_department not in (None, "") else None
    except (TypeError, ValueError):
        parsed_department = None
    try:
        parsed_score = float(raw_score) if raw_score not in (None, "") else None
    except (TypeError, ValueError):
        parsed_score = None
    transcription = _flag_text(row.get("transcription"))
    age_group = _normalise_age_group(row.get("age_group")) or "tat_ca"
    return FlaggedSymptomItem(
        id=parsed_id,
        flag_id=_flag_text(row.get("flag_id")),
        transcription=transcription,
        patient_description=transcription,
        age_group=age_group,
        age_group_label=AGE_GROUP_LABELS.get(age_group, age_group),
        ai_disease_guess=_flag_text(row.get("ai_disease_guess")),
        ai_icd10_code=_flag_text(row.get("ai_icd10_code")),
        ai_department_id=parsed_department,
        ai_department_name=_flag_text(row.get("ai_department_name")),
        ai_priority_level=_flag_text(row.get("ai_priority_level")),
        flag_reason=_flag_text(row.get("flag_reason")),
        similarity_score=parsed_score,
        ticket_number=_flag_text(row.get("ticket_number")),
        patient_name=_flag_text(row.get("patient_name")),
        confirmed_icd10_code=_flag_text(row.get("confirmed_icd10_code")),
        confirmed_disease_name=_flag_text(row.get("confirmed_disease_name")),
        confirmed_by_doctor=_flag_text(row.get("confirmed_by_doctor")),
        medical_symptom_standard=_flag_text(row.get("medical_symptom_standard")),
        red_flags=_flag_text(row.get("red_flags")),
        reception_action=_flag_text(row.get("reception_action")),
        status=_flag_text(row.get("status")) or "PENDING_DOCTOR",
        created_at=_flag_text(row.get("created_at")),
        updated_at=_flag_text(row.get("updated_at")),
    )


def _flag_stats(rows: list[dict[str, Any]]) -> FlaggedSymptomsStats:
    statuses = Counter(_flag_text(row.get("status")).upper() for row in rows)
    clusters: Counter[str] = Counter()
    for row in rows:
        label = (
            _flag_text(row.get("confirmed_disease_name")).strip()
            or _flag_text(row.get("ai_disease_guess")).strip()
            or _flag_text(row.get("flag_reason")).strip()
            or "Chưa phân loại"
        )
        clusters[label] += 1
    return FlaggedSymptomsStats(
        total=len(rows),
        pending_doctor=statuses.get("PENDING_DOCTOR", 0),
        doctor_confirmed=statuses.get("DOCTOR_CONFIRMED", 0),
        trained=statuses.get("TRAINED", 0),
        top_clusters=[
            FlaggedSymptomCluster(label=label, count=count)
            for label, count in sorted(clusters.items(), key=lambda item: -item[1])[:5]
        ],
    )


def _flag_report_values(row: dict[str, Any], index: int) -> list[str]:
    age_group = _normalise_age_group(row.get("age_group")) or "tat_ca"
    age_cell = f"{AGE_GROUP_LABELS[age_group]} [{age_group}]"
    transcription = _flag_text(row.get("transcription"))
    standard = _flag_text(row.get("medical_symptom_standard")).strip() or transcription
    disease = (
        _flag_text(row.get("confirmed_disease_name")).strip()
        or _flag_text(row.get("ai_disease_guess")).strip()
        or "Chưa xác định"
    )
    icd = (
        _flag_text(row.get("confirmed_icd10_code")).strip()
        or _flag_text(row.get("ai_icd10_code")).strip()
    )
    disease_cell = f"{disease} ({icd})" if icd else disease
    return [
        str(index),
        age_cell,
        transcription,
        standard,
        disease_cell,
        _flag_text(row.get("ai_priority_level")),
        _flag_text(row.get("red_flags")),
        _flag_text(row.get("reception_action")),
    ]


def _flag_export_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        dict(zip(FLAG_REPORT_COLUMNS, _flag_report_values(row, index)))
        for index, row in enumerate(rows, 1)
    ]


def _flag_md_cell(value: object) -> str:
    return _flag_text(value).replace("|", "\\|").replace("\n", " ")


def _flag_report_markdown(rows: list[dict[str, Any]]) -> str:
    records: list[tuple[int, list[str]]] = []
    for index, row in enumerate(rows, 1):
        try:
            department = int(row.get("ai_department_id") or 1)
        except (TypeError, ValueError):
            department = 1
        records.append((department if department in VALID_DEPARTMENT_IDS else 1,
                        _flag_report_values(row, index)))

    lines = [FLAG_REPORT_SECTION_START, ""]
    for department in VALID_DEPARTMENT_IDS:
        department_rows = [values for dept, values in records if dept == department]
        if not department_rows:
            continue
        lines.append(f"## KHOA {department}")
        lines.append("")
        lines.append("| " + " | ".join(FLAG_REPORT_COLUMNS) + " |")
        lines.append("|" + "|".join(["---"] * len(FLAG_REPORT_COLUMNS)) + "|")
        for values in department_rows:
            lines.append("| " + " | ".join(_flag_md_cell(value) for value in values) + " |")
        lines.append("")
    lines.append(FLAG_REPORT_SECTION_END)
    return "\n".join(lines)


def _merge_flag_report(path: Path, section: str) -> str:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if FLAG_REPORT_SECTION_START in existing and FLAG_REPORT_SECTION_END in existing:
        before, rest = existing.split(FLAG_REPORT_SECTION_START, 1)
        _, after = rest.split(FLAG_REPORT_SECTION_END, 1)
        return f"{before.rstrip()}\n\n{section.strip()}\n{after.lstrip()}"
    return f"{existing.rstrip()}\n\n{section.strip()}\n"


def _write_flag_reports(section: str) -> None:
    for path in (FLAG_REPORT_RAW_PATH, FLAG_REPORT_PARSED_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_merge_flag_report(path, section), encoding="utf-8")


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


@app.get("/api/v1/admin/flagged-symptoms", response_model=FlaggedSymptomsResponse)
async def list_flagged_symptoms():
    settings = get_settings()
    rows = await asyncio.to_thread(list_unmatched_logs, settings)
    return FlaggedSymptomsResponse(
        items=[_flag_item(row) for row in rows],
        stats=_flag_stats(rows),
    )


@app.get("/api/v1/admin/flagged-symptoms/export")
async def export_flagged_symptoms(format: Literal["csv", "md", "json"] = "csv"):
    settings = get_settings()
    rows = await asyncio.to_thread(list_unmatched_logs, settings)

    if format == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(FLAG_REPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(_flag_export_rows(rows))
        content = buffer.getvalue().encode("utf-8-sig")
        media_type = "text/csv; charset=utf-8"
        filename = "du_lieu_trieu_chung_ngoai.csv"
    elif format == "md":
        content = _flag_report_markdown(rows).encode("utf-8")
        media_type = "text/markdown; charset=utf-8"
        filename = "du_lieu_trieu_chung_ngoai.md"
    else:
        content = json.dumps(
            _flag_export_rows(rows), ensure_ascii=False, indent=2
        ).encode("utf-8")
        media_type = "application/json; charset=utf-8"
        filename = "du_lieu_trieu_chung_ngoai.json"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/v1/admin/flagged-symptoms/train-rag", response_model=FlaggedTrainResponse)
async def train_flagged_symptoms(payload: FlaggedTrainRequest):
    settings = get_settings()
    rows = await asyncio.to_thread(list_unmatched_logs, settings)
    requested = set(payload.flag_ids)
    if requested:
        selected = [row for row in rows if _flag_text(row.get("flag_id")) in requested]
    else:
        selected = [
            row for row in rows if _flag_text(row.get("status")).upper() == "DOCTOR_CONFIRMED"
        ]
    if not selected:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Không có ca nào được chọn để huấn luyện lại RAG.",
        )

    section = _flag_report_markdown(selected)
    await asyncio.to_thread(_write_flag_reports, section)
    selected_ids = [_flag_text(row.get("flag_id")) for row in selected]

    def _ingest() -> int:
        from app.ingestion.pipeline import ingest_markdown_file, purge_document

        purge_document(FLAG_REPORT_DOC_ID, settings)
        result = ingest_markdown_file(
            FLAG_REPORT_PARSED_PATH,
            FLAG_REPORT_DOC_ID,
            "Tổng hợp tiếp đón",
            "tiep_don",
            settings,
        )
        for flag_id in selected_ids:
            update_unmatched_log(flag_id, {"status": "TRAINED"}, settings)
        return result.total_chunks

    try:
        chunks = await asyncio.to_thread(_ingest)
    except Exception as exc:
        logger.exception("Không huấn luyện được dữ liệu ngoài tập: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Không nạp được dữ liệu vào pgvector: {_db_error_detail(exc, settings)}",
        ) from exc

    return FlaggedTrainResponse(
        message=f"Đã huấn luyện {len(selected_ids)} ca ngoài tập dữ liệu.",
        trained_count=len(selected_ids),
        chunks=chunks,
    )


@app.patch(
    "/api/v1/admin/flagged-symptoms/by-ticket/{ticket_number}/confirm",
    response_model=FlaggedSymptomConfirmResponse,
)
async def confirm_flagged_symptom_by_ticket(
    ticket_number: str, payload: FlaggedSymptomConfirmRequest
):
    settings = get_settings()
    rows = await asyncio.to_thread(list_unmatched_logs, settings)
    matches = [
        row for row in rows if _flag_text(row.get("ticket_number")) == ticket_number
    ]
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy ca nào gắn với ticket {ticket_number!r}.",
        )

    changes = {
        "confirmed_icd10_code": payload.confirmed_icd10_code,
        "confirmed_disease_name": payload.confirmed_disease_name,
        "confirmed_by_doctor": payload.confirmed_by_doctor or "Bác sĩ tiếp nhận",
        "medical_symptom_standard": payload.medical_symptom_standard,
        "red_flags": payload.red_flags,
        "reception_action": payload.reception_action,
    }

    def _confirm() -> list[dict[str, Any]]:
        updated: list[dict[str, Any]] = []
        for row in matches:
            status = (
                "TRAINED"
                if _flag_text(row.get("status")).upper() == "TRAINED"
                else "DOCTOR_CONFIRMED"
            )
            result = update_unmatched_log(
                _flag_text(row.get("flag_id")),
                {**changes, "status": status},
                settings,
            )
            if result:
                updated.append(result)
        return updated

    updated_rows = await asyncio.to_thread(_confirm)
    if not updated_rows:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Không cập nhật được ca đã khám; kiểm tra PostgreSQL hoặc tệp dự phòng.",
        )
    return FlaggedSymptomConfirmResponse(
        message=f"Đã xác nhận {len(updated_rows)} ca theo ticket {ticket_number}.",
        items=[_flag_item(row) for row in updated_rows],
    )


@app.patch("/api/v1/admin/flagged-symptoms/{flag_id}", response_model=FlaggedSymptomMutationResponse)
async def update_flagged_symptom(flag_id: str, payload: FlaggedSymptomUpdate):
    settings = get_settings()
    changes = {
        key: value
        for key, value in payload.model_dump(exclude_unset=True).items()
        if value is not None
    }
    if "age_group" in changes:
        age_group = _normalise_age_group(changes["age_group"])
        if not age_group:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Nhóm tuổi không hợp lệ: {changes['age_group']!r}.",
            )
        changes["age_group"] = age_group
    if "status" in changes and changes["status"] not in FLAG_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Trạng thái không hợp lệ: {changes['status']!r}.",
        )
    if not changes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Không có trường nào để cập nhật.",
        )

    updated = await asyncio.to_thread(update_unmatched_log, flag_id, changes, settings)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy ca {flag_id!r}.",
        )
    return FlaggedSymptomMutationResponse(
        message=f"Đã cập nhật ca {flag_id}.",
        item=_flag_item(updated),
    )


@app.delete(
    "/api/v1/admin/flagged-symptoms/{flag_id}",
    response_model=FlaggedSymptomDeleteResponse,
)
async def delete_flagged_symptom(flag_id: str):
    settings = get_settings()
    deleted = await asyncio.to_thread(delete_unmatched_log, flag_id, settings)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy ca {flag_id!r}.",
        )
    return FlaggedSymptomDeleteResponse(message=f"Đã xoá ca {flag_id}.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8008, reload=True)