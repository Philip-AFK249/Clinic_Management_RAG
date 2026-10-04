"""BHYT card extraction powered by Groq's Qwen multimodal VLM.

The entire visual pipeline (perspective correction, reading Vietnamese
dot-matrix print, diacritics) is delegated to `qwen/qwen3.8-27b`. No local
OCR engine (PaddleOCR/Tesseract) is involved.
"""

from __future__ import annotations

import base64
import io
import json
import re
import unicodedata
from datetime import date, datetime
from typing import Mapping

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.generator import get_groq_client

logger = get_logger(__name__)

# --- Image optimization budget -------------------------------------------
MAX_DIMENSION = 1200
MAX_IMAGE_BYTES = 200_000  # ~100-200 KB as required by the extraction budget
QUALITY_LADDER = (85, 75, 65, 55)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

# Number of characters on a Vietnamese BHYT card number (2 letters + 13 digits)
INSURANCE_CODE_LENGTH = 15

BHYT_SYSTEM_PROMPT = """Bạn là chuyên gia OCR bóc tách thông tin thẻ Bảo hiểm Y tế (BHYT) Việt Nam.

NHIỆM VỤ:
Đọc trực tiếp trên ẢNH thẻ BHYT được cung cấp và trả về dữ liệu có cấu trúc.

QUY TẮC BẮT BUỘC:
1. Ảnh có thể bị nghiêng 15-40 độ, chụp ngược, hoặc phản chiếu ánh sáng. Hãy tự căn chỉnh góc nhìn trước khi đọc.
2. TUYỆT ĐỐI KHÔNG bịa đặt, KHÔNG suy đoán thông tin. Nếu chữ bị mờ hoặc không có trên thẻ, trả về null.
3. Trả về DUY NHẤT một JSON object PHẲNG (không lồng nhau), đúng tên khoá sau đây, không thêm khoá khác.
4. TUYỆT ĐỐI KHÔNG bọc trong markdown ```json``` và không giải thích thêm.
5. Ngày tháng luôn trả về định dạng DD/MM/YYYY.
6. ma_so_bhyt ghi liền, không dấu cách, viết HOA (ví dụ: "HC4915361000392" hoặc "DN4797912345678").
7. ma_noi_dkkcb_ban_dau là mã nơi khám chữa bệnh ban đầu in trên thẻ, giữ nguyên dấu gạch nối (ví dụ: "79-014"). Nơi này KHÁC ma_so_bhyt.
8. noi_kham_chua_benh_ban_dau là TÊN nơi khám chữa bệnh ban đầu (ví dụ: "BV Đa Khoa Sài Gòn"), không kèm mã số.

SCHEMA JSON BẮT BUỘC:
{
  "ho_ten": "HỌ VÀ TÊN IN HOA",
  "ma_so_bhyt": "15 ký tự chữ và số viết liền, ví dụ: DN4797912345678",
  "ngay_sinh": "DD/MM/YYYY",
  "gioi_tinh": "Nam hoặc Nữ",
  "ma_noi_dkkcb_ban_dau": "Mã nơi KCB, ví dụ: 79-014 hoặc null",
  "noi_kham_chua_benh_ban_dau": "Tên nơi KCB, ví dụ: BV Đa Khoa Sài Gòn hoặc null",
  "gia_tri_su_dung_tu": "DD/MM/YYYY",
  "gia_tri_su_dung_den": "DD/MM/YYYY hoặc null",
  "con_han": true,
  "ghi_chu": "string hoặc null"
}"""


class BhytData(BaseModel):
    """Structured payload extracted from a BHYT card.

    Carries parallel representations of the same card so that the Vietnamese
    verification table, the React booking form at :5173 and the Spring Boot
    services can consume it without a mapping layer. The snake_case fields stay
    canonical; every derived display/ISO field is computed in
    `normalise_bhyt_payload` so all entry points stay consistent.
    """

    # --- Vietnamese fields (verification table display) ---
    ho_ten: str | None = None
    ma_so_bhyt: str | None = None  # raw 15 alnum code (e.g. DN4797912345678)
    ma_so_bhyt_formatted: str | None = None  # display code with spaces (DN 4 79 79 12345678)
    ngay_sinh: str | None = None  # card printed format: DD/MM/YYYY
    ngay_sinh_iso: str | None = None  # form input format: YYYY-MM-DD
    gioi_tinh: str | None = None
    ma_noi_dkkcb_ban_dau: str | None = None  # e.g. 79-014
    noi_kham_chua_benh_ban_dau: str | None = None  # e.g. BV Đa Khoa Sài Gòn
    noi_kcb_ban_dau_full: str | None = None  # combined: 79-014 (BV Đa Khoa Sài Gòn)
    gia_tri_su_dung_tu: str | None = None  # DD/MM/YYYY
    gia_tri_su_dung_den: str | None = None  # DD/MM/YYYY
    con_han: bool | None = None
    ghi_chu: str | None = None

    # --- camelCase fields (React frontend & Spring Boot API compatibility) ---
    fullName: str | None = None
    insuranceCode: str | None = None
    dateOfBirth: str | None = None  # YYYY-MM-DD
    gender: str | None = None  # "Nam" or "Nữ"
    initialHospitalCode: str | None = None
    validFrom: str | None = None  # YYYY-MM-DD
    validUntil: str | None = None  # YYYY-MM-DD
    isExpired: bool = False
    isOcrVerified: bool = True


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def optimize_image(
    image_bytes: bytes,
    max_dimension: int = MAX_DIMENSION,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> bytes:
    """Auto-orient via EXIF, downscale, flatten to RGB and compress to JPEG.

    Quality walks down a fixed ladder until the encoded payload fits the
    byte budget, so a 12MP phone capture still leaves under ~200 KB on the wire.
    """
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Tệp ảnh quá lớn ({len(image_bytes) // 1024 // 1024} MB). "
            f"Vui lòng chọn ảnh dưới {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
        )

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)  # honour camera orientation
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Tệp tải lên không phải là ảnh hợp lệ (.jpg, .jpeg, .png, .heic, .webp)."
        ) from exc

    # NOTE: Image.thumbnail() resizes in place and returns None.
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    image = image.convert("RGB")  # flatten alpha / palette / CMYK for JPEG

    # Walk down the quality ladder at full size; if the payload still misses the
    # budget (e.g. a very noisy camera capture), shrink further and try again.
    # This always converges for any input instead of silently overshooting.
    encoded = b""
    scale = 1.0
    for _ in range(4):
        if scale < 1.0:
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        for quality in QUALITY_LADDER:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            encoded = buffer.getvalue()
            if len(encoded) <= max_bytes:
                break
        if len(encoded) <= max_bytes:
            break
        scale *= 0.8

    logger.info(
        "Ảnh đã tối ưu: %sx%s -> %.1f KB (mục tiêu <= %d KB)",
        image.width,
        image.height,
        len(encoded) / 1024,
        max_bytes // 1024,
    )
    return encoded


def _parse_vn_long_date(text: str) -> str | None:
    """Handle Vietnamese long form, e.g. 'Ngày 10 tháng 12 năm 2020'."""
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    if len(numbers) != 3:
        return None
    day, month, year = numbers
    try:
        return date(year, month, day).strftime("%d/%m/%Y")
    except ValueError:
        return None


def _normalise_date(value: object) -> str | None:
    """Coerce any date shape the VLM emits into DD/MM/YYYY."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or _strip_accents(text).lower() in {"null", "none", "n/a", "khong"}:
        return None

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(text, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue

    return _parse_vn_long_date(text) or text


def to_iso_date(date_str: str | None) -> str | None:
    """DD/MM/YYYY -> YYYY-MM-DD for `<input type="date">` and Java LocalDate.

    Returns None for anything unparseable rather than passing the raw string
    through: Spring's LocalDate binding throws on a malformed date, so a null
    is the only safe fallback for an OCR field we could not read.
    """
    if not date_str:
        return None
    text = str(date_str).strip()
    if not text:
        return None

    # Already ISO, or a near miss such as "1980-1-8".
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    # Vietnamese long form, e.g. "Ngày 10 tháng 12 năm 2020".
    long_form = _parse_vn_long_date(text)
    if long_form:
        try:
            return datetime.strptime(long_form, "%d/%m/%Y").date().isoformat()
        except ValueError:
            return None
    return None


# Backwards-compatible private alias for the pre-rename call sites.
_to_iso_date = to_iso_date


def normalise_insurance_code(value: object) -> str | None:
    """Cards are printed with spaces/dashes; the canonical form is 15 alnum chars."""
    if value is None:
        return None
    code = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
    return code or None


def format_insurance_code(code: str | None) -> str | None:
    """Group a 15-character BHYT code the way the card prints it.

    `DN4797912345678` -> `DN 4 79 79 12345678` (2 / 1 / 2 / 2 / 8 groups).
    Short or malformed codes are returned uppercased but ungrouped, so a
    misread code stays visibly wrong instead of being silently reshaped.
    """
    if not code:
        return None
    c = re.sub(r"[^A-Za-z0-9]", "", str(code)).upper()
    if not c:
        return None
    if len(c) == INSURANCE_CODE_LENGTH:
        return f"{c[0:2]} {c[2]} {c[3:5]} {c[5:7]} {c[7:]}"
    return c


def _normalise_gender(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    flat = _strip_accents(text).lower()
    if flat in {"nam", "male", "name", "gioi tinh nam", "gioitinhnam"}:
        return "Nam"
    if flat in {"nu", "female", "gioi tinh nu", "gioitinhnu"}:
        return "Nữ"
    return text


def _parse_iso_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return datetime.strptime(text, "%d/%m/%Y").date()
    except ValueError:
        return None


def _extract_json_payload(raw_reply: str) -> dict:
    """Tolerate markdown fences or prose around the JSON object."""
    text = raw_reply.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("VLM trả về JSON không phải object.")
    return parsed


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


# camelCase -> snake_case aliases, so callers may send either convention.
_CAMEL_TO_SNAKE = {
    "fullName": "ho_ten",
    "insuranceCode": "ma_so_bhyt",
    "dateOfBirth": "ngay_sinh",
    "gender": "gioi_tinh",
    "initialHospitalCode": "noi_kham_chua_benh_ban_dau",
    "initialHospitalName": "noi_kham_chua_benh_ban_dau",
    "initialHospitalCodeNumber": "ma_noi_dkkcb_ban_dau",
    "maNoiDkkcbBanDau": "ma_noi_dkkcb_ban_dau",
    "validFrom": "gia_tri_su_dung_tu",
    "validUntil": "gia_tri_su_dung_den",
}


def alias_camel_keys(raw: Mapping[str, object]) -> dict:
    """Fold camelCase keys onto their snake_case equivalents.

    snake_case wins when both are supplied. `isExpired` is the inverse of
    `con_han`, and `isOcrVerified` is a derived output rather than an input, so
    both are handled explicitly.
    """
    data = dict(raw)
    for camel, snake in _CAMEL_TO_SNAKE.items():
        if data.get(snake) is None and data.get(camel) is not None:
            data[snake] = data[camel]
    if data.get("con_han") is None and data.get("isExpired") is not None:
        data["con_han"] = not bool(data["isExpired"])
    return data


def normalise_bhyt_payload(raw: Mapping[str, object]) -> BhytData:
    """Map any raw mapping onto the canonical schema and normalise every value.

    Shared by the VLM response path and the patient-confirmed save path so both
    produce identical, fully-populated records. Accepts snake_case or camelCase
    input and always emits both.
    """
    raw = alias_camel_keys(raw)

    raw_code = normalise_insurance_code(raw.get("ma_so_bhyt"))
    ngay_sinh_raw = _normalise_date(raw.get("ngay_sinh"))
    ma_kcb = _text_or_none(raw.get("ma_noi_dkkcb_ban_dau"))
    ten_kcb = _text_or_none(raw.get("noi_kham_chua_benh_ban_dau"))

    # The booking form shows one line: "79-014 (BV Đa Khoa Sài Gòn)". Fall back
    # to whichever half was actually legible on the card.
    if ma_kcb and ten_kcb:
        kcb_full = f"{ma_kcb} ({ten_kcb})"
    else:
        kcb_full = ten_kcb or ma_kcb

    data = BhytData(
        ho_ten=_text_or_none(raw.get("ho_ten")),
        ma_so_bhyt=raw_code,
        ma_so_bhyt_formatted=format_insurance_code(raw_code),
        ngay_sinh=ngay_sinh_raw,
        ngay_sinh_iso=to_iso_date(ngay_sinh_raw),
        gioi_tinh=_normalise_gender(raw.get("gioi_tinh")),
        ma_noi_dkkcb_ban_dau=ma_kcb,
        noi_kham_chua_benh_ban_dau=ten_kcb,
        noi_kcb_ban_dau_full=kcb_full,
        gia_tri_su_dung_tu=_normalise_date(raw.get("gia_tri_su_dung_tu")),
        gia_tri_su_dung_den=_normalise_date(raw.get("gia_tri_su_dung_den")),
        ghi_chu=_text_or_none(raw.get("ghi_chu")),
    )

    # Trust the card's printed expiry over the model's arithmetic when we can parse it.
    expiry = _parse_iso_date(data.gia_tri_su_dung_den)
    if expiry is not None:
        data.con_han = expiry >= date.today()
    elif raw.get("con_han") is not None:
        data.con_han = bool(raw["con_han"])

    # camelCase mirror for the React frontend / Spring Boot services.
    data.fullName = data.ho_ten
    data.insuranceCode = data.ma_so_bhyt
    data.dateOfBirth = to_iso_date(data.ngay_sinh)
    data.gender = data.gioi_tinh
    data.initialHospitalCode = data.noi_kham_chua_benh_ban_dau
    data.validFrom = to_iso_date(data.gia_tri_su_dung_tu)
    data.validUntil = to_iso_date(data.gia_tri_su_dung_den)
    # An unknown expiry is not evidence of expiry; only con_han=False implies it.
    data.isExpired = not (data.con_han if data.con_han is not None else True)
    data.isOcrVerified = bool(
        data.ma_so_bhyt and len(data.ma_so_bhyt) == INSURANCE_CODE_LENGTH
    )

    return data


def extract_bhyt_from_bytes(
    image_bytes: bytes, optimized: bytes | None = None
) -> tuple[BhytData, str, bytes]:
    """Run the optimized image through the Groq VLM and return parsed data.

    Pass `optimized` to reuse an encode already produced by the caller.
    The Groq client is sync/blocking, so call this from a worker thread.
    """
    settings = get_settings()
    if optimized is None:
        optimized = optimize_image(image_bytes)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(optimized).decode("utf-8")

    response = get_groq_client().client.chat.completions.create(
        model=settings.VLM_MODEL_ID,
        messages=[
            {"role": "system", "content": BHYT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Đây là ảnh thẻ BHYT. Hãy bóc tách dữ liệu theo schema JSON bắt buộc ở trên."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        temperature=0.0,
        max_completion_tokens=600,
        response_format={"type": "json_object"},
    )

    raw_reply = response.choices[0].message.content or "{}"
    logger.info("VLM (%s) trả về: %s", settings.VLM_MODEL_ID, raw_reply)

    data = normalise_bhyt_payload(_extract_json_payload(raw_reply))
    if not any(data.model_dump().values()):
        data.ghi_chu = "Mô hình không nhận ra nội dung thẻ. Vui lòng chụp lại rõ nét hơn."
    return data, raw_reply, optimized


def validate_bhyt(data: BhytData) -> list[str]:
    """Human-review warnings driving the status badges in the verification table."""
    warnings: list[str] = []

    code = data.ma_so_bhyt or ""
    if not code:
        warnings.append("ma_so_bhyt: Chưa đọc được mã số thẻ - vui lòng nhập tay.")
    elif len(code) != INSURANCE_CODE_LENGTH:
        warnings.append(
            f"ma_so_bhyt: Mã thẻ có {len(code)}/{INSURANCE_CODE_LENGTH} ký tự - cần kiểm tra lại."
        )

    if not data.ho_ten:
        warnings.append("ho_ten: Chưa đọc được họ tên - vui lòng nhập tay.")

    if not data.ngay_sinh:
        warnings.append("ngay_sinh: Chưa đọc được ngày sinh - vui lòng nhập tay.")
    elif _parse_iso_date(data.ngay_sinh) is None:
        warnings.append(f"ngay_sinh: Định dạng '{data.ngay_sinh}' không đúng DD/MM/YYYY.")

    if not data.gioi_tinh:
        warnings.append("gioi_tinh: Chưa đọc được giới tính - vui lòng chọn tay.")

    if not data.gia_tri_su_dung_den:
        warnings.append("gia_tri_su_dung_den: Không đọc được ngày hết hạn - không xác định được còn hạn.")
    elif data.con_han is False:
        warnings.append("gia_tri_su_dung_den: Thẻ đã HẾT HẠN - vui lòng liên hệ cơ quan BHXH.")

    return warnings