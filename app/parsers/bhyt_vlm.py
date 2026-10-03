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

SCHEMA JSON BẮT BUỘC:
{
  "ho_ten": "HỌ VÀ TÊN IN HOA",
  "ma_so_bhyt": "15 ký tự chữ và số",
  "ngay_sinh": "DD/MM/YYYY",
  "gioi_tinh": "Nam hoặc Nữ",
  "noi_kham_chua_benh_ban_dau": "string",
  "gia_tri_su_dung_tu": "DD/MM/YYYY",
  "gia_tri_su_dung_den": "DD/MM/YYYY hoặc null",
  "con_han": true,
  "ghi_chu": "string hoặc null"
}"""


class BhytData(BaseModel):
    """Structured payload extracted from a BHYT card."""

    ho_ten: str | None = None
    ma_so_bhyt: str | None = None
    ngay_sinh: str | None = None
    gioi_tinh: str | None = None
    noi_kham_chua_benh_ban_dau: str | None = None
    gia_tri_su_dung_tu: str | None = None
    gia_tri_su_dung_den: str | None = None
    con_han: bool | None = None
    ghi_chu: str | None = None


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


def normalise_insurance_code(value: object) -> str | None:
    """Cards are printed with spaces/dashes; the canonical form is 15 alnum chars."""
    if value is None:
        return None
    code = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
    return code or None


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


def normalise_bhyt_payload(raw: dict) -> BhytData:
    """Map any raw mapping onto the canonical schema and normalise every value.

    Shared by the VLM response path and the patient-confirmed save path so both
    produce identical, fully-populated records.
    """
    data = BhytData(
        ho_ten=_text_or_none(raw.get("ho_ten")),
        ma_so_bhyt=normalise_insurance_code(raw.get("ma_so_bhyt")),
        ngay_sinh=_normalise_date(raw.get("ngay_sinh")),
        gioi_tinh=_normalise_gender(raw.get("gioi_tinh")),
        noi_kham_chua_benh_ban_dau=_text_or_none(raw.get("noi_kham_chua_benh_ban_dau")),
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