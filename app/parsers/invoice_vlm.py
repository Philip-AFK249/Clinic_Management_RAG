"""Invoice & medical receipt extraction powered by Groq's Qwen multimodal VLM.

The VLM reads the document directly from the optimized image. Nothing is ever
written to disk: the caller hands over raw bytes, the optimized JPEG is kept in
memory, and only the structured JSON payload leaves this module.
"""

from __future__ import annotations

import base64
import re
from datetime import date, datetime
from typing import Literal, Mapping

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.parsers.bhyt_vlm import (
    _extract_json_payload,
    _normalise_date,
    optimize_image,
    to_iso_date,
)
from app.rag.generator import get_groq_client

logger = get_logger(__name__)

InvoiceType = Literal["SUPPLIER_DRUG", "PATIENT_MEDICAL", "RECEPTION_RECEIPT"]

# Dense invoice tables need more pixels than a BHYT card, otherwise batch
# numbers, expiry dates and unit prices blur together.
INVOICE_MAX_DIMENSION = 1600
INVOICE_MAX_IMAGE_BYTES = 350_000

# VND has no fractional unit in practice; anything beyond this is a misread.
AMOUNT_TOLERANCE_VND = 1_000.0

_NULL_TEXT = {"null", "none", "n/a", "na", "khong", "không"}

_INVOICE_TYPE_GUIDE = {
    "SUPPLIER_DRUG": (
        "Đây là HÓA ĐƠN NHẬP THUỐC từ nhà cung cấp.\n"
        "- issuerName = tên nhà cung cấp / công ty dược.\n"
        "- patientOrBuyerName = tên người hoặc đơn vị mua nếu có.\n"
        "- items = từng loại thuốc; bắt buộc đọc kỹ số lô (batchNumber), "
        "hạn sử dụng (expiryDate), nồng độ/hàm lượng và đơn giá.\n"
        "- clinicalSummary và detectedMedications có thể để null/[] nếu không có."
    ),
    "PATIENT_MEDICAL": (
        "Đây là HÓA ĐƠN / BIÊN LAI VIỆN PHÍ hoặc hóa đơn mua thuốc của BỆNH NHÂN "
        "tại bệnh viện, phòng khám hoặc nhà thuốc bên ngoài.\n"
        "- issuerName = tên bệnh viện / phòng khám / nhà thuốc phát hành.\n"
        "- patientOrBuyerName = tên bệnh nhân / người mua.\n"
        "- items = các dịch vụ, xét nghiệm, thuốc đã dùng kèm số lượng và đơn giá.\n"
        "- clinicalSummary = tóm tắt ngắn gọn bằng tiếng Việt các chẩn đoán, bệnh sử "
        "và thuốc đã dùng in trên hóa đơn để điền patient_history / chiefComplaint.\n"
        "- detectedMedications = danh sách tên thuốc đọc được trên hóa đơn.\n"
        "- allergyHints = các ghi chú dị ứng / chống chỉ định nếu có in trên hóa đơn."
    ),
    "RECEPTION_RECEIPT": (
        "Đây là BIÊN LAI THU TIỀN tại quầy tiếp đón.\n"
        "- issuerName = đơn vị thu tiền / đơn vị phát hành biên lai.\n"
        "- patientOrBuyerName = tên người nộp tiền / bệnh nhân.\n"
        "- items = các khoản thu (khám bệnh, thủ thuật, thuốc, ...).\n"
        "- Không bắt buộc có số lô hay hạn sử dụng cho loại này."
    ),
}

_INVOICE_JSON_SCHEMA = """{
  "invoiceType": "SUPPLIER_DRUG | PATIENT_MEDICAL | RECEPTION_RECEIPT",
  "invoiceNumber": "string hoặc null",
  "invoiceDate": "DD/MM/YYYY hoặc null",
  "issuerName": "string hoặc null",
  "patientOrBuyerName": "string hoặc null",
  "totalAmount": 0,
  "items": [
    {
      "name": "tên thuốc/dịch vụ",
      "concentration": "nồng độ/hàm lượng hoặc null",
      "unit": "đơn vị tính hoặc null",
      "batchNumber": "số lô hoặc null",
      "expiryDate": "DD/MM/YYYY hoặc null",
      "quantity": 0,
      "unitPrice": 0,
      "lineTotal": 0
    }
  ],
  "clinicalSummary": "string hoặc null",
  "detectedMedications": ["tên thuốc"],
  "allergyHints": ["ghi chú dị ứng/chống chỉ định"],
  "notes": "string hoặc null"
}"""


def build_invoice_system_prompt(invoice_type: InvoiceType) -> str:
    """Build the strict JSON prompt for one of the three supported document types."""
    return (
        "Bạn là chuyên gia OCR hóa đơn và biên lai y tế Việt Nam.\n\n"
        "NHIỆM VỤ:\n"
        "Đọc trực tiếp trên ẢNH được cung cấp và trả về dữ liệu có cấu trúc.\n\n"
        f"LOẠI CHỨNG TỪ: {invoice_type}\n"
        f"{_INVOICE_TYPE_GUIDE[invoice_type]}\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. Ảnh có thể bị nghiêng, chụp lệch hoặc lóa sáng; hãy tự căn chỉnh góc nhìn trước khi đọc.\n"
        "2. TUYỆT ĐỐI KHÔNG bịa đặt, KHÔNG suy đoán. Nếu chữ mờ hoặc không có trên chứng từ, trả về null.\n"
        "3. Trả về DUY NHẤT một JSON object đúng các khoá trong schema, không thêm khoá khác.\n"
        "4. KHÔNG bọc trong markdown ```json``` và KHÔNG giải thích thêm.\n"
        "5. Ngày tháng luôn trả về định dạng DD/MM/YYYY.\n"
        "6. Tiền tệ là VND: totalAmount, unitPrice, lineTotal phải là số (không kèm ký hiệu tiền tệ, dấu chấm phân cách nghìn cũng không).\n"
        "7. quantity là số nguyên; không thêm đơn vị vào trường số.\n"
        "8. Đọc đầy đủ từng dòng hàng, kể cả số lô và hạn sử dụng in nhỏ.\n"
        "9. invoiceType phải giữ đúng giá trị đã cho ở trên.\n\n"
        "SCHEMA JSON BẮT BUỘC:\n"
        f"{_INVOICE_JSON_SCHEMA}"
    )


class InvoiceLineItem(BaseModel):
    """One drug, service or fee line read from the invoice."""

    name: str = ""
    concentration: str | None = None
    unit: str | None = None
    batchNumber: str | None = None
    expiryDate: str | None = None  # DD/MM/YYYY as printed
    expiryDateIso: str | None = None  # YYYY-MM-DD for storage / comparisons
    quantity: int | None = None
    unitPrice: float | None = None
    lineTotal: float | None = None


class InvoiceOcrData(BaseModel):
    """Unified invoice payload shared by all three document types."""

    invoiceType: str
    invoiceNumber: str | None = None
    invoiceDate: str | None = None  # DD/MM/YYYY
    invoiceDateIso: str | None = None  # YYYY-MM-DD
    issuerName: str | None = None
    patientOrBuyerName: str | None = None
    totalAmount: float | None = None
    items: list[InvoiceLineItem] = Field(default_factory=list)
    clinicalSummary: str | None = None
    detectedMedications: list[str] = Field(default_factory=list)
    allergyHints: list[str] = Field(default_factory=list)
    notes: str | None = None


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text or text.lower() in _NULL_TEXT:
        return None
    return text


def _first_present(*values: object) -> object:
    """Pick the first non-null alias, preserving numeric zero values."""
    for value in values:
        if value is not None:
            return value
    return None


def _to_float(value: object) -> float | None:
    """Coerce VND amounts printed with dots/commas/currency symbols to float."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in _NULL_TEXT:
        return None

    cleaned = re.sub(r"[^\d.,-]", "", text)
    if not cleaned:
        return None

    last_dot = cleaned.rfind(".")
    last_comma = cleaned.rfind(",")

    if last_dot != -1 and last_comma != -1:
        # The right-most separator is the decimal one.
        if last_comma > last_dot:
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif last_comma != -1:
        parts = cleaned.split(",")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(",", ".")
    elif last_dot != -1:
        parts = cleaned.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            cleaned = cleaned.replace(".", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


def _to_int(value: object) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(round(number))


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in _NULL_TEXT:
            return []
        parts: list[object] = re.split(r"[\n;]+", text)
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]

    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = _text_or_none(part)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _as_mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [entry for entry in value if isinstance(entry, Mapping)]


def _normalise_items(raw_items: object) -> list[InvoiceLineItem]:
    items: list[InvoiceLineItem] = []
    for entry in _as_mapping_list(raw_items):
        expiry = _normalise_date(
            _first_present(
                entry.get("expiryDate"),
                entry.get("expiry_date"),
                entry.get("expiryDateIso"),
            )
        )
        items.append(
            InvoiceLineItem(
                name=_text_or_none(entry.get("name")) or "",
                concentration=_text_or_none(
                    _first_present(
                        entry.get("concentration"),
                        entry.get("strength"),
                        entry.get("nong_do"),
                    )
                ),
                unit=_text_or_none(
                    _first_present(entry.get("unit"), entry.get("don_vi"))
                ),
                batchNumber=_text_or_none(
                    _first_present(
                        entry.get("batchNumber"),
                        entry.get("batch_number"),
                        entry.get("batch"),
                    )
                ),
                expiryDate=expiry,
                expiryDateIso=to_iso_date(expiry),
                quantity=_to_int(
                    _first_present(entry.get("quantity"), entry.get("so_luong"))
                ),
                unitPrice=_to_float(
                    _first_present(entry.get("unitPrice"), entry.get("unit_price"))
                ),
                lineTotal=_to_float(
                    _first_present(entry.get("lineTotal"), entry.get("line_total"))
                ),
            )
        )
    return items


def normalise_invoice_payload(
    raw: Mapping[str, object], invoice_type: InvoiceType
) -> InvoiceOcrData:
    """Map the VLM JSON onto the unified invoice schema."""
    invoice_date = _normalise_date(raw.get("invoiceDate") or raw.get("invoice_date"))
    return InvoiceOcrData(
        invoiceType=invoice_type,
        invoiceNumber=_text_or_none(
            raw.get("invoiceNumber") or raw.get("invoice_number")
        ),
        invoiceDate=invoice_date,
        invoiceDateIso=to_iso_date(invoice_date),
        issuerName=_text_or_none(raw.get("issuerName") or raw.get("issuer_name")),
        patientOrBuyerName=_text_or_none(
            raw.get("patientOrBuyerName")
            or raw.get("patient_or_buyer_name")
            or raw.get("patientName")
            or raw.get("patient_name")
        ),
        totalAmount=_to_float(
            _first_present(raw.get("totalAmount"), raw.get("total_amount"))
        ),
        items=_normalise_items(raw.get("items")),
        clinicalSummary=_text_or_none(
            raw.get("clinicalSummary") or raw.get("clinical_summary")
        ),
        detectedMedications=_string_list(
            raw.get("detectedMedications") or raw.get("detected_medications")
        ),
        allergyHints=_string_list(
            raw.get("allergyHints") or raw.get("allergy_hints")
        ),
        notes=_text_or_none(raw.get("notes")),
    )


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _format_vnd(value: float) -> str:
    return f"{value:,.0f} VNĐ"


def validate_invoice(data: InvoiceOcrData) -> list[str]:
    """Human-review warnings for the invoice verification table."""
    warnings: list[str] = []

    if not data.invoiceNumber:
        warnings.append("invoiceNumber: Chưa đọc được số hóa đơn - vui lòng nhập tay.")

    if not data.invoiceDate:
        warnings.append("invoiceDate: Chưa đọc được ngày hóa đơn - vui lòng nhập tay.")
    elif not data.invoiceDateIso:
        warnings.append(
            f"invoiceDate: Định dạng '{data.invoiceDate}' không đúng DD/MM/YYYY."
        )

    requires_items = data.invoiceType in {"SUPPLIER_DRUG", "PATIENT_MEDICAL"}
    if requires_items and not data.items:
        warnings.append(
            "items: Không đọc được dòng hàng/dịch vụ nào trên hóa đơn - "
            "vui lòng kiểm tra và nhập tay."
        )

    line_totals: list[float] = []
    all_lines_priced = True

    for index, item in enumerate(data.items, start=1):
        label = item.name or f"dòng {index}"

        if item.quantity is None:
            warnings.append(
                f"items[{index}] ({label}): Thiếu số lượng - vui lòng kiểm tra."
            )
        if item.unitPrice is None:
            warnings.append(
                f"items[{index}] ({label}): Thiếu đơn giá - vui lòng kiểm tra."
            )

        effective_total: float | None = None
        if item.quantity is not None and item.unitPrice is not None:
            expected = item.quantity * item.unitPrice
            if item.lineTotal is None:
                warnings.append(
                    f"items[{index}] ({label}): Thiếu thành tiền - vui lòng kiểm tra."
                )
                effective_total = expected
            else:
                if abs(expected - item.lineTotal) > AMOUNT_TOLERANCE_VND:
                    warnings.append(
                        f"items[{index}] ({label}): Thành tiền "
                        f"{_format_vnd(item.lineTotal)} không khớp số lượng × đơn giá "
                        f"({_format_vnd(expected)}) - lệch quá 1.000 VNĐ."
                    )
                effective_total = item.lineTotal
        elif item.lineTotal is not None:
            effective_total = item.lineTotal
        else:
            all_lines_priced = False

        if effective_total is not None:
            line_totals.append(effective_total)

        if data.invoiceType == "SUPPLIER_DRUG":
            expiry = _parse_iso_date(item.expiryDateIso)
            if expiry is None:
                warnings.append(
                    f"items[{index}] ({label}): Thiếu hoặc không đọc được hạn sử dụng - "
                    "vui lòng kiểm tra."
                )
            elif expiry < date.today():
                warnings.append(
                    f"items[{index}] ({label}): Thuốc đã HẾT HẠN "
                    f"({item.expiryDate or item.expiryDateIso}) - không nhập kho."
                )

    if data.totalAmount is None:
        warnings.append("totalAmount: Chưa đọc được tổng tiền - vui lòng kiểm tra.")
    elif data.items and all_lines_priced:
        computed_total = sum(line_totals)
        if abs(computed_total - data.totalAmount) > AMOUNT_TOLERANCE_VND:
            warnings.append(
                f"totalAmount: Tổng tiền {_format_vnd(data.totalAmount)} không khớp "
                f"tổng các dòng ({_format_vnd(computed_total)}) - lệch quá 1.000 VNĐ."
            )

    return warnings


def extract_invoice_from_bytes(
    image_bytes: bytes,
    invoice_type: InvoiceType,
    optimized: bytes | None = None,
    model: str | None = None,
) -> tuple[InvoiceOcrData, str, bytes]:
    """Run the optimized image through the Groq VLM and normalise the result.

    Pass ``optimized`` to reuse an encode already produced by the caller.
    The Groq client is sync/blocking, so call this from a worker thread.
    """
    settings = get_settings()
    vlm_model = model or settings.VLM_MODEL_ID
    if optimized is None:
        optimized = optimize_image(
            image_bytes,
            max_dimension=INVOICE_MAX_DIMENSION,
            max_bytes=INVOICE_MAX_IMAGE_BYTES,
        )
    data_uri = "data:image/jpeg;base64," + base64.b64encode(optimized).decode("utf-8")

    response = get_groq_client().client.chat.completions.create(
        model=vlm_model,
        messages=[
            {"role": "system", "content": build_invoice_system_prompt(invoice_type)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Đây là ảnh hóa đơn/biên lai y tế. Hãy bóc tách dữ liệu "
                            "theo đúng schema JSON bắt buộc ở trên."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        temperature=0.0,
        max_completion_tokens=4000,
        response_format={"type": "json_object"},
    )

    raw_reply = response.choices[0].message.content or "{}"
    logger.info("VLM hóa đơn (%s, %s) trả về: %s", vlm_model, invoice_type, raw_reply)

    data = normalise_invoice_payload(_extract_json_payload(raw_reply), invoice_type)
    return data, raw_reply, optimized
