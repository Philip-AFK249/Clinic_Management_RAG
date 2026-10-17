"""Ambient consultation intelligence: diarization, red flags, SOAP/ICD/Rx draft.

Everything in this module works on **text only**. Audio never reaches it - the
gateway transcribes the in-memory buffer in ``app/api.py`` and drops the bytes
before any of these functions runs, so only structured clinical text (dialogue
transcripts, symptoms, SOAP notes, ICD-10 codes, prescription drafts) survives.

LLM answers are untrusted: every payload is parsed strictly (markdown fences
stripped, first JSON object extracted) and then sanitised against the clinical
contract before it can reach a response.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.clinical_client import AvailableDrug

logger = get_logger(__name__)

#: WHO ICD-10 morphology: letter, 2 digits, optional sub-code. 'U' is reserved
#: for special-purpose codes, so anything malformed is dropped, never guessed.
ICD10_RE = re.compile(r"^[A-TV-Z][0-9][0-9AB](\.[0-9A-TV-Z]{1,4})?$", re.IGNORECASE)

#: Models this pipeline is written for (the gateway may pick either).
PREFERRED_LLM_MODELS = ("openai/gpt-oss-120b", "llama-3.3-70b-versatile")

#: Bounded loudness so one runaway turn cannot blow up the prompt.
_MAX_TURN_CHARS = 800
_MAX_FIELD_CHARS = 2000

#: Token budget for every ambient JSON completion: short outputs serialise
#: fast, and a runaway generation cannot stall the consultation.
AMBIENT_MAX_COMPLETION_TOKENS = 600

#: Reasoning effort for the reasoning models this pipeline targets. gpt-oss
#: burns its completion budget on hidden reasoning first (600+ tokens at the
#: default effort), which alone exhausts the 600-token ceiling above; 'low'
#: keeps the JSON inside the budget and the reply fast. Models outside the
#: gpt-oss family reject the parameter, so only they get it.
AMBIENT_REASONING_EFFORT = "low"

#: Reasoning-capable models understood by ``AMBIENT_REASONING_EFFORT``.
_REASONING_MODELS = ("gpt-oss",)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_JSON_REMINDER = (
    "\n\nLƯU Ý BẮT BUỘC: Trả về đúng MỘT đối tượng JSON thuần (bắt đầu bằng '{', "
    "kết thúc bằng '}'), không markdown, không chú thích, không văn bản ngoài JSON."
)


class AmbientPipelineError(RuntimeError):
    """The model returned output the clinical contract cannot accept."""


# ---------------------------------------------------------------------------
# Response / domain models shared with the gateway
# ---------------------------------------------------------------------------


class DiarizedTurn(BaseModel):
    """One labelled utterance of the consultation (no voiceprint involved)."""

    speaker: str = Field(pattern="^(doctor|patient)$")
    text: str = Field(min_length=1, max_length=4000)


class SoapNote(BaseModel):
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""


class IcdDiagnosis(BaseModel):
    icd10Code: str
    diseaseName: str
    isPrimary: bool = False
    aiConfidence: int = Field(default=0, ge=0, le=100)

    @model_validator(mode="before")
    @classmethod
    def _accept_snake_case(cls, value: Any) -> Any:
        """Accept the snake_case spelling some clients send alongside camelCase."""
        if isinstance(value, dict):
            aliases = {
                "icd10_code": "icd10Code",
                "disease_name": "diseaseName",
                "is_primary": "isPrimary",
                "ai_confidence": "aiConfidence",
            }
            for source, target in aliases.items():
                if source in value and target not in value:
                    value[target] = value.pop(source)
        return value


class PrescriptionItemDraft(BaseModel):
    """A draft line already matched against the live formulary."""

    drugId: int
    drugCode: str = ""
    drugName: str
    routeFrequency: str = ""
    duration: str = ""
    quantity: int = Field(default=1, ge=0)
    currentStock: int = Field(default=0, ge=0)
    stockWarning: Optional[str] = None
    concentration: str = ""
    dosageForm: str = ""
    bhytCoverage: str = ""
    isPenicillinClass: bool = False

    @model_validator(mode="before")
    @classmethod
    def _accept_snake_case(cls, value: Any) -> Any:
        if isinstance(value, dict):
            aliases = {
                "drug_id": "drugId",
                "drug_code": "drugCode",
                "drug_name": "drugName",
                "route_frequency": "routeFrequency",
                "current_stock": "currentStock",
                "stock_warning": "stockWarning",
                "dosage_form": "dosageForm",
                "bhyt_coverage": "bhytCoverage",
                "is_penicillin_class": "isPenicillinClass",
            }
            for source, target in aliases.items():
                if source in value and target not in value:
                    value[target] = value.pop(source)
        return value


@dataclass
class DiarizationResult:
    turns: list[DiarizedTurn]
    chief_complaints: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)

    @property
    def transcript_text(self) -> str:
        """Canonical ``doctor: ...`` / ``patient: ...`` rendering."""
        return "\n".join(f"{turn.speaker}: {turn.text}" for turn in self.turns)


@dataclass
class RedFlagResult:
    has_red_flags: bool = False
    red_flag_alert: str = ""
    recommended_inquiry: str = ""
    matched_patterns: list[str] = field(default_factory=list)


@dataclass
class ClinicalDraft:
    """Structured outcome of one synthesis / revision pass."""

    soap_note: SoapNote
    diagnoses: list[IcdDiagnosis] = field(default_factory=list)
    prescription_draft: list[PrescriptionItemDraft] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strict JSON handling
# ---------------------------------------------------------------------------


def parse_llm_json(raw: Any) -> dict:
    """Parse one JSON object out of an LLM answer, tolerating markdown fences.

    Accepts ``{"a":1}``, ```` ```json {"a":1}``` ````, and prose wrapped around
    the object; anything that still is not a JSON object raises.
    """
    if isinstance(raw, dict):
        return raw
    text = (raw or "").strip() if isinstance(raw, str) else ""
    if not text:
        raise AmbientPipelineError("Mô hình trả về chuỗi rỗng thay vì JSON.")

    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text)

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1].strip())

    errors: list[str] = []
    for candidate in candidates:
        body = candidate.strip()
        if body.lower().startswith("json"):
            body = body[4:].lstrip(":\n\r\t ")
        if not body:
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            errors.append(exc.msg)
            continue
        if isinstance(payload, dict):
            return payload
        errors.append("không phải đối tượng JSON")

    raise AmbientPipelineError(
        "Không đọc được JSON hợp lệ từ mô hình (" + "; ".join(errors[:3]) + ")."
    )


def complete_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    settings: Settings | None = None,
    temperature: float = 0.1,
    max_completion_tokens: int | None = AMBIENT_MAX_COMPLETION_TOKENS,
) -> dict:
    """Run one chat completion and return a validated JSON object.

    Tries ``response_format=json_object`` first (best compliance), falls back to
    a plain completion when the model rejects it, and retries once with a
    stricter reminder before giving up. Temperature and the token budget are
    kept low so the answer comes back as small, deterministic JSON.
    """
    from app.rag.generator import get_groq_client

    client = get_groq_client()
    failures: list[str] = []
    reasoning_effort = (
        AMBIENT_REASONING_EFFORT
        if any(name in (model or "").lower() for name in _REASONING_MODELS)
        else None
    )

    for attempt in range(2):
        system = system_prompt if attempt == 0 else system_prompt + _JSON_REMINDER
        # Attempt 0 honours the tight latency budget. A long transcript can
        # push the answer past that ceiling (finish_reason=length, unparseable
        # JSON), so the retry - which exists anyway for malformed output - is
        # given a larger budget instead of failing the consultation.
        budget = max_completion_tokens
        if attempt == 1 and budget:
            budget = budget * 4
            logger.info(
                "Ambient JSON retry with a raised token ceiling (%d)", budget
            )
        try:
            try:
                raw = client.complete(
                    system,
                    user_prompt,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    model=model,
                    max_completion_tokens=budget,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as exc:  # model refuses json_object / transient API error
                logger.warning(
                    "json_object response_format unavailable for '%s' (%s); "
                    "retrying as a plain completion",
                    model,
                    exc,
                )
                raw = client.complete(
                    system,
                    user_prompt,
                    temperature=temperature,
                    model=model,
                    max_completion_tokens=budget,
                    reasoning_effort=reasoning_effort,
                )
            return parse_llm_json(raw)
        except AmbientPipelineError as exc:
            failures.append(str(exc))
            logger.warning(
                "Ambient LLM JSON parse failed (attempt %d/2): %s", attempt + 1, exc
            )
        except Exception as exc:
            raise AmbientPipelineError(
                f"Không gọi được mô hình '{model}': {type(exc).__name__}: {exc}"
            ) from exc

    raise AmbientPipelineError(" ".join(failures) or "Không đọc được JSON từ mô hình.")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DIARIZATION_SYSTEM_PROMPT = """Bạn là trợ lý lâm sàng phân loại hội thoại khám bệnh.
Dựa trên [TRANSCRIPT THÔ] do mô hình nhận dạng giọng nói tạo ra, hãy gán vai trò từng câu thoại KHÔNG dùng dấu vết giọng nói, chỉ dùng ngữ cảnh và ngôn từ, rồi trả về đúng MỘT đối tượng JSON:

{"diarized_turns": [{"speaker": "doctor", "text": "..."}], "chief_complaints": ["..."], "symptoms": ["..."]}

Analyze Vietnamese conversational doctor-patient interaction. Any interrogative inquiry asking about medical history, allergies, symptom duration, dietary intake, or physical examination requests (e.g., 'Anh/chị/bác có bị... không?', 'Bị bao lâu rồi?', 'Ăn uống gì lạ không?', 'Để bác sĩ xem/khám...') MUST be attributed to speaker: 'doctor'. When a single paragraph contains both a question and a subsequent answer, you MUST split them into alternating 'doctor' and 'patient' turns. Opening greetings inviting the patient into the examination room (e.g. 'Rồi, mời anh vào', 'Chào bác, mời ngồi', 'Hôm nay anh thấy thế nào?') are 100% spoken by the 'doctor'. Never assign room entry invitations to the patient.

Quy tắc phân vai:
1. doctor: câu hỏi thăm khám, chỉ định thăm khám/vận động ("nào, bác sĩ khám bụng nhé", "hít thở sâu nào"), yêu cầu xét nghiệm, giải thích chẩn đoán, tư vấn và kê đơn. Câu hỏi khai thác bệnh sử/dị ứng/thời gian triệu chứng/ăn uống đều là doctor, kể cả khi Whisper ghi sai chính tả.
2. patient: lời phàn nàn, mô tả cảm giác đau/khó chịu, diễn biến theo thời gian, tiền sử, trả lời câu hỏi của bác sĩ.
3. speaker CHỈ được là "doctor" hoặc "patient". Không bịa thêm nội dung không có trong transcript; giữ nguyên tiếng Việt y khoa.
4. Gộp các câu liên tiếp của cùng một người nói thành một turn; tách turn khi đổi người nói. Mỗi turn 1-2 câu. Một câu hỏi và câu trả lời LUÔN ở hai turn riêng biệt xen kẽ doctor/patient.
5. chief_complaints: 2-6 vấn đề chính rút gọn (ví dụ: "Zona đau rát vùng liên sườn bên trái 5 ngày").
6. symptoms: danh sách triệu chứng cụ thể nghe được (kèm thời gian nếu có), tối đa 12 mục."""

RED_FLAG_SYSTEM_PROMPT = """Bạn là bác sĩ tầm soát "cờ đỏ" (red flags) trong lúc khám bệnh.
Đọc [HỘI THOẠI] và [TRIỆU CHỨNG], chỉ báo true khi có DẤU HIỆU CỤ THỂ trong hội thoại. Ưu tiên rà soát:

1. Suy giảm miễn dịch / nhiễm trùng cơ hội (HIV, Lao):
   - Zona diện rộng hoặc nhiều rễ da, zona ở người trẻ, nổi mụn nước rải rác;
   - Lở miệng / nấm Candida miệng tái phát (oral candidiasis);
   - Sốt kéo dài không rõ nguyên nhân, đổ mồ hôi đêm, gầy sút cân nhanh;
   - Ho kéo dài, khạc đờm lẫn máu (hemoptysis), đau ngực, mệt mỏi kéo dài.
2. Phản ứng da nặng do thuốc: Stevens-Johnson / toxic epidermal necrolysis (ban đỏ mụn nước lan nhanh, bong tróc da, lở loét niêm mạc mắt, miệng, sinh dục, sốt sau khi dùng thuốc mới).
3. Cấp cứu khác: sốc phản vệ, đau ngực dữ dội, khó thở nặng, lú lẫn, hạ huyết áp.

Trả về đúng MỘT đối tượng JSON:
{"has_red_flags": false, "red_flag_alert": "", "recommended_inquiry": "", "matched_patterns": []}

- has_red_flags=true thì red_flag_alert = cảnh báo chi tiết cho bác sĩ (tiếng Việt, có thể mở đầu bằng "⚠️"), nêu rõ hội chứng nghi ngờ và đề xuất chỉ định/khai thác;
  recommended_inquiry = MỘT câu bác sĩ nên hỏi bệnh nhân ngay;
  matched_patterns = các dấu hiệu khớp (ngắn gọn).
- Không có dấu hiệu → false, chuỗi rỗng, mảng rỗng. Tuyệt đối không kết luận chắc chắn bệnh, chỉ cảnh báo để khai thác thêm."""

SYNTHESIS_SYSTEM_PROMPT = """Bạn là bác sĩ sĩ soạn hồ sơ bệnh án điện tử cho bác sĩ lâm sàng.
Dựa trên hội thoại khám bệnh, triệu chứng, tiền sử, phác đồ và danh thuốc trong kho, trả về đúng MỘT đối tượng JSON:

{"soap_note": {"subjective": "...", "objective": "...", "assessment": "...", "plan": "..."},
 "diagnoses": [{"icd10Code": "B02.9", "diseaseName": "...", "isPrimary": true, "aiConfidence": 90}],
 "prescription_draft": [{"drugId": 1, "drugName": "...", "routeFrequency": "...", "duration": "...", "quantity": 35}]}

Bắt buộc:
1. SOAP: subjective = lời bệnh nhân tự kể + bệnh sử; objective = dấu hiệu thăm khám khách quan (ghi "Chưa có thông tin" nếu hội thoại không nêu); assessment = chẩn đoán phân biệt; plan = chỉ định xét nghiệm, điều trị, theo dõi, tái khám. Tiếng Việt y khoa, không đoán thừa.
2. diagnoses: mã ICD-10 đúng định dạng (chữ + 2 chữ số, ví dụ B02.9, J06.9, L51.2, B20); CHỈ dùng mã có thật; isPrimary=true đúng một cái (chẩn đoán chính); aiConfidence integer 0-100. Không tự bịa mã.
3. prescription_draft: CHỈ chọn thuốc có trong [THUỐC TRONG KHO] và ghi đúng drugId từ danh sách đó; tuyệt đối không kê thuốc ngoài danh sách.
   - Tôn trọng [DỊ ỨNG] (ví dụ dị ứng Penicillin thì không được kê Amoxicillin/Cephalosporin nếu tiền sử ghi rõ).
   - Tôn trọng cảnh báo [CỜ ĐỎ]/[TƯƠNG TÁC]: nếu thuốc có nguy cơ thì ghi chú vào plan thay vì kê.
   - Ưu tiên thuốc còn tồn kho tốt (bỏ qua mục đánh dấu LOW_STOCK khi còn lựa chọn khác).
   - routeFrequency: ví dụ "Uống 3 lần/ngày sau ăn"; duration: "7 ngày"; quantity = tổng số viên/ml cần cho toàn bộ thời gian điều trị (vd 3 lần/ngày x 7 ngày = 21 viên).
4. Thiếu dữ kiện thì ghi "Chưa có thông tin", không bịa."""

REVISION_SYSTEM_PROMPT = """Bạn là bác sĩ cập nhật hồ sơ bệnh án sau phản hồi bổ sung của bác sĩ trong lúc khám.
Nhận [HỒ SƠ TRƯỚC] và [PHẢN HỒI BÁC SĨ], trả về đúng MỘT đối tượng JSON:

{"assistant_message": "tóm tắt thay đổi bằng tiếng Việt",
 "soap_note": {"subjective": "...", "objective": "...", "assessment": "...", "plan": "..."},
 "diagnoses": [{"icd10Code": "...", "diseaseName": "...", "isPrimary": true, "aiConfidence": 90}],
 "prescription_draft": [{"drugId": 1, "drugName": "...", "routeFrequency": "...", "duration": "...", "quantity": 35}]}

Bắt buộc:
1. Cập nhật chủ động: thêm/bỏ chẩn đoán phụ phù hợp phản hồi (vd xác nhận nhiễm HIV 2 năm → thêm B20; đang dùng ARV → ghi vào subjective và cân nhắc tương tác thuốc).
2. Giữ nguyên phần hồ sơ không bị ảnh hưởng; chỉ sửa những gì phản hồi yêu cầu.
3. Cập nhật plan: chỉ định xét nghiệm, điều trị, theo dõi, tái khám theo thông tin mới.
4. prescription_draft CHỈ dùng drugId có trong [THUỐC TRONG KHO]; loại thuốc bệnh nhân dị ứng/đang chống chỉ định; quantity tính theo tổng thời gian điều trị.
5. ICD-10 có thật, đúng định dạng, đúng một isPrimary=true, aiConfidence 0-100.
6. Trả về JSON NHỎ: chỉ đưa vào những mục THAY ĐỊNH (assistant_message luôn có);
   bỏ qua hoàn toàn soap_note / diagnoses / prescription_draft không đổi - hệ thống
   tự giữ nguyên phần hồ sơ cũ đó. Không diễn đạt lại nội dung không thay đổi."""


# ---------------------------------------------------------------------------
# Step 2 - semantic diarization & role labelling
# ---------------------------------------------------------------------------

_DOCTOR_CUES = re.compile(
    r"^(nào|rồi,? mời|mời (anh|chị|bác|em|bạn)|để (tôi|bác sĩ)|tôi (khám|sẽ|nghe|cho)|"
    r"bác sĩ|chào (anh|chị|bác|em|bạn)|hôm nay (sao|thế nào|bị gì)|"
    r"(anh|chị|bạn|em|bác)\s+(có|đang|bị|ăn|uống|dùng|thấy)|"
    r"hít thở sâu|nằm xuống|tái khám|uống thuốc|xét nghiệm)",
    re.IGNORECASE,
)

_PATIENT_CUES = re.compile(
    r"(đau|rát|sốt|ho |khó thở|ngứa|mệt|chóng mặt|buồn nôn|nôn|tiêu chảy|nổi|mẩn|"
    r"chảy máu|sưng|tê|kéo dài|mấy ngày|từ hồi|tiền sử)",
    re.IGNORECASE,
)


def _guess_speaker(text: str, previous: Optional[str]) -> str:
    """Heuristic fallback used only when the model returned an unknown label."""
    stripped = text.strip()
    if _DOCTOR_CUES.search(stripped):
        return "doctor"
    if _PATIENT_CUES.search(stripped):
        return "patient"
    # Consultations open with the doctor, then alternate.
    return "patient" if previous == "doctor" else "doctor"


#: Whisper routinely drops punctuation, but when a question mark does survive
#: it is a reliable boundary between the doctor's inquiry and the answer.
_QUESTION_SPLIT = re.compile(r"\s*\?\s*")


def _split_question_then_answer(text: str, speaker: str) -> list[tuple[str, str]]:
    """Break a turn holding a doctor question *and* the answer into two turns.

    The model occasionally lumps the pair into a single "patient" turn, which
    then swallows the clinical history question. The split only fires when the
    sentence before the question mark reads as a doctor's inquiry, so patient
    speech is never chopped up.
    """
    match = _QUESTION_SPLIT.search(text)
    if not match:
        return [(speaker, text)]
    question = text[: match.start()].strip()
    answer = text[match.end() :].strip()
    if len(question) < 5 or not answer or not _DOCTOR_CUES.search(question):
        return [(speaker, text)]
    return [("doctor", question), ("patient", answer)]


def _clean_lines(value: Any, limit: int, max_chars: int) -> list[str]:
    """Coerce a model-provided list of strings into bounded clean strings."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    lines: list[str] = []
    for item in value:
        text = " ".join(str(item).split()).strip(" -;:")
        if not text:
            continue
        lines.append(text[:max_chars])
        if len(lines) >= limit:
            break
    return lines


def diarize_dialogue(
    raw_transcript: str,
    clinical_profile: str,
    model: str,
    settings: Settings | None = None,
) -> DiarizationResult:
    """Step 2: label turns doctor/patient and extract complaints + symptoms."""
    payload = complete_json(
        DIARIZATION_SYSTEM_PROMPT,
        f"[HỒ SƠ LÂM SÀNG]\n{clinical_profile}\n\n[TRANSCRIPT THÔ - WHISPER]\n{raw_transcript}",
        model,
        settings,
        temperature=0.1,
    )

    raw_turns = payload.get("diarized_turns") or payload.get("diarized_transcript") or []
    if isinstance(raw_turns, dict):
        raw_turns = [raw_turns]

    turns: list[DiarizedTurn] = []
    previous: Optional[str] = None
    for item in raw_turns:
        if isinstance(item, dict):
            speaker = str(item.get("speaker") or "").strip().lower()
            text = str(item.get("text") or item.get("content") or "").strip()
        elif isinstance(item, str):
            match = re.match(r"^\s*(doctor|patient|bác sĩ|bac si|bệnh nhân|benh nhan)\s*[:\-]\s*(.*)$",
                             item, re.IGNORECASE)
            speaker = (match.group(1) if match else "").lower()
            text = (match.group(2) if match else item).strip()
        else:
            continue

        if not text:
            continue
        speaker = _normalise_speaker(speaker) or _guess_speaker(text, previous)
        text = " ".join(text.split())[:_MAX_TURN_CHARS]
        for segment_speaker, segment_text in _split_question_then_answer(text, speaker):
            if turns and turns[-1].speaker == segment_speaker:
                # Merge consecutive utterances of the same speaker into one turn.
                merged = f"{turns[-1].text} {segment_text}".strip()[:_MAX_TURN_CHARS]
                turns[-1] = DiarizedTurn(speaker=segment_speaker, text=merged)
                continue
            turns.append(DiarizedTurn(speaker=segment_speaker, text=segment_text))
            previous = segment_speaker

    if not turns:
        raise AmbientPipelineError(
            "Không phân loại được hội thoại (mô hình không trả về turn nào)."
        )

    chief_complaints = _clean_lines(
        payload.get("chief_complaints"), limit=8, max_chars=240
    )
    symptoms = _clean_lines(payload.get("symptoms"), limit=16, max_chars=200)

    logger.info(
        "Diarization: %d turn(s), %d chief complaint(s), %d symptom(s)",
        len(turns),
        len(chief_complaints),
        len(symptoms),
    )
    return DiarizationResult(
        turns=turns,
        chief_complaints=chief_complaints,
        symptoms=symptoms,
    )


def _normalise_speaker(value: str) -> Optional[str]:
    if not value:
        return None
    if value in {"doctor", "bác sĩ", "bac si", "bac_si", "bs", "md", "physician"}:
        return "doctor"
    if value in {"patient", "bệnh nhân", "benh nhan", "benh_nhan"}:
        return "patient"
    return None


# ---------------------------------------------------------------------------
# Step 3 - red flags & infectious disease scanner
# ---------------------------------------------------------------------------


def scan_red_flags(
    diarization: DiarizationResult,
    clinical_profile: str,
    model: str,
    settings: Settings | None = None,
) -> RedFlagResult:
    """Step 3: look for opportunistic-infection / severe-drug-reaction patterns."""
    findings = " | ".join(
        diarization.chief_complaints + diarization.symptoms
    ) or "(không tóm tắt được triệu chứng)"
    payload = complete_json(
        RED_FLAG_SYSTEM_PROMPT,
        (
            f"[HỒ SƠ LÂM SÀNG]\n{clinical_profile}\n\n"
            f"[HỘI THOẠI ĐÃ PHÂN LOẠI]\n{diarization.transcript_text}\n\n"
            f"[TRIỆU CHỨNG TÓM TẮT]\n{findings}"
        ),
        model,
        settings,
        temperature=0.1,
    )

    return RedFlagResult(
        has_red_flags=_as_bool(payload.get("has_red_flags")),
        red_flag_alert=str(payload.get("red_flag_alert") or "").strip()[:_MAX_FIELD_CHARS],
        recommended_inquiry=str(payload.get("recommended_inquiry") or "").strip()[:600],
        matched_patterns=_clean_lines(payload.get("matched_patterns"), limit=8, max_chars=200),
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "có", "co"}
    return bool(value)


# ---------------------------------------------------------------------------
# Step 4 - knowledge RAG (degrades instead of failing)
# ---------------------------------------------------------------------------

#: Hard ceiling for one retrieval call: 2.5s. bge-m3 embedding is CPU-bound
#: and used to stall consultations for up to 90s on a cold cluster, so a
#: retrieval that cannot finish inside this budget is simply skipped (with a
#: warning) - a prompt without MoH guideline context beats a doctor waiting.
RAG_TIMEOUT_SECONDS = 2.5


def pgvector_reachable(settings: Settings) -> bool:
    """Cheap liveness probe - a down cluster must never trigger the bge-m3 load."""
    import psycopg

    try:
        with psycopg.connect(settings.db_dsn, connect_timeout=3, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return True
    except Exception:
        return False


def retrieve_knowledge_context(
    query: str,
    settings: Settings | None = None,
    top_k: int | None = None,
) -> tuple[str, list[str], list[str]]:
    """Vietnamese MoH guideline retrieval from ``clinic_knowledge_nodes`` (bge-m3).

    Returns ``(context, sources, warnings)``; every failure path returns an empty
    context plus a warning so synthesis can proceed on the transcript alone.
    """
    settings = settings or get_settings()
    if not query.strip():
        return "", [], []

    if not pgvector_reachable(settings):
        return (
            "",
            [],
            [
                "Kho tri thức pgvector chưa khả dụng nên đã bỏ qua bước RAG phác đồ "
                "Bộ Y tế; hồ sơ chỉ dựa trên hội thoại và danh mục thuốc."
            ],
        )

    from app.ingestion.embedder import warm_up_embedder
    from app.rag.retriever import retrieve_context

    try:
        # Single-flight the model load: two parallel consultations must not start
        # two multi-gigabyte bge-m3 loads at once.
        warm_up_embedder(settings)
        results = retrieve_context(
            query, top_k=top_k or settings.RETRIEVER_TOP_K, settings=settings
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}".strip().splitlines()[0][:200]
        logger.warning("Knowledge retrieval failed: %s", detail)
        return "", [], [f"Không truy xuất được phác đồ Bộ Y tế ({detail})."]

    if not results:
        return "", [], ["Không tìm thấy phác đồ phù hợp trong kho tri thức."]

    context = "\n\n---\n\n".join(result.content for result in results)
    sources = []
    for result in results:
        label = f"{result.doc_id} ({result.specialty})" if result.specialty else str(result.doc_id)
        if label not in sources:
            sources.append(label)
    return context, sources, []


# ---------------------------------------------------------------------------
# Step 5 - structured synthesis (SOAP, ICD-10, prescription draft)
# ---------------------------------------------------------------------------


def sanitise_soap(raw: Any) -> SoapNote:
    """Fold the model's SOAP object onto the four required sections."""
    raw = raw if isinstance(raw, dict) else {}

    def section(key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str):
            value = "" if value is None else str(value)
        text = " ".join(value.split()).strip()
        return text[:4000] or "Chưa có thông tin"

    return SoapNote(
        subjective=section("subjective"),
        objective=section("objective"),
        assessment=section("assessment"),
        plan=section("plan"),
    )


def sanitise_diagnoses(raw: Any) -> tuple[list[IcdDiagnosis], list[str]]:
    """Validate ICD-10 codes/confidence and enforce exactly one primary."""
    warnings: list[str] = []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return [], ["Mô hình không trả về danh sách chẩn đoán; hồ sơ chưa có ICD-10."]

    diagnoses: list[IcdDiagnosis] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        code = str(item.get("icd10Code") or item.get("icd10_code") or "").strip().upper()
        if not code:
            continue
        if not ICD10_RE.match(code):
            warnings.append(f"Mã ICD-10 không hợp lệ ({code}) đã bị bỏ qua.")
            continue
        if code in seen:
            continue
        seen.add(code)
        name = " ".join(str(item.get("diseaseName") or item.get("disease_name") or "").split())
        if not name:
            warnings.append(f"Bỏ qua chẩn đoán {code} vì thiếu tên bệnh.")
            continue
        try:
            confidence = int(float(item.get("aiConfidence", item.get("ai_confidence", 0)) or 0))
        except (TypeError, ValueError):
            confidence = 0
        diagnoses.append(
            IcdDiagnosis(
                icd10Code=code,
                diseaseName=name[:240],
                isPrimary=_as_bool(item.get("isPrimary", item.get("is_primary", False))),
                aiConfidence=max(0, min(100, confidence)),
            )
        )

    primaries = [d for d in diagnoses if d.isPrimary]
    if not primaries and diagnoses:
        best = max(diagnoses, key=lambda d: d.aiConfidence)
        best.isPrimary = True
        warnings.append(
            f"Mô hình không chọn chẩn đoán chính; tự đặt {best.icd10Code} làm chính."
        )
    elif len(primaries) > 1:
        for extra in primaries[1:]:
            extra.isPrimary = False
        warnings.append("Mô hình chọn nhiều chẩn đoán chính; chỉ giữ một.")
    return diagnoses, warnings


def reconcile_prescription(
    raw_items: Any, drugs: list[AvailableDrug]
) -> tuple[list[PrescriptionItemDraft], list[str]]:
    """Match the model's draft lines against the live formulary (strictly).

    A line that cannot be resolved to an in-stock ``drugs.id`` is dropped and
    reported; stock fields always come from the formulary, never from the model.
    """
    warnings: list[str] = []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raw_items = []
    if not drugs:
        if raw_items:
            warnings.append(
                "Danh mục thuốc không khả dụng nên dự thảo đơn thuốc bị bỏ trống "
                "(không thể đối chiếu drugId với kho)."
            )
        return [], warnings

    by_id = {drug.id: drug for drug in drugs}
    by_name: dict[str, AvailableDrug] = {}
    for drug in drugs:
        for key in {drug.name.strip().lower(), drug.label.strip().lower()}:
            if key and key not in by_name:
                by_name[key] = drug

    draft: list[PrescriptionItemDraft] = []
    seen: set[int] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        drug = _resolve_drug(item, by_id, by_name, drugs)
        if drug is None:
            quoted = item.get("drugName") or item.get("drug_name") or item.get("drugId") or "?"
            warnings.append(
                f"Bỏ qua '{quoted}': không có trong danh mục thuốc đang còn tồn kho."
            )
            continue
        if drug.id in seen:
            continue
        seen.add(drug.id)

        quantity = _to_int(item.get("quantity"), default=0)
        if quantity <= 0:
            quantity = 1
            warnings.append(f"Số lượng của {drug.name} không hợp lệ; đặt lại bằng 1.")
        stock_warning = drug.stock_warning
        if quantity > drug.stock_quantity:
            stock_warning = "INSUFFICIENT_STOCK"
            warnings.append(
                f"{drug.name}: cần {quantity} nhưng kho chỉ còn {drug.stock_quantity}."
            )
        if drug.is_penicillin_class:
            warnings.append(
                f"{drug.name} thuộc nhóm Penicillin - kiểm tra kỹ tiền sử dị ứng trước khi duyệt."
            )

        draft.append(
            PrescriptionItemDraft(
                drugId=drug.id,
                drugCode=drug.code,
                drugName=drug.name,
                routeFrequency=_text(item.get("routeFrequency") or item.get("route_frequency"), 200),
                duration=_text(item.get("duration"), 80),
                quantity=min(quantity, 100000),
                currentStock=drug.stock_quantity,
                stockWarning=stock_warning,
                concentration=drug.concentration,
                dosageForm=drug.dosage_form,
                bhytCoverage=drug.bhyt_coverage,
                isPenicillinClass=drug.is_penicillin_class,
            )
        )
    return draft, warnings


def _resolve_drug(
    item: dict,
    by_id: dict[int, AvailableDrug],
    by_name: dict[str, AvailableDrug],
    drugs: list[AvailableDrug],
) -> Optional[AvailableDrug]:
    """Resolve a draft line by drugId first, then by name/label."""
    raw_id = item.get("drugId", item.get("drug_id"))
    if raw_id is not None:
        try:
            drug = by_id.get(int(raw_id))
        except (TypeError, ValueError):
            drug = None
        if drug is not None:
            return drug

    name = _text(item.get("drugName") or item.get("drug_name"), 200).strip().lower()
    if not name:
        return None
    if name in by_name:
        return by_name[name]
    # Tolerate "amoxicillin 500mg" style echoes of name + concentration.
    for drug in drugs:
        if name.startswith(drug.name.strip().lower()):
            return drug
    return None


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.split())[:limit]


def _formulary_block(drugs: list[AvailableDrug]) -> str:
    if not drugs:
        return "(Danh mục thuốc đang trống - không được kê đơn trong lần này.)"
    return "\n".join(drug.compact_row() for drug in drugs)


def _build_user_prompt(
    diarization: DiarizationResult,
    red_flags: RedFlagResult,
    clinical_profile: str,
    knowledge_context: str,
    drugs: list[AvailableDrug],
) -> str:
    if red_flags.has_red_flags:
        patterns = ", ".join(red_flags.matched_patterns) or "có"
        red_flag_block = (
            f"CỜ ĐỎ ({patterns}): {red_flags.red_flag_alert}\n"
            "Câu hỏi nên hỏi bệnh nhân: "
            + (red_flags.recommended_inquiry or "-")
        )
    else:
        red_flag_block = "Không phát hiện cờ đỏ trong hội thoại."
    complaints = "\n".join(f"- {c}" for c in diarization.chief_complaints) or "- (không có)"
    symptoms = "\n".join(f"- {s}" for s in diarization.symptoms) or "- (không có)"
    return (
        f"[HỒ SƠ LÂM SÀNG]\n{clinical_profile}\n\n"
        f"[HỘI THOẠI ĐÃ PHÂN LOẠI]\n{diarization.transcript_text}\n\n"
        f"[CHỦ PHÀN CHÍNH]\n{complaints}\n\n"
        f"[TRIỆU CHỨNG]\n{symptoms}\n\n"
        f"[CỜ ĐỎ]\n{red_flag_block}\n\n"
        f"[PHÁC ĐỒ BỘ Y TẾ]\n{knowledge_context or '(Không tra được phác đồ trong phiên này)'}\n\n"
        "[THUỐC TRONG KHO] (id | mã | tên | hàm lượng | dạng | tồn kho | BHYT)\n"
        f"{_formulary_block(drugs)}"
    )


def synthesise_clinical_note(
    diarization: DiarizationResult,
    red_flags: RedFlagResult,
    clinical_profile: str,
    knowledge_context: str,
    drugs: list[AvailableDrug],
    model: str,
    settings: Settings | None = None,
) -> ClinicalDraft:
    """Step 5: SOAP note + ICD-10 diagnoses + formulary-matched prescription."""
    payload = complete_json(
        SYNTHESIS_SYSTEM_PROMPT,
        _build_user_prompt(diarization, red_flags, clinical_profile, knowledge_context, drugs),
        model,
        settings,
        temperature=0.1,
    )

    warnings: list[str] = []
    soap_note = sanitise_soap(payload.get("soap_note"))
    diagnoses, diagnosis_warnings = sanitise_diagnoses(payload.get("diagnoses"))
    warnings.extend(diagnosis_warnings)
    prescription, rx_warnings = reconcile_prescription(payload.get("prescription_draft"), drugs)
    warnings.extend(rx_warnings)

    if red_flags.has_red_flags and red_flags.recommended_inquiry:
        soap_note.plan = (
            f"{soap_note.plan} | CẦN KHAI THÁC THÊM: {red_flags.recommended_inquiry}"
        )

    return ClinicalDraft(
        soap_note=soap_note,
        diagnoses=diagnoses,
        prescription_draft=prescription,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Copilot chat - refine an existing draft from the doctor's follow-up
# ---------------------------------------------------------------------------


def revise_clinical_note(
    doctor_message: str,
    previous: ClinicalDraft,
    clinical_profile: str,
    knowledge_context: str,
    drugs: list[AvailableDrug],
    model: str,
    settings: Settings | None = None,
    conversation_history: str = "",
) -> tuple[ClinicalDraft, str]:
    """Apply the doctor's follow-up to the existing SOAP/ICD/Rx draft."""

    def _diagnosis_line(d: IcdDiagnosis) -> str:
        return f"- {d.icd10Code} {d.diseaseName} (chính={d.isPrimary}, tin cậy={d.aiConfidence})"

    def _rx_line(item: PrescriptionItemDraft) -> str:
        return (
            f"- {item.drugId} {item.drugName} {item.concentration}: {item.routeFrequency}, "
            f"{item.duration}, x{item.quantity} (tồn kho {item.currentStock})"
        )

    previous_block = (
        "SOAP:\n"
        f"- S: {previous.soap_note.subjective}\n"
        f"- O: {previous.soap_note.objective}\n"
        f"- A: {previous.soap_note.assessment}\n"
        f"- P: {previous.soap_note.plan}\n\n"
        "Chẩn đoán:\n" + ("\n".join(_diagnosis_line(d) for d in previous.diagnoses) or "- (chưa có)") + "\n\n"
        "Đơn thuốc dự thảo:\n" + ("\n".join(_rx_line(i) for i in previous.prescription_draft) or "- (chưa có)")
    )

    user_prompt = (
        f"[HỒ SƠ LÂM SÀNG]\n{clinical_profile}\n\n"
        f"[HỒ SƠ TRƯỚC]\n{previous_block}\n\n"
        + (f"[LỊCH SỬ HỘI THOẠI]\n{conversation_history}\n\n" if conversation_history else "")
        + f"[PHẢN HỒI BÁC SĨ]\n{doctor_message}\n\n"
        f"[PHÁC ĐỒ BỘ Y TẾ]\n{knowledge_context or '(Không tra được phác đồ trong phiên này)'}\n\n"
        f"[THUỐC TRONG KHO] (id | mã | tên | hàm lượng | dạng | tồn kho | BHYT)\n"
        f"{_formulary_block(drugs)}"
    )

    payload = complete_json(
        REVISION_SYSTEM_PROMPT, user_prompt, model, settings, temperature=0.1
    )

    assistant_message = _text(payload.get("assistant_message"), 800) or (
        "Đã cập nhật hồ sơ theo phản hồi của bác sĩ."
    )
    draft = apply_revision_payload(payload, previous, drugs)
    return draft, assistant_message


def _has_any_soap_section(raw: Any) -> bool:
    return isinstance(raw, dict) and any(
        _text(raw.get(key), 1).strip()
        for key in ("subjective", "objective", "assessment", "plan")
    )


def apply_revision_payload(
    payload: dict, previous: ClinicalDraft, drugs: list[AvailableDrug]
) -> ClinicalDraft:
    """Merge one revision payload onto the previous draft.

    A section the model left out keeps the previous content (stock is still
    re-checked against the live formulary) instead of silently wiping it; a
    section it returned - even an empty list - is taken at face value so the
    doctor can deliberately clear a prescription.
    """
    warnings: list[str] = []

    if _has_any_soap_section(payload.get("soap_note")):
        soap_note = sanitise_soap(payload.get("soap_note"))
    else:
        soap_note = previous.soap_note.model_copy(deep=True)
        warnings.append("Mô hình không trả về SOAP mới; giữ nguyên hồ sơ trước đó.")

    if "diagnoses" in payload:
        diagnoses, diagnosis_warnings = sanitise_diagnoses(payload["diagnoses"])
        warnings.extend(diagnosis_warnings)
    else:
        diagnoses = [item.model_copy(deep=True) for item in previous.diagnoses]
        warnings.append("Mô hình không trả về chẩn đoán mới; giữ nguyên danh sách cũ.")

    if "prescription_draft" in payload:
        prescription, rx_warnings = reconcile_prescription(
            payload["prescription_draft"], drugs
        )
    else:
        prescription, rx_warnings = reconcile_prescription(
            [item.model_dump() for item in previous.prescription_draft], drugs
        )
        warnings.append("Mô hình không trả về đơn thuốc mới; giữ nguyên dự thảo cũ.")
    warnings.extend(rx_warnings)

    return ClinicalDraft(
        soap_note=soap_note,
        diagnoses=diagnoses,
        prescription_draft=prescription,
        warnings=warnings,
    )
