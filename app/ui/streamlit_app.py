"""Streamlit web chat interface (migrated from web_chat.py)."""
import io
import sys
from pathlib import Path

# Đưa thư mục gốc dự án (clinic_rag_backend) vào sys.path để nhận diện package 'app'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.generator import DOCTOR_SYSTEM_PROMPT, TRIAGE_SYSTEM_PROMPT, get_groq_client
from app.rag.retriever import retrieve_context
from app.ingestion.embedder import get_embedder

logger = get_logger(__name__)
settings: Settings = get_settings()

# Streamlit's audio_input hands back a browser blob whose filename is often
# just ``audio.wav`` regardless of the real container. Trust the MIME type when
# it is available, otherwise fall back to the filename suffix.
_AUDIO_TYPE_TO_EXTENSION = {
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mp4": ".m4a",
    "audio/m4a": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
}
_AUDIO_EXTENSION_TO_MIME = {
    ".webm": "audio/webm",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
}
_STT_MODEL_FALLBACKS = ("whisper-large-v3-turbo", "whisper-large-v3")


def _audio_container(audio_value) -> tuple[str, str]:
    """Return ``(filename, mime)`` matching the uploaded container."""
    mime = (getattr(audio_value, "type", "") or "").split(";", 1)[0].strip().lower()
    name = getattr(audio_value, "name", "") or ""
    suffix = Path(name).suffix.lower()
    extension = _AUDIO_TYPE_TO_EXTENSION.get(mime) or suffix
    if extension not in _AUDIO_EXTENSION_TO_MIME:
        extension = ".webm"
    mime = _AUDIO_EXTENSION_TO_MIME[extension]
    return (f"voice_input{extension}", mime)


def _stt_models(configured: str) -> list[str]:
    candidates: list[str] = []
    if configured and configured.strip():
        candidates.append(configured.strip())
    for model in _STT_MODEL_FALLBACKS:
        if model not in candidates:
            candidates.append(model)
    return candidates


def _transcribe_audio(audio_value, audio_bytes: bytes | None = None) -> str:
    """Send a fresh BytesIO to Groq using bytes read once by the caller."""
    filename, mime = _audio_container(audio_value)
    audio_value.seek(0)
    if audio_bytes is None:
        audio_bytes = audio_value.read()
    if not audio_bytes:
        raise ValueError("Microphone recording was empty; please record again.")

    prompt = (
        "Khám bệnh, đau thắt ngực, khó thở, huyết áp, hồi hộp, "
        "tim mạch, sốt cao, ho khạc đờm, ho khan, đau rát họng, "
        "khò khè, hen suyễn, sổ mũi, viêm xoang, dị ứng, nổi mẩn đỏ, "
        "ngứa ngáy, mụn nước, dát sẩn, vảy nến, zona, sốc phản vệ, "
        "mề đay."
    )
    last_error: Exception | None = None
    for model in _stt_models(settings.LLM_STT_MODEL_ID):
        response = None
        serialization_failed = False
        try:
            response = client.client.audio.transcriptions.create(
                file=(filename, io.BytesIO(audio_bytes), mime),
                model=model,
                language="vi",
                prompt=prompt,
                response_format="text",
            )
        except Exception as exc:  # noqa: BLE001 - Groq raises several STT errors
            last_error = exc
            serialization_failed = True
            logger.warning("STT model '%s' failed: %s", model, exc)

        if response is None and serialization_failed:
            try:
                response = client.client.audio.transcriptions.create(
                    file=(filename, io.BytesIO(audio_bytes), mime),
                    model=model,
                    language="vi",
                    prompt=prompt,
                    response_format="json",
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("STT model '%s' failed: %s", model, exc)
                continue

        if response is None:
            continue

        if hasattr(response, "text"):
            text = response.text
        elif isinstance(response, dict):
            text = response.get("text", "")
        else:
            text = str(response)
        text = (text or "").strip()
        if text:
            return text
        last_error = RuntimeError("Empty transcription")
    raise RuntimeError(f"Không nhận diện được giọng nói: {last_error}")


SPECIALTY_MAPPING = {
    "Tất cả chuyên khoa": "all",
    "Tim mạch (tim_mach)": "tim_mach",
    "Nội khoa (noi_khoa)": "noi_khoa",
    "Da liễu (da_lieu)": "da_lieu",
}

st.set_page_config(
    page_title="Clinic AI Portal - Đa Chuyên Khoa",
    page_icon="🏥",
    layout="wide",
)


@st.cache_resource
def init_resources():
    embedder = get_embedder(settings)
    groq_client = get_groq_client()
    return embedder, groq_client


embed_model, client = init_resources()


def doctor_interface() -> None:
    st.title("👨‍⚕️ Tra Cứu Phác Đồ Bộ Y Tế (Tim Mạch - Nội Khoa - Da Liễu)")

    spec_choice = st.sidebar.selectbox("Lọc chuyên khoa:", list(SPECIALTY_MAPPING.keys()))
    selected_spec = SPECIALTY_MAPPING[spec_choice]

    if "doctor_messages" not in st.session_state:
        st.session_state.doctor_messages = []

    for msg in st.session_state.doctor_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Tài liệu trích dẫn"):
                    for s in msg["sources"]:
                        st.write(s)

    if doc_query := st.chat_input("Nhập câu hỏi chuyên môn: vảy nến, zona, nong van, chụp mạch vành..."):
        st.session_state.doctor_messages.append({"role": "user", "content": doc_query})
        with st.chat_message("user"):
            st.markdown(doc_query)

        with st.chat_message("assistant"):
            with st.status("Đang truy xuất phác đồ từ pgvector...", expanded=False):
                results = retrieve_context(
                    doc_query, top_k=3, specialty=selected_spec, settings=settings
                )
                context_text = "\n\n---\n\n".join(r.content for r in results)
                augmented_prompt = (
                    f"[NGỮ CẢNH PHÁC ĐỒ]:\n{context_text}\n\n[CÂU HỎI]: {doc_query}"
                )

            full_reply = st.write_stream(
                client.stream(DOCTOR_SYSTEM_PROMPT, augmented_prompt, temperature=0.1)
            )

            sources_info = []
            with st.expander("📚 Tài liệu trích dẫn (Top-3)"):
                for idx, r in enumerate(results, 1):
                    info = f"**[{idx}] {r.doc_id}** (Khoa: `{r.specialty}`) — Khoảng cách Cosine: `{r.distance:.4f}`"
                    st.markdown(info)
                    sources_info.append(info)

        st.session_state.doctor_messages.append({
            "role": "assistant",
            "content": full_reply,
            "sources": sources_info,
        })


def patient_interface() -> None:
    st.title("🎙️ Tiếp Đón & Sàng Lọc Triệu Chứng")
    st.caption("Nhận diện giọng nói đa chuyên khoa qua Whisper-large-v3")

    if "patient_messages" not in st.session_state:
        st.session_state.patient_messages = []

    for msg in st.session_state.patient_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    audio_value = st.audio_input("Nhấn mic để nói mô tả cảm giác khó chịu hoặc tổn thương...")

    patient_input = ""
    if audio_value is not None:
        # Streamlit reruns the script from top to bottom. The uploaded blob may
        # already be at EOF, so always rewind before reading and keep the bytes
        # in a local variable; a second read after rerender must not be b"".
        audio_value.seek(0)
        audio_bytes = audio_value.read()
        if audio_bytes:
            audio_key = (
                getattr(audio_value, "name", ""),
                len(audio_bytes),
                audio_bytes[:64],
            )
            if st.session_state.get("last_audio_key") != audio_key:
                st.session_state.last_audio_key = audio_key
                with st.spinner("Đang nhận diện giọng nói tiếng Việt..."):
                    try:
                        # Reuse these bytes (no second read) and wrap them in a
                        # fresh BytesIO with the correct container MIME type.
                        patient_input = _transcribe_audio(audio_value, audio_bytes)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Lỗi nhận diện âm thanh: {exc}")
        else:
            st.warning("Tệp ghi âm rỗng sau khi đọc lại. Vui lòng ghi âm lại.")

    text_fallback = st.chat_input("Hoặc gõ mô tả triệu chứng tại đây...")
    if text_fallback:
        patient_input = text_fallback

    if patient_input:
        st.session_state.patient_messages.append({"role": "user", "content": patient_input})
        with st.chat_message("user"):
            st.markdown(f"🗣️ **Triệu chứng ghi nhận:** {patient_input}")

        with st.chat_message("assistant"):
            with st.spinner("Đang phân loại và đưa ra hướng dẫn tiếp đón..."):
                triage_reply = client.triage_patient(patient_input)
                st.markdown(triage_reply)

        st.session_state.patient_messages.append({"role": "assistant", "content": triage_reply})


def main() -> None:
    st.sidebar.title("🏥 Hệ Thống Y Tế Phòng Khám")
    role = st.sidebar.radio(
        "Chọn phân hệ:",
        ["👨‍⚕️ Cổng Bác sĩ (EHR RAG)", "🎙️ Bệnh nhân (Tiếp đón Voice)"],
    )

    if role == "👨‍⚕️ Cổng Bác sĩ (EHR RAG)":
        doctor_interface()
    else:
        patient_interface()


if __name__ == "__main__":
    main()