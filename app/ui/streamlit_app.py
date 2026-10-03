"""Streamlit web chat interface (migrated from web_chat.py)."""
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
        if "last_audio" not in st.session_state or st.session_state.last_audio != audio_value:
            st.session_state.last_audio = audio_value
            with st.spinner("Đang nhận diện giọng nói tiếng Việt..."):
                try:
                    transcription = client.client.audio.transcriptions.create(
                        file=("voice_input.wav", audio_value.read()),
                        model=settings.LLM_STT_MODEL_ID,
                        language="vi",
                        prompt=(
                            "Khám bệnh, đau thắt ngực, khó thở, huyết áp, da liễu, "
                            "nổi mẩn đỏ, ngứa ngáy, mụn nước, dát sẩn, vảy nến, "
                            "bong vảy, viêm da cơ địa, zona, lở loét."
                        ),
                        response_format="text",
                    )
                    patient_input = transcription.strip()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Lỗi nhận diện âm thanh: {exc}")

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