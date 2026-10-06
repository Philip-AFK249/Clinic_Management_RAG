"""Groq LLM client for clinical SOAP drafting and patient triage."""

from functools import lru_cache
from typing import Iterable

from groq import Groq, NotFoundError

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

FALLBACK_MODEL_PREFERENCES = (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
)

DOCTOR_SYSTEM_PROMPT = """Bạn là trợ lý y khoa AI hỗ trợ bác sĩ lâm sàng tra cứu phác đồ điều trị của Bộ Y tế.
Nguyên tắc trả lời:
1. CHỈ sử dụng thông tin trong [NGỮ CẢNH PHÁC ĐỒ] được cung cấp.
2. Trình bày rõ ràng theo bố cục: Chỉ định, Chống chỉ định, Quy trình kỹ thuật hoặc Biến chứng/Xử trí.
3. Nếu ngữ cảnh không có thông tin, hãy trả lời thẳng: 'Phác đồ hiện tại không có thông tin này', tuyệt đối không tự bịa đặt.
4. Diễn đạt bằng tiếng Việt y khoa tự nhiên, chuẩn chính tả, dấu câu rõ ràng.
5. Tuyệt đối không dùng từ ngữ dịch thô, không ghép từ Hán-Việt gượng gạo.
6. Xưng hô lịch sự, ân cần và chuyên nghiệp với người bệnh."""

TRIAGE_SYSTEM_PROMPT = """Bạn là trợ lý tiếp đón và phân loại triệu chứng tại phòng khám (Medical Triage Assistant).
Nhiệm vụ của bạn là lắng nghe triệu chứng từ người bệnh và:
1. ĐÁNH GIÁ MỨC ĐỘ NGUY CẤP:
   - Cấp cứu (Đỏ): Khó thở dữ dội, đau thắt ngực, sốc phản vệ, hoại tử/loét da diện rộng cấp tính -> Gọi cấp cứu 115 ngay lập tức.
   - Ưu tiên khám sớm (Vàng): Đau tức ngực gắng sức, tổn thương da lan nhanh, ngứa rát dữ dội, mụn mủ có sốt.
   - Khám thông thường (Xanh): Tái khám định kỳ, ngứa nhẹ, sẩn đỏ rải rác mạn tính.
2. GỢI Ý CHUYÊN KHOA PHÙ HỢP: (Tim mạch, Da liễu, Hô hấp, Tiêu hóa...).
3. TÓM TẮT BỆNH SỬ: Tóm tắt ngắn gọn triệu chứng để bệnh nhân trình bày với bác sĩ.

CẢNH BÁO: Không kết luận tên bệnh xác định, không kê đơn thuốc hoặc liều lượng."""

SOAP_SYSTEM_PROMPT = """Bạn là trợ lý AI soạn hồ sơ bệnh án SOAP (Subjective, Objective, Assessment, Plan)
cho bác sĩ lâm sàng. Dựa trên thông tin phác đồ được cung cấp:
1. Chỉ trích dẫn thông tin có trong [NGỮ CẢNH PHÁC ĐỒ].
2. Nếu thiếu thông tin cho một mục SOAP, ghi rõ 'Chưa có thông tin'.
3. Trình bày theo 4 mục S / O / A / P bằng tiếng Việt y khoa."""

BHYT_VLM_GUIDE = """Thẻ Bảo hiểm Y tế được bóc tách bởi VLM Groq (xem app/parsers/bhyt_vlm.py),
nên không còn bước chuyển văn bản OCR thô sang JSON ở tầng LLM này."""


class GroqClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY in .env")
        self.client = Groq(api_key=self.settings.GROQ_API_KEY)
        self._available_models: set[str] | None = None

    # ---------- model availability / fallback ----------

    def _available_model_ids(self) -> set[str]:
        """Query the account whitelist once and cache the model IDs."""
        if self._available_models is None:
            try:
                response = self.client.models.list()
                models = getattr(response, "data", response)
                self._available_models = {m.id for m in models}
                logger.debug("Available Groq models: %s", sorted(self._available_models))
            except Exception:
                logger.warning("Failed to list Groq models; disabling fallback.", exc_info=True)
                self._available_models = set()
        return self._available_models

    def _pick_fallback_model(self) -> str:
        available = self._available_model_ids()
        for candidate in FALLBACK_MODEL_PREFERENCES:
            if candidate in available:
                return candidate
        if available:
            fallback = next(iter(available))
            logger.warning("No preferred fallback model available; using '%s'.", fallback)
            return fallback
        raise RuntimeError("No fallback LLM model is available on this account.")

    def _create_completion(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        stream: bool,
        response_format: dict | None = None,
    ):
        kwargs = {
            "model": model,
            "temperature": temperature,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if response_format:
            kwargs["response_format"] = response_format
        return self.client.chat.completions.create(**kwargs)

    def _create_with_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        stream: bool,
        response_format: dict | None = None,
    ):
        model = self.settings.LLM_MODEL_ID
        try:
            return self._create_completion(
                model, system_prompt, user_prompt, temperature, stream, response_format
            )
        except NotFoundError:
            logger.warning(
                "LLM model '%s' is not accessible (404); switching to a whitelisted fallback.",
                model,
            )
            fallback_model = self._pick_fallback_model()
            logger.info("Falling back to LLM model '%s'.", fallback_model)
            return self._create_completion(
                fallback_model, system_prompt, user_prompt, temperature, stream, response_format
            )

    # ---------- building blocks ----------

    def build_doctor_prompt(self, context: str, question: str) -> str:
        return f"[NGỮ CẢNH PHÁC ĐỒ]:\n{context}\n\n[CÂU HỎI]: {question}"

    def build_soap_prompt(self, context: str, patient_note: str) -> str:
        return (
            f"[NGỮ CẢNH PHÁC ĐỒ]:\n{context}\n\n"
            f"[MÔ TẢ BỆNH NHÂN]: {patient_note}"
        )

    @staticmethod
    def context_from_results(results) -> str:
        contents = getattr(results, "search_results", results)
        if not contents:
            return "(Không có ngữ cảnh)"
        return "\n\n---\n\n".join(r.content for r in contents)

    # ---------- chat completions ----------

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        response_format: dict | None = None,
    ) -> str:
        completion = self._create_with_fallback(
            system_prompt, user_prompt, temperature, stream=False, response_format=response_format
        )
        return completion.choices[0].message.content  # type: ignore[return-value]

    def stream(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.1
    ) -> Iterable[str]:
        completion = self._create_with_fallback(
            system_prompt, user_prompt, temperature, stream=True
        )
        for chunk in completion:
            yield chunk.choices[0].delta.content or ""

    # ---------- high-level RAG helpers ----------

    def answer_doctor_question(self, query: str, results) -> str:
        user_prompt = self.build_doctor_prompt(self.context_from_results(results), query)
        return self.complete(DOCTOR_SYSTEM_PROMPT, user_prompt, temperature=0.1)

    def stream_doctor_answer(self, query: str, results) -> Iterable[str]:
        user_prompt = self.build_doctor_prompt(self.context_from_results(results), query)
        return self.stream(DOCTOR_SYSTEM_PROMPT, user_prompt, temperature=0.1)

    def draft_soap(self, patient_note: str, results) -> str:
        user_prompt = self.build_soap_prompt(self.context_from_results(results), patient_note)
        return self.complete(SOAP_SYSTEM_PROMPT, user_prompt, temperature=0.15)

    def triage_patient(self, patient_description: str) -> str:
        user_prompt = f"Bệnh nhân mô tả: {patient_description}"
        return self.complete(TRIAGE_SYSTEM_PROMPT, user_prompt, temperature=0.2)

    # NOTE: the OCR-text -> JSON hop (extract_bhyt_info) was removed together with
    # PaddleOCR. BHYT cards are now read end-to-end by the VLM in app/parsers/bhyt_vlm.py.


@lru_cache(maxsize=1)
def get_groq_client() -> GroqClient:
    """Trả về singleton GroqClient dựa trên settings mặc định."""
    return GroqClient()


def run_chat(settings: Settings | None = None) -> None:
    """Interactive console chat for doctor-facing RAG evaluation."""
    from app.rag.retriever import retrieve_context

    settings = settings or get_settings()
    client = GroqClient(settings)
    print("\n" + "=" * 65)
    print(" 🩺 KHUNG CHAT HỖ TRỢ BÁC SĨ (CLINIC RAG)")
    print(f" 🤖 LLM Engine: {settings.LLM_MODEL_ID}")
    print(" 💡 Gõ câu hỏi phác đồ cần tra cứu (Gõ 'exit' để thoát)")
    print("=" * 65 + "\n")

    while True:
        try:
            user_query = input("\nBác sĩ: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ("exit", "quit"):
                print("Đã kết thúc phiên làm việc.")
                break

            print("\n[Đang truy xuất dữ liệu từ PostgreSQL...]")
            results = retrieve_context(user_query, top_k=3, settings=settings)
            for token in client.stream_doctor_answer(user_query, results):
                print(token, end="", flush=True)

            print("\n\n" + "-" * 50)
            print("Tài liệu trích dẫn:")
            for idx, r in enumerate(results, 1):
                print(f"  [{idx}] {r.doc_id} ({r.specialty}) - Cosine: {r.distance:.4f}")
            print("-" * 50)
        except KeyboardInterrupt:
            print("\nĐã ngắt phiên trò chuyện.")
            break
        except Exception as exc:  # noqa: BLE001 - keep console loop alive
            print(f"\nLỗi: {exc}")


if __name__ == "__main__":
    run_chat()