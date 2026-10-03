from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger
from app.parsers.bhyt_ocr import ocr_image_bytes
from app.rag.generator import get_groq_client

logger = get_logger(__name__)

app = FastAPI(title="Smart Clinic - AI Gateway API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
groq_client = get_groq_client()

class BhytCardResponse(BaseModel):
    fullName: str | None = None
    insuranceCode: str | None = None
    dateOfBirth: str | None = None
    gender: str | None = None
    initialHospitalCode: str | None = None
    validFrom: str | None = None
    validUntil: str | None = None
    isExpired: bool = False
    isOcrVerified: bool = True
    rawText: str = ""

@app.post("/api/v1/ocr/bhyt", response_model=BhytCardResponse)
async def process_bhyt_upload(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Vui lòng tải lên tệp ảnh hợp lệ (.jpg, .png)")

    try:
        image_bytes = await file.read()
        
        # 1. OCR bóc tách văn bản thô
        raw_text = ocr_image_bytes(image_bytes)
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="Ảnh không rõ hoặc không nhận diện được ký tự nào.")

        # 2. Sử dụng Groq LLM (openai/gpt-oss-120b) chuẩn hóa thông tin
        parsed_data = groq_client.extract_bhyt_info(raw_text)
        
        # 3. Gán metadata phản hồi
        parsed_data["rawText"] = raw_text
        parsed_data["isOcrVerified"] = bool(parsed_data.get("insuranceCode"))

        return parsed_data

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Lỗi trong quá trình xử lý bóc tách thẻ BHYT: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ: {str(exc)}")