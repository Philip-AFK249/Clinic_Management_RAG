"""FastAPI gateway: BHYT card upload -> Groq VLM extraction -> human verification."""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import PROJECT_ROOT, get_settings
from app.core.logging import get_logger
from app.parsers.bhyt_vlm import (
    BhytData,
    alias_camel_keys,
    extract_bhyt_from_bytes,
    normalise_bhyt_payload,
    normalise_insurance_code,
    optimize_image,
    validate_bhyt,
)
from app.services import bhyt_store

logger = get_logger(__name__)

INDEX_HTML = PROJECT_ROOT / "index.html"

app = FastAPI(title="Smart Clinic - AI Gateway API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "vlm_model": settings.VLM_MODEL_ID,
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
    try:
        # Optimize once so the preview the patient checks against is exactly the
        # pixels the VLM saw; then run the (blocking) Groq call off the event loop.
        optimized = await asyncio.to_thread(optimize_image, image_bytes)
        data, raw_text, _ = await asyncio.to_thread(
            extract_bhyt_from_bytes, image_bytes, optimized
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
        model=get_settings().VLM_MODEL_ID,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)