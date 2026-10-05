"""FastAPI gateway: BHYT card upload -> Groq VLM extraction -> human verification."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import PARSED_MARKDOWN_DIR, PROJECT_ROOT, RAW_DOCS_DIR, get_settings
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
PIPELINE_HTML = PROJECT_ROOT / "app" / "static" / "pipeline.html"

# Document types the pipeline console is allowed to ingest.
RAW_DOC_EXTENSIONS = (".pdf", ".docx")

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
    parser_type: Literal["llama", "docling"] = "docling"
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
    """Map a bare filename onto data/raw_docs/, refusing anything that escapes it."""
    candidate = (RAW_DOCS_DIR / filename).resolve()
    try:
        candidate.relative_to(RAW_DOCS_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Đường dẫn không hợp lệ.",
        )
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
    # Heavy imports stay inside the worker thread: pulling in sentence-transformers
    # and docling at module scope would add ~30s to API startup.
    target_dir = PARSED_MARKDOWN_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

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
        for path in sorted(RAW_DOCS_DIR.iterdir(), key=lambda p: p.name.lower()):
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)