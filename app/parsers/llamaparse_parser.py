"""Cloud Vision PDF-to-Markdown extraction using LlamaParse.

Includes a medical-layout system prompt optimized for Vietnamese
Ministry of Health treatment guidelines (Phác đồ điều trị).
"""
from pathlib import Path
from typing import Optional

from llama_parse import LlamaParse

from app.core.config import PARSED_MARKDOWN_DIR, Settings, get_settings
from app.core.logging import get_logger
from app.parsers.base import BaseParser

logger = get_logger(__name__)

MEDICAL_PARSING_INSTRUCTION = """
Đây là tài liệu Phác đồ điều trị y tế của Bộ Y tế Việt Nam.
Khi chuyển đổi sang Markdown, hãy tuân thủ nghiêm ngặt các quy tắc sau:
1. Giữ nguyên cấu trúc phân cấp: Phần, Chương, I, II, 1, 2, a, b.
2. Tất cả các bảng danh mục thuốc, bảng liều lượng (mg, ml, liều nạp,
   liều duy trì) và bảng chỉ số sinh hiệu phải được định dạng chuẩn
   Markdown Table (| Cột 1 | Cột 2 |). Tuyệt đối không làm gãy hàng/cột.
3. Không bỏ sót các mã chẩn đoán ICD-10, chống chỉ định, liều dùng đặc biệt
   cho suy gan/suy thận.
4. Giữ nguyên các danh sách gạch đầu dòng triệu chứng, tiêu chuẩn chẩn đoán.
5. Ma trận sàng lọc (P1 đỏ, P2 vàng, P3 xanh) và bảng định tuyến tiếp đón phải
   là bảng Markdown đầy đủ, giữ nguyên từng dòng và không gộp ô.
"""


class LlamaParseParser(BaseParser):
    """Cloud (GPU) parser preserving complex medical layouts."""

    name = "llamaparse"

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(output_dir or PARSED_MARKDOWN_DIR)
        settings = settings or get_settings()
        if not settings.LLAMA_CLOUD_API_KEY:
            raise ValueError("Missing LLAMA_CLOUD_API_KEY in .env")

        # split_by_page=False keeps the whole guideline as one document: a
        #   medication table split across a page boundary otherwise arrives as
        #   two fragments and loses its header row.
        # annotate_links=True preserves the reference URLs in the guideline
        #   bibliography so citations stay resolvable after parsing.
        # language="vi" keeps Vietnamese diacritics intact instead of letting
        #   the model transliterate them.
        self._parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            language="vi",
            parsing_instruction=MEDICAL_PARSING_INSTRUCTION,
            split_by_page=False,
            annotate_links=True,
            num_workers=4,
            verbose=True,
        )

    def parse(self, file_path: Path) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing source file: {file_path}")

        logger.info("Sending %s to LlamaParse (medical layout mode)", file_path)
        documents = self._parser.load_data(str(file_path))
        markdown_content = "\n\n".join(doc.text for doc in documents)
        out_path = self.save(file_path, markdown_content)
        logger.info("Saved markdown to %s (%d chars)", out_path, len(markdown_content))
        return markdown_content


def main(source_dir: Path, document_stem: str | None = None) -> Path:
    from app.core.config import RAW_DOCS_DIR

    sources = sorted(source_dir.glob("*.pdf"))
    if document_stem:
        sources = [s for s in sources if s.stem == document_stem]
    if not sources:
        raise SystemExit(f"No PDF documents found in {source_dir}")

    parser = LlamaParseParser()
    for source in sources:
        parser.parse(source)
    return PARSED_MARKDOWN_DIR


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LlamaParse PDF -> Markdown (Cloud Vision)")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw_docs"),
        help="Directory containing source PDFs",
    )
    parser.add_argument("--stem", type=str, default=None, help="Optional file stem to parse")
    args = parser.parse_args()
    main(args.source_dir, args.stem)