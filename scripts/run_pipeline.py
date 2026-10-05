"""Master ingestion orchestrator: parse PDFs -> chunk -> extract -> embed -> insert.

Runs the whole knowledge-base lifecycle in one command:

    python scripts/run_pipeline.py                    # skip existing markdown, ingest all
    python scripts/run_pipeline.py --parser docling   # local parsing, no cloud key
    python scripts/run_pipeline.py --force-reparse    # re-run LlamaParse/Docling
    python scripts/run_pipeline.py --clean-db         # truncate first, then reindex

Every stage is idempotent-by-skip: a PDF whose Markdown already exists is left
alone unless ``--force-reparse`` is passed. Use ``--clean-db`` when re-ingesting
documents that were already embedded, otherwise their chunks are duplicated.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import (  # noqa: E402
    PARSED_MARKDOWN_DIR,
    RAW_DOCS_DIR,
    Settings,
    get_settings,
)
from app.core.logging import get_logger  # noqa: E402
from app.database.schema import truncate_table  # noqa: E402
from app.ingestion.pipeline import (  # noqa: E402
    IngestResult,
    classify_document,
    ingest_markdown_file,
)
from app.parsers.base import BaseParser  # noqa: E402

logger = get_logger(__name__)

RULE = "=" * 72


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full clinic knowledge-base ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parser",
        choices=("llama", "docling"),
        default="llama",
        help="PDF parsing backend (default: llama; requires LLAMA_CLOUD_API_KEY)",
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Re-parse PDFs even when the target Markdown already exists",
    )
    parser.add_argument(
        "--clean-db",
        action="store_true",
        help="Truncate clinic_knowledge_nodes before ingesting (full reindex)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=RAW_DOCS_DIR,
        help="Directory containing source PDFs (default: data/raw_docs)",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=PARSED_MARKDOWN_DIR,
        help="Directory for Markdown output (default: data/parsed_markdown)",
    )
    return parser


def preflight_db(settings: Settings) -> bool:
    """Fail fast with an actionable message when PostgreSQL is unreachable.

    ``run_migrations`` would otherwise block for the full TCP connect timeout
    and surface a raw driver exception on the way out.
    """
    import psycopg

    try:
        with psycopg.connect(settings.db_dsn, connect_timeout=5):
            return True
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Cannot reach PostgreSQL at %s:%s/%s (%s)",
            settings.DB_HOST, settings.DB_PORT, settings.DB_NAME, exc,
        )
        logger.error("Start the database first:  docker compose up -d postgres")
        return False


def build_pdf_parser(kind: str, parsed_dir: Path, settings: Settings) -> Optional[BaseParser]:
    """Instantiate the requested parser, falling back to Docling on failure."""
    if kind == "llama":
        try:
            from app.parsers.llamaparse_parser import LlamaParseParser

            parser: BaseParser = LlamaParseParser(output_dir=parsed_dir, settings=settings)
            logger.info("PDF backend: LlamaParse (cloud, medical layout mode)")
            return parser
        except Exception as exc:  # noqa: BLE001 - missing key or SDK/config error
            logger.warning(
                "LlamaParse unavailable (%s) - falling back to the local Docling parser",
                exc,
            )

    from app.parsers.docling_parser import DoclingParser

    logger.info("PDF backend: Docling (local, no OCR)")
    return DoclingParser(output_dir=parsed_dir)


def parse_pdfs(
    pdf_parser: BaseParser,
    source_dir: Path,
    parsed_dir: Path,
    force: bool,
) -> tuple[int, int]:
    """Parse ``source_dir/*.pdf``, skipping PDFs whose Markdown already exists."""
    pdfs = sorted(source_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDF documents found in %s", source_dir)
        return 0, 0

    parsed_count = 0
    skipped = 0
    for pdf in pdfs:
        target = parsed_dir / f"{pdf.stem}.md"
        if target.exists() and not force:
            logger.info("SKIP  %s (markdown already exists: %s)", pdf.name, target.name)
            skipped += 1
            continue
        logger.info("PARSE %s -> %s", pdf.name, target.name)
        try:
            pdf_parser.parse(pdf)
            parsed_count += 1
        except Exception as exc:  # noqa: BLE001 - keep going on a single bad PDF
            logger.error("FAILED %s: %s", pdf.name, exc)
    return parsed_count, skipped


def ingest_markdown_dir(parsed_dir: Path, settings: Settings) -> List[IngestResult]:
    """Chunk, extract, embed and insert every Markdown document in ``parsed_dir``."""
    documents = sorted(parsed_dir.glob("*.md"))
    if not documents:
        logger.warning("No Markdown documents found in %s", parsed_dir)
        return []

    results: List[IngestResult] = []
    for md_path in documents:
        specialty, audience = classify_document(md_path.stem)
        logger.info(
            "INGEST %s (doc_id=%s, specialty=%s, audience=%s)",
            md_path.name, md_path.stem, specialty, audience,
        )
        try:
            results.append(
                ingest_markdown_file(
                    md_path,
                    doc_id=md_path.stem,
                    specialty=specialty,
                    target_audience=audience,
                    settings=settings,
                    migrate=False,  # DDL already applied once below
                )
            )
        except Exception as exc:  # noqa: BLE001 - isolate one document's failure
            logger.error("FAILED ingest for %s: %s", md_path.name, exc)
    return results


def print_summary(
    results: List[IngestResult],
    parsed: int,
    skipped: int,
    elapsed_s: float,
    cleaned: bool,
) -> None:
    total_chunks = sum(r.total_chunks for r in results)
    total_entities = sum(r.entities_found for r in results)

    print()
    print(RULE)
    print(" ✅ CLINIC KNOWLEDGE PIPELINE - SUMMARY")
    print(RULE)
    print(f" PDFs parsed              : {parsed}")
    print(f" PDFs skipped (existing)  : {skipped}")
    print(f" Markdown documents       : {len(results)}")
    print(f" Chunks created           : {total_chunks}")
    print(f" Entities extracted       : {total_entities} (ICD-10 + medications)")
    print(f" DB rows committed        : {total_chunks}")
    print(f" Table truncated first    : {'yes' if cleaned else 'no'}")
    print(f" Elapsed                  : {elapsed_s:.1f}s")
    print(RULE)
    for r in results:
        print(
            f"  • {r.doc_id:<34} {r.specialty:<15} {r.target_audience:<10}"
            f" {r.total_chunks:>5} chunks  {r.entities_found:>4} entities"
        )
    print(RULE)


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    started = time.perf_counter()

    # 1. Directories + schema/index migrations.
    source_dir: Path = args.source_dir
    parsed_dir: Path = args.parsed_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Source PDFs : %s", source_dir)
    logger.info("Markdown out: %s", parsed_dir)

    if not preflight_db(settings):
        return 2

    logger.info("Running migrations (table + specialty + HNSW + GIN indexes)...")
    from app.database.schema import run_migrations

    run_migrations(settings)

    if args.clean_db:
        logger.warning("--clean-db: truncating clinic_knowledge_nodes before reindexing")
        truncate_table(settings)

    # 2. Step 1 - PDF -> Markdown.
    pdf_parser = build_pdf_parser(args.parser, parsed_dir, settings)
    parsed, skipped = parse_pdfs(pdf_parser, source_dir, parsed_dir, args.force_reparse)

    # 3. Step 2 - chunk, extract entities, embed, insert.
    results = ingest_markdown_dir(parsed_dir, settings)

    print_summary(results, parsed, skipped, time.perf_counter() - started, args.clean_db)
    if not results:
        logger.error("Nothing was ingested - check the PDFs in %s", source_dir)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())