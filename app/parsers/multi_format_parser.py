"""Direct-to-Markdown parser for text, tabular, structured and web documents."""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional, Sequence

from app.core.config import PARSED_MARKDOWN_DIR
from app.core.logging import get_logger
from app.parsers.base import BaseParser

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = (
    ".md", ".markdown", ".txt", ".csv", ".tsv", ".xlsx", ".xls",
    ".json", ".html", ".htm",
)
_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1258", "latin-1")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_IGNORED_TAGS = {"script", "style", "noscript", "iframe"}


def _decode(raw: bytes) -> str:
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        try:
            return raw.decode("utf-16").replace("\x00", "")
        except UnicodeDecodeError:
            pass
    for encoding in _TEXT_ENCODINGS:
        try:
            text = raw.decode(encoding)
            return text.replace("\x00", "")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").replace("\x00", "")


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return ", ".join(_stringify(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return ", ".join(_stringify(item) for item in value)
    return str(value)


def _inline(value: Any) -> str:
    text = _stringify(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).replace("|", "\\|").strip()


def _cell(value: Any) -> str:
    text = _stringify(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")
    return re.sub(r"[ \t]+", " ", text).strip()


def _markdown_table(rows: Sequence[Sequence[Any]]) -> str:
    matrix = [
        [_cell(cell) for cell in row]
        for row in rows
        if any(_stringify(cell).strip() for cell in row)
    ]
    if not matrix:
        return ""
    width = max(len(row) for row in matrix)
    matrix = [list(row) + [""] * (width - len(row)) for row in matrix]
    header = [cell or f"Cột {index + 1}" for index, cell in enumerate(matrix[0])]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in matrix[1:])
    return "\n".join(lines)


def _dict_table(items: Sequence[dict]) -> str:
    keys = list(items[0])
    rows = [keys] + [[item.get(key) for key in keys] for item in items]
    return _markdown_table(rows)


def _is_uniform_dict_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value)
        and len({tuple(sorted(item)) for item in value}) == 1
    )


def _primary_list(value: Any) -> Optional[list]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for item in value.values():
            if _is_uniform_dict_list(item):
                return item
        for item in value.values():
            if isinstance(item, list):
                return item
    return None


class _MarkdownHTMLParser(HTMLParser):
    def __init__(self, stem: str) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = [f"# {_inline(stem)}"]
        self._buffer: list[str] = []
        self._ignore_depth = 0
        self._rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell_buffer: list[str] = []
        self._in_cell = False

    def _flush_line(self, prefix: str = "") -> None:
        text = _inline("".join(self._buffer))
        self._buffer = []
        if text:
            self.lines.append(prefix + text)

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        tag = tag.lower()
        if tag in _IGNORED_TAGS:
            self._ignore_depth += 1
            return
        if self._ignore_depth:
            return
        if tag in _HEADING_TAGS or tag in {"p", "li", "table"}:
            self._flush_line()
        if tag in {"th", "td"}:
            self._cell_buffer = []
            self._in_cell = True
        elif tag == "br":
            self._buffer.append(" ")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        if self._in_cell:
            self._cell_buffer.append(data)
        else:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _IGNORED_TAGS:
            self._ignore_depth = max(0, self._ignore_depth - 1)
            return
        if self._ignore_depth:
            return
        if tag in _HEADING_TAGS:
            text = _inline("".join(self._buffer))
            self._buffer = []
            if text:
                self.lines.extend(["", f"{'#' * int(tag[1])} {text}"])
        elif tag == "p":
            self._flush_line()
            self.lines.append("")
        elif tag == "li":
            self._flush_line("- ")
        elif tag in {"th", "td"}:
            self._row.append(_inline("".join(self._cell_buffer)))
            self._cell_buffer = []
            self._in_cell = False
        elif tag == "tr":
            if any(cell.strip() for cell in self._row):
                self._rows.append(self._row)
            self._row = []
        elif tag == "table":
            table = _markdown_table(self._rows)
            if table:
                self.lines.extend(["", table, ""])
            self._rows = []

    def result(self) -> str:
        self._flush_line()
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self.lines)).strip() + "\n"


def _html_with_bs4(html_text: str, stem: str) -> str:
    from bs4 import BeautifulSoup, NavigableString, Tag

    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(list(_IGNORED_TAGS)):
        tag.decompose()
    lines = [f"# {_inline(stem)}"]

    def render(node: Any) -> None:
        if isinstance(node, NavigableString):
            text = _inline(str(node))
            if text:
                lines.append(text)
            return
        if not isinstance(node, Tag):
            return
        name = node.name.lower()
        if name in _HEADING_TAGS:
            text = _inline(node.get_text(" ", strip=True))
            if text:
                lines.extend(["", f"{'#' * int(name[1])} {text}"])
            return
        if name == "table":
            rows = []
            for tr in node.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                rows.append([cell.get_text(" ", strip=True) for cell in cells])
            table = _markdown_table(rows)
            if table:
                lines.extend(["", table, ""])
            return
        if name in {"ul", "ol"}:
            for item in node.find_all("li", recursive=False):
                text = _inline(item.get_text(" ", strip=True))
                if text:
                    lines.append(f"- {text}")
            lines.append("")
            return
        if name == "br":
            lines.append("")
            return
        for child in node.children:
            render(child)
        if name == "p":
            lines.append("")

    for child in (soup.body or soup).children:
        render(child)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


class MultiFormatParser(BaseParser):
    """Convert supported raw documents into retrievable Markdown."""

    name = "multi_format"

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        super().__init__(output_dir or PARSED_MARKDOWN_DIR)

    def parse(self, file_path: Path) -> str:
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing source file: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {suffix or '(none)'}")

        logger.info("Parsing %s directly to Markdown", file_path.name)
        if suffix in {".md", ".markdown"}:
            content = self._parse_markdown(file_path)
        elif suffix == ".txt":
            content = self._parse_text(file_path)
        elif suffix in {".csv", ".tsv"}:
            content = self._parse_delimited(file_path)
        elif suffix == ".xlsx":
            content = self._parse_xlsx(file_path)
        elif suffix == ".xls":
            content = self._parse_xls(file_path)
        elif suffix == ".json":
            content = self._parse_json(file_path)
        else:
            content = self._parse_html(file_path)

        out_path = self.save(file_path, content)
        logger.info("Saved markdown to %s", out_path)
        return content

    def _read_text(self, file_path: Path) -> str:
        return _decode(file_path.read_bytes())

    def _parse_markdown(self, file_path: Path) -> str:
        text = _normalise_newlines(self._read_text(file_path))
        return (text.rstrip("\n") + "\n") if text.strip() else f"# {_inline(file_path.stem)}\n"

    def _parse_text(self, file_path: Path) -> str:
        lines = [line.rstrip() for line in _normalise_newlines(self._read_text(file_path)).split("\n")]
        if not any(_HEADING_RE.match(line.strip()) for line in lines if line.strip()):
            lines = [f"# {_inline(file_path.stem)}", ""] + lines
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip("\n"))
        return body + "\n"

    def _parse_delimited(self, file_path: Path) -> str:
        raw = file_path.read_bytes()
        if raw[:4] == b"PK\\x03\\x04":
            try:
                return self._parse_xlsx(file_path)
            except Exception as exc:
                raise ValueError(
                    "File có định dạng Excel (.xlsx) nhưng mang đuôi .csv/.tsv; "
                    "vui lòng tải lên dưới dạng .xlsx."
                ) from exc
        text = _normalise_newlines(_decode(raw))
        sample = text[:8192]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
        except csv.Error:
            delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
        if file_path.suffix.lower() == ".tsv" and "\t" in sample:
            delimiter = "\t"
        rows = [
            row
            for row in csv.reader(io.StringIO(text), delimiter=delimiter)
            if any(str(cell).strip() for cell in row)
        ]
        table = _markdown_table(rows)
        body = table or f"_(Không có dữ liệu trong {_inline(file_path.name)})_"
        return f"# {_inline(file_path.stem)}\n\n{body}\n"

    def _parse_xlsx(self, file_path: Path) -> str:
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise ValueError("Cần cài đặt openpyxl để đọc file .xlsx.") from exc

        workbook = load_workbook(file_path, read_only=True, data_only=True)
        try:
            parts = [f"# {_inline(file_path.stem)}"]
            for sheet in workbook.worksheets:
                rows = [
                    row
                    for row in sheet.iter_rows(values_only=True)
                    if any(_stringify(cell).strip() for cell in row)
                ]
                if not rows:
                    continue
                parts.extend(["", f"## {_inline(sheet.title)}", "", _markdown_table(rows)])
            if len(parts) == 1:
                parts.append("_(Không có dữ liệu trong workbook)_")
        finally:
            workbook.close()
        return "\n".join(parts).strip() + "\n"

    def _parse_xls(self, file_path: Path) -> str:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ValueError("Cần cài đặt pandas và xlrd để đọc file .xls cũ.") from exc
        try:
            sheets = pd.read_excel(file_path, sheet_name=None)
        except Exception as exc:
            raise ValueError(f"Không đọc được file .xls ({type(exc).__name__}).") from exc

        parts = [f"# {_inline(file_path.stem)}"]
        for name, frame in sheets.items():
            rows = [list(frame.columns)] + frame.fillna("").values.tolist()
            if not any(_stringify(cell).strip() for row in rows for cell in row):
                continue
            parts.extend(["", f"## {_inline(name)}", "", _markdown_table(rows)])
        if len(parts) == 1:
            parts.append("_(Không có dữ liệu trong workbook)_")
        return "\n".join(parts).strip() + "\n"

    def _parse_json(self, file_path: Path) -> str:
        try:
            payload = json.loads(self._read_text(file_path))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON không hợp lệ ({exc.msg} tại dòng {exc.lineno}).") from exc
        return self._render_json(payload, file_path.stem)

    def _render_json(self, payload: Any, title: str) -> str:
        lines = [f"# {_inline(title)}"]
        if isinstance(payload, dict):
            primary = _primary_list(payload)
            others = {
                key: value
                for key, value in payload.items()
                if primary is None or value is not primary
            }
            if others:
                lines.extend(["", "## Thông tin chung", ""])
                lines.extend(self._json_pairs(others, 3))
            if primary is not None:
                key = next(
                    (name for name, value in payload.items() if value is primary),
                    "Dữ liệu",
                )
                lines.extend(["", f"## {_inline(key)}", ""])
                if _is_uniform_dict_list(primary):
                    lines.append(_dict_table(primary))
                else:
                    lines.extend(f"- {_inline(item)}" for item in primary)
        elif _is_uniform_dict_list(payload):
            lines.extend(["", _dict_table(payload)])
        elif isinstance(payload, list):
            lines.append("")
            lines.extend(f"- {_inline(item)}" for item in payload)
        else:
            lines.extend(["", f"- **Giá trị:** {_inline(payload)}"])
        return "\n".join(lines).strip() + "\n"

    def _json_pairs(self, mapping: dict, level: int) -> list[str]:
        lines: list[str] = []
        heading = "#" * max(level, 2)
        for key, value in mapping.items():
            label = _inline(key)
            if isinstance(value, dict):
                lines.extend(["", f"{heading} {label}", ""])
                lines.extend(self._json_pairs(value, level + 1))
            elif _is_uniform_dict_list(value):
                lines.extend(["", f"{heading} {label}", "", _dict_table(value)])
            elif isinstance(value, list):
                if not value:
                    lines.append(f"- **{label}:** _(trống)_")
                elif all(not isinstance(item, (dict, list)) for item in value):
                    lines.append(
                        f"- **{label}:** " + ", ".join(_inline(item) for item in value)
                    )
                else:
                    lines.extend(["", f"{heading} {label}", ""])
                    for item in value:
                        if isinstance(item, dict):
                            lines.extend(self._json_pairs(item, level + 1))
                        else:
                            lines.append(f"- {_inline(item)}")
            else:
                lines.append(f"- **{label}:** {_inline(value)}")
        return lines

    def _parse_html(self, file_path: Path) -> str:
        html_text = _normalise_newlines(self._read_text(file_path))
        try:
            import bs4
        except ImportError:
            parser = _MarkdownHTMLParser(file_path.stem)
            parser.feed(html_text)
            parser.close()
            return parser.result()
        del bs4
        return _html_with_bs4(html_text, file_path.stem)
