"""Markdown-aware chunker that never splits tables, lists or fenced code.

Clinical guidelines are dominated by medication-dosage tables and bulleted
diagnostic criteria. Cutting them at a raw character offset destroys the
Markdown syntax and detaches a dose from the drug it belongs to, so this module
first splits a document into *atomic blocks* (headings, paragraphs, bullet
lists, whole tables, fenced code) and only ever cuts on block boundaries. A
block that is itself larger than the budget is split with structure-aware rules
instead of a character slice: table rows are kept whole and the header row is
repeated, list items stay intact, and free text falls back to word wrapping.
"""
import re
from typing import List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_BLANK_RE = re.compile(r"^\s*$")
# Page breaks emitted by the PDF parsers ("---", "----"): no clinical signal, so
# they must not become chunks of their own.
_RULE_ONLY_RE = re.compile(r"^\s*(?:-{2,}|\*{2,}|_{2,})\s*$")

# Separator between two blocks kept in the same chunk.
_BLOCK_JOIN = "\n\n"


def chunk_markdown(path: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Split a Markdown file into overlapping, structure-preserving chunks.

    Chunks break on heading and block boundaries and never exceed ``max_chars``
    except when a single atomic block is larger than that (e.g. one huge table
    with no splittable row). ``overlap`` carries whole trailing blocks into the
    next chunk; a block larger than the overlap budget is dropped from the carry
    instead of being cut, which keeps table rows and list items intact.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # A carry-over larger than the window can never advance the window, so it
    # would loop forever; clamp instead of trusting the caller.
    overlap = max(0, min(overlap, max_chars - 1))

    blocks = _split_blocks(text)
    chunks = _pack_blocks(blocks, max_chars, overlap)

    # Drop chunks that carry no retrievable signal (whitespace / lone fence).
    chunks = [c for c in chunks if c.strip()]
    logger.info(
        "Produced %d chunks from %s (%d blocks, max_chars=%d, overlap=%d)",
        len(chunks), path, len(blocks), max_chars, overlap,
    )
    return chunks


# --- Block segmentation ---------------------------------------------------


def _split_blocks(text: str) -> List[str]:
    """Break Markdown into atomic blocks that must not be split mid-way."""
    blocks: List[str] = []
    lines = text.splitlines()
    buffer: List[str] = []
    in_fence = False
    fence_marker = ""

    idx = 0
    total = len(lines)
    while idx < total:
        line = lines[idx]
        fence = _FENCE_RE.match(line)

        if in_fence:
            buffer.append(line)
            if fence and fence.group(0).strip().startswith(fence_marker):
                in_fence = False
            idx += 1
            continue

        if fence:
            if buffer:
                blocks.append("\n".join(buffer).strip("\n"))
                buffer = []
            buffer.append(line)
            in_fence = True
            fence_marker = fence.group(0).strip()[:3]
            idx += 1
            continue

        if _BLANK_RE.match(line):
            if buffer:
                blocks.append("\n".join(buffer).strip("\n"))
                buffer = []
            idx += 1
            continue

        if _HEADING_RE.match(line):
            if buffer:
                blocks.append("\n".join(buffer).strip("\n"))
                buffer = []
            blocks.append(line.strip())
            idx += 1
            continue

        if _TABLE_ROW_RE.match(line):
            if buffer:
                blocks.append("\n".join(buffer).strip("\n"))
                buffer = []
            # A table is atomic: gather the full run of rows, delimiter included.
            start = idx
            while idx < total and _TABLE_ROW_RE.match(lines[idx]):
                idx += 1
            blocks.append("\n".join(ln.rstrip() for ln in lines[start:idx]).strip())
            continue

        buffer.append(line)
        idx += 1

    if buffer:
        blocks.append("\n".join(buffer).strip("\n"))
    return [
        b for b in blocks
        if b.strip() and not _RULE_ONLY_RE.match(b)
    ]


# --- Block packing --------------------------------------------------------


def _pack_blocks(blocks: List[str], max_chars: int, overlap: int) -> List[str]:
    """Greedily pack blocks into chunks, honouring headings and the overlap."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append(_BLOCK_JOIN.join(current))
        current = []
        current_len = 0

    for block in blocks:
        block_len = len(block)

        if block_len > max_chars:
            # Oversized atomic block: flush what we have, then split the block
            # with its own structure-aware rule set.
            flush()
            chunks.extend(_split_oversized_block(block, max_chars, overlap))
            continue

        is_heading = bool(_HEADING_RE.match(block))
        if current:
            will_not_fit = current_len + block_len + len(_BLOCK_JOIN) > max_chars
            # A heading always opens a new chunk so its section stays together,
            # mirroring the heading-delimited sections of the original splitter.
            if is_heading or will_not_fit:
                carry = _fit_carry(_overlap_tail(current, overlap), block, max_chars)
                flush()
                current = carry
                current_len = sum(len(c) + len(_BLOCK_JOIN) for c in current)

        current.append(block)
        current_len += block_len + len(_BLOCK_JOIN)

    flush()
    return chunks


def _fit_carry(carry: List[str], block: str, max_chars: int) -> List[str]:
    """Trim carried blocks from the front until ``block`` fits beside them.

    Without this the overlap budget could push a chunk to
    ``max_chars + overlap`` and silently break the size contract.
    """
    while carry and len(_BLOCK_JOIN.join([*carry, block])) > max_chars:
        carry = carry[1:]
    return carry


def _overlap_tail(blocks: List[str], overlap: int) -> List[str]:
    """Trailing whole blocks that fit in the ``overlap`` budget.

    Only complete blocks are carried: a partially copied table or list item is
    exactly the corruption this chunker exists to prevent.
    """
    if overlap <= 0 or not blocks:
        return []

    tail: List[str] = []
    budget = overlap
    for block in reversed(blocks):
        cost = len(block) + (len(_BLOCK_JOIN) if tail else 0)
        if cost > budget:
            break
        tail.insert(0, block)
        budget -= cost
    return tail


# --- Oversized block splitting -------------------------------------------


def _split_oversized_block(block: str, max_chars: int, overlap: int) -> List[str]:
    """Split a block that alone exceeds ``max_chars`` without breaking syntax.

    Order matters: a fenced block is checked first because parsed guideline
    pages embed whole ```` ```markdown ```` regions that contain their own
    tables and bullet lists, and those must keep their fence markers instead of
    being re-interpreted as a list.
    """
    lines = block.splitlines()
    if lines and _FENCE_RE.match(lines[0]):
        parts = _split_fenced(block, max_chars)
    elif len(lines) > 1 and all(_TABLE_ROW_RE.match(ln) for ln in lines if ln.strip()):
        parts = _split_table(lines, max_chars)
    elif any(_BULLET_RE.match(ln) for ln in lines):
        parts = _split_list(lines, max_chars, overlap)
    else:
        parts = _split_paragraph(block, max_chars, overlap)
    return _bound_parts(parts, max_chars)


def _bound_parts(parts: List[str], max_chars: int) -> List[str]:
    """Guarantee the size budget for fragments no rule above could shrink.

    Table fragments are passed through untouched on purpose: the only way to
    shrink them further would be to cut a row or a cell apart.
    """
    bounded: List[str] = []
    for part in parts:
        if len(part) <= max_chars or _is_table_part(part):
            bounded.append(part)
            continue
        lines = part.splitlines()
        if lines and _FENCE_RE.match(lines[0]):
            bounded.extend(_split_fenced(part, max_chars))
        else:
            bounded.extend(_split_paragraph(part, max_chars, 0))
    return bounded


def _is_table_part(text: str) -> bool:
    """True when every non-blank line of ``text`` is a Markdown table row."""
    rows = [ln for ln in text.splitlines() if ln.strip()]
    return bool(rows) and all(_TABLE_ROW_RE.match(ln) for ln in rows)


def _table_header(lines: List[str]) -> List[str]:
    """Header row plus the delimiter row, which define the column layout."""
    header = [lines[0]]
    if len(lines) > 1 and _TABLE_SEPARATOR_RE.match(lines[1]):
        header.append(lines[1])
    return header


def _split_table(lines: List[str], max_chars: int) -> List[str]:
    """Split an oversized table at row boundaries, repeating the header.

    Re-emitting the header keeps every fragment interpretable: a lone row of
    ``| Amoxicillin | 500mg |`` means little, but with its header it still tells
    the model which column is the drug and which is the dose.
    """
    header = _table_header(lines)
    body = lines[len(header):]
    header_cost = len("\n".join(header)) + 1

    # A table whose header alone exceeds the budget has no body rows to fall
    # back on: emit it whole rather than dropping it.
    if not body:
        return ["\n".join(header)]

    parts: List[str] = []
    current: List[str] = []
    for row in body:
        if current and header_cost + len("\n".join(current + [row])) > max_chars:
            parts.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join(header + current))

    # A row that alone exceeds the budget is emitted whole: cutting a dosage
    # row mid-cell would detach the drug from its dose, which is exactly the
    # corruption this chunker exists to prevent.
    return parts


def _split_list(lines: List[str], max_chars: int, overlap: int) -> List[str]:
    """Split a long bullet list on item boundaries, carrying whole items."""
    items: List[str] = [lines[0]]
    for line in lines[1:]:
        if _BULLET_RE.match(line):
            items.append(line)
        else:  # continuation line of the previous item
            items[-1] += "\n" + line

    parts: List[str] = []
    current: List[str] = []
    for item in items:
        if current and len("\n".join(current + [item])) > max_chars:
            parts.append("\n".join(current))
            current = _fit_carry(_overlap_tail(current, overlap), item, max_chars)
        current.append(item)
    if current:
        parts.append("\n".join(current))
    return parts


def _split_fenced(block: str, max_chars: int) -> List[str]:
    """Keep a fenced code block closed in every fragment.

    The real opening and closing lines are re-emitted around each fragment. The
    closing marker must be the block's own last line (usually a bare ```` ``` ````)
    rather than a copy of the opening one: a closing fence may not carry an info
    string, so ```` ```markdown ```` would not close the block and the fragment
    would render as unterminated code.
    """
    lines = block.splitlines()
    if len(lines) < 3:
        return [block]

    opening, closing = lines[0], lines[-1]
    # Budget left for the body once both fence lines and their newlines are paid.
    body_budget = max_chars - (len(opening) + len(closing) + 2)
    if body_budget <= 0:
        return [block]

    body = lines[1:-1]
    parts: List[str] = []
    current: List[str] = []

    def emit() -> None:
        parts.append("\n".join([opening, *current, closing]))

    for line in body:
        if len(line) > body_budget:
            if current:
                emit()
                current = []
            for wrapped in _split_paragraph(line, body_budget, 0):
                parts.append("\n".join([opening, wrapped, closing]))
            continue
        if current and len("\n".join(current)) + 1 + len(line) > body_budget:
            emit()
            current = []
        current.append(line)
    if current or not parts:
        emit()
    return parts


def _split_paragraph(block: str, max_chars: int, overlap: int) -> List[str]:
    """Word-wrap free text that has no usable structure to cut on."""
    lines = block.splitlines()
    parts: List[str] = []
    current: List[str] = []
    for line in lines:
        while len(line) > max_chars:
            # A single unbreakable line: emit as much as the budget allows.
            if current:
                parts.append("\n".join(current))
                current = []
            parts.append(line[:max_chars])
            line = line[max_chars - min(overlap, max_chars // 4):]
        if current and len("\n".join(current + [line])) > max_chars:
            parts.append("\n".join(current))
            current = _overlap_tail(current, overlap)
        current.append(line)
    if current:
        parts.append("\n".join(current))
    return [p for p in parts if p.strip()]


def chunk_markdown_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Same chunking rules as :func:`chunk_markdown` for an in-memory document."""
    overlap = max(0, min(overlap, max_chars - 1))
    return [c for c in _pack_blocks(_split_blocks(text), max_chars, overlap) if c.strip()]


def table_row_count(chunk: str) -> int:
    """Number of Markdown table rows in ``chunk`` (diagnostics helper)."""
    return sum(1 for line in chunk.splitlines() if _TABLE_ROW_RE.match(line))


def find_heading(chunk: str) -> Optional[str]:
    """First Markdown heading in ``chunk`` (diagnostics helper)."""
    for line in chunk.splitlines():
        if _HEADING_RE.match(line):
            return line.strip()
    return None