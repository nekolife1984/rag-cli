"""Paragraph-aware chunking with sliding window for long paragraphs."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    text: str
    char_start: int
    char_end: int


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("need 0 <= overlap < chunk_size")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    # Markdown headings are section boundaries: isolate them as own paragraphs
    text = re.sub(r"\n(?=#{1,6}\s)", "\n\n", text)
    text = re.sub(r"(#{1,6}[^\n]*)\n(?!\n)", r"\1\n\n", text)

    # Split into paragraphs keeping offsets
    paras: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.S):
        paras.append((m.start(), m.end(), m.group(0)))
    if not paras:
        paras = [(0, len(text), text)]

    raw: list[tuple[int, int, str]] = []
    buf_start, buf_end, buf = None, None, ""

    def flush():
        nonlocal buf_start, buf_end, buf
        if buf:
            raw.append((buf_start, buf_end, buf))
        buf_start, buf_end, buf = None, None, ""

    for start, end, para in paras:
        # A heading starts a new chunk (keeps sections from diluting each other)
        if para.lstrip().startswith("#") and buf:
            flush()
        # Long paragraph: sliding window
        if len(para) > chunk_size:
            flush()
            step = chunk_size - overlap
            for i in range(0, len(para), step):
                piece = para[i : i + chunk_size]
                raw.append((start + i, start + i + len(piece), piece))
                if i + chunk_size >= len(para):
                    break
            continue
        if buf and len(buf) + 2 + len(para) > chunk_size:
            flush()
        if buf:
            buf = buf + "\n\n" + para
            buf_end = end
        else:
            buf_start, buf_end, buf = start, end, para
    flush()

    return [Chunk(i, t.strip(), s, e) for i, (s, e, t) in enumerate(raw) if t.strip()]
