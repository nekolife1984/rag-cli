"""Load text from files. txt/md/csv/json/docx are stdlib-only; pdf uses pypdf if installed."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

SUPPORTED = {".txt", ".md", ".markdown", ".csv", ".json", ".docx", ".pdf", ".html", ".htm"}


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp932", "euc_jp"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _load_docx(path: Path) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    paras = []
    for p in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        text = "".join(t.text or "" for t in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
        if text.strip():
            paras.append(text)
    return "\n\n".join(paras)


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise SystemExit(f"PDF needs: pip install pypdf  ({path})") from e
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _load_json(path: Path) -> str:
    data = json.loads(_read_text_file(path))
    return json.dumps(data, ensure_ascii=False, indent=2)


def load_file(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise SystemExit(f"unsupported file type: {suffix} ({path}) supported={sorted(SUPPORTED)}")
    if suffix in (".txt", ".md", ".markdown", ".csv", ".html", ".htm"):
        return _read_text_file(path)
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    raise AssertionError(suffix)


def iter_files(paths: list[str], recursive: bool = False) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            glob = path.rglob("*") if recursive else path.glob("*")
            out.extend(f for f in glob if f.is_file() and f.suffix.lower() in SUPPORTED)
        elif path.is_file():
            out.append(path)
        else:
            raise SystemExit(f"not found: {p}")
    return sorted(out)
