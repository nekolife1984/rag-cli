"""Load text from files. txt/md/csv/json/docx/xlsx are stdlib-only; pdf uses pypdf if installed."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

SUPPORTED = {".txt", ".md", ".markdown", ".csv", ".json", ".docx", ".pdf", ".html", ".htm",
             ".xlsx", ".xlsm"}


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


_BUILTIN_DATE_FMTS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31,
                      32, 33, 34, 35, 36, 45, 46, 47, 50, 57}


def _is_date_format(fmt_id: int, custom: dict[int, str]) -> bool:
    if fmt_id in _BUILTIN_DATE_FMTS:
        return True
    code = custom.get(fmt_id)
    if not code:
        return False
    import re as _re

    code = _re.sub(r'"[^"]*"|\[[^\]]*\]', "", code.lower())
    return "y" in code or "d" in code or ("h" in code and ("m" in code or "s" in code))


def _xlsx_date_flags(z: zipfile.ZipFile, names: set[str], ns: str) -> list[bool]:
    """Per-style-index flags: does this cell style use a date/time number format?"""
    if "xl/styles.xml" not in names:
        return []
    root = ElementTree.fromstring(z.read("xl/styles.xml"))
    custom: dict[int, str] = {}
    numfmts = root.find(ns + "numFmts")
    if numfmts is not None:
        for nf in numfmts.iter(ns + "numFmt"):
            try:
                custom[int(nf.get("numFmtId"))] = nf.get("formatCode") or ""
            except (TypeError, ValueError):
                pass
    flags: list[bool] = []
    cellxfs = root.find(ns + "cellXfs")
    if cellxfs is not None:
        for xf in cellxfs.iter(ns + "xf"):
            try:
                flags.append(_is_date_format(int(xf.get("numFmtId") or 0), custom))
            except ValueError:
                flags.append(False)
    return flags


def _xlsx_cell(c, shared: list[str], ns: str, date_flags: list[bool]) -> str:
    t = c.get("t")
    if t == "inlineStr":
        is_elem = c.find(ns + "is")
        if is_elem is None:
            return ""
        return "".join(x.text or "" for x in is_elem.iter(ns + "t"))
    v = c.find(ns + "v")
    if v is None or v.text is None:
        return ""
    if t == "s":
        try:
            return shared[int(v.text)]
        except (ValueError, IndexError):
            return ""
    if t == "b":
        return "TRUE" if v.text == "1" else "FALSE"
    if t in (None, "n"):
        s_idx = c.get("s")
        if s_idx is not None and date_flags:
            try:
                if date_flags[int(s_idx)]:
                    from datetime import datetime, timedelta

                    dt = datetime(1899, 12, 30) + timedelta(days=float(v.text))
                    return dt.strftime("%Y-%m-%d") if dt.time().isoformat() == "00:00:00" \
                        else dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, IndexError):
                pass
    return v.text


def _load_xlsx(path: Path) -> str:
    """Excel (.xlsx/.xlsm) via stdlib zip+XML. Cell values only (formula cells
    yield their cached value); date/time-styled cells are converted from serial
    numbers to ISO dates. Legacy binary .xls is not supported."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    relns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            sroot = ElementTree.fromstring(z.read("xl/sharedStrings.xml"))
            for si in sroot.iter(ns + "si"):
                shared.append("".join(t.text or "" for t in si.iter(ns + "t")))
        date_flags = _xlsx_date_flags(z, names, ns)
        wb = ElementTree.fromstring(z.read("xl/workbook.xml"))
        rels: dict[str, str] = {}
        if "xl/_rels/workbook.xml.rels" in names:
            rroot = ElementTree.fromstring(z.read("xl/_rels/workbook.xml.rels"))
            for rel in rroot.iter(relns + "Relationship"):
                rels[rel.get("Id")] = rel.get("Target") or ""
        sheets: list[tuple[str, str]] = []
        for sh in wb.iter(ns + "sheet"):
            target = rels.get(sh.get(rns + "id"), "").lstrip("/")
            if target and not target.startswith("xl/"):
                target = "xl/" + target
            sheets.append((sh.get("name") or "Sheet", target))
        blocks: list[str] = []
        for sheet_name, target in sheets:
            if target not in names:
                continue
            sroot = ElementTree.fromstring(z.read(target))
            lines: list[str] = []
            for row in sroot.iter(ns + "row"):
                vals: list[str] = []
                for c in row.iter(ns + "c"):
                    ref = c.get("r") or ""
                    letters = "".join(ch for ch in ref if ch.isalpha())
                    if letters:  # keep column positions for sparse rows
                        col = 0
                        for ch in letters:
                            col = col * 26 + (ord(ch.upper()) - 64)
                        while len(vals) < col - 1:
                            vals.append("")
                    vals.append(_xlsx_cell(c, shared, ns, date_flags))
                while vals and vals[-1] == "":
                    vals.pop()
                if any(vals):
                    lines.append(", ".join(vals))
            blocks.append(f"# Sheet: {sheet_name}\n\n" + "\n".join(lines))
        return "\n\n".join(blocks)


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
    if suffix in (".xlsx", ".xlsm"):
        return _load_xlsx(path)
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
