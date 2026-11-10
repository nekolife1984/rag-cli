"""Core logic tests for rag-cli. Stdlib only (unittest + hash backend).

Run from the project root:
    python3 -m unittest discover -s tests -v
"""
import tempfile
import unittest
import zipfile
from pathlib import Path

from rag_cli.chunking import chunk_text
from rag_cli.embed import HashBackend
from rag_cli.hybrid import bm25_scores, run_search
from rag_cli.loaders import load_file
from rag_cli.store import Store

NS = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'


def make_xlsx(path: Path) -> None:
    parts = {
        "xl/sharedStrings.xml":
            f'<sst {NS}><si><t>品名</t></si><si><t>マウス</t></si></sst>',
        "xl/workbook.xml":
            f'<workbook {NS} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="在庫" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels":
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/styles.xml":
            f'<styleSheet {NS}><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="14"/></cellXfs></styleSheet>',
        "xl/worksheets/sheet1.xml":
            f'<worksheet {NS}><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>45</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2" s="1"><v>46331</v></c></row>'
            '</sheetData></worksheet>',
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, xml in parts.items():
            z.writestr(name, xml)


class TestChunking(unittest.TestCase):
    def test_long_paragraph_overlap(self):
        para = "あ" * 1000
        chunks = chunk_text(para, chunk_size=500, overlap=80)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[1].char_start, 420)  # 500 - 80
        self.assertEqual(chunks[0].text[-80:], chunks[1].text[:80])  # shared region

    def test_heading_starts_new_chunk(self):
        text = "# 見出し1\n\n本文です。\n\n# 見出し2\n\n次の本文です。"
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].text.startswith("# 見出し2"))

    def test_empty(self):
        self.assertEqual(chunk_text("   \n\n  "), [])


class TestBM25(unittest.TestCase):
    def test_hiragana_only_match_is_not_a_hit(self):
        # Regression: chunks matching only function-word bigrams (して/てい/いる)
        # must score 0, or they pollute keyword/hybrid results.
        hit = "データを取得する手順は簡単です"
        noise = "設定を変更している最中の話"
        scores = bm25_scores([hit, noise], "データを取得している")
        self.assertGreater(scores[0], 0)
        self.assertEqual(scores[1], 0.0)


class TestStoreSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = str(Path(self.tmp.name) / "t.db")
        self.store = Store(self.db)
        self.backend = HashBackend()
        self.store.ensure_backend("hash", self.backend.dim)

    def _add(self, name: str, text: str):
        path = Path(self.tmp.name) / name
        path.write_text(text, encoding="utf-8")
        chunks = chunk_text(text)
        embs = self.backend.embed([c.text for c in chunks])
        self.store.add_document(path, self.store.file_hash(path), chunks, embs)
        return path

    def test_keyword_and_hybrid_find_model_number(self):
        self._add("a.md", "ワイヤレスマウス WM-200 の在庫は45個です。")
        self._add("b.md", "家庭用NASのファン停止アラートへの対処法。再起動して確認する。")
        kw = run_search(self.store, "WM-200", None, mode="keyword")
        self.assertEqual(kw[0]["source_name"], "a.md")
        qv = self.backend.embed(["マウスの在庫"], is_query=True)[0]
        hy = run_search(self.store, "マウスの在庫", qv, mode="hybrid")
        self.assertEqual(hy[0]["source_name"], "a.md")
        self.assertIn("keyword_score", hy[0])

    def test_get_context_fulltext(self):
        self._add("multi.md", "# 一\n\nあああ。\n\n# 二\n\nいいい。\n\n# 三\n\nううう。")
        docs = self.store.list_documents()
        self.assertEqual(docs[0]["chunk_count"], 3)
        mid = self.store.get_chunk(2)  # chunk ids are global, doc order here
        self.assertEqual(mid["chunk_index"], 1)
        ctx = self.store.get_context(2, before=1, after=1)
        self.assertEqual([c["chunk_index"] for c in ctx["before"]], [0])
        self.assertEqual([c["chunk_index"] for c in ctx["after"]], [2])
        ft = self.store.fulltext(target="multi.md")
        self.assertIn("いいい", ft["text"])
        self.assertFalse(ft["changed_since_ingest"])
        self.assertIsNone(self.store.get_chunk(9999))
        with self.assertRaises(LookupError):
            self.store.fulltext(target=9999)


class TestXlsxLoader(unittest.TestCase):
    def test_sheet_strings_numbers_and_date(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.xlsx"
            make_xlsx(p)
            text = load_file(p)
        self.assertIn("# Sheet: 在庫", text)
        self.assertIn("品名, 45", text)
        self.assertIn("マウス, 2026-11-05", text)  # serial 46331 with date style -> ISO


if __name__ == "__main__":
    unittest.main()
