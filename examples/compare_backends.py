"""Compare hash vs sbert on direct + paraphrase queries.

Usage (from rag-cli/):
  python3 examples/compare_backends.py                 # hash only (stdlib)
  .venv/bin/python examples/compare_backends.py        # both, if sbert installed
Expected chunk_index is known for examples/sample.md:
  1=観光列車の予約 2=認定試験 3=体重管理 4=NASファン
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_cli.chunking import chunk_text
from rag_cli.embed import get_backend
from rag_cli.loaders import load_file
from rag_cli.store import Store
import tempfile, os

CASES = [
    # (query, expected_chunk_index, kind)
    ("観光列車の予約はいつから可能？", 1, "direct"),
    ("体重の目標は何キロ？", 3, "direct"),
    ("ファンが止まったらどうすればいい？", 4, "direct"),
    ("認定試験はいつまで受験できる？", 2, "direct"),
    ("渓谷を走る観光列車の切符はどれくらい前に買える？", 1, "paraphrase"),
    ("ダイエットで目指している数値を教えて", 3, "paraphrase"),
    ("ネットワークストレージの冷却装置が止まった時の対処法", 4, "paraphrase"),
    ("機械学習の認定試験の有効期限はいつ？", 2, "paraphrase"),
]


def run(backend_name: str, chunks, texts):
    backend = get_backend(backend_name)
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "t.db")
        store = Store(db)
        store.ensure_backend(backend.name, backend.dim)
        embs = backend.embed(texts, is_query=False)
        from pathlib import Path as P
        store.add_document(P("examples/sample.md").resolve(), "x", chunks, embs)
        hits = 0
        for q, exp, kind in CASES:
            qv = backend.embed([q], is_query=True)[0]
            res = store.search(qv, top_k=3)
            top = res[0]["chunk_index"] if res else None
            ok = top == exp
            hits += ok
            print(f"  [{'OK ' if ok else 'MISS'}] ({kind}) {q}")
            print(f"        top1=chunk#{top} score={res[0]['score']:.4f} (expected #{exp})"
                  f"  top2=chunk#{res[1]['chunk_index']} score={res[1]['score']:.4f}")
        print(f"  => {backend_name}: {hits}/{len(CASES)} top1 correct")
        return hits


def main():
    text = load_file("examples/sample.md")
    chunks = chunk_text(text)
    texts = [c.text for c in chunks]
    print(f"chunks={len(chunks)}")
    run("hash", chunks, texts)
    try:
        import sentence_transformers  # noqa
    except ImportError:
        print("  sbert: skipped (sentence-transformers not installed in this python)")
        return
    run("sbert", chunks, texts)


if __name__ == "__main__":
    main()
