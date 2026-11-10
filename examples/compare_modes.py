"""Compare search modes (vector / keyword / hybrid) against the resident server DB.

Usage (from rag-cli/):
  .venv/bin/python examples/compare_modes.py
Requires the sbert.db built from examples/sample.md (chunk indices:
1=観光列車 2=認定試験 3=体重管理 4=NASファン).
Query kinds:
  exact     -- contains a literal term/number from the text (keyword-friendly)
  paraphrase-- no shared keywords (vector-friendly)
  mixed     -- both a literal term and a paraphrase
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_cli.embed import get_backend
from rag_cli.hybrid import run_search
from rag_cli.store import Store

CASES = [
    ("72.5キログラム", 3, "exact"),
    ("2027年3月31日まで受験できる試験", 2, "exact"),
    ("ファン停止のアラート", 4, "exact"),
    ("紅葉の季節の列車", 1, "exact"),
    ("ダイエットで目指している数値を教えて", 3, "paraphrase"),
    ("ネットワークストレージの冷却装置が止まった時の対処法", 4, "paraphrase"),
    ("渓谷を走る観光列車の切符はどれくらい前に買える？", 1, "paraphrase"),
    ("機械学習の認定試験の有効期限はいつ？", 2, "paraphrase"),
    ("72.5キロから何キロ痩せる計画？", 3, "mixed"),
    ("ファンが故障したら高温で使い続けていい？", 4, "mixed"),
]


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "sbert.db"
    backend_name = sys.argv[2] if len(sys.argv) > 2 else "sbert"
    backend = get_backend(backend_name)
    store = Store(db)
    totals = {"vector": 0, "keyword": 0, "hybrid": 0}
    for q, exp, kind in CASES:
        qv = backend.embed([q], is_query=True)[0]
        line = [f"({kind}) {q}"]
        for mode in ("vector", "keyword", "hybrid"):
            res = run_search(store, q, qv, top_k=3, mode=mode)
            top = res[0]["chunk_index"] if res else None
            ok = top == exp
            totals[mode] += ok
            line.append(f"{mode}={'OK' if ok else f'x#{top}'}")
        print("  ".join(line))
    print({m: f"{t}/{len(CASES)}" for m, t in totals.items()})


if __name__ == "__main__":
    main()
