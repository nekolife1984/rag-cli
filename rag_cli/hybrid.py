"""Hybrid search: dense (vector) + sparse (BM25 keyword) fused with RRF.

Modes:
- vector : cosine similarity over embeddings only (original behaviour)
- keyword: BM25 over chunk text only (no embedding model needed)
- hybrid : Reciprocal Rank Fusion of the two rankings (k=60).
           RRF fuses *ranks*, so it needs no score-scale calibration --
           important because sbert cosines cluster in 0.74-0.91 while
           BM25 scores are unbounded.

Japanese tokenization for BM25: character bigrams over the
whitespace-stripped text, plus ASCII word/number tokens (keeps things
like "72.5" and "NAS" as single terms).
"""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
K1, B = 1.2, 0.75
RRF_K = 60


def tokenize(text: str) -> list[str]:
    t = text.lower()
    tokens = _WORD_RE.findall(t)
    compact = re.sub(r"\s+", "", t)
    tokens.extend(compact[i : i + 2] for i in range(max(0, len(compact) - 1)))
    return tokens


_CONTENT_RE = re.compile(r"[^\u3040-\u309f]")  # any char outside hiragana


def _is_content(term: str) -> bool:
    """Content terms carry meaning (kanji/katakana/alnum); hiragana-only
    bigrams (して, てい, いる...) are function fragments. A chunk whose only
    matches are function fragments is noise, not a keyword hit."""
    return bool(_CONTENT_RE.search(term))


def bm25_scores(chunks_texts: list[str], query: str) -> list[float]:
    n = len(chunks_texts)
    if n == 0:
        return []
    docs = [tokenize(t) for t in chunks_texts]
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    q_terms = set(tokenize(query))
    scores: list[float] = []
    for d in docs:
        tf = Counter(d)
        dl = len(d) or 1
        s = 0.0
        content_hit = False
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            if _is_content(term):
                content_hit = True
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / avgdl))
        scores.append(s if content_hit else 0.0)
    return scores


def _rank_order(scores: list[float], positive_only: bool = False) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if positive_only:
        order = [i for i in order if scores[i] > 0]
    return order


def run_search(store, query: str, query_vec: list[float] | None,
               top_k: int = 5, mode: str = "hybrid",
               min_score: float | None = None) -> list[dict]:
    rows = store.all_chunks()
    if not rows:
        return []
    # shortcut: BM25 re-tokenizes every chunk on each query; fine up to ~10k
    # chunks, cache the tokenized corpus (keyed by db state) beyond that.
    bm25 = bm25_scores([r["text"] for r in rows], query)

    vec_scores: list[float] | None = None
    if query_vec is not None:
        vec_scores = [sum(a * b for a, b in zip(query_vec, r.pop("_vec"))) for r in rows]
    else:
        for r in rows:
            r.pop("_vec", None)

    def base_result(i: int) -> dict:
        r = rows[i]
        return {
            "chunk_id": r["id"], "source": r["source_path"], "source_name": r["source_name"],
            "chunk_index": r["chunk_index"], "char_start": r["char_start"], "char_end": r["char_end"],
            "text": r["text"],
            "vector_score": round(vec_scores[i], 4) if vec_scores is not None else None,
            "keyword_score": round(bm25[i], 4),
        }

    if mode == "vector":
        if vec_scores is None:
            raise ValueError("vector mode needs a query embedding")
        out = []
        for i in _rank_order(vec_scores):
            if min_score is not None and vec_scores[i] < min_score:
                continue
            res = base_result(i)
            res["score"] = round(vec_scores[i], 4)
            out.append(res)
        return out[:top_k]

    if mode == "keyword":
        out = []
        for i in _rank_order(bm25, positive_only=True):
            if min_score is not None and bm25[i] < min_score:
                continue
            res = base_result(i)
            res["score"] = round(bm25[i], 4)
            out.append(res)
        return out[:top_k]

    if mode == "hybrid":
        if vec_scores is None:
            raise ValueError("hybrid mode needs a query embedding")
        fused: dict[int, float] = {}
        for rank, i in enumerate(_rank_order(vec_scores), 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        for rank, i in enumerate(_rank_order(bm25, positive_only=True), 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        out = []
        for i, s in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            res = base_result(i)
            res["score"] = round(s, 4)  # RRF score: rank-based, not comparable across modes
            out.append(res)
        return out[:top_k]

    raise ValueError(f"unknown mode: {mode} (choose: vector, keyword, hybrid)")
