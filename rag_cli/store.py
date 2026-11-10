"""SQLite store. Embeddings are float32 BLOBs; search is brute-force cosine.

Brute force is fine for personal scale (~tens of thousands of chunks).
When you outgrow it, swap this layer for sqlite-vec / LanceDB / Chroma
without touching cli.py -- the API below is the seam.
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct
import time
from pathlib import Path


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


class Store:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, source_path TEXT UNIQUE, source_name TEXT,
                file_hash TEXT, ingested_at REAL, chunk_count INTEGER
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER, text TEXT, char_start INTEGER, char_end INTEGER,
                embedding BLOB, dim INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )

    # --- meta / backend lock ---
    def ensure_backend(self, backend: str, dim: int) -> None:
        cur = self.conn.execute("SELECT key, value FROM meta WHERE key IN ('backend','dim')")
        meta = dict(cur.fetchall())
        if meta.get("backend") and (meta["backend"] != backend or int(meta["dim"]) != dim):
            raise SystemExit(
                f"DB backend mismatch: db={meta['backend']}({meta['dim']}d) vs requested={backend}({dim}d). "
                "Use the same --backend as ingest, or use a different --db."
            )
        self.conn.executemany(
            "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
            [("backend", backend), ("dim", str(dim))],
        )
        self.conn.commit()

    def backend_info(self) -> dict:
        cur = self.conn.execute("SELECT key, value FROM meta")
        return dict(cur.fetchall())

    # --- ingest ---
    @staticmethod
    def file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def already_ingested(self, source_path: str, file_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT file_hash FROM documents WHERE source_path=?", (source_path,)
        ).fetchone()
        return row is not None and row["file_hash"] == file_hash

    def add_document(self, path: Path, file_hash: str, chunks, embeddings) -> int:
        source = str(path.resolve())
        self.conn.execute("DELETE FROM chunks WHERE doc_id IN (SELECT id FROM documents WHERE source_path=?)", (source,))
        self.conn.execute("DELETE FROM documents WHERE source_path=?", (source,))
        cur = self.conn.execute(
            "INSERT INTO documents(source_path, source_name, file_hash, ingested_at, chunk_count) VALUES(?,?,?,?,?)",
            (source, path.name, file_hash, time.time(), len(chunks)),
        )
        doc_id = int(cur.lastrowid)
        self.conn.executemany(
            "INSERT INTO chunks(doc_id, chunk_index, text, char_start, char_end, embedding, dim) VALUES(?,?,?,?,?,?,?)",
            [
                (doc_id, c.index, c.text, c.char_start, c.char_end, _pack(e), len(e))
                for c, e in zip(chunks, embeddings)
            ],
        )
        self.conn.commit()
        return doc_id

    def all_chunks(self):
        """All chunks with metadata and unpacked vectors (key `_vec`)."""
        rows = self.conn.execute(
            """
            SELECT c.id, c.chunk_index, c.text, c.char_start, c.char_end, c.embedding,
                   d.source_path, d.source_name
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            """
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["_vec"] = _unpack(d.pop("embedding"))
            out.append(d)
        return out

    # --- search ---
    def search(self, query_vec: list[float], top_k: int = 5, min_score: float | None = None):
        rows = self.conn.execute(
            """
            SELECT c.id, c.chunk_index, c.text, c.char_start, c.char_end, c.embedding,
                   d.source_path, d.source_name
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            """
        ).fetchall()
        scored = []
        for r in rows:
            vec = _unpack(r["embedding"])
            score = sum(a * b for a, b in zip(query_vec, vec))  # vectors are normalized
            if min_score is not None and score < min_score:
                continue
            scored.append({
                "chunk_id": r["id"],
                "source": r["source_path"],
                "source_name": r["source_name"],
                "chunk_index": r["chunk_index"],
                "char_start": r["char_start"],
                "char_end": r["char_end"],
                "score": round(float(score), 4),
                "text": r["text"],
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # --- management ---
    def list_documents(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT id, source_path, source_name, ingested_at, chunk_count FROM documents ORDER BY id"
        ).fetchall()]

    def stats(self):
        docs = self.conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
        chunks = self.conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
        return {"db": self.db_path, "documents": docs, "chunks": chunks, **self.backend_info()}

    def remove(self, source: str) -> int:
        target = str(Path(source).resolve())
        cur = self.conn.execute("DELETE FROM documents WHERE source_path=? OR source_name=?", (target, source))
        self.conn.execute(
            "DELETE FROM chunks WHERE doc_id NOT IN (SELECT id FROM documents)"
        )
        self.conn.commit()
        return cur.rowcount

    def clear(self) -> None:
        self.conn.executescript("DELETE FROM chunks; DELETE FROM documents; DELETE FROM meta;")
        self.conn.commit()
