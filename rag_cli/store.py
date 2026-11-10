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

    def get_chunk(self, chunk_id: int):
        row = self.conn.execute(
            "SELECT c.id, c.doc_id, c.chunk_index, c.char_start, c.char_end, c.text, d.source_path "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "chunk_id": row["id"],
            "doc_id": row["doc_id"],
            "source": row["source_path"],
            "source_name": Path(row["source_path"]).name,
            "chunk_index": row["chunk_index"],
            "char_start": row["char_start"],
            "char_end": row["char_end"],
            "text": row["text"],
        }

    def get_context(self, chunk_id: int, before: int = 1, after: int = 1):
        """Target chunk plus neighboring chunks from the SAME document.

        Neighbors are selected by chunk_index distance; at document edges
        fewer chunks are returned. Returns None if the chunk does not exist.
        """
        target = self.get_chunk(chunk_id)
        if target is None:
            return None
        idx, doc_id = target["chunk_index"], target["doc_id"]
        rows = self.conn.execute(
            "SELECT c.id, c.doc_id, c.chunk_index, c.char_start, c.char_end, c.text, d.source_path "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "WHERE c.doc_id = ? AND c.chunk_index BETWEEN ? AND ? ORDER BY c.chunk_index",
            (doc_id, idx - max(0, before), idx + max(0, after)),
        ).fetchall()
        chunks = [{
            "chunk_id": r["id"], "doc_id": r["doc_id"], "source": r["source_path"],
            "source_name": Path(r["source_path"]).name, "chunk_index": r["chunk_index"],
            "char_start": r["char_start"], "char_end": r["char_end"], "text": r["text"],
        } for r in rows]
        return {
            "target": target,
            "before": [c for c in chunks if c["chunk_index"] < idx],
            "after": [c for c in chunks if c["chunk_index"] > idx],
        }

    def find_document(self, target):
        """Find a document row by id, source path, or source name."""
        t = str(target)
        if t.isdigit():
            row = self.conn.execute("SELECT * FROM documents WHERE id=?", (int(t),)).fetchone()
            if row is not None:
                return dict(row)
        row = self.conn.execute(
            "SELECT * FROM documents WHERE source_path=? OR source_name=?",
            (str(Path(t).resolve()), t),
        ).fetchone()
        return dict(row) if row is not None else None

    def fulltext(self, target=None, chunk_id=None) -> dict:
        """Full extracted text of a source document, re-read from the file on disk.

        The DB stores chunks only, so the text is loaded fresh via the same
        loader used at ingest time. Raises LookupError if the document (or
        chunk) is unknown, FileNotFoundError if the file is gone.
        """
        from .loaders import load_file

        if chunk_id is not None:
            chunk = self.get_chunk(int(chunk_id))
            if chunk is None:
                raise LookupError(f"chunk not found: id={chunk_id}")
            doc = self.find_document(chunk["doc_id"])
        elif target is not None:
            doc = self.find_document(target)
        else:
            raise ValueError("fulltext needs a target document or a chunk_id")
        if doc is None:
            raise LookupError(f"document not found: {target}")
        path = Path(doc["source_path"])
        if not path.is_file():
            raise FileNotFoundError(f"source file not found: {path}")
        text = load_file(path)
        return {
            "doc_id": doc["id"],
            "source": doc["source_path"],
            "source_name": doc["source_name"],
            "ingested_at": doc["ingested_at"],
            "chunk_count": doc["chunk_count"],
            "chars": len(text),
            "changed_since_ingest": self.file_hash(path) != doc["file_hash"],
            "text": text,
        }

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
