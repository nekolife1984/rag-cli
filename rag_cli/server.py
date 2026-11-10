"""Resident server: loads the embedding model once and serves search/ingest over HTTP.

Why: the plain CLI reloads the model on every invocation (~20s on a small VM).
The server keeps it in memory, so a search answers in well under a second.
Binds to 127.0.0.1 only, no auth -- local use only.

Endpoints (JSON in/out):
  GET  /health                 -> {status, backend, dim, default_db, uptime_sec}
  GET  /stats?db=<path>        -> store stats
  POST /search  {query, db?, top_k?, min_score?}        -> {results: [chunk, ...]}
  POST /ingest  {paths, db?, chunk_size?, overlap?, force?, recursive?}
                               -> {files: [{path, chunks, skipped}], total_chunks}
Responses return hit chunks only -- no generation, same as the CLI.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .chunking import chunk_text
from .embed import get_backend
from .loaders import iter_files, load_file
from .store import Store


class RagServer:
    def __init__(self, backend_name: str, model: str | None, default_db: str):
        self.started = time.time()
        self.backend = get_backend(backend_name, model)
        self.backend_name = self.backend.name
        self.default_db = default_db
        self.lock = threading.Lock()  # serialize model encodes

    def _store(self, db: str | None) -> Store:
        return Store(db or self.default_db)

    def search(self, payload: dict) -> dict:
        query = payload["query"]
        store = self._store(payload.get("db"))
        try:
            with self.lock:
                qv = self.backend.embed([query], is_query=True)[0]
            results = store.search(
                qv,
                top_k=int(payload.get("top_k", 5)),
                min_score=payload.get("min_score"),
            )
            return {"results": results}
        finally:
            store.conn.close()

    def ingest(self, payload: dict) -> dict:
        files = iter_files(payload["paths"], recursive=bool(payload.get("recursive", False)))
        chunk_size = int(payload.get("chunk_size", 500))
        overlap = int(payload.get("overlap", 80))
        force = bool(payload.get("force", False))
        store = self._store(payload.get("db"))
        try:
            store.ensure_backend(self.backend.name, self.backend.dim)
            out, total = [], 0
            for path in files:
                file_hash = store.file_hash(path)
                if not force and store.already_ingested(str(path.resolve()), file_hash):
                    out.append({"path": str(path), "chunks": 0, "skipped": True})
                    continue
                text = load_file(path)
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                if not chunks:
                    out.append({"path": str(path), "chunks": 0, "skipped": True})
                    continue
                with self.lock:
                    embs = self.backend.embed([c.text for c in chunks], is_query=False)
                store.add_document(path, file_hash, chunks, embs)
                total += len(chunks)
                out.append({"path": str(path), "chunks": len(chunks), "skipped": False})
            return {"files": out, "total_chunks": total}
        finally:
            store.conn.close()

    def stats(self, db: str | None) -> dict:
        store = self._store(db)
        try:
            return store.stats()
        finally:
            store.conn.close()

    def list_documents(self, db: str | None) -> dict:
        store = self._store(db)
        try:
            return {"documents": store.list_documents()}
        finally:
            store.conn.close()

    def health(self) -> dict:
        return {
            "status": "ok",
            "backend": self.backend_name,
            "dim": self.backend.dim,
            "default_db": self.default_db,
            "uptime_sec": round(time.time() - self.started, 1),
        }


def make_handler(rag: RagServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter logs
            import sys
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            try:
                url = urlparse(self.path)
                if url.path == "/health":
                    self._send(200, rag.health())
                elif url.path == "/stats":
                    db = parse_qs(url.query).get("db", [None])[0]
                    self._send(200, rag.stats(db))
                elif url.path == "/list":
                    db = parse_qs(url.query).get("db", [None])[0]
                    self._send(200, rag.list_documents(db))
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:  # keep the server alive on a bad request
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            try:
                payload = self._read_json()
                if self.path == "/search":
                    self._send(200, rag.search(payload))
                elif self.path == "/ingest":
                    self._send(200, rag.ingest(payload))
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def serve(host: str, port: int, backend: str, model: str | None, default_db: str) -> int:
    t0 = time.time()
    rag = RagServer(backend, model, default_db)
    print(f"model loaded: backend={rag.backend_name} dim={rag.backend.dim} ({time.time()-t0:.1f}s)", flush=True)
    httpd = ThreadingHTTPServer((host, port), make_handler(rag))
    print(f"rag-server listening on http://{host}:{port} default_db={default_db}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
