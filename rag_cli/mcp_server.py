"""MCP (Model Context Protocol) server for rag-cli, stdio transport, stdlib only.

Two modes:
- bridge (default): forwards tool calls to the resident rag-server
  (http://127.0.0.1:8765). Lightweight -- this process needs no torch,
  startup is instant, and search uses the model already in memory.
- direct (--direct): loads the embedding backend in this process and works
  standalone. The MCP host keeps this process alive for the session, so the
  model load (~20s for sbert) is paid only once, at startup.

Protocol notes: stdio transport is newline-delimited JSON-RPC 2.0.
stdout carries ONLY protocol messages; everything else goes to stderr.

Tools (results are hit chunks only -- no generation, same as the CLI):
  rag_search {query, top_k?, min_score?, db?}
  rag_get_chunk {chunk_id, db?}
  rag_get_context {chunk_id, before?, after?, db?}
  rag_fulltext {target?, chunk_id?, db?}
  rag_ingest {paths, db?, chunk_size?, overlap?, force?, recursive?}
  rag_list   {db?}
  rag_stats  {db?}
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

SUPPORTED_PROTOCOLS = ["2024-11-05", "2025-03-26", "2025-06-18"]
SERVER_INFO = {"name": "rag-cli", "version": "0.3.5"}

TOOLS = [
    {
        "name": "rag_search",
        "description": "Search the local RAG database and return the hit chunks only "
                       "(score, source file, chunk index, char range, text). No answer generation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (any language)"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                "min_score": {"type": "number", "description": "Optional minimum score (vector: cosine, keyword: BM25; ignored in hybrid)"},
                "mode": {"type": "string", "enum": ["vector", "keyword", "hybrid"], "default": "hybrid",
                         "description": "vector=embedding only, keyword=BM25 only, hybrid=RRF fusion of both"},
                "db": {"type": "string", "description": "Optional DB path (default: server default DB)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rag_get_chunk",
        "description": "Fetch one chunk by its chunk_id (as returned by rag_search): "
                       "source file, chunk index, char range and full text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer", "description": "Chunk ID from a rag_search result"},
                "db": {"type": "string", "description": "Optional DB path (default: server default DB)"},
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "rag_get_context",
        "description": "Fetch a chunk plus its neighboring chunks from the same document "
                       "(before/after counts by chunk position). Returns {target, before, after} "
                       "in document order within each list.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer", "description": "Center chunk ID from a rag_search result"},
                "before": {"type": "integer", "default": 1, "minimum": 0, "description": "How many preceding chunks"},
                "after": {"type": "integer", "default": 1, "minimum": 0, "description": "How many following chunks"},
                "db": {"type": "string", "description": "Optional DB path (default: server default DB)"},
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "rag_fulltext",
        "description": "Return the full extracted text of a source document, re-read from the "
                       "file on disk. Identify the document by target (doc_id, source path or "
                       "file name) or by chunk_id of any of its chunks. The result flags "
                       "changed_since_ingest if the file differs from ingest time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": ["string", "integer"], "description": "Document id, source path, or file name"},
                "chunk_id": {"type": "integer", "description": "Resolve the document via this chunk id"},
                "db": {"type": "string", "description": "Optional DB path (default: server default DB)"},
            },
        },
    },
    {
        "name": "rag_ingest",
        "description": "Ingest files (or directories) into the RAG database: load, chunk, embed, store. "
                       "Unchanged files are skipped unless force=true.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Files or directories"},
                "db": {"type": "string"},
                "chunk_size": {"type": "integer", "default": 500},
                "overlap": {"type": "integer", "default": 80},
                "force": {"type": "boolean", "default": False},
                "recursive": {"type": "boolean", "default": False},
            },
            "required": ["paths"],
        },
    },
    {
        "name": "rag_list",
        "description": "List documents registered in the RAG database.",
        "inputSchema": {"type": "object", "properties": {"db": {"type": "string"}}},
    },
    {
        "name": "rag_stats",
        "description": "Show RAG database stats (document/chunk counts, backend, dimension).",
        "inputSchema": {"type": "object", "properties": {"db": {"type": "string"}}},
    },
]


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# ---------- operations: bridge to resident server ----------
class BridgeOps:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def _post(self, route: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + route,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    def _get(self, route: str, db: str | None) -> dict:
        url = self.base + route
        if db:
            url += "?db=" + urllib.parse.quote(db)
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())

    def search(self, args: dict) -> dict:
        return self._post("/search", {
            "query": args["query"], "db": args.get("db"),
            "top_k": args.get("top_k", 5), "min_score": args.get("min_score"),
            "mode": args.get("mode", "hybrid"),
        })

    def ingest(self, args: dict) -> dict:
        return self._post("/ingest", {
            "paths": args["paths"], "db": args.get("db"),
            "chunk_size": args.get("chunk_size", 500), "overlap": args.get("overlap", 80),
            "force": args.get("force", False), "recursive": args.get("recursive", False),
        })

    def get_chunk(self, args: dict) -> dict:
        import urllib.error

        url = self.base + "/chunk?id=" + str(int(args["chunk_id"]))
        if args.get("db"):
            url += "&db=" + urllib.parse.quote(args["db"])
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(f"chunk not found: id={args['chunk_id']}") from e
            raise

    def get_context(self, args: dict) -> dict:
        import urllib.error

        url = (self.base + "/context?id=" + str(int(args["chunk_id"]))
               + "&before=" + str(int(args.get("before", 1)))
               + "&after=" + str(int(args.get("after", 1))))
        if args.get("db"):
            url += "&db=" + urllib.parse.quote(args["db"])
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(f"chunk not found: id={args['chunk_id']}") from e
            raise

    def fulltext(self, args: dict) -> dict:
        import urllib.error

        params = {}
        if args.get("target") is not None:
            params["target"] = str(args["target"])
        if args.get("chunk_id") is not None:
            params["chunk_id"] = str(int(args["chunk_id"]))
        if args.get("db"):
            params["db"] = args["db"]
        url = self.base + "/fulltext?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                try:
                    msg = json.loads(e.read()).get("error", "not found")
                except Exception:
                    msg = "not found"
                raise ValueError(msg) from e
            raise

    def list_documents(self, args: dict) -> dict:
        return self._get("/list", args.get("db"))

    def stats(self, args: dict) -> dict:
        return self._get("/stats", args.get("db"))


# ---------- operations: direct (in-process backend) ----------
class DirectOps:
    def __init__(self, backend: str, model: str | None, default_db: str):
        from .server import RagServer

        self.rag = RagServer(backend, model, default_db)

    def search(self, args: dict) -> dict:
        return self.rag.search({
            "query": args["query"], "db": args.get("db"),
            "top_k": args.get("top_k", 5), "min_score": args.get("min_score"),
            "mode": args.get("mode", "hybrid"),
        })

    def ingest(self, args: dict) -> dict:
        return self.rag.ingest({
            "paths": args["paths"], "db": args.get("db"),
            "chunk_size": args.get("chunk_size", 500), "overlap": args.get("overlap", 80),
            "force": args.get("force", False), "recursive": args.get("recursive", False),
        })

    def get_chunk(self, args: dict) -> dict:
        chunk = self.rag.get_chunk(args.get("db"), int(args["chunk_id"]))
        if chunk is None:
            raise ValueError(f"chunk not found: id={args['chunk_id']}")
        return {"chunk": chunk}

    def get_context(self, args: dict) -> dict:
        ctx = self.rag.get_context(args.get("db"), int(args["chunk_id"]),
                                   int(args.get("before", 1)), int(args.get("after", 1)))
        if ctx is None:
            raise ValueError(f"chunk not found: id={args['chunk_id']}")
        return ctx

    def fulltext(self, args: dict) -> dict:
        return self.rag.fulltext(args.get("db"), target=args.get("target"),
                                 chunk_id=args.get("chunk_id"))

    def list_documents(self, args: dict) -> dict:
        return self.rag.list_documents(args.get("db"))

    def stats(self, args: dict) -> dict:
        return self.rag.stats(args.get("db"))


# ---------- MCP protocol ----------
class McpServer:
    def __init__(self, ops):
        self.ops = ops

    def handle(self, msg: dict) -> dict | None:
        method = msg.get("method", "")
        msg_id = msg.get("id")
        if msg_id is None:  # notification: no response
            return None
        try:
            if method == "initialize":
                requested = (msg.get("params") or {}).get("protocolVersion")
                version = requested if requested in SUPPORTED_PROTOCOLS else "2025-03-26"
                result = {
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self.call_tool(msg.get("params") or {})
            elif method in ("resources/list", "prompts/list"):
                result = {"resources": []} if method.startswith("resources") else {"prompts": []}
            else:
                return self._error(msg_id, -32601, f"method not found: {method}")
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}
        except Exception as e:
            return self._error(msg_id, -32603, f"{type(e).__name__}: {e}")

    @staticmethod
    def _error(msg_id, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    def call_tool(self, params: dict) -> dict:
        name = params.get("name", "")
        args = params.get("arguments") or {}
        try:
            if name == "rag_search":
                out = self.ops.search(args)
                payload = {"results": out["results"]}
            elif name == "rag_get_chunk":
                payload = self.ops.get_chunk(args)
            elif name == "rag_get_context":
                payload = self.ops.get_context(args)
            elif name == "rag_fulltext":
                payload = self.ops.fulltext(args)
            elif name == "rag_ingest":
                out = self.ops.ingest(args)
                if "error" in out:
                    raise RuntimeError(out["error"])
                payload = out
            elif name == "rag_list":
                payload = self.ops.list_documents(args)
            elif name == "rag_stats":
                payload = self.ops.stats(args)
            else:
                return {
                    "content": [{"type": "text", "text": f"unknown tool: {name}"}],
                    "isError": True,
                }
            if isinstance(payload, dict) and "error" in payload:
                raise RuntimeError(payload["error"])
            return {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                "structuredContent": payload,
                "isError": False,
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                "isError": True,
            }


def run_mcp(ops) -> int:
    server = McpServer(ops)
    log(f"rag-cli MCP server ready (serverInfo={SERVER_INFO})")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        messages = msg if isinstance(msg, list) else [msg]
        for m in messages:
            resp = server.handle(m)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
    return 0
