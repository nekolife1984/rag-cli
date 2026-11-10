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
SERVER_INFO = {"name": "rag-cli", "version": "0.3.0"}

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
