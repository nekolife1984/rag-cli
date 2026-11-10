"""rag CLI: ingest files into a DB, search returns hit chunks only. No generation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .chunking import chunk_text
from .embed import get_backend
from .loaders import iter_files, load_file
from .store import Store

DEFAULT_DB = os.environ.get("RAG_DB", "rag.db")
DEFAULT_SERVER = os.environ.get("RAG_SERVER")  # e.g. http://127.0.0.1:8765


def _server_post(server: str, route: str, payload: dict) -> dict:
    import urllib.request

    req = urllib.request.Request(
        server.rstrip("/") + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _print_search_results(results: list, as_json: bool) -> None:
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    if not results:
        print("no hits")
        return
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.4f}  {r['source_name']}  chunk#{r['chunk_index']}  chars {r['char_start']}-{r['char_end']}")
        print(r["text"])
        print("---")


def _add_common(p):
    p.add_argument("--db", default=DEFAULT_DB, help=f"DB file (default: {DEFAULT_DB}, env RAG_DB)")
    p.add_argument("--backend", default=os.environ.get("RAG_BACKEND", "hash"),
                   choices=["hash", "sbert", "openai"], help="embedding backend (default: hash, env RAG_BACKEND)")
    p.add_argument("--model", default=None, help="model name for sbert/openai backend")


def cmd_ingest(args) -> int:
    if getattr(args, "server", None):
        resp = _server_post(args.server, "/ingest", {
            "paths": args.paths, "db": None if args.db == DEFAULT_DB else args.db, "chunk_size": args.chunk_size,
            "overlap": args.overlap, "force": args.force, "recursive": args.recursive,
        })
        if "error" in resp:
            print(f"server error: {resp['error']}", file=sys.stderr)
            return 1
        for f in resp["files"]:
            print(f"{'skip' if f['skipped'] else 'ingested'}: {f['path']}  chunks={f['chunks']}")
        print(f"done: new_chunks={resp['total_chunks']} via server {args.server} (db: server default if --db omitted)")
        return 0
    files = iter_files(args.paths, recursive=args.recursive)
    if not files:
        print("no supported files found", file=sys.stderr)
        return 1
    backend = get_backend(args.backend, args.model)
    store = Store(args.db)
    store.ensure_backend(backend.name, backend.dim)
    total_chunks = 0
    for path in files:
        file_hash = store.file_hash(path)
        if not args.force and store.already_ingested(str(path.resolve()), file_hash):
            print(f"skip (unchanged): {path}")
            continue
        text = load_file(path)
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        if not chunks:
            print(f"skip (empty): {path}")
            continue
        embeddings = backend.embed([c.text for c in chunks], is_query=False)
        store.add_document(path, file_hash, chunks, embeddings)
        total_chunks += len(chunks)
        print(f"ingested: {path}  chunks={len(chunks)}")
    print(f"done: files={len(files)} new_chunks={total_chunks} db={args.db} backend={backend.name}")
    return 0


def cmd_search(args) -> int:
    if getattr(args, "server", None):
        resp = _server_post(args.server, "/search", {
            "query": args.query, "db": None if args.db == DEFAULT_DB else args.db, "top_k": args.top_k, "min_score": args.min_score,
        })
        if "error" in resp:
            print(f"server error: {resp['error']}", file=sys.stderr)
            return 1
        _print_search_results(resp["results"], args.json)
        return 0
    backend = get_backend(args.backend, args.model)
    store = Store(args.db)
    info = store.backend_info()
    if info.get("backend") and info["backend"] != backend.name:
        print(f"warning: db backend={info['backend']} but requested {backend.name}; "
              "results may be meaningless. Use the ingest backend.", file=sys.stderr)
    query_vec = backend.embed([args.query], is_query=True)[0]
    results = store.search(query_vec, top_k=args.top_k, min_score=args.min_score)
    _print_search_results(results, args.json)
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    return serve(args.host, args.port, args.backend, args.model, args.db)


def cmd_mcp(args) -> int:
    from .mcp_server import BridgeOps, DirectOps, run_mcp

    if args.direct:
        ops = DirectOps(args.backend, args.model, args.db)
    else:
        ops = BridgeOps(args.server)
    return run_mcp(ops)


def cmd_list(args) -> int:
    store = Store(args.db)
    docs = store.list_documents()
    if args.json:
        print(json.dumps(docs, ensure_ascii=False, indent=2))
        return 0
    for d in docs:
        print(f"{d['id']}: {d['source_name']}  chunks={d['chunk_count']}  {d['source_path']}")
    print(f"documents={len(docs)}")
    return 0


def cmd_stats(args) -> int:
    store = Store(args.db)
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_remove(args) -> int:
    store = Store(args.db)
    n = store.remove(args.source)
    print(f"removed documents={n}")
    return 0


def cmd_clear(args) -> int:
    if not args.yes:
        print("refuse: pass --yes to clear all data", file=sys.stderr)
        return 1
    Store(args.db).clear()
    print(f"cleared: {args.db}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Local RAG retrieval CLI (chunks only, no generation)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="add files to the DB")
    p.add_argument("paths", nargs="+", help="files or directories")
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--overlap", type=int, default=80)
    p.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    p.add_argument("--server", default=DEFAULT_SERVER, help="resident server URL (env RAG_SERVER); skips local model load")
    _add_common(p)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="search and return hit chunks only")
    p.add_argument("query")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--server", default=DEFAULT_SERVER, help="resident server URL (env RAG_SERVER); skips local model load")
    _add_common(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("serve", help="run the resident server (model stays loaded)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    _add_common(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mcp", help="run as an MCP server (stdio); bridges to the resident server by default")
    p.add_argument("--server", default=DEFAULT_SERVER or "http://127.0.0.1:8765",
                   help="resident server URL for bridge mode (env RAG_SERVER)")
    p.add_argument("--direct", action="store_true",
                   help="load the embedding backend in this process instead of bridging")
    _add_common(p)
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("list", help="list ingested documents")
    p.add_argument("--json", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("stats", help="DB stats")
    _add_common(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("remove", help="remove a document by path or name")
    p.add_argument("source")
    _add_common(p)
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("clear", help="delete everything in the DB")
    p.add_argument("--yes", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_clear)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
