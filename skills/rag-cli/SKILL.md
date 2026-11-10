---
name: rag-cli
description: Search the user's locally ingested documents with rag-cli, a retrieval-only RAG tool that returns scored text chunks (no answer generation). Use when a question may be answered by the user's local files, manuals, notes, or spreadsheets; when the user asks to search their documents/files/資料; or when you need source text with provenance (file name and character position). Covers the MCP tools (rag_search, rag_get_chunk, rag_get_context, rag_fulltext, rag_ingest, rag_list, rag_stats), the CLI, and the resident HTTP server, including hybrid/vector/keyword mode selection and the search → context → fulltext workflow.
---

# rag-cli — local document retrieval for agents

rag-cli indexes local files into a SQLite database and, given a query, returns
**hit chunks only**. It never generates answers: you read the chunks and compose
the answer yourself, citing the source.

## Ground rules

- **Check the corpus before searching blindly.** Run `rag_list` / `rag_stats`
  (or `rag list` / `rag stats`) to see what documents and which embedding
  backend the DB holds. If the topic is not in the corpus, say so instead of
  guessing.
- **One backend per DB.** The embedding backend (`hash` / `sbert` / `openai`)
  is fixed when documents are ingested. Searching a DB with a different
  backend is meaningless; ingest and search must use the same one.
- **Prefer the resident server or MCP.** The plain CLI reloads the embedding
  model on every call (slow for `sbert`, ~20s). The resident server
  (`http://127.0.0.1:8765`) keeps the model in memory and answers in under a
  second. When MCP tools are available, use them (they bridge to the server).
- **Scores are only comparable within one mode** (see below).

## The workflow: search → context → fulltext

1. **Search** for relevant chunks.
   - MCP: `rag_search {query, top_k?, mode?, min_score?, db?}`
   - CLI: `rag search "<query>" [-k N] [--mode ...] [--json] [--server http://127.0.0.1:8765]`
   - HTTP: `POST /search {"query": ..., "top_k": ..., "mode": ...}`
2. **Widen** a promising hit with its neighbors in the same document.
   - MCP: `rag_get_context {chunk_id, before?, after?}` (defaults: 1 and 1)
   - CLI: `rag context <chunk_id> [--before N] [--after N] [--json]`
   - Returns `{target, before, after}` in document order. Neighbors never cross
     into another document; at document edges you get fewer chunks.
3. **Fetch one chunk exactly** when you already know its id.
   - MCP: `rag_get_chunk {chunk_id}` / CLI: `rag get <chunk_id> [--json]`
4. **Read the whole source file** when chunks are not enough.
   - MCP: `rag_fulltext {target | chunk_id}` where target is a doc_id, source
     path, or file name (doc ids come from `rag_list`).
   - CLI: `rag fulltext <doc_id|path|name>` or `rag fulltext --chunk-id N`
   - The DB stores chunks only, so fulltext re-reads the file from disk.
     Check `changed_since_ingest`: if true, you are reading the *current*
     file, which may differ from what was indexed. A missing file is an error.

## Choosing a search mode (`--mode` / `mode`)

| Mode | Mechanism | Use it for |
|---|---|---|
| `hybrid` (default) | BM25 keyword + vector, fused by RRF | General questions; mixed wording |
| `vector` | Embedding cosine similarity only | Paraphrases, conceptual questions, "the thing about X" |
| `keyword` | BM25 only (no embedding model needed, instant) | Exact terms: model numbers, dates, names, codes |

Score semantics differ per mode and **must not be compared across modes**:

- `vector`: `score` = cosine similarity (roughly 0–1).
- `keyword`: `score` = BM25 (unbounded).
- `hybrid`: `score` = RRF rank fusion (small values, max ≈ 0.033). Each hit
  also carries `vector_score` and `keyword_score` for diagnosis.
- `--min-score` applies to vector/keyword modes only; it is ignored in hybrid.

If a search returns nothing useful: retry with the other mode (a paraphrase
query in `vector`, exact terms in `keyword`), broaden the query wording, or
raise `top_k` before concluding the corpus lacks the topic.

## Result format

Each hit: `chunk_id`, `doc_id`, `source` (absolute path), `source_name`,
`chunk_index` (position inside the document), `char_start`/`char_end`
(character offsets), `text`, `score`, `vector_score`, `keyword_score`.

Text fidelity notes:

- Chunk text joins paragraphs with a blank line, and a chunk covering a long
  paragraph is split with an **80-character overlap** — concatenating adjacent
  chunks verbatim can duplicate the overlap region.
- Chunks are contiguous within a document: between neighboring chunks there
  is no missing content (only blank lines), except that a heading always
  starts a new chunk, so a boundary can be "next section" rather than a
  continuing sentence.

## Ingesting documents

- MCP: `rag_ingest {paths, chunk_size?, overlap?, force?, recursive?}`
- CLI: `rag ingest <file|dir...> [-r] [--force]`
- Formats: txt / md / csv / json / html / docx / **xlsx / xlsm** (all with the
  Python standard library only), pdf (requires `pypdf`). Legacy binary `.xls`
  is **not** supported.
- Excel files are split per sheet under a `# Sheet: <name>` heading; date/time
  cells are converted from serial values to ISO dates (`2026-11-05`).
- Unchanged files are skipped automatically (content hash); a changed file is
  replaced. Use `--force` only when you must re-ingest identical content.
- Default chunking: 500 characters, 80 overlap, paragraph- and
  Markdown-heading-aware.

## Interfaces at a glance

MCP tools: `rag_search`, `rag_get_chunk`, `rag_get_context`, `rag_fulltext`,
`rag_ingest`, `rag_list`, `rag_stats`. Start the server with `rag mcp`
(bridges to the resident server; `--direct` loads the model in-process).

CLI commands: `ingest`, `search`, `get`, `context`, `fulltext`, `list`,
`stats`, `remove`, `clear`, `serve`, `mcp`. Most read commands accept
`--server <url>` (or env `RAG_SERVER`) and `--json`.

HTTP endpoints: `GET /health`, `GET /stats`, `GET /list`, `GET /chunk?id=`,
`GET /context?id=&before=&after=`, `GET /fulltext?target=&chunk_id=`,
`POST /search`, `POST /ingest`. Binds to 127.0.0.1 only, no auth — local use.

Errors for a missing chunk/document are consistent: CLI exit code 1, HTTP
404, MCP `isError: true`.
