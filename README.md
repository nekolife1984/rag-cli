# rag-cli

ファイルをデータベースに登録し、検索クエリに対して **ヒットしたチャンクのリストだけ** を返すローカル RAG（検索）ツールです。LLM による回答文の生成は行いません。検索結果をそのまま使うことも、別のプログラムや LLM へ渡す材料にすることもできます。

- リポジトリ: https://github.com/nekolife1984/rag-cli
- ライセンス: 未設定

## 特徴

- **生成なし・検索専用**: 返すのはスコア付きのチャンク一覧のみ。出力は人間向けテキストと JSON の2形式
- **依存が軽い**: コア（CLI・DB・MCP）は Python 標準ライブラリのみで動作する
- **埋め込みは3方式から選択**: オフラインで即動く `hash`、意味検索の `sbert`、API の `openai`
- **ハイブリッド検索**: ベクトル検索と BM25 キーワード検索を RRF で融合するモードを用意。固有名詞や数値に強いキーワード検索と、言い換えに強いベクトル検索の弱点を補い合う
- **対応形式**: txt / md / csv / json / html / docx / xlsx / xlsm（標準ライブラリのみ）、pdf（要 pypdf）。旧形式の .xls（バイナリ）には未対応。Excel はシートごとに `# Sheet: <シート名>` を付け、日付・時刻セルのシリアル値を ISO 日付に変換して取り込む
- **3つの利用形態**: 単発 CLI / 常駐サーバー（HTTP）/ MCP サーバー（stdio）
- **日本語対応**: チャンク分割と埋め込みのいずれも日本語文書を前提に調整している

## アーキテクチャ

```
ファイル ──ingest──▶ ローダ ─▶ チャンク分割 ─▶ 埋め込み ─▶ SQLite
                                                              │
クエリ ───────────────────────────── 埋め込み ─▶ コサイン類似度で検索
                                                              │
                                            ヒットしたチャンクのリスト
```

| モジュール | 役割 |
|---|---|
| `rag_cli/cli.py` | CLI 本体（ingest / search / list / stats / remove / clear / serve / mcp） |
| `rag_cli/loaders.py` | ファイル読み込み（txt / md / csv / json / html / docx / xlsx / xlsm、pdf は要 pypdf） |
| `rag_cli/chunking.py` | 段落と Markdown 見出しを優先したチャンク分割 |
| `rag_cli/embed.py` | 埋め込みバックエンド（hash / sbert / openai） |
| `rag_cli/store.py` | SQLite への保存と総当たりコサイン検索 |
| `rag_cli/hybrid.py` | BM25 キーワード検索と、ベクトル検索との RRF 融合 |
| `rag_cli/server.py` | 常駐 HTTP サーバー（モデルをメモリに保持） |
| `rag_cli/mcp_server.py` | MCP サーバー（stdio、JSON-RPC） |

設計上の要点:

- DB は SQLite の単一ファイル。チャンク本文と埋め込みベクトル（float32）を保存する
- チャンクは既定で500文字、重なり80文字。`--chunk-size` / `--overlap` で変更できる
- ベクトルはモデルごとに互換性がないため、DB には最初に使ったバックエンドが記録され、異なるバックエンドでの取り込みはエラーになる。検索は取り込みと同じバックエンドを使う

## インストール

要件: Python 3.10 以上

```bash
# このフォルダでそのまま使う（インストール不要）
python3 -m rag_cli --help

# `rag` コマンドとして入れる
pip install -e .
pip install -e ".[sbert]"   # 意味検索（sentence-transformers）を使う場合
pip install -e ".[pdf]"     # PDF を読む場合
```

## 使い方

### CLI

```bash
python3 -m rag_cli ingest examples/sample.md --db ./test.db
python3 -m rag_cli search "観光列車の予約はいつから？" --db ./test.db
python3 -m rag_cli search "ファンが止まったらどうする？" --db ./test.db --json
python3 -m rag_cli list --db ./test.db
python3 -m rag_cli stats --db ./test.db
python3 -m rag_cli remove sample.md --db ./test.db
python3 -m rag_cli clear --db ./test.db --yes
```

| コマンド | 説明 |
|---|---|
| `ingest <file\|dir...>` | ファイルを取り込む。`-r` でフォルダを再帰処理。内容が同じファイルは自動でスキップし、変わっていれば差し替える（`--force` で強制再取り込み） |
| `search "<query>"` | ヒットしたチャンクをスコア順に返す。`-k` 件数（既定5）、`--min-score` 下限、`--mode` 検索モード、`--json` で JSON 出力 |
| `get <chunk_id>` | 検索結果の `chunk_id` を指定して、そのチャンク1件（出典・位置・全文）を取得。存在しない ID はエラー（終了コード1） |
| `context <chunk_id>` | そのチャンクと、同じ文書内での前後のチャンクを取得。`--before N` / `--after N` で前後の個数を指定（既定は各1）。文書の端では取れた分だけ返す |
| `fulltext <doc_id\|path\|name>` | 元ファイルの全文（抽出テキスト）を返す。`--chunk-id N` でチャンクから文書を引くこともできる。DB はチャンクしか保持しないため、実ファイルをその場で読み直す。取り込み後にファイルが変わっていれば警告し、現在の内容を返す。ファイルが消えている場合はエラー |
| `list` | 登録済み文書の一覧 |
| `stats` | 文書数・チャンク数・バックエンド・次元数 |
| `remove <file>` | 指定した文書を DB から削除 |
| `clear --yes` | DB の中身をすべて削除 |

#### 検索モード

| モード | 中身 | 向いているクエリ |
|---|---|---|
| `vector` | 埋め込みのコサイン類似度のみ | 言い換え、意味の近い表現 |
| `keyword` | BM25 のみ（埋め込みモデル不要で高速） | 固有名詞、型番、日付、数値などの完全一致寄りの語 |
| `hybrid`（既定） | 両方のランキングを RRF（Reciprocal Rank Fusion、k=60）で融合 | 両方の性質が混ざったクエリ全般 |

- 融合に順位ベースの RRF を使うため、方式ごとに水準が違うスコア同士を調整なしに組み合わせられる
- 結果には統合後の `score` のほかに、内訳として `vector_score` と `keyword_score` が付く（hybrid の `score` は RRF スコアであり、他モードと絶対値を比較できない）
- BM25 の日本語トークン化は文字バイグラム＋英数トークン。ひらがなだけの断片（「して」「てい」等）しか一致しないチャンクはキーワードヒットとして扱わない
- `--min-score` は vector ではコサイン、keyword では BM25 スコアへの下限として働き、hybrid では使われない

検索結果（JSON）の形式:

```json
[{"chunk_id": 2, "source": "/path/to/sample.md", "source_name": "sample.md",
  "chunk_index": 1, "char_start": 61, "char_end": 164, "score": 0.8713,
  "vector_score": 0.8713, "keyword_score": 13.3045,
  "text": "## 渓谷観光列車の予約について\n\n渓谷を走る観光列車は、川沿いの景色を楽しめる人気の列車である。…"}]
```

### 常駐サーバー（HTTP）

sbert はプロセス起動のたびにモデルを読み込むため、単発 CLI では検索1回に時間がかかります。常駐サーバーはモデルをメモリに保持し、HTTP で検索と取り込みを受け付けます。

```bash
# 手動起動
.venv/bin/python -m rag_cli serve --host 127.0.0.1 --port 8765 --db ./sbert.db --backend sbert

# クライアント（通常の python3 でよい。--db を省くとサーバーの既定 DB を使う）
python3 -m rag_cli search "ダイエットで目指している数値を教えて" --server http://127.0.0.1:8765
python3 -m rag_cli ingest <ファイル> --server http://127.0.0.1:8765
export RAG_SERVER=http://127.0.0.1:8765   # 設定すると --server を省略できる
```

| エンドポイント | 説明 |
|---|---|
| `GET /health` | 稼働状態、バックエンド、次元数、既定 DB |
| `GET /stats?db=` | DB の統計 |
| `GET /list?db=` | 登録済み文書の一覧 |
| `GET /chunk?id=&db=` | `chunk_id` 指定でチャンク1件を返す。存在しなければ 404 |
| `GET /context?id=&before=&after=&db=` | チャンクと同一文書内の前後を `{"target", "before", "after"}` で返す。存在しなければ 404 |
| `GET /fulltext?target=&chunk_id=&db=` | 元ファイルの全文を `{doc_id, source, chars, changed_since_ingest, text, ...}` で返す。文書が無い・ファイルが消えている場合は 404 |
| `POST /search` | `{"query", "db?", "top_k?", "min_score?", "mode?"}` → `{"results": [...], "mode": "..."}` |
| `POST /ingest` | `{"paths", "db?", "chunk_size?", "overlap?", "force?", "recursive?"}` |

- バインドは 127.0.0.1 のみ、認証なし。ローカル専用であり、外部には公開しないこと
- systemd で常駐させる場合のユニット例が `rag-server.service`。モデルをキャッシュから読む前提のため `HF_HUB_OFFLINE=1` を付けてある（未ダウンロードの環境では、先に通常の ingest でモデルを取得すること）

### MCP サーバー

`rag mcp` で MCP（Model Context Protocol）サーバーとして起動します。stdio トランスポート、標準ライブラリのみの実装です。

| ツール | 引数 | 返り値 |
|---|---|---|
| `rag_search` | `query`, `top_k?`, `min_score?`, `mode?`, `db?` | ヒットしたチャンクのリスト |
| `rag_get_chunk` | `chunk_id`, `db?` | 指定チャンク1件（出典・位置・全文） |
| `rag_get_context` | `chunk_id`, `before?`, `after?`, `db?` | チャンクと同一文書内の前後（`{target, before, after}`） |
| `rag_fulltext` | `target?`（doc_id・パス・ファイル名）, `chunk_id?`, `db?` | 元ファイルの全文（実ファイルを読み直し、変更検知フラグ付き） |
| `rag_ingest` | `paths`, `db?`, `chunk_size?`, `overlap?`, `force?`, `recursive?` | ファイルごとの取り込み結果 |
| `rag_list` | `db?` | 登録済み文書の一覧 |
| `rag_stats` | `db?` | DB の統計 |

モードは2つあります。

- **ブリッジ（既定）**: 常駐サーバー（`http://127.0.0.1:8765`）へ中継する。MCP プロセス自体にモデル読み込みがなく、軽量で起動が速い
- **直接（`--direct`）**: このプロセス内でバックエンドを読み込む。常駐サーバーがない環境用。MCP ホストがプロセスを維持するため、モデル読み込みはセッションの初回だけになる

クライアント設定例（リポジトリのパスは環境に合わせて読み替えること）:

```json
{"mcpServers": {"rag": {
  "command": "/usr/bin/python3",
  "args": ["-m", "rag_cli", "mcp"],
  "cwd": "/path/to/rag-cli",
  "env": {"PYTHONPATH": "/path/to/rag-cli"}
}}}
```

```yaml
# Hermes の ~/.hermes/config.yaml
mcp_servers:
  rag:
    command: /usr/bin/python3
    args: ["-m", "rag_cli", "mcp"]
    env:
      PYTHONPATH: /path/to/rag-cli
```

## 埋め込みバックエンド

| backend | 中身 | 意味検索 | 追加要件 |
|---|---|---|---|
| `hash`（既定） | 文字 n-gram（2〜4）のハッシュ、2048次元 | △ キーワード寄り | なし（オフラインで即動作） |
| `sbert` | sentence-transformers `intfloat/multilingual-e5-small`、384次元 | ◎ 日本語対応 | `sentence-transformers`（モデル約471MB、torch を含む環境は約1.2GB） |
| `openai` | `text-embedding-3-small`、1536次元 | ◎ | 環境変数 `OPENAI_API_KEY`（従量課金） |

sbert の導入例:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend cpu sentence-transformers
.venv/bin/python -m rag_cli ingest examples/sample.md --db ./sbert.db --backend sbert
```

スコアの読み方:

- スコアはコサイン類似度だが、絶対値の水準はバックエンドで大きく異なる。hash では正解が 0.06〜0.28 程度、sbert では 0.87〜0.91 程度（2位でも 0.76〜0.80 程度）に出る
- `--min-score` のしきい値はバックエンドごとに別々に調整すること

## 設定

| 環境変数 | 対応する引数 | 内容 |
|---|---|---|
| `RAG_DB` | `--db` | DB ファイル（既定 `rag.db`） |
| `RAG_BACKEND` | `--backend` | `hash` / `sbert` / `openai`（既定 `hash`） |
| `RAG_SERVER` | `--server` | 常駐サーバーの URL |
| `RAG_MODEL` | `--model` | sbert / openai のモデル名 |
| `OPENAI_API_KEY` | — | openai バックエンド用 |

## 性能と規模

バックエンド比較（サンプル文書5チャンク、直接クエリ4問＋言い換えクエリ4問で1位正解率を測定。再現手順は `examples/compare_backends.py`）:

| backend | 直接クエリ | 言い換えクエリ | 合計 |
|---|---|---|---|
| hash | 4/4 | 2/4 | 6/8 |
| sbert | 4/4 | 4/4 | 8/8 |

言い換えクエリの例: 「ネットワークストレージの冷却装置が止まった時の対処法」では、本文に該当語を含まない NAS ファン警告のチャンクを、sbert は1位（0.870）で返し、hash は外した。

検索モード比較（サンプル文書5チャンク、完全一致寄り4問＋言い換え4問＋混合2問の10問で1位正解率を測定。再現手順は `examples/compare_modes.py`）:

| バックエンド | vector | keyword | hybrid |
|---|---|---|---|
| sbert | 10/10 | 9/10 | 10/10 |
| hash | 8/10 | 9/10 | 9/10 |

ハイブリッドはどちらのバックエンドでも単独方式の最良と同等以上になった。たとえば hash ではベクトル単独が外した言い換えクエリをキーワード側が拾って正解に届いている。ただし対象が5チャンクの小規模サンプルであり、実文書での優劣はコーパスの規模やクエリの性質に依存する。

応答時間とリソース（小規模 VM での参考値）:

| 形態 | 検索1回 |
|---|---|
| 単発 CLI（sbert） | 約20秒（大半はモデル読み込み） |
| 常駐サーバー経由 | 0.36〜1.5秒 |
| 常駐サーバーのメモリ使用量 | 約810MB |

規模の目安と拡張:

- 検索は総当たりのため、快適なのは数万チャンク程度まで
- それを超える場合は `store.py` の検索層だけを sqlite-vec / LanceDB / Chroma 等に差し替えられるよう、保存と検索を分離してある
- 回答生成を足す場合は、search の JSON 出力を LLM に渡す薄い層を別コマンドとして追加する想定であり、本体には含めていない

## ファイル構成

```
rag-cli/
├── README.md
├── pyproject.toml          # rag コマンドと任意依存（sbert / pdf）の定義
├── rag-server.service      # systemd ユニットの例
├── rag_cli/
│   ├── cli.py              # CLI 本体
│   ├── loaders.py          # ファイル読み込み
│   ├── chunking.py         # チャンク分割
│   ├── embed.py            # 埋め込みバックエンド
│   ├── store.py            # SQLite 保存・検索
│   ├── hybrid.py           # BM25 と RRF 融合
│   ├── server.py           # 常駐 HTTP サーバー
│   └── mcp_server.py       # MCP サーバー
└── examples/
    ├── sample.md           # 動作確認用の架空のサンプル文書
    ├── compare_backends.py # バックエンド比較スクリプト
    ├── compare_modes.py    # 検索モード比較スクリプト
    └── mcp_test.py         # MCP クライアントのテストスクリプト
```

## テスト

```bash
# スモークテスト（hash、追加インストール不要）
python3 -m rag_cli ingest examples/sample.md --db /tmp/test.db --force
python3 -m rag_cli search "体重の目標は何キロ？" --db /tmp/test.db

# バックエンド比較（sbert は .venv の python で実行）
.venv/bin/python examples/compare_backends.py

# MCP テスト（ブリッジモードは常駐サーバーが必要）
python3 examples/mcp_test.py
```
