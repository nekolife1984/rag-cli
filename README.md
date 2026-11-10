# rag-cli

Repository: https://github.com/nekolife1984/rag-cli

ファイルを DB に登録し、検索すると **ヒットしたチャンクのリストだけ** を返す CLI。
AI による文章生成はしない（検索 = Retrieval まで）。

## 設計

| 項目 | 採用 | 理由 |
|---|---|---|
| 言語 | Python 3.10+ | 標準ライブラリだけでコアが動く |
| DB | SQLite（単一ファイル） | セットアップ不要、持ち運びやすい |
| 保存 | チャンク本文 + 埋め込みベクトル(float32 BLOB) | 検索はコサイン類似度の総当たり |
| チャンク | 段落優先、500文字 / 重なり80文字 | 日本語文書の既定値、引数で変更可 |
| 対応形式 | .txt .md .csv .json .docx（標準のみ）/ .pdf（要 pypdf） | |

### 埋め込みバックエンド（DBごとに1つ固定）

| backend | 中身 | 意味検索 | 追加インストール |
|---|---|---|---|
| `hash`（既定） | 文字 n-gram ハッシュ 2048次元 | △ キーワード寄り | なし（即動く・オフライン） |
| `sbert` | sentence-transformers `intfloat/multilingual-e5-small` | ◎ 日本語対応 | `pip install sentence-transformers`（初回 約470MB DL） |
| `openai` | `text-embedding-3-small` | ◎ | なし（要 `OPENAI_API_KEY`、従量課金） |

> ベクトルはモデルごとに互換性がないため、DB には最初に使った backend が記録され、
> 違う backend で取り込むとエラーになる。検索は取り込みと同じ backend を指定すること。

## 使い方

インストールなしで使う場合（このフォルダで）:

```bash
cd rag-cli
python3 -m rag_cli ingest examples/sample.md --db ./test.db
python3 -m rag_cli search "観光列車の予約はいつから？" --db ./test.db
python3 -m rag_cli search "ファンが止まったらどうする？" --db ./test.db --json
python3 -m rag_cli list --db ./test.db
python3 -m rag_cli stats --db ./test.db
python3 -m rag_cli remove sample.md --db ./test.db
```

コマンドとして入れる場合:

```bash
pip install -e .            # `rag` コマンドが使えるようになる
pip install -e ".[sbert]"   # 意味検索まで使う場合
pip install -e ".[pdf]"     # PDF を読む場合
rag ingest docs/ -r
rag search "SageMaker のチューニング" -k 5
```

### 主な引数

- `ingest <file|dir...> [-r] [--chunk-size 500] [--overlap 80] [--force]`
  - 同じファイル（中身が同じ）は自動でスキップ、変わっていれば差し替え
- `search "<query>" [-k 5] [--min-score 0.3] [--json]`
  - 返り値はヒットしたチャンクのリストだけ（score / 出典ファイル / チャンク番号 / 文字位置 / 本文）
  - `--json` で他プログラムからそのまま使える形になる
- 共通: `--db rag.db`（環境変数 `RAG_DB`）、`--backend hash|sbert|openai`（環境変数 `RAG_BACKEND`）

## 規模の目安と拡張ポイント

- 総当たり検索のため、快適なのは数万チャンク程度まで。それを超えたら
  `store.py` の検索層だけ sqlite-vec / LanceDB / Chroma に差し替えればよい
  （CLI と取り込み側はそのまま使えるよう分離してある）。
- 生成（Generation）を足したくなったら、search の JSON 出力を LLM に渡す薄い層を別コマンドにする想定。今は入れていない。

## 意味検索（sbert）の使い方と実測（2026-11-10）

```bash
# 初回だけ: venv にインストール（CPU版 torch、約1.2GB）＋ モデルDL（約471MB）
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend cpu sentence-transformers

.venv/bin/python -m rag_cli ingest examples/sample.md --db ./sbert.db --backend sbert
.venv/bin/python -m rag_cli search "ダイエットで目指している数値を教えて" --db ./sbert.db --backend sbert
```

実測比較（サンプル文書5チャンク、直接クエリ4問＋言い換えクエリ4問、`examples/compare_backends.py`）:

| backend | 直接クエリ | 言い換えクエリ | 合計 |
|---|---|---|---|
| hash | 4/4 | 2/4 | 6/8 |
| sbert (multilingual-e5-small, 384次元) | 4/4 | **4/4** | **8/8** |

言い換え例: 「ネットワークストレージの冷却装置が止まった時の対処法」
→ hash は外し、sbert は NAS ファン警告のチャンクが1位（score 0.870）。

注意点:
- sbert の score は全体的に高く出る（正解 0.87〜0.91 / 2位 0.76〜0.80 程度）。
  `--min-score` のしきい値は backend ごとに別物と考え、sbert では 0.80 前後から調整すること。
- CLI は起動のたびにモデルを読み込むため、このVMでは検索1回 約20秒（torch読込＋モデル読込が大半）。
  頻繁に検索するなら常駐サーバー化するか、hash との使い分けが現実的。

## 常駐サーバー（2026-11-10 導入）

モデルを読みっぱなしにする HTTP サーバーを systemd サービス `rag-server` で常駐化した。
起動のたびのモデル読込（約20秒）がなくなり、検索は実測 0.36〜1.5秒で返る。
常駐コストはメモリ約810MB（torch＋モデル）。

```bash
# クライアントは通常の python3 でOK（torch不要）。--db を省くとサーバーの既定DB（sbert.db）を使う
python3 -m rag_cli search "ダイエットで目指している数値を教えて" --server http://127.0.0.1:8765
python3 -m rag_cli ingest <ファイル> --server http://127.0.0.1:8765
export RAG_SERVER=http://127.0.0.1:8765   # 毎回 --server を付けたくない場合

systemctl status rag-server               # 状態確認
curl http://127.0.0.1:8765/health         # {"status":"ok", ...}
```

- API: `GET /health`、`GET /stats?db=`、`POST /search`、`POST /ingest`（JSON、127.0.0.1のみ・認証なし＝ローカル専用）
- サーバー手動起動（systemd を使わない場合）: `.venv/bin/python -m rag_cli serve --host 127.0.0.1 --port 8765 --db ./sbert.db --backend sbert`
- ユニット原稿は `rag-server.service`（このフォルダ）。モデルはキャッシュ済み前提で
  `HF_HUB_OFFLINE=1` を付けてある（起動時に HF Hub へ接続しない。モデル未DLの環境では先に通常の ingest でDLすること）。
- VM交換で /etc のユニットが消えた場合の復旧: `cp rag-server.service /etc/systemd/system/ && systemctl daemon-reload && systemctl enable --now rag-server`
- 既存の見守りフック（service-watchdog、5分間隔）の対象に rag-server も追加済み。停止を検知したら原稿から復旧し、成功時は無報告・失敗時だけユーザーへ報告する。

## MCP 対応（2026-11-10 導入）

`rag mcp` で MCP サーバー（stdio、JSON-RPC）として起動できる。標準ライブラリのみの実装で追加依存なし。
公開ツールは4つ: `rag_search` / `rag_ingest` / `rag_list` / `rag_stats`（いずれも結果はチャンクのリスト等のみ、生成なし）。

2つのモード:
- **ブリッジ（既定）**: 常駐サーバー（http://127.0.0.1:8765）へ中継。MCPプロセス自体は軽く、起動即時・検索は常駐モデルで高速。
- **直接（`--direct`）**: このプロセス内でバックエンドを読む。常駐サーバーがない環境用で、MCPホストがプロセスを維持するので sbert の読込はセッション初回だけ。

クライアント設定例:

```json
// Claude Desktop / Claude Code などの mcpServers 形式
{"mcpServers": {"rag": {
  "command": "/usr/bin/python3",
  "args": ["-m", "rag_cli", "mcp"],
  "cwd": "/home/hatch/workspace/rag-cli",
  "env": {"PYTHONPATH": "/home/hatch/workspace/rag-cli"}
}}}
```

```yaml
# Hermes (~/.hermes/config.yaml の mcp_servers)
mcp_servers:
  rag:
    command: /usr/bin/python3
    args: ["-m", "rag_cli", "mcp"]
    env:
      PYTHONPATH: /home/hatch/workspace/rag-cli
    # ブリッジモードなので DB は常駐サーバーの既定（sbert.db）が使われる
```

動作確認: `python3 examples/mcp_test.py`（initialize → tools/list → tools/call を一通り実行。ブリッジモード実測 全体1.4秒、言い換えクエリでも正解チャンク1位）

## テスト

```bash
python3 -m rag_cli ingest examples/sample.md --db /tmp/test.db --force
python3 -m rag_cli search "体重の目標は何キロ？" --db /tmp/test.db
```
