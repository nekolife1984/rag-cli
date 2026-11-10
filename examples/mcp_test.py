"""Minimal MCP client test: spawns `rag_cli mcp` and speaks JSON-RPC over stdio.

Usage (from rag-cli/):
  python3 examples/mcp_test.py            # bridge mode (needs rag-server running)
  python3 examples/mcp_test.py --direct   # direct mode with hash backend (no server needed)
"""
import json
import subprocess
import sys
import time

def main():
    direct = "--direct" in sys.argv
    cmd = [sys.executable, "-m", "rag_cli", "mcp"]
    if direct:
        cmd += ["--direct", "--backend", "hash", "--db", "test_hash.db"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    seq = 0

    def send(obj):
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def request(method, params=None):
        nonlocal seq
        seq += 1
        send({"jsonrpc": "2.0", "id": seq, "method": method, "params": params or {}})
        line = proc.stdout.readline()
        return json.loads(line)

    t0 = time.time()
    r = request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                               "clientInfo": {"name": "mcp_test", "version": "0"}})
    print("initialize:", r["result"]["protocolVersion"], r["result"]["serverInfo"])
    send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    r = request("tools/list")
    print("tools:", [t["name"] for t in r["result"]["tools"]])

    t = time.time()
    r = request("tools/call", {"name": "rag_search", "arguments": {
        "query": "ネットワークストレージの冷却装置が止まった時の対処法", "top_k": 2}})
    res = r["result"]
    payload = json.loads(res["content"][0]["text"])
    top = payload["results"][0]
    print(f"rag_search ({time.time()-t:.2f}s) isError={res['isError']} "
          f"top1=chunk#{top['chunk_index']} score={top['score']} src={top['source_name']}")
    if direct:
        # hash backend is keyword-ish and legitimately misses this paraphrase query
        # (measured: hash 6/8 vs sbert 8/8) -- protocol check only here.
        print("note: direct/hash mode -- top1 not asserted (known hash limitation)")
    else:
        assert top["chunk_index"] == 4, "expected NAS fan chunk #4"

    r = request("tools/call", {"name": "rag_stats", "arguments": {}})
    print("rag_stats:", json.loads(r["result"]["content"][0]["text"]))

    r = request("tools/call", {"name": "rag_list", "arguments": {}})
    docs = json.loads(r["result"]["content"][0]["text"])["documents"]
    print("rag_list:", [(d["source_name"], d["chunk_count"]) for d in docs])

    r = request("tools/call", {"name": "rag_ingest", "arguments": {"paths": ["examples/sample.md"]}})
    print("rag_ingest (unchanged):", json.loads(r["result"]["content"][0]["text"]))

    r = request("ping")
    print("ping:", r["result"])
    print(f"total: {time.time()-t0:.2f}s  ALL OK")
    proc.stdin.close()
    proc.wait(timeout=10)

if __name__ == "__main__":
    main()
