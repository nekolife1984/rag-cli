"""Embedding backends. Core (hash) uses only stdlib.

- hash   : zero-dependency, offline. Char n-gram (2-4) hashing -> 2048-dim.
           Good for keyword-ish / near-duplicate matching and instant demos.
           NOT true semantic search.
- sbert  : sentence-transformers, intfloat/multilingual-e5-small (default).
           True semantic search, Japanese OK. Needs: pip install sentence-transformers
           First run downloads ~470MB model.
- openai : text-embedding-3-small via stdlib urllib. Needs OPENAI_API_KEY.
           True semantic search, no local model download.
A DB is tied to one backend/dim (stored in meta). Mixing backends in one DB
is rejected on purpose: vectors from different models are not comparable.
"""
from __future__ import annotations

import hashlib
import math
import os
import re

DIM_HASH = 2048
DEFAULT_SBERT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ---------- hash backend ----------
_WORD_RE = re.compile(r"[a-z0-9_]+")

def _hash_embed_one(text: str) -> list[float]:
    vec = [0.0] * DIM_HASH
    t = text.lower()
    # ascii word tokens
    tokens = _WORD_RE.findall(t)
    for token in tokens:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(h[:4], "little") % DIM_HASH] += 2.0
    # char n-grams over whitespace-stripped text (works for Japanese);
    # longer grams are more discriminative, so weight them higher.
    # Unsigned counts: signed hashing caused cancellations and spurious hits.
    compact = re.sub(r"\s+", "", t)
    for n, weight in ((2, 1.0), (3, 1.5), (4, 2.0)):
        for i in range(max(0, len(compact) - n + 1)):
            g = compact[i : i + n]
            h = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
            vec[int.from_bytes(h[:4], "little") % DIM_HASH] += weight
    return _normalize(vec)


# ---------- backends ----------
class HashBackend:
    name = "hash"
    dim = DIM_HASH

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        return [_hash_embed_one(t) for t in texts]


class SbertBackend:
    name = "sbert"

    def __init__(self, model: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise SystemExit(
                "sbert backend needs: pip install sentence-transformers\n"
                "or use --backend hash / --backend openai"
            ) from e
        self.model_name = model or os.environ.get("RAG_MODEL") or DEFAULT_SBERT_MODEL
        self._model = SentenceTransformer(self.model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        # e5 models expect instruction prefixes
        if "e5" in self.model_name:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class OpenAIBackend:
    name = "openai"
    dim = 1536  # text-embedding-3-small

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("RAG_OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise SystemExit("openai backend needs env OPENAI_API_KEY")

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        import json
        import urllib.request

        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            req = urllib.request.Request(
                "https://api.openai.com/v1/embeddings",
                data=json.dumps({"model": self.model, "input": batch}).encode(),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            out.extend(_normalize(list(map(float, d["embedding"]))) for d in data["data"])
        return out


def get_backend(name: str, model: str | None = None):
    if name == "hash":
        return HashBackend()
    if name == "sbert":
        return SbertBackend(model)
    if name == "openai":
        return OpenAIBackend(model)
    raise SystemExit(f"unknown backend: {name} (choose: hash, sbert, openai)")
