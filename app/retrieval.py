"""Recuperação sobre as regras: chunking por seção, embeddings e ranking.

Adaptado de sessions/rag_course_support.py (aulas 3 e 4).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from app.metrics import MetricsTracker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES_DIR = PROJECT_ROOT / "data" / "regras"
CACHE_FILE = PROJECT_ROOT / "data" / ".cache" / "embeddings.json"

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    title: str
    section: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def _slug(text: str) -> str:
    text = text.casefold()
    for accented, plain in zip("áàâãéêíóôõúç", "aaaaeeiooouc"):
        text = text.replace(accented, plain)
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def load_chunks(directory: Path = RULES_DIR) -> list[Chunk]:
    """Um chunk por seção '##' de cada arquivo Markdown, mantendo a citação estável."""
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        title = lines[0].lstrip("# ").strip()
        section, body = title, []

        def flush() -> None:
            text = "\n".join(body).strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{path.stem}::{_slug(section)}",
                        source=path.name,
                        title=title,
                        section=section,
                        text=text,
                    )
                )

        for line in lines[1:]:
            if line.startswith("## "):
                flush()
                section, body = line[3:].strip(), []
            else:
                body.append(line)
        flush()

    if not chunks:
        raise FileNotFoundError(f"Nenhuma regra encontrada em {directory}")
    return chunks


def embedding_text(chunk: Chunk) -> str:
    """Inclui título e seção para o embedding saber de que regra se trata."""
    return f"{chunk.title} - {chunk.section}\n{chunk.text}"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


class RulesIndex:
    """Índice em memória; os embeddings dos chunks ficam em cache em disco."""

    def __init__(self, client: OpenAI, chunks: list[Chunk] | None = None) -> None:
        self.client = client
        self.chunks = chunks or load_chunks()
        self.vectors = self._load_or_create_embeddings()

    def _embed(
        self, texts: list[str], metrics: MetricsTracker | None = None
    ) -> list[list[float]]:
        started_at = time.perf_counter()
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        if metrics is not None:
            metrics.record("embedding da consulta", EMBEDDING_MODEL, response.usage, started_at)
        return [item.embedding for item in response.data]

    def _load_or_create_embeddings(self) -> list[list[float]]:
        texts = [embedding_text(chunk) for chunk in self.chunks]
        keys = [
            hashlib.sha256(f"{EMBEDDING_MODEL}\n{text}".encode()).hexdigest()
            for text in texts
        ]
        cache: dict[str, list[float]] = {}
        if CACHE_FILE.is_file():
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

        missing = [i for i, key in enumerate(keys) if key not in cache]
        if missing:
            new_vectors = self._embed([texts[i] for i in missing])
            for i, vector in zip(missing, new_vectors, strict=True):
                cache[keys[i]] = vector
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps({k: cache[k] for k in keys}), encoding="utf-8")
        return [cache[key] for key in keys]

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.3,
        metrics: MetricsTracker | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k deve ser pelo menos 1.")
        query_vector = self._embed([query], metrics)[0]
        results = [
            SearchResult(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors, strict=True)
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return [result for result in results[:top_k] if result.score >= min_score]


def format_context(results: list[SearchResult]) -> str:
    """Contexto com rótulos estáveis para o modelo citar."""
    return "\n\n---\n\n".join(
        f"[FONTE: {r.chunk.source} | CHUNK: {r.chunk.chunk_id}]\n"
        f"{r.chunk.title} - {r.chunk.section}\n{r.chunk.text}"
        for r in results
    )
