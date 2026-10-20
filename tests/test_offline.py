"""Testes rápidos que não chamam a API (chunking, validação de citações e métricas).

Uso: uv run python projects/assistente-regras-futebol/tests/test_offline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.metrics import MetricsTracker, estimate_cost  # noqa: E402
from app.rag import validate_citations  # noqa: E402
from app.retrieval import SearchResult, load_chunks  # noqa: E402
from app.schemas import Citacao, Decisao  # noqa: E402


def test_chunks_have_unique_ids_and_text() -> None:
    chunks = load_chunks()
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids)), "IDs de chunk duplicados"
    assert all(chunk.text for chunk in chunks)
    assert "03-lei-11-impedimento::excecoes" in ids


def _impedimento_results() -> list[SearchResult]:
    chunk = next(c for c in load_chunks() if c.chunk_id == "03-lei-11-impedimento::excecoes")
    return [SearchResult(chunk=chunk, score=0.9)]


def test_valid_citation_passes() -> None:
    results = _impedimento_results()
    decision = Decisao(
        respondivel=True,
        decisao="Não é impedimento",
        explicacao="...",
        citacoes=[
            Citacao(
                fonte="03-lei-11-impedimento.md",
                chunk_id="03-lei-11-impedimento::excecoes",
                trecho="diretamente de um tiro de meta, de um arremesso lateral",
            )
        ],
    )
    assert validate_citations(decision, results) == []


def test_invented_quote_fails() -> None:
    results = _impedimento_results()
    decision = Decisao(
        respondivel=True,
        decisao="Impedimento",
        explicacao="...",
        citacoes=[
            Citacao(
                fonte="03-lei-11-impedimento.md",
                chunk_id="03-lei-11-impedimento::excecoes",
                trecho="o impedimento vale sempre",
            )
        ],
    )
    assert validate_citations(decision, results)


def test_quote_with_ellipsis_checks_each_fragment() -> None:
    def decision(quote: str) -> Decisao:
        return Decisao(
            respondivel=True,
            decisao="Não é impedimento",
            explicacao="...",
            citacoes=[
                Citacao(
                    fonte="03-lei-11-impedimento.md",
                    chunk_id="03-lei-11-impedimento::excecoes",
                    trecho=quote,
                )
            ],
        )

    results = _impedimento_results()
    assert validate_citations(decision("Não há infração ... de um arremesso lateral"), results) == []
    assert validate_citations(decision("Não há infração ... de um pênalti"), results)


def test_answer_without_citation_fails() -> None:
    decision = Decisao(respondivel=True, decisao="X", explicacao="...", citacoes=[])
    assert validate_citations(decision, _impedimento_results())


def test_metrics_sum_tokens_and_cost() -> None:
    llm_usage = SimpleNamespace(
        input_tokens=1_000_000,
        output_tokens=500_000,
        input_tokens_details=SimpleNamespace(cached_tokens=200_000),
    )
    embedding_usage = SimpleNamespace(prompt_tokens=1_000_000)

    tracker = MetricsTracker()
    tracker.record("decisao", "gpt-4.1-mini", llm_usage, time.perf_counter())
    tracker.record("embedding da consulta", "text-embedding-3-small", embedding_usage, time.perf_counter())
    summary = tracker.summary()

    assert summary.tokens_entrada == 2_000_000
    assert summary.tokens_saida == 500_000
    # 800k x 0.40 + 200k x 0.10 + 500k x 1.60 + 1M x 0.02 (por 1M de tokens)
    assert abs(summary.custo_usd - (0.32 + 0.02 + 0.80 + 0.02)) < 1e-9
    assert [e.etapa for e in summary.etapas] == ["decisao", "embedding da consulta"]


def test_unknown_model_has_no_cost() -> None:
    assert estimate_cost("modelo-desconhecido", 100, 100) is None


if __name__ == "__main__":
    tests = [value for name, value in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} testes passaram")
