"""Métricas de cada pedido: latência, tokens e custo estimado das chamadas à OpenAI."""

from __future__ import annotations

import time
from typing import Any

from app.schemas import MetricaEtapa, Metricas


# Preço em USD por 1M de tokens: (entrada, entrada em cache, saída).
# Confirme os valores atuais em https://openai.com/api/pricing e acrescente outros modelos aqui.
PRICES_PER_MILLION: dict[str, tuple[float, float, float]] = {
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "text-embedding-3-small": (0.02, 0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.13, 0.0),
}


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0
) -> float | None:
    """Custo em USD, ou None se o modelo não estiver na tabela de preços."""
    prices = PRICES_PER_MILLION.get(model)
    if prices is None:
        return None
    input_price, cached_price, output_price = prices
    cost = (
        (input_tokens - cached_tokens) * input_price
        + cached_tokens * cached_price
        + output_tokens * output_price
    )
    return round(cost / 1_000_000, 8)


class MetricsTracker:
    """Acumula as chamadas de um pedido; a latência total conta desde a criação."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.etapas: list[MetricaEtapa] = []

    def record(self, etapa: str, model: str, usage: Any, started_at: float) -> None:
        """Regista uma chamada. `usage` vem da Responses API ou da API de embeddings."""
        if hasattr(usage, "input_tokens"):  # Responses API
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            details = getattr(usage, "input_tokens_details", None)
            cached_tokens = getattr(details, "cached_tokens", 0) or 0
        else:  # Embeddings
            input_tokens, output_tokens, cached_tokens = usage.prompt_tokens, 0, 0

        self.etapas.append(
            MetricaEtapa(
                etapa=etapa,
                modelo=model,
                latencia_ms=round((time.perf_counter() - started_at) * 1000),
                tokens_entrada=input_tokens,
                tokens_saida=output_tokens,
                custo_usd=estimate_cost(model, input_tokens, output_tokens, cached_tokens),
            )
        )

    def summary(self) -> Metricas:
        costs = [etapa.custo_usd for etapa in self.etapas]
        tokens_in = sum(etapa.tokens_entrada for etapa in self.etapas)
        tokens_out = sum(etapa.tokens_saida for etapa in self.etapas)
        return Metricas(
            latencia_ms=round((time.perf_counter() - self._start) * 1000),
            tokens_entrada=tokens_in,
            tokens_saida=tokens_out,
            tokens_total=tokens_in + tokens_out,
            custo_usd=None if None in costs else round(sum(costs), 8),
            etapas=self.etapas,
        )
