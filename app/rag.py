"""Passo 2 (LLM): decisão fundamentada nas regras recuperadas, com citações validadas.

Adaptado de sessions/04-rag-pipeline/starter/03_grounded_rag.py.
"""

from __future__ import annotations

import os
import re
import time

from openai import OpenAI

from app.metrics import MetricsTracker
from app.retrieval import SearchResult, format_context
from app.schemas import Decisao


GENERATION_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

INSTRUCTIONS = """És um árbitro assistente que explica decisões com base nas Leis do Jogo.

Usa SOMENTE as regras fornecidas em "Regras". Não uses conhecimento próprio.
Considera como factos tudo o que a descrição do lance já diz, e NÃO presumas
detalhes que não foram ditos (local do lance, intenção, posição dos jogadores).

Segue estes passos por ordem:
1. Cobertura: se nenhuma das regras fornecidas trata o tema da pergunta
   (ex.: a pergunta é sobre o VAR e nenhuma regra fala do VAR), devolve
   respondivel=false, informacao_em_falta vazia, citacoes vazia e explica que
   a base de regras não cobre esse caso. Não faças perguntas neste caso.
2. Detalhes em falta: se a decisão muda conforme um detalhe que a descrição
   NÃO dá (ex.: dentro ou fora da área decide entre tiro livre e pênalti),
   devolve respondivel=false, até 2 perguntas curtas em informacao_em_falta,
   citacoes vazia e uma explicação curta de porque esse detalhe importa.
   Não perguntes por detalhes que não mudam a decisão nem por detalhes já
   descritos. Perguntas gerais sobre regras nunca têm detalhes em falta.
3. Decisão: respondivel=true, informacao_em_falta vazia, citações de todas as
   regras usadas, uma explicação simples em português e uma decisão curta
   (ex.: "Pênalti e cartão amarelo").

Cada citação deve usar exatamente a FONTE e o CHUNK indicados e um trecho curto
copiado literalmente e de forma contínua do texto da regra, sem reticências.
"""


def generate_decision(
    client: OpenAI,
    question: str,
    results: list[SearchResult],
    previous_errors: list[str] | None = None,
    metrics: MetricsTracker | None = None,
) -> Decisao:
    context = format_context(results) or "Nenhuma regra foi encontrada."
    prompt = f"Lance:\n{question}\n\nRegras:\n{context}"
    if previous_errors:
        # Com temperature=0, repetir o mesmo pedido daria a mesma resposta
        prompt += (
            "\n\nA tua resposta anterior falhou a validação das citações:\n- "
            + "\n- ".join(previous_errors)
            + "\nCorrige as citações copiando o texto exatamente como aparece nas regras."
        )
    started_at = time.perf_counter()
    response = client.responses.parse(
        model=GENERATION_MODEL,
        instructions=INSTRUCTIONS,
        input=prompt,
        text_format=Decisao,
        temperature=0,
    )
    if metrics is not None:
        etapa = "decisao (nova tentativa)" if previous_errors else "decisao"
        metrics.record(etapa, GENERATION_MODEL, response.usage, started_at)
    if response.output_parsed is None:
        raise RuntimeError("O modelo não devolveu uma decisão válida.")
    return response.output_parsed


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip(".\"'“”").casefold()


def validate_citations(decision: Decisao, results: list[SearchResult]) -> list[str]:
    """Não confia nas citações do modelo: cada trecho tem de existir no chunk citado."""
    errors: list[str] = []
    available = {(r.chunk.source, r.chunk.chunk_id): r.chunk.text for r in results}

    if decision.respondivel and not decision.citacoes:
        errors.append("Uma resposta deve incluir pelo menos uma citação.")
    if not decision.respondivel and decision.citacoes:
        errors.append("Uma recusa não deve incluir citações.")

    for citation in decision.citacoes:
        key = (citation.fonte, citation.chunk_id)
        if key not in available:
            errors.append(f"Citação para um trecho desconhecido: {citation.chunk_id}")
            continue
        # Reticências são aceites, mas cada fragmento tem de existir no texto
        fragments = [_normalize(f) for f in re.split(r"\.\.\.|…", citation.trecho)]
        fragments = [f for f in fragments if f]
        chunk_text = _normalize(available[key])
        if not fragments:
            errors.append(f"Trecho vazio em {citation.chunk_id}")
        elif any(fragment not in chunk_text for fragment in fragments):
            errors.append(f"Trecho não encontrado em {citation.chunk_id}: {citation.trecho!r}")
    return errors
