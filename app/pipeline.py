"""Fluxo completo: classificar -> recuperar -> decidir (ou perguntar) -> validar citações."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from openai import OpenAI

from app.classify import classify
from app.metrics import MetricsTracker
from app.rag import generate_decision, validate_citations
from app.retrieval import RulesIndex
from app.schemas import Evidencia, Resultado


load_dotenv()


class RefereeAssistant:
    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or OpenAI()
        self.index = RulesIndex(self.client)

    def analyze(self, question: str, top_k: int = 4, min_score: float = 0.3) -> Resultado:
        question = question.strip()
        if not question:
            raise ValueError("Descreva o lance ou faça uma pergunta.")

        metrics = MetricsTracker()
        result = self._analyze(question, top_k, min_score, metrics)
        result.metricas = metrics.summary()
        return result

    def _analyze(
        self, question: str, top_k: int, min_score: float, metrics: MetricsTracker
    ) -> Resultado:
        # Passo 1: classificar
        classification = classify(self.client, question, metrics)

        if classification.categoria == "fora_do_tema":
            return Resultado(
                status="fora_do_tema",
                categoria="fora_do_tema",
                explicacao=(
                    "Só consigo ajudar com as regras do futebol. "
                    "Descreva um lance ou faça uma pergunta sobre as regras."
                ),
            )

        # Recuperação: a consulta reescrita usa o vocabulário das regras
        query = f"{classification.consulta_busca}\n{question}"
        results = self.index.search(query, top_k=top_k, min_score=min_score, metrics=metrics)
        evidence = [
            Evidencia(
                chunk_id=r.chunk.chunk_id,
                fonte=r.chunk.source,
                titulo=f"{r.chunk.title} - {r.chunk.section}",
                score=round(r.score, 3),
                texto=r.chunk.text,
            )
            for r in results
        ]

        if not results:
            return Resultado(
                status="sem_evidencia",
                categoria=classification.categoria,
                explicacao="Não encontrei nenhuma regra relevante para este lance.",
            )

        # Passo 2: decidir com base nas regras (nova tentativa com feedback se as citações falharem)
        decision = generate_decision(self.client, question, results, metrics=metrics)
        errors = validate_citations(decision, results)
        if errors:
            decision = generate_decision(
                self.client, question, results, previous_errors=errors, metrics=metrics
            )
            errors = validate_citations(decision, results)

        if decision.informacao_em_falta:
            return Resultado(
                status="precisa_de_mais_informacao",
                categoria=classification.categoria,
                explicacao=decision.explicacao,
                perguntas=decision.informacao_em_falta,
                evidencias=evidence,
            )

        if not decision.respondivel:
            return Resultado(
                status="sem_evidencia",
                categoria=classification.categoria,
                explicacao=decision.explicacao,
                evidencias=evidence,
            )

        return Resultado(
            status="respondido",
            categoria=classification.categoria,
            decisao=decision.decisao,
            explicacao=decision.explicacao,
            citacoes=decision.citacoes,
            evidencias=evidence,
            avisos=errors,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisa um lance de futebol.")
    parser.add_argument("pergunta")
    args = parser.parse_args()
    print(RefereeAssistant().analyze(args.pergunta).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
