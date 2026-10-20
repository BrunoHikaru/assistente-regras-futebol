"""Schemas validados com Pydantic: saída de cada passo do LLM e resposta final."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Categoria = Literal[
    "impedimento",
    "mao_na_bola",
    "falta",
    "cartao",
    "penalti",
    "reinicio_de_jogo",
    "bola_e_gol",
    "jogadores",
    "outra_regra",
    "fora_do_tema",
]


# Passo 1: classificação do lance
class Classificacao(BaseModel):
    categoria: Categoria
    consulta_busca: str = Field(
        description="Reformulação do lance em linguagem de regulamento, para a busca"
    )


# Passo 2: decisão fundamentada nas regras
class Citacao(BaseModel):
    fonte: str
    chunk_id: str
    trecho: str = Field(description="Citação curta e literal do trecho usado")


class Decisao(BaseModel):
    # A ordem importa: o modelo cita e explica antes de escrever a decisão final.
    respondivel: bool
    informacao_em_falta: list[str] = Field(
        default_factory=list,
        description="Perguntas curtas sobre detalhes sem os quais as regras não permitem decidir",
    )
    citacoes: list[Citacao] = Field(default_factory=list)
    explicacao: str
    decisao: str = Field(description="Decisão curta, ex.: 'Impedimento - tiro livre indireto'")


# Resposta final devolvida pela API
Status = Literal[
    "respondido",
    "precisa_de_mais_informacao",
    "fora_do_tema",
    "sem_evidencia",
]


class Evidencia(BaseModel):
    chunk_id: str
    fonte: str
    titulo: str
    score: float
    texto: str


class MetricaEtapa(BaseModel):
    etapa: str
    modelo: str
    latencia_ms: int
    tokens_entrada: int
    tokens_saida: int
    custo_usd: float | None = Field(description="None se o modelo não tiver preço conhecido")


class Metricas(BaseModel):
    latencia_ms: int = Field(description="Tempo total do pedido, incluindo a recuperação")
    tokens_entrada: int
    tokens_saida: int
    tokens_total: int
    custo_usd: float | None = Field(description="None se alguma etapa não tiver preço conhecido")
    etapas: list[MetricaEtapa] = Field(default_factory=list)


class Resultado(BaseModel):
    status: Status
    categoria: Categoria
    decisao: str | None = None
    explicacao: str
    perguntas: list[str] = Field(default_factory=list)
    citacoes: list[Citacao] = Field(default_factory=list)
    evidencias: list[Evidencia] = Field(default_factory=list)
    avisos: list[str] = Field(default_factory=list)
    metricas: Metricas | None = None
