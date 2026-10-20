"""API FastAPI (adaptada de sessions/07-agent-api/starter/02_fastapi_service.py)."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.pipeline import RefereeAssistant
from app.schemas import Resultado


class AnalyzeRequest(BaseModel):
    pergunta: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=8)


@lru_cache(maxsize=1)
def get_assistant() -> RefereeAssistant:
    return RefereeAssistant()


app = FastAPI(title="Assistente de Regras de Futebol", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analisar", response_model=Resultado)
def analyze(
    request: AnalyzeRequest,
    assistant: RefereeAssistant = Depends(get_assistant),
) -> Resultado:
    try:
        return assistant.analyze(request.pergunta, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
