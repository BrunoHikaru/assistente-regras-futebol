"""Avaliação ponta a ponta com os casos de tests/casos.json (usa a API da OpenAI).

Uso: uv run python projects/assistente-regras-futebol/tests/run_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import RefereeAssistant  # noqa: E402


CASES_FILE = PROJECT_ROOT / "tests" / "casos.json"


def check_case(case: dict, result) -> list[str]:
    failures: list[str] = []
    if result.status != case["status_esperado"]:
        failures.append(f"status={result.status} (esperado {case['status_esperado']})")
    expected_category = case["categoria_esperada"]
    if expected_category and result.categoria != expected_category:
        failures.append(f"categoria={result.categoria} (esperado {expected_category})")

    # Conteúdo: a decisão + explicação tem de mencionar cada termo ("a|b" = a ou b)
    answer_text = f"{result.decisao or ''} {result.explicacao}".casefold()
    for term in case.get("decisao_deve_conter", []):
        if not any(option.casefold() in answer_text for option in term.split("|")):
            failures.append(f"resposta não menciona: {term!r}")

    # Recuperação: as fontes esperadas têm de aparecer nas evidências e nas citações
    retrieved = {ev.fonte for ev in result.evidencias}
    cited = {c.fonte for c in result.citacoes}
    for source in case["fontes_esperadas"]:
        if source not in retrieved:
            failures.append(f"fonte não recuperada: {source}")
        elif source not in cited:
            failures.append(f"fonte não citada: {source}")
    failures.extend(f"citação inválida: {aviso}" for aviso in result.avisos)
    return failures


def main() -> None:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    assistant = RefereeAssistant()
    passed = 0

    for case in cases:
        result = assistant.analyze(case["pergunta"])
        failures = check_case(case, result)
        passed += not failures
        print(f"{'PASS' if not failures else 'FAIL'}  {case['id']:<28} {result.status}")
        if result.decisao:
            print(f"      -> {result.decisao}")
        for failure in failures:
            print(f"      ! {failure}")

    print(f"\n{passed}/{len(cases)} casos passaram")
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
