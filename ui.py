"""Interface Streamlit: comunica com a API FastAPI via HTTP."""

from __future__ import annotations

import os

import httpx
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

EXEMPLOS = [
    "O goleiro pegou com a mão uma bola recuada com o pé pelo zagueiro, dentro da área.",
    "O atacante estava impedido, mas não tocou na bola nem atrapalhou ninguém. É impedimento?",
    "O zagueiro deu um carrinho por trás com força excessiva no meio-campo.",
    "Teve mão na bola do defensor.",
    "Quem ganhou a Copa de 2002?",
]

STATUS_LABEL = {
    "respondido": ("✅ Decisão", "success"),
    "precisa_de_mais_informacao": ("❓ Preciso de mais detalhes", "warning"),
    "fora_do_tema": ("🚫 Fora do tema", "info"),
    "sem_evidencia": ("📭 Sem regra que cubra o caso", "info"),
}

st.set_page_config(page_title="Assistente de Regras de Futebol", page_icon="⚽")
st.title("⚽ Assistente de Regras de Futebol")
st.caption("Descreva um lance e receba a decisão com as regras citadas (Leis do Jogo da IFAB).")

exemplo = st.selectbox("Exemplos", [""] + EXEMPLOS)
pergunta = st.text_area("Lance ou pergunta", value=exemplo, height=100)

if st.button("Analisar", type="primary", disabled=not pergunta.strip()):
    try:
        with st.spinner("Consultando as regras..."):
            response = httpx.post(
                f"{API_URL}/analisar", json={"pergunta": pergunta}, timeout=60
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        st.error(f"Erro da API ({exc.response.status_code}): {exc.response.text}")
        st.stop()
    except httpx.HTTPError:
        st.error(f"Não foi possível ligar à API em {API_URL}. Ela está a correr?")
        st.stop()

    result = response.json()
    label, kind = STATUS_LABEL[result["status"]]
    getattr(st, kind)(f"**{label}** · categoria: `{result['categoria']}`")

    if result["decisao"]:
        st.subheader(result["decisao"])
    st.write(result["explicacao"])

    for pergunta_extra in result["perguntas"]:
        st.markdown(f"- {pergunta_extra}")

    if result["citacoes"]:
        st.markdown("#### 📖 Regras citadas")
        for citacao in result["citacoes"]:
            st.markdown(f"> {citacao['trecho']}\n\n`{citacao['chunk_id']}`")

    for aviso in result["avisos"]:
        st.warning(aviso)

    if result["evidencias"]:
        with st.expander("Trechos recuperados (evidência)"):
            for ev in result["evidencias"]:
                st.markdown(f"**{ev['titulo']}** · score {ev['score']}")
                st.caption(ev["texto"])

    metricas = result.get("metricas")
    if metricas:
        st.divider()
        col_latencia, col_tokens, col_custo = st.columns(3)
        col_latencia.metric("⏱️ Latência", f"{metricas['latencia_ms'] / 1000:.2f} s")
        col_tokens.metric(
            "🔢 Tokens",
            f"{metricas['tokens_total']:,}",
            help=f"Entrada: {metricas['tokens_entrada']:,} · Saída: {metricas['tokens_saida']:,}",
        )
        custo = metricas["custo_usd"]
        col_custo.metric(
            "💲 Custo",
            "n/d" if custo is None else f"US$ {custo:.5f}",
            help="Estimativa com a tabela de preços em app/metrics.py",
        )
        with st.expander("Métricas por etapa"):
            st.dataframe(
                [
                    {
                        "Etapa": etapa["etapa"],
                        "Modelo": etapa["modelo"],
                        "Latência (ms)": etapa["latencia_ms"],
                        "Tokens entrada": etapa["tokens_entrada"],
                        "Tokens saída": etapa["tokens_saida"],
                        "Custo (US$)": etapa["custo_usd"],
                    }
                    for etapa in metricas["etapas"]
                ],
                hide_index=True,
            )
