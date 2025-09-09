import time
import streamlit as st
import c_backend
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="BOT Colocación | Análisis SQL", page_icon="💼")
st.title("💼 BOT de PREGUNTAS DE NEGOCIO sobre Colocación")

st.write("")  # Línea en blanco
st.write("💡 **Ejemplos de consultas útiles (Colocación):**")
ejemplos = [
    "📊 ¿CUÁLES FUERON LAS 5 SUCURSALES CON MAYOR MONTO COLOCADO EN 2025?",
    "🏪 TOP 10 SUCURSALES POR MONTO COLOCADO EN 2025",
    "🗓️ DIME LAS COLOCACIONES POR MES EN 2025 ME INTERESA CONOCER EL CONTEO Y EL MONTO",
    "🧭 MONTO TOTAL COLOCADO POR PRODUCTO EN EL TERCER TRIMESTRE DE 2025",
    "🔎 DAME UNA LISTA DE TODAS LAS SUCURSALES",
    "💳 CUÁNTAS COLOCACIONES DE ‘PERSONAL CMV’ HUBO EN SUCURSAL LA BARCA EN JULIO 2025"
]
for ej in ejemplos:
    st.write(f"- {ej}")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if "rewriter_llm" not in st.session_state:
    st.session_state.rewriter_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

REWRITE_PROMPT = PromptTemplate.from_template(
    """Eres un reescritor de preguntas de análisis de datos de colocación.
Reescribe la pregunta para que sea totalmente autónoma y completa si depende del historial.

Historial:
{history}

Pregunta nueva:
{question}

Pregunta autónoma:"""
)

def _historial(max_turnos=2):
    msgs = st.session_state.mensajes
    pares, i = [], len(msgs)-1
    while i >= 1 and len(pares) < max_turnos:
        if msgs[i]["role"] == "assistant" and msgs[i-1]["role"] == "user":
            pares.append(f"Usuario: {msgs[i-1]['content']}\nAsistente: {msgs[i]['content']}")
            i -= 2
        else:
            i -= 1
    pares.reverse()
    return "\n\n".join(pares) if pares else "(sin historial)"

def reescribir_pregunta_si_aplica(pregunta):
    try:
        h = _historial()
        out = st.session_state.rewriter_llm.invoke(REWRITE_PROMPT.format(history=h, question=pregunta))
        return (out.content or "").strip() or pregunta
    except Exception:
        return pregunta

def _altura_para_df(df_len, max_height=420):
    return min(max_height, 42 + (32 * max(df_len, 1)))

# Render historial
for i, m in enumerate(st.session_state.mensajes):
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("df") is not None:
            height = _altura_para_df(len(m["df"]))
            st.dataframe(m["df"], use_container_width=True, height=height)
            st.download_button("📥 Exportar este resultado a CSV",
                               m["df"].to_csv(index=False).encode("utf-8"),
                               f"resultado_{i}.csv", mime="text/csv",
                               key=f"dl_hist_{i}")

# Input chat
if prompt := st.chat_input("¿En qué te puedo ayudar sobre colocación?"):
    st.session_state.mensajes.append({"role": "user", "content": prompt, "df": None})
    with st.chat_message("user"):
        st.write(prompt)

    pregunta_final = reescribir_pregunta_si_aplica(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            texto, df, sql = c_backend.consulta(pregunta_final)

        if "bloqueada por seguridad" in (texto or "").lower():
            st.error("🔒 Consulta bloqueada por seguridad: solo se permiten operaciones SELECT.")
        else:
            st.write(texto or "")
            if sql is not None:
                with st.expander("📄 Ver consulta SQL generada"):
                    st.code(sql, language="sql")

        if df is not None:
            height = _altura_para_df(len(df))
            st.dataframe(df, use_container_width=True, height=height)
            st.download_button("📥 Exportar este resultado a CSV",
                               df.to_csv(index=False).encode("utf-8"),
                               f"resultado_{int(time.time())}.csv", mime="text/csv",
                               key=f"dl_new_{int(time.time()*1000)}")

    st.session_state.mensajes.append({
        "role": "assistant",
        "content": texto,
        "df": df
    })

# Reset
if st.button("🧹Nueva conversación"):
    st.session_state.mensajes = []
    st.rerun()
