# c_backend.py
import os
import re
import time
import sqlite3
import requests
import streamlit as st
import pandas as pd
import unicodedata
import gdown

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate

# =========================
#   Rutas cross-platform (raíz del proyecto)
# =========================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_LOCAL_PATH = os.path.join(PROJECT_DIR, "colocacion.db")

DB_CANDIDATES = [DB_LOCAL_PATH]

# =========================
#   Descargar base de datos (versión robusta) — (adaptada de tu código bueno)
# =========================
@st.cache_data(ttl=3600)
def download_database():
    """
    Descarga colocacion.db desde Google Drive a la RAÍZ del proyecto.
    """
    db_path = DB_LOCAL_PATH

    # Si ya existe y pesa "bien", úsala
    if os.path.exists(db_path):
        file_size = os.path.getsize(db_path)
        if file_size > 1000:
            return db_path
        else:
            try:
                os.remove(db_path)
            except Exception:
                pass

    try:
        # ID de Drive que me diste para colocación
        file_id = "1QG7X70hoO5kc0f-nsX6Qy-NqBqoq-hZY"
        url = f"https://drive.google.com/uc?id={file_id}"

        progress_container = st.container()
        with progress_container:
            st.info("🔄 Descargando base de datos de colocación... Esto puede tardar unos segundos.")
            progress_bar = st.progress(10)
            status_text = st.empty()

            # Intento 1: gdown
            try:
                status_text.text("Conectando con Google Drive...")
                output = gdown.download(url, db_path, quiet=True)
                if output:
                    progress_bar.progress(100)
                    status_text.text("✅ Base de datos descargada exitosamente!")
                    time.sleep(1)
                    return db_path
                else:
                    raise Exception("gdown no pudo descargar el archivo")
            except Exception:
                status_text.text("Intentando método alternativo...")
                progress_bar.progress(50)

                session = requests.Session()
                response = session.get(f"https://drive.google.com/uc?export=download&id={file_id}", stream=True)

                # token de confirmación (archivos “grandes”)
                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                        break

                urls = [f"https://drive.google.com/uc?export=download&id={file_id}"]
                if token:
                    urls.insert(0, f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}")

                for durl in urls:
                    try:
                        response = session.get(durl, stream=True, timeout=300)
                        if response.status_code == 200 and 'text/html' not in response.headers.get('content-type', ''):
                            total_size = int(response.headers.get('content-length', 0))
                            block_size = 8192
                            downloaded = 0
                            temp_path = db_path + ".tmp"

                            with open(temp_path, 'wb') as f:
                                for chunk in response.iter_content(block_size):
                                    if chunk:
                                        f.write(chunk)
                                        downloaded += len(chunk)
                                        if total_size > 0:
                                            progress = int(50 + (downloaded / total_size) * 50)
                                            progress_bar.progress(progress)
                                            status_text.text(f"Descargando... {downloaded / 1024 / 1024:.1f} MB")

                            # Verificación rápida de SQLite
                            try:
                                conn = sqlite3.connect(temp_path)
                                cursor = conn.cursor()
                                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                                cursor.close()
                                conn.close()

                                os.replace(temp_path, db_path)
                                progress_bar.progress(100)
                                status_text.text("✅ Base de datos descargada y verificada!")
                                time.sleep(1)
                                return db_path

                            except sqlite3.DatabaseError:
                                try:
                                    os.remove(temp_path)
                                except Exception:
                                    pass
                                status_text.text("❌ Archivo descargado no válido (no es SQLite)")
                                continue
                    except:
                        continue

                raise Exception("No se pudo descargar el archivo.")
    except Exception as e:
        st.error(f"❌ Error al descargar base de datos: {str(e)}")
        st.error("Verifica que el archivo sea público en Drive.")
        return None
    finally:
        if 'progress_container' in locals():
            progress_container.empty()

# =========================
#   Inicialización de BD (usa candidatos y, si falta, descarga)
# =========================
@st.cache_resource
def init_database():
    db_path = None
    for p in DB_CANDIDATES:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            db_path = os.path.abspath(p)
            break

    if not db_path:
        dl_path = download_database()
        if dl_path and os.path.exists(dl_path) and os.path.getsize(dl_path) > 0:
            db_path = os.path.abspath(dl_path)
        else:
            st.error("No se encontró la base de datos 'colocacion.db' y no se pudo descargar desde Drive.")
            return None

    try:
        return SQLDatabase.from_uri(f"sqlite:///{db_path}")
    except Exception as e:
        st.error(f"Error al abrir SQLite: {e}")
        return None

db = init_database()

# =========================
#   API KEY
# =========================
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    try:
        import a_env_vars
        os.environ["OPENAI_API_KEY"] = a_env_vars.OPENAI_API_KEY
    except Exception:
        pass

# =========================
#   Helpers Seguridad/UX
# =========================
def limpiar_fences_y_espacios(sql: str) -> str:
    if sql is None:
        return ""
    s = sql.strip()
    s = re.sub(r"^```(?:sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s, flags=re.IGNORECASE)
    return s.strip()

def es_consulta_segura(sql: str) -> bool:
    s = re.sub(r'--.*?(\n|$)', '', sql, flags=re.IGNORECASE|re.DOTALL)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.IGNORECASE|re.DOTALL)
    s = limpiar_fences_y_espacios(s)
    s_low = s.lower()
    if not (s_low.startswith("select") or s_low.startswith("with")):
        return False
    peligrosas = [
        r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b", r"\balter\b",
        r"\bcreate\b", r"\btruncate\b", r"\breplace\b", r"\battach\b",
        r"\bdetach\b", r"\bpragma\b", r"\bexec\b", r"\bexecute\b", r"\bvacuum\b"
    ]
    for pat in peligrosas:
        if re.search(pat, s_low):
            return False
    return True

def quitar_acentos(texto: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def corregir_sql_sucursal(sql: str) -> str:
    patron = re.compile(r'("?SUCURSAL"?\s*=\s*)\'([^\']+)\'', re.IGNORECASE)
    return patron.sub(lambda m: f"UPPER(SUCURSAL) = '{quitar_acentos(m.group(2)).upper()}'", sql)

def eliminar_limit_si_lista_sucursales(sql: str) -> str:
    if re.search(r'select\s+distinct\s+"?sucursal"?\s+from', sql, re.IGNORECASE):
        return re.sub(r'\blimit\s+\d+\b', '', sql, flags=re.IGNORECASE)
    return sql

def expandir_fecha_igual_a_dia(sql: str) -> str:
    s = sql
    m = re.search(r"fecha_colocacion\s*=\s*'(\d{4})-(\d{2})-(\d{2})'", s, re.IGNORECASE)
    if m:
        y, mm, _dd = m.group(1), m.group(2), m.group(3)
        rango = (f"(fecha_colocacion >= date('{y}-{mm}-01') "
                 f"AND fecha_colocacion < date('{y}-{mm}-01','+1 month'))")
        s = re.sub(r"fecha_colocacion\s*=\s*'\d{4}-\d{2}-\d{2}'", rango, s, flags=re.IGNORECASE)
    m2 = re.search(r"fecha_colocacion\s+like\s+'(\d{4})-(\d{2})-%'", s, re.IGNORECASE)
    if m2:
        y, mm = m2.group(1), m2.group(2)
        rango = (f"(fecha_colocacion >= date('{y}-{mm}-01') "
                 f"AND fecha_colocacion < date('{y}-{mm}-01','+1 month'))")
        s = re.sub(r"fecha_colocacion\s+like\s+'\d{4}-\d{2}-%'", rango, s, flags=re.IGNORECASE)
    return s

def forzar_vista_detalle(sql: str) -> str:
    return re.sub(r'\bvw_fact_colocacion_bruta_agrupado\b',
                  'vw_fact_colocacion_bruta_detalle', sql, flags=re.IGNORECASE)

def dejar_solo_un_statement(sql: str) -> str:
    return sql.split(";")[0].strip()

def primer_select_o_with(sql: str) -> str:
    s = limpiar_fences_y_espacios(sql)
    s_sin_com = re.sub(r'--.*?(\n|$)', '', s, flags=re.IGNORECASE|re.DOTALL)
    s_sin_com = re.sub(r'/\*.*?\*/', '', s_sin_com, flags=re.IGNORECASE|re.DOTALL)
    m = re.search(r'\b(select|with)\b', s_sin_com, flags=re.IGNORECASE)
    if not m:
        return s
    m2 = re.search(r'\b(select|with)\b', s, flags=re.IGNORECASE)
    if m2:
        return s[m2.start():].strip()
    return s[m.start():].strip()

def actualizar_ultima_sucursal(sql: str):
    m = re.search(r"UPPER\(SUCURSAL\)\s*=\s*'([^']+)'", sql, re.IGNORECASE)
    if m:
        st.session_state["ultima_sucursal"] = m.group(1)

# =========================
#   LangChain Chain
# =========================
@st.cache_resource
def init_chain():
    global db
    if db is None:
        db = init_database()
        if db is None:
            return None, None, None, None
    try:
        llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
        query_chain = create_sql_query_chain(llm, db)

        answer_prompt = PromptTemplate.from_template(
            """
Base de datos de COLocación (vista principal: `vw_fact_colocacion_bruta_detalle`).

Columnas útiles:
- region, sucursal, producto, monto
- fecha_colocacion, fecha_colocacion_año, fecha_colocacion_mes, fecha_colocacion_trimestre

REGLAS:
1) Usa siempre `vw_fact_colocacion_bruta_detalle` (no uses *_agrupado).
2) Para fechas: usa *_año, *_mes, *_trimestre cuando aplique.
3) Para sucursal: UPPER(SUCURSAL) = 'MAYUSCULAS_SIN_ACENTO'.
4) Agrega LIMIT si no lo pones (1000).
5) Si piden lista de sucursales (distinct), sin LIMIT.

Pregunta del usuario: {question}
Consulta SQL generada: {query}
Resultado SQL (muestra): {result}

Redacta respuesta breve y clara.
"""
        )
        return query_chain, db, answer_prompt, llm
    except Exception as e:
        st.error(f"Error al inicializar la cadena: {str(e)}")
        return None, None, None, None

# =========================
#   FLUJO PRINCIPAL
# =========================
def consulta(pregunta_usuario: str):
    try:
        if "OPENAI_API_KEY" not in os.environ:
            return "❌ No se configuró la API Key.", None, None

        query_chain, db_sql, prompt, llm = init_chain()
        if not query_chain or not db_sql:
            return "⚠️ No se pudo inicializar el sistema.", None, None

        with st.spinner("🔍 Generando consulta SQL..."):
            consulta_sql = query_chain.invoke({"question": pregunta_usuario})

        consulta_sql = limpiar_fences_y_espacios(consulta_sql)
        consulta_sql = forzar_vista_detalle(consulta_sql)
        consulta_sql = corregir_sql_sucursal(consulta_sql)
        consulta_sql = expandir_fecha_igual_a_dia(consulta_sql)
        consulta_sql = primer_select_o_with(consulta_sql)
        consulta_sql = eliminar_limit_si_lista_sucursales(consulta_sql)
        consulta_sql = dejar_solo_un_statement(consulta_sql)

        if not es_consulta_segura(consulta_sql):
            return "❌ Consulta bloqueada por seguridad. Solo se permiten operaciones SELECT.", None, consulta_sql

        if "limit" not in consulta_sql.lower():
            consulta_sql += " LIMIT 1000"

        # Resolver ruta de BD (candidato o descarga on-demand)
        path = None
        for p in DB_CANDIDATES:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                path = p
                break
        if not path:
            path = download_database()

        if not path or not os.path.exists(path):
            return "⚠️ No se encontró/descargó la base de datos colocacion.db.", None, consulta_sql

        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute(consulta_sql)
        columnas = [d[0] for d in cur.description]
        filas = cur.fetchall()
        conn.close()

        actualizar_ultima_sucursal(consulta_sql)

        muestra = str(filas[:3]) + (" ..." if len(filas) > 3 else "")
        with st.spinner("💬 Generando respuesta..."):
            respuesta = llm.invoke(prompt.format_prompt(
                question=pregunta_usuario,
                query=consulta_sql,
                result=muestra
            ).to_string())

        df = pd.DataFrame(filas, columns=columnas)
        return (respuesta.content if hasattr(respuesta, "content") else str(respuesta)), df, consulta_sql

    except Exception as e:
        return f"⚠️ Error: {str(e)}", None, None
