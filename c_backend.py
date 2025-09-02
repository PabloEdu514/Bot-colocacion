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
#   Descargar base de datos (versión robusta)
# =========================
@st.cache_data(ttl=3600)
def download_database():
    """
    Descarga colocacion.db desde Google Drive a la RAÍZ del proyecto si no existe localmente.
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
        # Cambia este file_id por el tuyo si usas Google Drive
        file_id = "1QG7X70hoO5kc0f-nsX6Qy-NqBqoq-hZY"
        url = f"https://drive.google.com/uc?id={file_id}"

        progress_container = st.container()
        with progress_container:
            st.info("🔄 Descargando base de datos de colocación... Puede tardar unos segundos.")
            progress_bar = st.progress(10)
            status_text = st.empty()

            # Intento 1: gdown
            try:
                status_text.text("Conectando con Google Drive...")
                output = gdown.download(url, db_path, quiet=True)
                if output:
                    progress_bar.progress(100)
                    status_text.text("✅ Base de datos descargada exitosamente")
                    time.sleep(0.8)
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
                                status_text.text("✅ Base de datos descargada y verificada")
                                time.sleep(0.8)
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
        st.error("Verifica que el archivo sea público en Drive o coloca 'colocacion.db' junto a este script.")
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
    # Quitar comentarios y fences
    s = re.sub(r'--.*?(\n|$)', '', sql or '', flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.IGNORECASE | re.DOTALL)
    s = limpiar_fences_y_espacios(s)
    s_low = s.lower().lstrip()

    # Solo permitimos SELECT o WITH
    if not (s_low.startswith("select") or s_low.startswith("with")):
        return False

    # Palabras peligrosas (DML/DDL/PRAGMA/etc.)
    peligrosas = [
        r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b", r"\balter\b",
        r"\bcreate\b", r"\btruncate\b", r"\battach\b", r"\bdetach\b",
        r"\bpragma\b", r"\bexec\b", r"\bexecute\b", r"\bvacuum\b"
    ]
    for pat in peligrosas:
        if re.search(pat, s_low):
            return False

    # Bloquear únicamente la sentencia DML "REPLACE INTO" (permitir la función REPLACE())
    if re.search(r"\breplace\s+into\b", s_low):
        return False

    return True

def quitar_acentos(texto: str) -> str:
    """
    Normaliza a NFD y elimina marcas diacríticas; retorna en MAYÚSCULAS.
    (ñ → n por normalización; luego .upper())
    """
    base = ''.join(
        c for c in unicodedata.normalize('NFD', texto or '')
        if unicodedata.category(c) != 'Mn'
    )
    return base.upper()

def expr_unaccent_col(colname: str) -> str:
    """
    Construye una expresión SQL con UPPER + REPLACE encadenados
    para normalizar ÁÉÍÓÚÜ y Ñ (paréntesis balanceados).
    """
    e = f"UPPER({colname})"
    e = f"REPLACE({e}, 'Á','A')"
    e = f"REPLACE({e}, 'É','E')"
    e = f"REPLACE({e}, 'Í','I')"
    e = f"REPLACE({e}, 'Ó','O')"
    e = f"REPLACE({e}, 'Ú','U')"
    e = f"REPLACE({e}, 'Ü','U')"
    e = f"REPLACE({e}, 'Ñ','N')"
    return e

def corregir_sql_sucursal(sql: str) -> str:
    """
    Reescribe condiciones SUCURSAL = 'texto' para que comparen con columna normalizada.
    """
    patron = re.compile(r'("?SUCURSAL"?\s*=\s*)\'([^\']+)\'', re.IGNORECASE)
    return patron.sub(lambda m: f"{expr_unaccent_col('SUCURSAL')} = '{quitar_acentos(m.group(2))}'", sql)

def corregir_sql_nombre(sql: str) -> str:
    """
    Reescribe condiciones NOMBRE = 'texto' para que comparen con columna normalizada.
    """
    patron = re.compile(r'("?NOMBRE"?\s*=\s*)\'([^\']+)\'', re.IGNORECASE)
    return patron.sub(lambda m: f"{expr_unaccent_col('NOMBRE')} = '{quitar_acentos(m.group(2))}'", sql)

def eliminar_limit_si_lista_sucursales(sql: str) -> str:
    """
    Si es una lista de sucursales (distinct), elimina cualquier LIMIT (queremos todas).
    """
    if re.search(r'^\s*select\s+distinct\s+"?sucursal"?\s+from\s+vw_fact_colocacion_bruta_detalle', sql, re.IGNORECASE):
        return re.sub(r'\blimit\s+\d+\b', '', sql, flags=re.IGNORECASE)
    return sql

def expandir_fecha_igual_a_dia(sql: str) -> str:
    """
    Cambia fecha_colocacion = 'YYYY-MM-DD' por rango mensual del mes correspondiente.
    También maneja 'LIKE YYYY-MM%'.
    """
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
    """
    Reemplaza cualquier referencia a *_agrupado por la vista *_detalle.
    """
    return re.sub(r'\bvw_fact_colocacion_bruta_agrupado\b',
                  'vw_fact_colocacion_bruta_detalle', sql, flags=re.IGNORECASE)

def dejar_solo_un_statement(sql: str) -> str:
    """
    Ejecuta solo el primer statement (antes de cualquier ';').
    """
    return (sql or '').split(";")[0].strip()

def primer_select_o_with(sql: str) -> str:
    """
    Quita comentarios y devuelve desde el primer SELECT/WITH real (evita prólogos).
    """
    s = limpiar_fences_y_espacios(sql or '')
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
    m = re.search(r"UPPER\(SUCURSAL\)\s*=\s*'([^']+)'", sql or '', re.IGNORECASE)
    if m:
        st.session_state["ultima_sucursal"] = m.group(1)

def es_lista_solo_sucursales(sql: str) -> bool:
    return bool(re.search(
        r'^\s*select\s+distinct\s+"?sucursal"?\s+from\s+vw_fact_colocacion_bruta_detalle',
        sql or '', re.IGNORECASE
    ))

def _normalizar_sql_modelo(consulta_sql: str) -> str:
    """
    Aplica todas las normalizaciones/saneos en orden seguro.
    """
    s = limpiar_fences_y_espacios(consulta_sql)
    s = forzar_vista_detalle(s)
    s = corregir_sql_sucursal(s)
    s = corregir_sql_nombre(s)
    s = expandir_fecha_igual_a_dia(s)
    s = primer_select_o_with(s)
    s = eliminar_limit_si_lista_sucursales(s)
    s = dejar_solo_un_statement(s)
    return s

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

        # Prompt de interpretación (no afecta al SQL, sólo a la explicación)
        answer_prompt = PromptTemplate.from_template(
"""
Base de datos: Caja Morelia Valladolid – Colocación.
Vista principal: `vw_fact_colocacion_bruta_detalle`.

COLUMNAS CLAVE:
- nombre: Nombre del ejecutivo (quien colocó el crédito).
- usuario: Código corto del ejecutivo.
- numero_empleado: Número de empleado del ejecutivo.
- sucursal: Nombre de la sucursal.
- region: Región de la sucursal.
- producto: Producto financiero colocado.
- monto: Monto colocado.
- fecha_colocacion: Fecha exacta de la colocación.
- fecha_colocacion_año, fecha_colocacion_mes, fecha_colocacion_trimestre: Derivados de la fecha.

REGLAS OBLIGATORIAS PARA LA CONSULTA:
1. Siempre usa la vista `vw_fact_colocacion_bruta_detalle`.
2. Para fechas:
   - Usa `fecha_colocacion_año` y `fecha_colocacion_mes` para meses.
   - Usa `fecha_colocacion_año` y `fecha_colocacion_trimestre` para trimestres.
   - Nunca uses directamente `fecha_colocacion = 'YYYY-MM-DD'`.
3. Para sucursal: usa `UPPER(SUCURSAL) = 'MAYUSCULAS_SIN_ACENTO'`.
4. Para región: usa `UPPER(REGION) = 'MAYUSCULAS_SIN_ACENTO'`.
5. Para ejecutivos: incluye columnas `nombre`, `numero_empleado`, `usuario`.
6. Agrega `LIMIT 1000` por defecto (excepto en listas DISTINCT).
7. Si es ranking: usa `ORDER BY valor DESC LIMIT n`.

SINÓNIMOS DE NEGOCIO:
- "ejecutivo" ↔ nombre / usuario / numero_empleado.
- "colocación", "créditos otorgados" ↔ SUM(monto).
- "trimestre" ↔ fecha_colocacion_trimestre.
- "año" ↔ fecha_colocacion_año.
- "mes" ↔ fecha_colocacion_mes.

EJEMPLOS (sólo para guiar la forma de los querys, NO repetir en la respuesta final):

Pregunta: "¿Qué sucursal tuvo mayor colocación en junio 2025?"
SQL → (se genera internamente, no se repite en la explicación)
Interpretación: En junio 2025, la sucursal con mayor colocación fue X con un total aproximado de Y.

Pregunta: "Colocación por región en Q2 2025"
SQL → (se genera internamente, no se repite en la explicación)
Interpretación: En el segundo trimestre de 2025, la región con más colocación fue X con Z monto total.

---

Pregunta del usuario: {question}
Consulta SQL generada: {query}
Resultado SQL (muestra): {result}

Tarea:
1. NO muestres la consulta SQL en la respuesta (el usuario la verá en su botón).
2. Redacta SOLO la interpretación en lenguaje natural, breve y clara:
   - Explica qué filtros se aplicaron (sucursal, producto, fechas, etc.).
   - Resume el resultado en términos de negocio (conteos, montos, tendencias).
   - Si el valor es 0, dilo explícitamente como "no hubo registros".
   - Si hay un número mayor a 0, indica que hubo actividad moderada/alta.
   - Sé conciso (máx. 2–3 frases).
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
    """
    Retorna: (texto_respuesta, dataframe_resultado, sql_final)
    """
    try:
        if "OPENAI_API_KEY" not in os.environ:
            return "❌ No se configuró la API Key.", None, None

        query_chain, db_sql, prompt, llm = init_chain()
        if not query_chain or not db_sql:
            return "⚠️ No se pudo inicializar el sistema.", None, None

        with st.spinner("🔍 Generando consulta SQL..."):
            consulta_sql_generada = query_chain.invoke({"question": pregunta_usuario})

        # Normalizaciones y saneo del SQL
        consulta_sql = _normalizar_sql_modelo(consulta_sql_generada)

        # Seguridad
        if not es_consulta_segura(consulta_sql):
            return "❌ Consulta bloqueada por seguridad. Solo se permiten operaciones SELECT.", None, consulta_sql

        # LIMIT por defecto, excepto listas de sucursales
        if ("limit" not in (consulta_sql or '').lower()) and (not es_lista_solo_sucursales(consulta_sql)):
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

        # Ejecutar
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(consulta_sql)
        filas = cur.fetchall()
        columnas = [d[0] for d in cur.description]
        conn.close()

        actualizar_ultima_sucursal(consulta_sql)

        # Muestra pequeña para la interpretación LLM
        muestra = [dict(row) for row in filas[:3]]
        muestra_str = str(muestra) + (" ..." if len(filas) > 3 else "")

        with st.spinner("💬 Generando respuesta..."):
            respuesta = llm.invoke(prompt.format_prompt(
                question=pregunta_usuario,
                query=consulta_sql,
                result=muestra_str
            ).to_string())

        df = pd.DataFrame([{k: row[k] for k in columnas} for row in filas])
        return (respuesta.content if hasattr(respuesta, "content") else str(respuesta)), df, consulta_sql

    except Exception as e:
        return f"⚠️ Error: {str(e)}", None, None

# =========================
#   (Opcional) Self-test rápido al importar
# =========================
if __name__ == "__main__":
    # Prueba básica sin Streamlit UI (ejecuta: python c_backend.py)
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    txt, df, sql = consulta("¿A qué sucursal pertenece Cesar Rafael Lopez Villaseñor?")
    print("=== SQL FINAL ===")
    print(sql)
    print("=== RESPUESTA ===")
    print(txt)
    if df is not None:
        print("=== MUESTRA DF ===")
        print(df.head(5).to_string(index=False))
