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
from typing import Optional

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate

# =========================
#   Rutas cross-platform (raíz del proyecto)
# =========================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_LOCAL_PATH = os.path.join(PROJECT_DIR, "colocacion.db")

# Puedes agregar candidatos extra si lo necesitas (ej. cuando corres en otro entorno)
DB_CANDIDATES = [DB_LOCAL_PATH, "/mnt/data/colocacion.db"]

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

def _strip_string_literals(s: str) -> str:
    """
    Reemplaza literales '...' y "..." por marcadores para evitar falsos positivos al buscar keywords peligrosas.
    """
    # '...' con '' escapadas
    s = re.sub(r"'.*?(?:''.*?)*'", "''", s, flags=re.DOTALL)
    # "..." con "" escapadas
    s = re.sub(r'".*?(?:"".*?)*"', '""', s, flags=re.DOTALL)
    return s

def es_consulta_segura(sql: str) -> bool:
    # Quitar comentarios y fences
    s = re.sub(r'--.*?(\n|$)', '', sql or '', flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.IGNORECASE | re.DOTALL)
    s = limpiar_fences_y_espacios(s)
    s_low = s.lower().lstrip()

    # Solo permitimos SELECT o WITH
    if not (s_low.startswith("select") or s_low.startswith("with")):
        return False

    # Escanear sin literales
    s_scan = _strip_string_literals(s_low)

    # Palabras peligrosas (DML/DDL/PRAGMA/etc.)
    peligrosas = [
        r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b", r"\balter\b",
        r"\bcreate\b", r"\btruncate\b", r"\battach\b", r"\bdetach\b",
        r"\bpragma\b", r"\bexec\b", r"\bexecute\b", r"\bvacuum\b",
        r"\breplace\s+into\b"
    ]
    for pat in peligrosas:
        if re.search(pat, s_scan):
            return False
    return True

def quitar_acentos(texto: str) -> str:
    """
    Normaliza a NFD y elimina marcas diacríticas; retorna en MAYÚSCULAS.
    """
    base = ''.join(
        c for c in unicodedata.normalize('NFD', texto or '')
        if unicodedata.category(c) != 'Mn'
    )
    return base.upper()

def expr_unaccent_col(colname: str) -> str:
    """
    Construye una expresión SQL con UPPER + REPLACE encadenados
    para normalizar ÁÉÍÓÚÜ y Ñ.
    Acepta identificadores entre comillas (p.ej. "sucursal").
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

# ===== PARCHE CLAVE: soportar identificadores entre comillas y normalizar literal =====
def _normalize_eq_like_in(col: str, sql: str) -> str:
    """
    Normaliza comparaciones con =, LIKE e IN para columnas de texto (alias opcional),
    aceptando identificadores entre comillas dobles, p. ej.: "sucursal", a."sucursal".
    """
    def norm_expr(lhs_before_op: str) -> str:
        # lhs_before_op incluye el LHS con espacios, ej: ' "sucursal"  =' o ' a."sucursal"   like'
        # nos quedamos solo con la referencia a columna (con o sin alias), p.ej.: a."sucursal"
        col_ref = lhs_before_op.strip()
        # Quitar el operador si viene pegado (por seguridad)
        col_ref = re.sub(r'(=|like|in)\s*$', '', col_ref, flags=re.IGNORECASE).strip()
        return expr_unaccent_col(col_ref)

    # alias opcional + nombre de columna con o sin comillas
    # ejemplos válidos: sucursal | "sucursal" | a.sucursal | a."sucursal"
    col_rx = rf'(?:["]?[A-Za-z_]\w*["]?\s*\.\s*)?["]?{col}["]?'

    # 1) = 'texto'
    sql = re.sub(
        rf'({col_rx}\s*=\s*)\'([^\']+)\'',
        lambda m: f"{norm_expr(m.group(1))} = '{quitar_acentos(m.group(2))}'",
        sql,
        flags=re.IGNORECASE
    )

    # 2) LIKE '%texto%' (con o sin % en los extremos)
    sql = re.sub(
        rf'({col_rx}\s+like\s*)\'%?([^%\']+)%?\'',
        lambda m: f"{norm_expr(m.group(1))} LIKE '%{quitar_acentos(m.group(2))}%'",
        sql,
        flags=re.IGNORECASE
    )

    # 3) IN ('a','b',...)
    def _repl_in(m):
        lhs = norm_expr(m.group(1))
        raw = m.group(2)
        items = [x.strip().strip("'\"") for x in raw.split(',')]
        items = [f"'{quitar_acentos(x)}'" for x in items if x]
        return f"{lhs} IN ({', '.join(items)})"

    sql = re.sub(
        rf'({col_rx}\s+in\s*)\(([^)]+)\)',
        _repl_in,
        sql,
        flags=re.IGNORECASE
    )

    return sql

def corregir_sql_sucursal(sql: str) -> str:
    return _normalize_eq_like_in('SUCURSAL', sql)

def corregir_sql_nombre(sql: str) -> str:
    return _normalize_eq_like_in('NOMBRE', sql)

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

# ===== PARCHE FECHAS: BETWEEN 'YYYY-MM-01' AND 'YYYY-MM-31' -> rango mensual seguro =====
def expandir_fecha_between_mes(sql: str) -> str:
    """
    Transforma BETWEEN 'YYYY-MM-dd' AND 'YYYY-MM-dd' en rango mensual seguro:
    (fecha >= 'YYYY-MM-01' AND fecha < 'YYYY-MM-01','+1 month')
    Solo si ambas fechas son del mismo año/mes.
    """
    pat = r"(fecha_colocacion)\s+between\s*'(\d{4})-(\d{2})-(\d{2})'\s*and\s*'(\d{4})-(\d{2})-(\d{2})'"
    def repl(m):
        col, y1, mm1, d1, y2, mm2, d2 = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)
        if y1 == y2 and mm1 == mm2:
            ini = f"{y1}-{mm1}-01"
            return f"({col} >= date('{ini}') AND {col} < date('{ini}','+1 month'))"
        return m.group(0)  # distinto mes: no tocamos

    return re.sub(pat, repl, sql, flags=re.IGNORECASE)

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
        return s.strip()
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

# ---- NUEVOS HELPERS PARA MANEJO INTELIGENTE DE LIMIT ----

def quitar_limits_global(sql: str) -> str:
    """
    Elimina cualquier cláusula LIMIT/OFFSET/FETCH del SQL generado por el modelo.
    """
    if not sql:
        return sql

    s = sql
    s = re.sub(r'(?is)\blimit\s+\d+(?:\s+offset\s+\d+)?\s*;?', ' ', s)  # LIMIT [n] [OFFSET m]
    s = re.sub(r'(?is)\bfetch\s+first\s+\d+\s+rows\s+only\s*;?', ' ', s)  # FETCH FIRST n ROWS ONLY
    s = re.sub(r'(?is)\boffset\s+\d+\s+rows\s+fetch\s+next\s+\d+\s+rows\s+only\s*;?', ' ', s)  # OFFSET ... FETCH ...
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# --- mapea numeros en palabras -> digitos (básico)
_NUM_WORDS = {
    "un":1, "uno":1, "una":1,
    "dos":2, "tres":3, "cuatro":4, "cinco":5, "seis":6, "siete":7, "ocho":8, "nueve":9, "diez":10,
    "once":11, "doce":12, "trece":13, "catorce":14, "quince":15, "veinte":20
}

def _palabras_a_digitos(t: str) -> str:
    return re.sub(
        r'\b(un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|veinte)\b',
        lambda m: str(_NUM_WORDS[m.group(1)]), t, flags=re.IGNORECASE
    )

def extraer_limit_de_pregunta(texto: str) -> Optional[int]:
    """
    Detecta si el usuario pidió explícitamente un límite (top/primeros/los N ...).
    Soporta: dígitos, números en palabras, 'top5', 'top-5', 'los/las 5 sucursales', 'primeros 5', etc.
    Evita confundir años (2024/2025).
    """
    if not texto:
        return None

    t = texto.lower()

    # 0) quitar años (para no confundir 2025 con un límite)
    t = re.sub(r'\b(?:en|del|de|para)?\s*20\d{2}\b', ' ', t)

    # 1) normalizar palabras-numero -> dígitos
    t = _palabras_a_digitos(t)

    # 2) normalizar variantes de "top5" / "top-5" -> "top 5"
    t = re.sub(r'\btop\s*-\s*(\d+)\b', r'top \1', t)
    t = re.sub(r'\btop(\d+)\b', r'top \1', t)

    # 3) patrones
    patrones = [
        r'\btop\s+(\d+)\b',
        r'\bprimer(?:os|as)?\s+(\d+)\b',
        r'\b(?:los|las)\s+(\d+)\s+(?:sucursales|productos|regiones|ejecutivos|filas|registros|resultados|clientes)\b',
        r'\b(?:muestrame|muéstrame|mostrar|lista|listame)\s+(\d+)\b',
        r'\blos\s+(\d+)\s+m[aá]s\b',
    ]
    for pat in patrones:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1))
                if n > 0:
                    return n
            except Exception:
                pass
    return None

def _pide_todo(texto: str) -> bool:
    """
    True si el usuario pide explicitamente todo: 'todas', 'todo', 'completa', 'sin límite'.
    """
    if not texto:
        return False
    t = texto.lower()
    return bool(re.search(r'\b(todas?|todo|completa|sin\s+l[ií]mite)\b', t))

def agregar_limit_si_no_existe(sql: str, n: int) -> str:
    """
    Agrega 'LIMIT n' al final del statement si no existe ya.
    Asume que ya corriste quitar_limits_global() antes para evitar duplicados.
    """
    if not sql:
        return sql
    if re.search(r'\blimit\s+\d+\b', sql, flags=re.IGNORECASE):
        return sql
    return sql.rstrip(';') + f' LIMIT {n}'

# =========================
#   Normalizador maestro
# =========================
def _normalizar_sql_modelo(consulta_sql: str) -> str:
    """
    Aplica normalizaciones/saneos en orden seguro.
    - Fuerza vista *_detalle
    - Corrige sucursal/nombre con UPPER+sin acentos (soporta identificadores con comillas)
    - Expande fecha = 'YYYY-MM-DD' a rango mensual
    - Expande BETWEEN 'YYYY-MM-01' AND 'YYYY-MM-31' a rango mensual (si mismo mes)
    - Elimina fences/comentarios
    - Deja un solo statement
    - Quita cualquier LIMIT/OFFSET/FETCH que haya metido el modelo (solo se reinsertará si usuario lo pidió)
    """
    s = limpiar_fences_y_espacios(consulta_sql)
    s = forzar_vista_detalle(s)
    s = corregir_sql_sucursal(s)
    s = corregir_sql_nombre(s)
    s = expandir_fecha_igual_a_dia(s)
    s = expandir_fecha_between_mes(s)  # <<<<<< PARCHE
    s = primer_select_o_with(s)
    s = dejar_solo_un_statement(s)
    s = quitar_limits_global(s)  # 🔥 NO LIMIT a menos que el usuario lo pida
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
- nombre, usuario, numero_empleado
- sucursal, region, producto, monto
- fecha_colocacion, fecha_colocacion_año, fecha_colocacion_mes, fecha_colocacion_trimestre

REGLAS OBLIGATORIAS PARA LA CONSULTA:
1. Siempre usa la vista `vw_fact_colocacion_bruta_detalle`.
2. Para fechas:
   - Usa `fecha_colocacion_año` y `fecha_colocacion_mes` para meses.
   - Usa `fecha_colocacion_año` y `fecha_colocacion_trimestre` para trimestres.
   - Nunca uses directamente `fecha_colocacion = 'YYYY-MM-DD'`.
3. Para sucursal/región, normaliza a MAYÚSCULAS SIN ACENTO (se corrige en backend).
4. Para ranking por grupo (por región/sucursal/ejecutivo) prefiere `ROW_NUMBER() OVER (PARTITION BY ...)`.
5. Evita usar LIMIT, salvo que el usuario lo pida explícitamente (top/primeros/limit N).

SINÓNIMOS DE NEGOCIO:
- "ejecutivo" ↔ nombre / usuario / numero_empleado
- "colocación", "créditos otorgados" ↔ SUM(monto)
- "trimestre" ↔ fecha_colocacion_trimestre
- "año" ↔ fecha_colocacion_año
- "mes" ↔ fecha_colocacion_mes

Pregunta del usuario: {question}
Consulta SQL generada: {query}
Resultado SQL (muestra): {result}

Tarea:
1. NO muestres la consulta SQL en la respuesta.
2. Redacta SOLO la interpretación en lenguaje natural, breve y clara (2–3 frases).
3. Explica filtros aplicados y resume hallazgos (conteos, montos, tendencias).
4. Si no hay registros, dilo explícitamente.
"""
)
        return query_chain, db, answer_prompt, llm
    except Exception as e:
        st.error(f"Error al inicializar la cadena: {str(e)}")
        return None, None, None, None

# =========================
#   FLUJO PRINCIPAL
# =========================
# c_backend.py
# ... (todo igual hasta la función consulta)

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

        # 0) Detectar si el usuario pidió límite explícito (tras rewriter)
        user_limit = extraer_limit_de_pregunta(pregunta_usuario)

        # Si explícitamente pide "todas/todo/completa/sin límite", no imponemos LIMIT
        if _pide_todo(pregunta_usuario):
            user_limit = None

        # 1) Generar SQL desde LangChain
        with st.spinner("🔍 Generando consulta SQL..."):
            consulta_sql_generada = query_chain.invoke({"question": pregunta_usuario})

        # DEBUG consola
        # print("🔍 SQL ORIGINAL DEL MODELO:\n", consulta_sql_generada)

        # 2) Normalizar y QUITAR cualquier LIMIT/OFFSET/FETCH del modelo
        consulta_sql = _normalizar_sql_modelo(consulta_sql_generada)

        # 3) Si el usuario pidió límite explícito, lo agregamos nosotros (y solo nosotros)
        if user_limit is not None:
            consulta_sql = agregar_limit_si_no_existe(consulta_sql, user_limit)

        # DEBUG consola
        # print("✅ SQL FINAL NORMALIZADO:\n", consulta_sql)

        # # (Opcional) Mostrar en UI para depuración
        # with st.expander("🔎 SQL generado por el modelo (antes de correcciones)"):
        #     st.code(consulta_sql_generada, language="sql")
        # with st.expander("✅ SQL final corregido y ejecutado"):
        #     st.code(consulta_sql, language="sql")
        # with st.expander("🧪 DEBUG: detección de límite del usuario"):
        #     st.write(f"LIMIT solicitado: {user_limit!r}")

        # 4) Seguridad sobre el SQL final
        if not es_consulta_segura(consulta_sql):
            return "❌ Consulta bloqueada por seguridad. Solo se permiten operaciones SELECT.", None, consulta_sql

        # 5) Resolver ruta de BD (candidato o descarga on-demand)
        path = None
        for p in DB_CANDIDATES:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                path = p
                break
        if not path:
            path = download_database()

        if not path or not os.path.exists(path):
            return "⚠️ No se encontró/descargó la base de datos colocacion.db.", None, consulta_sql

        # 6) Ejecutar
        t0 = time.time()
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(consulta_sql)
        filas = cur.fetchall()
        columnas = [d[0] for d in cur.description]
        conn.close()
        ms = int((time.time() - t0) * 1000)

        actualizar_ultima_sucursal(consulta_sql)

        # 7) Muestra pequeña para interpretación LLM
        muestra = [dict(row) for row in filas[:3]]
        muestra_str = str(muestra) + (" ..." if len(filas) > 3 else "")

        with st.spinner("💬 Generando respuesta..."):
            respuesta = llm.invoke(prompt.format_prompt(
                question=pregunta_usuario,
                query=consulta_sql,
                result=muestra_str
            ).to_string())

        df = pd.DataFrame(filas, columns=columnas)

        # with st.expander("⏱️ Métrica ejecución"):
        #     st.write(f"Filas: {len(df)} | Tiempo: {ms} ms")

        return (respuesta.content if hasattr(respuesta, "content") else str(respuesta)), df, consulta_sql

    except Exception as e:
        return f"⚠️ Error: {str(e)}", None, None


# =========================
#   (Opcional) Self-test rápido al importar
# =========================
if __name__ == "__main__":
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    txt, df, sql = consulta("¿Cuáles fueron las 5 sucursales con mayor monto colocado en 2025?")
    print("=== SQL FINAL ===")
    print(sql)
    print("=== RESPUESTA ===")
    print(txt)
    if df is not None:
        print("=== MUESTRA DF ===")
        print(df.head(5).to_string(index=False))
