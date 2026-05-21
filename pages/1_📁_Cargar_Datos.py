"""Página 1 — Carga de datos: uno o múltiples archivos con combinación opcional."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.data_loader import load_file
from src.data_profiler import profile_dataset

st.set_page_config(page_title="Cargar Datos — CustomerIQ", layout="wide")

st.title("📁 Cargar Datos")
st.markdown(
    "Sube uno o **varios archivos** de datos. Puedes trabajar con uno solo o combinarlos "
    "(apilar filas o unir por columna clave)."
)

# ---------------------------------------------------------------------------
# File uploader — múltiples archivos
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Selecciona uno o más archivos",
    type=["csv", "xlsx", "xls", "json", "tsv"],
    accept_multiple_files=True,
    help="Formatos soportados: CSV, Excel (.xlsx/.xls), JSON y TSV. Puedes subir varios a la vez.",
)

if not uploaded_files:
    if st.session_state.get("df_raw") is not None:
        st.info(
            f"Dataset activo: **{st.session_state.file_name}** — "
            f"{st.session_state.df_raw.shape[0]:,} filas × {st.session_state.df_raw.shape[1]} columnas. "
            "Sube archivos nuevos para reemplazarlo."
        )
        st.dataframe(st.session_state.df_raw.head(20), use_container_width=True)
    else:
        st.warning("Sube al menos un archivo para comenzar el análisis.")
    st.stop()

# ---------------------------------------------------------------------------
# Cargar todos los archivos
# ---------------------------------------------------------------------------
loaded: dict[str, pd.DataFrame] = {}
errors: list[str] = []

for f in uploaded_files:
    try:
        df_i, _ = load_file(f)
        loaded[f.name] = df_i
    except ValueError as e:
        errors.append(f"**{f.name}**: {e}")

if errors:
    for err in errors:
        st.error(err)

if not loaded:
    st.stop()

st.success(f"{len(loaded)} archivo(s) cargado(s) correctamente.")

# ---------------------------------------------------------------------------
# Modo de combinación (solo aparece si hay más de un archivo)
# ---------------------------------------------------------------------------
if len(loaded) == 1:
    file_name, df_final = next(iter(loaded.items()))
    mode = "único"
else:
    st.markdown("### ¿Cómo quieres combinar los archivos?")
    mode = st.radio(
        "Modo de combinación",
        options=["Usar solo uno", "Apilar filas (concat)", "Unir por columna clave (merge)"],
        horizontal=True,
        help=(
            "**Usar solo uno**: selecciona cuál dataset usar.\n"
            "**Apilar filas**: une todos los archivos verticalmente (deben tener las mismas columnas).\n"
            "**Unir por columna clave**: hace un JOIN entre dos archivos por una columna común."
        ),
    )

    # ---- Modo: elegir uno ----
    if mode == "Usar solo uno":
        file_name = st.selectbox("Selecciona el archivo a usar", list(loaded.keys()))
        df_final = loaded[file_name]

    # ---- Modo: concat ----
    elif mode == "Apilar filas (concat)":
        files_to_stack = st.multiselect(
            "Archivos a apilar (en ese orden)",
            options=list(loaded.keys()),
            default=list(loaded.keys()),
        )
        if not files_to_stack:
            st.error("Selecciona al menos un archivo.")
            st.stop()

        # Advertir sobre columnas distintas
        col_sets = [set(loaded[f].columns) for f in files_to_stack]
        common_cols = set.intersection(*col_sets)
        all_cols = set.union(*col_sets)
        if common_cols != all_cols:
            st.warning(
                f"Los archivos no tienen exactamente las mismas columnas. "
                f"Columnas en común: **{len(common_cols)}** | "
                f"Total únicas: **{len(all_cols)}**. "
                "Las columnas faltantes se completarán con NaN."
            )

        df_final = pd.concat(
            [loaded[f] for f in files_to_stack],
            ignore_index=True,
            sort=False,
        )
        file_name = " + ".join(files_to_stack)

    # ---- Modo: merge ----
    elif mode == "Unir por columna clave (merge)":
        col_left, col_right = st.columns(2)
        with col_left:
            left_file = st.selectbox("Archivo izquierdo (base)", list(loaded.keys()), key="merge_left")
        with col_right:
            right_options = [f for f in loaded.keys() if f != left_file]
            if not right_options:
                st.error("Necesitas al menos dos archivos para hacer merge.")
                st.stop()
            right_file = st.selectbox("Archivo derecho", right_options, key="merge_right")

        df_left = loaded[left_file]
        df_right = loaded[right_file]

        common_keys = list(set(df_left.columns) & set(df_right.columns))
        if not common_keys:
            st.error("Los archivos no comparten ninguna columna para hacer el join.")
            st.stop()

        col_k, col_how = st.columns(2)
        with col_k:
            join_key = st.selectbox("Columna clave (key)", common_keys)
        with col_how:
            join_how = st.selectbox(
                "Tipo de join",
                ["inner", "left", "right", "outer"],
                help="inner: solo filas que coinciden. left: todas del izquierdo. outer: todas.",
            )

        df_final = pd.merge(df_left, df_right, on=join_key, how=join_how, suffixes=("_L", "_R"))
        file_name = f"{left_file} ⋈ {right_file}"

# ---------------------------------------------------------------------------
# Vista previa y metadata del dataset final
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"### Dataset resultante: `{file_name}`")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Filas", f"{df_final.shape[0]:,}")
m2.metric("Columnas", df_final.shape[1])
m3.metric("Nulos totales", int(df_final.isnull().sum().sum()))
m4.metric("Duplicados", int(df_final.duplicated().sum()))

# Preview individual de cada archivo subido
if len(loaded) > 1:
    with st.expander("Vista previa por archivo", expanded=False):
        for fname, dfi in loaded.items():
            st.markdown(f"**{fname}** — {dfi.shape[0]:,} filas × {dfi.shape[1]} cols")
            st.dataframe(dfi.head(5), use_container_width=True)

n_preview = st.slider("Filas a mostrar", 5, 100, 20)
st.dataframe(df_final.head(n_preview), use_container_width=True)

# Tipos de datos
with st.expander("Tipos de datos por columna", expanded=False):
    dtypes_df = pd.DataFrame(
        df_final.dtypes.astype(str).reset_index().values,
        columns=["Columna", "Tipo"],
    )
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Confirmar y guardar en session_state
# ---------------------------------------------------------------------------
st.markdown("---")
if st.button("Usar este dataset para el análisis", type="primary", use_container_width=True):
    st.session_state.df_raw = df_final
    st.session_state.file_name = file_name
    # Resetear todo el pipeline al cambiar el dataset
    st.session_state.df_clean = None
    st.session_state.X_train = None
    st.session_state.X_val = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_val = None
    st.session_state.y_test = None
    st.session_state.models_trained = {}
    st.session_state.experiments_log = []
    st.session_state.clustering_results = {}
    st.session_state.profile = profile_dataset(df_final)

    st.success(
        f"Dataset **{file_name}** cargado: "
        f"{df_final.shape[0]:,} filas × {df_final.shape[1]} columnas."
    )
    st.info("Continúa en la página **2 — Exploración**.")
