"""Página 3 — Limpieza de datos: imputación, duplicados y outliers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.cleaner import handle_missing, handle_outliers, outlier_summary, remove_duplicates
from src.data_profiler import profile_dataset


def _try_numeric(val: str):
    """Intenta convertir string a número, si falla retorna el string."""
    try:
        return float(val)
    except ValueError:
        return val

st.set_page_config(page_title="Limpieza — CustomerIQ", layout="wide")

st.title("Limpieza de Datos")

if st.session_state.get("df_raw") is None:
    st.warning("Primero debes cargar un dataset en la página **1 — Cargar Datos**.")
    st.stop()

df_raw: pd.DataFrame = st.session_state.df_raw
df_work = st.session_state.df_clean if st.session_state.df_clean is not None else df_raw.copy()

# ---------------------------------------------------------------------------
# 1. Valores nulos
# ---------------------------------------------------------------------------
st.markdown("## 1. Imputación de valores nulos")

null_pct = df_work.isnull().mean() * 100
cols_with_nulls = null_pct[null_pct > 0].index.tolist()

if not cols_with_nulls:
    st.success("No hay valores nulos en el dataset actual.")
    strategy_dict: dict = {}
else:
    st.markdown(
        f"Se detectaron **{len(cols_with_nulls)}** columnas con valores nulos. "
        "Configura la estrategia de imputación para cada una:"
    )

    strategy_dict = {}
    numeric_cols = df_work.select_dtypes(include="number").columns.tolist()

    for col in cols_with_nulls:
        pct = null_pct[col]
        is_numeric = col in numeric_cols
        options = (
            ["media", "mediana", "moda", "eliminar fila", "valor constante"]
            if is_numeric
            else ["moda", "eliminar fila", "valor constante"]
        )
        choice = st.selectbox(
            f"**{col}** ({pct:.1f}% nulos)",
            options=options,
            key=f"impute_{col}",
            help=f"{'Columna numérica' if is_numeric else 'Columna categórica'} — {pct:.1f}% de valores faltantes",
        )

        if choice == "valor constante":
            const_val = st.text_input(
                f"Valor constante para {col}",
                value="0" if is_numeric else "desconocido",
                key=f"const_{col}",
            )
            strategy_dict[col] = const_val if not is_numeric else _try_numeric(const_val)
        else:
            map_choice = {
                "media": "mean", "mediana": "median", "moda": "mode",
                "eliminar fila": "drop", "valor constante": "drop",
            }
            strategy_dict[col] = map_choice[choice]

# ---------------------------------------------------------------------------
# 2. Duplicados
# ---------------------------------------------------------------------------
st.markdown("## 2. Filas duplicadas")
n_dup = int(df_work.duplicated().sum())
if n_dup > 0:
    st.warning(f"Hay **{n_dup}** filas duplicadas ({n_dup/len(df_work)*100:.1f}%).")
    remove_dups = st.checkbox("Eliminar duplicados", value=True)
else:
    st.success("No hay filas duplicadas.")
    remove_dups = False

# ---------------------------------------------------------------------------
# 3. Outliers
# ---------------------------------------------------------------------------
st.markdown("## 3. Valores atípicos (Outliers)")

num_cols_all = df_work.select_dtypes(include="number").columns.tolist()

if not num_cols_all:
    st.info("No hay columnas numéricas para analizar outliers.")
    outlier_method = "iqr"
    outlier_action = "keep"
else:
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        outlier_method = st.selectbox(
            "Método de detección",
            ["iqr", "zscore"],
            help="IQR: basado en cuartiles. Z-Score: basado en desviaciones estándar.",
        )
    with col_out2:
        outlier_action = st.selectbox(
            "Acción",
            ["keep", "remove", "cap"],
            format_func=lambda x: {"keep": "Conservar", "remove": "Eliminar filas", "cap": "Limitar (winsorize)"}[x],
        )

    summary = outlier_summary(df_work, method=outlier_method)
    total_outliers = summary["n_outliers"].sum()
    st.dataframe(summary, use_container_width=True, hide_index=True)
    if total_outliers > 0:
        st.warning(f"Total de filas con al menos un outlier detectado por columna: {int(total_outliers)}")

# ---------------------------------------------------------------------------
# Aplicar limpieza
# ---------------------------------------------------------------------------
st.markdown("---")
if st.button("Aplicar limpieza", type="primary", use_container_width=True):
    df_clean = df_work.copy()

    # Imputación
    if strategy_dict:
        df_clean = handle_missing(df_clean, strategy_dict)

    # Duplicados
    if remove_dups:
        df_clean = remove_duplicates(df_clean)

    # Outliers
    if num_cols_all and outlier_action != "keep":
        df_clean = handle_outliers(df_clean, method=outlier_method, action=outlier_action)

    st.session_state.df_clean = df_clean
    # Resetear pasos posteriores
    st.session_state.X_train = None
    st.session_state.models_trained = {}
    st.session_state.experiments_log = []

    st.success("Limpieza aplicada correctamente.")

    # Antes / después
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Antes**")
        st.metric("Filas", f"{df_raw.shape[0]:,}")
        st.metric("Nulos totales", int(df_raw.isnull().sum().sum()))
    with col_b:
        st.markdown("**Después**")
        st.metric("Filas", f"{df_clean.shape[0]:,}")
        st.metric("Nulos totales", int(df_clean.isnull().sum().sum()))

    st.dataframe(df_clean.head(20), use_container_width=True)
    st.info("Continúa en la página **4 — Partición** para dividir el dataset.")

elif st.session_state.df_clean is not None:
    st.info("Limpieza ya aplicada. Modifica las opciones y vuelve a hacer clic en 'Aplicar limpieza' para recalcular.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Filas originales", f"{df_raw.shape[0]:,}")
    with col_b:
        st.metric("Filas tras limpieza", f"{st.session_state.df_clean.shape[0]:,}")
