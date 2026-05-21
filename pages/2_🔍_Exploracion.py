"""Página 2 — Exploración y análisis descriptivo del dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.data_profiler import profile_dataset
from src.visualizer import (
    plot_boxplot,
    plot_correlation_matrix,
    plot_countplot,
    plot_distribution,
    plot_missing_values,
)

st.set_page_config(page_title="Exploración — CustomerIQ", layout="wide")

st.title("Exploración de Datos")

if st.session_state.get("df_raw") is None:
    st.warning("Primero debes cargar un dataset en la página **1 — Cargar Datos**.")
    st.stop()

df: pd.DataFrame = st.session_state.df_raw

# Regenerar perfil si no existe
if st.session_state.get("profile") is None:
    st.session_state.profile = profile_dataset(df)

profile = st.session_state.profile

# ---------------------------------------------------------------------------
# Estadísticas descriptivas
# ---------------------------------------------------------------------------
st.markdown("## Estadísticas descriptivas")

tab_num, tab_cat = st.tabs(["Numéricas", "Categóricas"])

with tab_num:
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        st.info("No hay columnas numéricas.")
    else:
        st.dataframe(num_df.describe().T.round(4), use_container_width=True)

with tab_cat:
    cat_df = df.select_dtypes(exclude="number")
    if cat_df.empty:
        st.info("No hay columnas categóricas.")
    else:
        rows = []
        for col in cat_df.columns:
            vc = df[col].value_counts()
            rows.append({
                "Columna": col,
                "Únicos": df[col].nunique(),
                "Más frecuente": str(vc.index[0]) if len(vc) > 0 else "—",
                "Frecuencia (%)": f"{vc.iloc[0]/len(df)*100:.1f}%" if len(vc) > 0 else "—",
                "Nulos (%)": f"{df[col].isnull().mean()*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Valores nulos
# ---------------------------------------------------------------------------
st.markdown("## Valores nulos")
null_pct = profile["null_pct"]
has_nulls = any(v > 0 for v in null_pct.values())
if has_nulls:
    st.plotly_chart(plot_missing_values(df), use_container_width=True)
else:
    st.success("No hay valores nulos en el dataset.")

# ---------------------------------------------------------------------------
# Duplicados
# ---------------------------------------------------------------------------
dup = profile["duplicates"]
if dup["n_duplicates"] > 0:
    st.warning(
        f"Se encontraron **{dup['n_duplicates']}** filas duplicadas "
        f"({dup['pct_duplicates']}% del total)."
    )
else:
    st.success("No hay filas duplicadas.")

# ---------------------------------------------------------------------------
# Correlación
# ---------------------------------------------------------------------------
st.markdown("## Correlaciones")
num_cols = df.select_dtypes(include="number").columns.tolist()
if len(num_cols) >= 2:
    st.plotly_chart(plot_correlation_matrix(df), use_container_width=True)
else:
    st.info("Se necesitan al menos 2 columnas numéricas para calcular correlaciones.")

# ---------------------------------------------------------------------------
# Distribuciones por columna
# ---------------------------------------------------------------------------
st.markdown("## Distribución por columna")
col_select = st.selectbox("Selecciona una columna", options=df.columns.tolist())

col_type = profile["column_types"].get(col_select, "categorical")
vis_col1, vis_col2 = st.columns(2)

if col_type == "numeric":
    with vis_col1:
        st.plotly_chart(plot_distribution(df, col_select), use_container_width=True)
    with vis_col2:
        st.plotly_chart(plot_boxplot(df, col_select), use_container_width=True)
else:
    st.plotly_chart(plot_countplot(df, col_select), use_container_width=True)

# ---------------------------------------------------------------------------
# Sugerencia de variable objetivo
# ---------------------------------------------------------------------------
st.markdown("## Sugerencia de variable objetivo")
suggestions = profile["suggested_target"]
if suggestions:
    st.info(
        f"Las columnas más adecuadas como variable objetivo son: "
        f"**{', '.join(suggestions[:5])}**\n\n"
        "Son binarias o tienen pocas clases únicas y bajo porcentaje de nulos."
    )
else:
    st.warning(
        "No se detectó automáticamente una variable objetivo clara. "
        "Selecciónala manualmente en la página **4 — Partición**."
    )

# ---------------------------------------------------------------------------
# Tipos detectados
# ---------------------------------------------------------------------------
with st.expander("Tipos de columna detectados", expanded=False):
    types_df = pd.DataFrame(
        [(col, t) for col, t in profile["column_types"].items()],
        columns=["Columna", "Tipo detectado"],
    )
    st.dataframe(types_df, use_container_width=True, hide_index=True)
