"""Página principal de CustomerIQ — inicializa session_state global."""

import streamlit as st

st.set_page_config(
    page_title="CustomerIQ — Análisis de Clientes con ML",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inicializar session_state global
# ---------------------------------------------------------------------------
_defaults = {
    "df_raw": None,
    "df_clean": None,
    "X_train": None,
    "X_val": None,
    "X_test": None,
    "y_train": None,
    "y_val": None,
    "y_test": None,
    "feature_cols": [],
    "target_col": None,
    "scaler": None,
    "encoding_info": {},
    "models_trained": {},      # {nombre: modelo}
    "experiments_log": [],     # lista de dicts
    "clustering_results": {},  # {algorithm: {labels, score, ...}}
    "profile": None,
    "file_name": None,
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Contenido de la página principal
# ---------------------------------------------------------------------------

st.title(" CustomerIQ")
st.subheader("Sistema Interactivo de Análisis de Clientes con Machine Learning")

st.markdown("""
Bienvenido a **CustomerIQ**, una plataforma de análisis de datos de clientes que combina
técnicas de **segmentación no supervisada** y **clasificación supervisada** para extraer
insights accionables de cualquier dataset tabular.

---
""")

# Indicadores de progreso del flujo
col1, col2, col3, col4 = st.columns(4)
col5, col6, col7, col8 = st.columns(4)

def _status(condition: bool) -> str:
    return "✅" if condition else "⬜"

with col1:
    st.metric(
        label=f"{_status(st.session_state.df_raw is not None)} 1. Cargar Datos",
        value="Listo" if st.session_state.df_raw is not None else "Pendiente",
    )
with col2:
    st.metric(
        label=f"{_status(st.session_state.profile is not None)} 2. Exploración",
        value="Listo" if st.session_state.profile is not None else "Pendiente",
    )
with col3:
    st.metric(
        label=f"{_status(st.session_state.df_clean is not None)} 3. Limpieza",
        value="Listo" if st.session_state.df_clean is not None else "Pendiente",
    )
with col4:
    st.metric(
        label=f"{_status(st.session_state.X_train is not None)} 4. Partición",
        value="Listo" if st.session_state.X_train is not None else "Pendiente",
    )
with col5:
    has_clustering = bool(st.session_state.clustering_results)
    st.metric(
        label=f"{_status(has_clustering)} 5. Segmentación",
        value="Listo" if has_clustering else "Pendiente",
    )
with col6:
    has_models = bool(st.session_state.models_trained)
    st.metric(
        label=f"{_status(has_models)} 6. Clasificación",
        value="Listo" if has_models else "Pendiente",
    )
with col7:
    has_exp = bool(st.session_state.experiments_log)
    st.metric(
        label=f"{_status(has_exp)} 7. Comparación",
        value="Listo" if has_exp else "Pendiente",
    )
with col8:
    st.metric(
        label="📑 8. Reporte",
        value="Disponible" if has_models else "Pendiente",
    )

st.markdown("---")

# Guía de uso
st.markdown("""
### Flujo recomendado

Usa la barra lateral para navegar entre páginas siguiendo este orden:

| Paso | Página | Qué hacer |
|------|--------|-----------|
| 1 | **Cargar Datos** | Sube tu archivo CSV, Excel, JSON o TSV |
| 2 | **Exploración** | Analiza distribuciones, correlaciones y valores nulos |
| 3 | **Limpieza** | Imputa valores faltantes y trata outliers |
| 4 | **Partición** | Divide en train/val/test, aplica encoding y escalado |
| 5 | **Segmentación** | Descubre grupos de clientes con K-Means y clustering jerárquico |
| 6 | **Clasificación** | Entrena Árbol de Decisión y Random Forest |
| 7 | **Comparación** | Compara modelos con ROC, matrices de confusión e interpretación gerencial |
| 8 | **Reporte** | Descarga el informe ejecutivo y los modelos entrenados |

### Requisitos del dataset
- Formato: CSV, XLSX, JSON o TSV
- Al menos una columna numérica y una columna objetivo (categórica o binaria)
- Sin requisitos sobre nombres de columnas — el sistema se adapta automáticamente
""")

if st.session_state.df_raw is not None:
    st.success(
        f"Dataset activo: **{st.session_state.file_name}** — "
        f"{st.session_state.df_raw.shape[0]:,} filas × {st.session_state.df_raw.shape[1]} columnas"
    )
else:
    st.info("Comienza cargando tu dataset en la página **1 — Cargar Datos**.")
