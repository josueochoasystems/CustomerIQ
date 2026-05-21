"""Página 4 — Partición train/val/test, encoding, escalado y baseline.

Maneja automáticamente clases raras en la variable objetivo para evitar
errores de stratify en train_test_split. Soporta datasets combinados.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from src.classification import train_baseline
from src.evaluator import evaluate_classifier, log_experiment
from src.preprocessor import (
    encode_categoricals,
    prepare_features_target,
    scale_features,
    split_data,
)

st.set_page_config(page_title="Partición — CustomerIQ", layout="wide")

st.title("✂️ Partición de Datos")

# ---------------------------------------------------------------------------
# Guardia de estado
# ---------------------------------------------------------------------------
if st.session_state.get("df_raw") is None:
    st.warning("Primero debes cargar un dataset en la página **1 — Cargar Datos**.")
    st.stop()

df_source: pd.DataFrame = (
    st.session_state.df_clean
    if st.session_state.df_clean is not None
    else st.session_state.df_raw
)

if st.session_state.df_clean is None:
    st.warning(
        "No se ha aplicado limpieza. Se usará el dataset original. "
        "Se recomienda pasar primero por la página **3 — Limpieza**."
    )


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def get_class_diagnosis(y: pd.Series, min_count: int) -> dict:
    """Diagnostica la distribución del target y cuántas clases sobrevivirían el agrupamiento.

    Args:
        y: Serie con la variable objetivo.
        min_count: Umbral mínimo de registros por clase.

    Returns:
        Dict con freq_df, n_total_classes, n_rare, n_surviving, pct_rare_rows.
    """
    freq = y.value_counts().reset_index()
    freq.columns = ["Clase", "Frecuencia"]
    freq["% del total"] = (freq["Frecuencia"] / len(y) * 100).round(2)
    freq["¿Rara?"] = freq["Frecuencia"] < min_count

    n_total = freq["Clase"].nunique()
    n_rare = int(freq["¿Rara?"].sum())
    n_surviving = n_total - n_rare
    # Si se agrupan todas las raras en "Otros", cuenta como +1 clase solo si hay raras
    effective_classes = n_surviving + (1 if n_rare > 0 else 0)
    pct_rare_rows = float(freq.loc[freq["¿Rara?"], "Frecuencia"].sum() / len(y) * 100)

    return {
        "freq_df": freq,
        "n_total_classes": n_total,
        "n_rare": n_rare,
        "n_surviving": n_surviving,
        "effective_classes": effective_classes,
        "pct_rare_rows": pct_rare_rows,
    }


def fix_rare_classes(
    y: pd.Series, min_count: int = 2, enabled: bool = True
) -> tuple[pd.Series, list]:
    """Agrupa clases con pocos registros bajo una etiqueta genérica.

    Args:
        y: Serie con la variable objetivo.
        min_count: Mínimo de registros para conservar una clase.
        enabled: Si False, retorna y sin cambios.

    Returns:
        Tupla (y_corregida, lista_de_clases_raras).
    """
    if not enabled:
        return y.copy(), []

    freq = y.value_counts()
    rare = freq[freq < min_count].index.tolist()
    if not rare:
        return y.copy(), []

    fill_value = -999 if pd.api.types.is_numeric_dtype(y) else "Otros"
    y_fixed = y.copy()
    y_fixed[y_fixed.isin(rare)] = fill_value
    return y_fixed, rare


def validate_classes(y: pd.Series) -> bool:
    """Verifica que la serie tenga al menos 2 clases únicas.

    Args:
        y: Variable objetivo ya procesada.

    Returns:
        True si hay al menos 2 clases únicas.
    """
    return y.nunique() >= 2


# ---------------------------------------------------------------------------
# Selección de variables
# ---------------------------------------------------------------------------
st.markdown("## Configuración de variables")

all_cols = df_source.columns.tolist()
profile = st.session_state.get("profile")
suggestions = profile["suggested_target"] if profile else []

target_col = st.selectbox(
    "Variable objetivo (y)",
    options=all_cols,
    index=all_cols.index(suggestions[0]) if suggestions and suggestions[0] in all_cols else 0,
    help="Columna que el modelo aprenderá a predecir.",
)

feature_options = [c for c in all_cols if c != target_col]
feature_cols = st.multiselect(
    "Features (X)",
    options=feature_options,
    default=feature_options,
    help="Columnas a usar como variables predictoras.",
)

if not feature_cols:
    st.error("Debes seleccionar al menos una feature.")
    st.stop()

# ---------------------------------------------------------------------------
# Diagnóstico en tiempo real del target seleccionado
# ---------------------------------------------------------------------------
st.markdown("## Diagnóstico de la variable objetivo")

# Opciones avanzadas de agrupamiento
with st.expander("⚙️ Opciones avanzadas de clases raras", expanded=False):
    group_rare = st.checkbox(
        "Agrupar clases con pocos registros automáticamente",
        value=True,
        help=(
            "Agrupa clases con menos de N registros para evitar errores en el split. "
            "Desactiva si quieres conservar todas las clases tal cual."
        ),
    )
    min_count_thresh = st.slider(
        "Mínimo de registros por clase (N)",
        min_value=1,
        max_value=10,
        value=2,
        help="Clases con menos de N registros se agruparán como 'Otros'.",
        disabled=not group_rare,
    )

# Calcular diagnóstico con los parámetros actuales
diag = get_class_diagnosis(df_source[target_col], min_count=min_count_thresh if group_rare else 1)

# Métricas de diagnóstico
d1, d2, d3, d4 = st.columns(4)
d1.metric("Clases únicas totales", diag["n_total_classes"])
d2.metric(
    "Clases raras (< N registros)" if group_rare else "Clases raras (desactivado)",
    diag["n_rare"] if group_rare else "—",
)
d3.metric(
    "Clases efectivas tras agrupamiento",
    diag["effective_classes"] if group_rare else diag["n_total_classes"],
)
d4.metric("Filas afectadas por agrupamiento", f"{diag['pct_rare_rows']:.1f}%" if group_rare else "0%")

# Tabla de distribución
with st.expander("Ver distribución completa del target", expanded=diag["n_rare"] > 0):
    freq_df = diag["freq_df"].copy()
    if group_rare:
        freq_df["Estado"] = freq_df["¿Rara?"].map({True: "⚠️ Se agrupará", False: "✅ Se conserva"})
    st.dataframe(freq_df.drop(columns=["¿Rara?"]), use_container_width=True, hide_index=True)

# --- ALERTA CRÍTICA: si el agrupamiento deja < 2 clases ---
effective = diag["effective_classes"] if group_rare else diag["n_total_classes"]

if effective < 2:
    st.error("⛔ **Problema detectado con la variable objetivo**")

    n_surviving = diag["n_surviving"]
    n_rare = diag["n_rare"]
    n_total = diag["n_total_classes"]

    if n_total == 1:
        cause = "La columna tiene **una sola clase** — no sirve como variable objetivo para clasificación."
    elif n_surviving == 0 and group_rare:
        cause = (
            f"**Todas las {n_total} clases** tienen menos de {min_count_thresh} registros. "
            "Al agruparlas, quedan todas en 'Otros' → 1 clase."
        )
    else:
        cause = "Tras el agrupamiento queda solo 1 clase válida."

    st.markdown(f"**Causa:** {cause}")
    st.markdown("**Soluciones disponibles:**")

    sol1, sol2, sol3 = st.columns(3)
    with sol1:
        st.info(
            "**Opción A — Desactivar agrupamiento**\n\n"
            "En ⚙️ Opciones avanzadas, desactiva *Agrupar clases raras* y "
            "desactiva también la *Estratificación* abajo."
        )
    with sol2:
        st.info(
            "**Opción B — Otra columna target**\n\n"
            "Selecciona una columna diferente como variable objetivo. "
            f"Columnas sugeridas: `{', '.join(suggestions[:3]) if suggestions else 'ninguna detectada'}`"
        )
    with sol3:
        st.info(
            "**Opción C — Revisar el dataset**\n\n"
            "Si combinaste datasets con distintos formatos de target "
            "(ej: 'Sí/No' + '1/0'), aplica limpieza/mapeo manual antes de volver aquí."
        )
    st.stop()

elif diag["n_rare"] > 0 and group_rare:
    st.warning(
        f"Se agruparán **{diag['n_rare']}** clases raras → quedarán "
        f"**{diag['effective_classes']}** clases efectivas. "
        "Puedes ajustar el umbral en ⚙️ Opciones avanzadas."
    )
else:
    st.success(f"✅ Variable objetivo válida con **{diag['n_total_classes']}** clases.")

# ---------------------------------------------------------------------------
# Proporciones
# ---------------------------------------------------------------------------
st.markdown("## Proporciones del split")

col1, col2, col3 = st.columns(3)
with col1:
    train_pct = st.slider("Train (%)", 50, 85, 70, step=5)
with col2:
    val_pct = st.slider("Validación (%)", 5, 30, 15, step=5)
with col3:
    test_pct = 100 - train_pct - val_pct
    st.metric("Test (%)", test_pct)

if test_pct <= 0:
    st.error("Las proporciones no son válidas. Ajusta Train y Validación.")
    st.stop()

stratify = st.checkbox(
    "Estratificar partición",
    value=True,
    help="Mantiene la proporción de clases en cada subconjunto (recomendado para clasificación).",
)

# Sugerir desactivar estratificación si hay muchas clases con pocos registros
if diag["n_rare"] > 0 and stratify and group_rare:
    min_required = int(1 / (min(val_pct, test_pct) / 100)) + 1
    if any(diag["freq_df"]["Frecuencia"] < min_required):
        st.warning(
            f"⚠️ Algunas clases tienen muy pocos registros para estratificar correctamente. "
            "Si el split falla, desactiva la estratificación."
        )

# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
st.markdown("## Preprocesamiento")
col_enc, col_scl = st.columns(2)
with col_enc:
    encoding_method = st.selectbox(
        "Encoding categórico",
        ["auto", "onehot", "label"],
        help="auto: One-Hot si < 10 categorías, Label si >= 10.",
    )
with col_scl:
    scaling_method = st.selectbox(
        "Escalado numérico",
        ["standard", "minmax", "robust"],
        format_func=lambda x: {
            "standard": "StandardScaler (Z-score)",
            "minmax": "MinMaxScaler (0-1)",
            "robust": "RobustScaler (mediana/IQR)",
        }[x],
    )

# ---------------------------------------------------------------------------
# Sección educativa: Data Leakage
# ---------------------------------------------------------------------------
with st.expander("¿Qué es el Data Leakage y cómo se previene aquí?"):
    st.markdown("""
**Data Leakage** ocurre cuando información del conjunto de **prueba** "contamina" el
proceso de entrenamiento, generando modelos que parecen más precisos de lo que realmente son.

**Cómo lo previene este pipeline:**
1. La **partición** se realiza *antes* de cualquier transformación estadística.
2. El **escalador** se ajusta **solo** con `X_train` y luego se aplica a `X_val` y `X_test`.
3. El **encoding** se aplica sin aprender estadísticas del conjunto completo.
""")

# ---------------------------------------------------------------------------
# Aplicar partición
# ---------------------------------------------------------------------------
st.markdown("---")
if st.button("Aplicar partición y entrenar baseline", type="primary", use_container_width=True):
    with st.spinner("Procesando..."):
        try:
            X, y = prepare_features_target(df_source, target_col, feature_cols)

            # 1. Corregir clases raras según configuración del usuario
            y_fixed, rare_classes = fix_rare_classes(
                y, min_count=min_count_thresh, enabled=group_rare
            )
            if rare_classes:
                st.info(
                    f"Se agruparon **{len(rare_classes)}** clase(s) raras "
                    f"(< {min_count_thresh} registros) bajo etiqueta genérica."
                )

            # 2. Validar clases resultantes
            if not validate_classes(y_fixed):
                st.error(
                    "Tras el agrupamiento quedó solo 1 clase. "
                    "Prueba: desactivar agrupamiento + desactivar estratificación, "
                    "o seleccionar otra columna como target."
                )
                st.stop()

            # 3. Encoding de features
            X_enc, enc_info = encode_categoricals(
                X, method=encoding_method, exclude_cols=[]
            )

            # 4. Codificar target categórico a entero
            le_target = None
            if not pd.api.types.is_numeric_dtype(y_fixed):
                le_target = LabelEncoder()
                y_fixed = pd.Series(
                    le_target.fit_transform(y_fixed.astype(str)),
                    index=y_fixed.index,
                )

            # 5. Split — con fallback automático si estratificado falla
            try:
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                    X_enc, y_fixed,
                    train_size=train_pct / 100,
                    val_size=val_pct / 100,
                    test_size=test_pct / 100,
                    stratify=stratify,
                )
            except ValueError:
                if stratify:
                    st.warning(
                        "La estratificación falló (alguna clase tiene muy pocos registros "
                        "para repartir en los 3 conjuntos). Se aplicó el split **sin estratificar** automáticamente."
                    )
                    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                        X_enc, y_fixed,
                        train_size=train_pct / 100,
                        val_size=val_pct / 100,
                        test_size=test_pct / 100,
                        stratify=False,
                    )
                else:
                    raise

            # 6. Escalado — fit SOLO en train
            X_train_s, X_val_s, X_test_s, scaler = scale_features(
                X_train, X_val, X_test, method=scaling_method
            )

            # 7. Persistir en session_state
            st.session_state.X_train = X_train_s
            st.session_state.X_val = X_val_s
            st.session_state.X_test = X_test_s
            st.session_state.y_train = y_train
            st.session_state.y_val = y_val
            st.session_state.y_test = y_test
            st.session_state.feature_cols = list(X_train_s.columns)
            st.session_state.target_col = target_col
            st.session_state.scaler = scaler
            st.session_state.encoding_info = enc_info
            st.session_state.le_target = le_target
            st.session_state.models_trained = {}
            st.session_state.experiments_log = []

            # 8. Baseline
            baseline = train_baseline(X_train_s, y_train)
            st.session_state.models_trained["Baseline"] = baseline
            bl_metrics_train = evaluate_classifier(baseline, X_train_s, y_train)
            bl_metrics_test = evaluate_classifier(baseline, X_test_s, y_test)
            log_experiment(
                "Baseline",
                {"strategy": "most_frequent"},
                bl_metrics_test,
                split="test",
            )

        except Exception as e:
            st.error(f"Error inesperado durante la partición: {e}")
            st.stop()

    st.success("✅ Partición y baseline completados.")

    s1, s2, s3 = st.columns(3)
    s1.metric("Train", f"{len(X_train_s):,} muestras")
    s2.metric("Validación", f"{len(X_val_s):,} muestras")
    s3.metric("Test", f"{len(X_test_s):,} muestras")

    st.markdown("### Rendimiento del Baseline")
    st.caption("Un modelo útil debe superar estas métricas.")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Accuracy (test)", f"{bl_metrics_test['accuracy']:.3f}")
    b2.metric("F1 (test)", f"{bl_metrics_test['f1']:.3f}")
    b3.metric("Precision (test)", f"{bl_metrics_test['precision']:.3f}")
    b4.metric("Recall (test)", f"{bl_metrics_test['recall']:.3f}")

    st.info(
        f"Features finales: **{len(st.session_state.feature_cols)}** columnas. "
        "Continúa en **5 — Segmentación** o **6 — Clasificación**."
    )

elif st.session_state.get("X_train") is not None:
    st.info("Partición ya aplicada. Modifica los parámetros y vuelve a hacer clic para recalcular.")
    s1, s2, s3 = st.columns(3)
    s1.metric("Train", f"{len(st.session_state.X_train):,}")
    s2.metric("Validación", f"{len(st.session_state.X_val):,}")
    s3.metric("Test", f"{len(st.session_state.X_test):,}")
