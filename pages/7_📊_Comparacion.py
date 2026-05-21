"""Página 7 — Comparación de modelos con ROC, matrices de confusión e interpretación gerencial."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.evaluator import (
    confusion_matrix_data,
    evaluate_classifier,
    experiments_to_dataframe,
)
from src.visualizer import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_roc_curves,
)

st.set_page_config(page_title="Comparación — CustomerIQ", layout="wide")

st.title("Comparación de Modelos")

if st.session_state.get("X_train") is None:
    st.warning("Completa la partición en la página **4 — Partición**.")
    st.stop()

if not st.session_state.get("models_trained"):
    st.warning("Entrena al menos un modelo en la página **6 — Clasificación**.")
    st.stop()

X_test = st.session_state.X_test
y_test = st.session_state.y_test
models = st.session_state.models_trained

y_all = pd.concat([st.session_state.y_train, st.session_state.y_val, y_test])
class_names = [str(c) for c in sorted(y_all.unique())]
is_binary = len(class_names) == 2

# ---------------------------------------------------------------------------
# Tabla comparativa de métricas
# ---------------------------------------------------------------------------
st.markdown("## Tabla comparativa de métricas (conjunto de test)")

rows = []
for name, model in models.items():
    mets = evaluate_classifier(model, X_test, y_test)
    rows.append({"Modelo": name, **mets})

comparison_df = pd.DataFrame(rows)

# Resaltar el mejor valor en cada métrica
metric_cols = [c for c in ["accuracy", "f1", "precision", "recall", "auc"] if c in comparison_df.columns]

def highlight_max(s: pd.Series):
    if s.dtype == object or s.isna().all():
        return [""] * len(s)
    max_val = s.max()
    return ["background-color: #d4edda; font-weight: bold" if v == max_val else "" for v in s]

styled_df = comparison_df.style.apply(highlight_max, subset=metric_cols)
st.dataframe(styled_df, use_container_width=True, hide_index=True)

# Selección de métrica para ranking
rank_metric = st.selectbox(
    "Métrica principal para recomendación",
    options=[c for c in metric_cols if c in comparison_df.columns],
    index=1 if "f1" in metric_cols else 0,
    help="El modelo recomendado se determina según esta métrica.",
)

best_model_name = comparison_df.loc[
    comparison_df[rank_metric].idxmax(), "Modelo"
]
best_val = comparison_df[rank_metric].max()
st.success(
    f"Modelo recomendado según **{rank_metric.upper()}**: **{best_model_name}** ({best_val:.4f})"
)

# Gráfico comparativo
st.plotly_chart(plot_metrics_comparison(comparison_df), use_container_width=True)

# ---------------------------------------------------------------------------
# Historial de experimentos
# ---------------------------------------------------------------------------
st.markdown("## Historial de experimentos")
exp_df = experiments_to_dataframe()
if exp_df is not None:
    st.dataframe(exp_df, use_container_width=True)
    csv_exp = exp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar historial (CSV)",
        data=csv_exp,
        file_name="historial_experimentos.csv",
        mime="text/csv",
    )
else:
    st.info("No hay experimentos registrados aún.")

# ---------------------------------------------------------------------------
# Curvas ROC
# ---------------------------------------------------------------------------
st.markdown("## Curvas ROC")
if is_binary:
    st.plotly_chart(
        plot_roc_curves(models, X_test, y_test),
        use_container_width=True,
    )
else:
    st.info(
        "La curva ROC individual solo está disponible para clasificación binaria. "
        "Este dataset tiene múltiples clases."
    )

# ---------------------------------------------------------------------------
# Matrices de confusión
# ---------------------------------------------------------------------------
st.markdown("## Matrices de confusión")
cm_cols = st.columns(min(len(models), 3))

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm_data = confusion_matrix_data(np.array(y_test), y_pred)
    with cm_cols[i % len(cm_cols)]:
        st.plotly_chart(
            plot_confusion_matrix(cm_data["matrix"], cm_data["classes"], title=name),
            use_container_width=True,
        )
        if is_binary and "sensitivity" in cm_data:
            st.markdown(
                f"**Sensibilidad**: {cm_data['sensitivity']:.3f} | "
                f"**Especificidad**: {cm_data['specificity']:.3f}"
            )

# ---------------------------------------------------------------------------
# Sección gerencial
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Para la Gerencia — Interpretación en lenguaje no técnico")

best_model_obj = models.get(best_model_name)
if best_model_obj is not None:
    mets_best = evaluate_classifier(best_model_obj, X_test, y_test)
    acc = mets_best["accuracy"]
    prec = mets_best["precision"]
    rec = mets_best["recall"]
    f1 = mets_best["f1"]
    auc_val = mets_best.get("auc")

    n_test = len(y_test)
    n_correct = int(acc * n_test)

    st.info(f"""
**¿Qué tan bueno es el modelo {best_model_name}?**

De cada **100 clientes** que el modelo analiza en un conjunto nuevo:

- ✅ Clasifica correctamente a **{int(acc*100)}** de ellos (Accuracy = {acc:.1%}).
- 🎯 De los clientes que **identifica como positivos**, {int(prec*100)} de cada 100 realmente lo son
  (Precisión = {prec:.1%}). Esto reduce los "falsos alarmas".
- 🔍 De los clientes que **realmente son positivos**, el modelo encuentra a {int(rec*100)} de cada 100
  (Recall = {rec:.1%}). Esto reduce los casos perdidos.
- ⚖️ El balance entre precisión y detección (F1) es de **{f1:.2f}** sobre 1.0 posible.
""")

    if auc_val is not None and is_binary:
        st.markdown(f"""
**¿Qué significa el AUC de {auc_val:.3f}?**

El AUC (Área bajo la curva ROC) mide qué tan bien el modelo distingue entre
clientes positivos y negativos. Un valor de **{auc_val:.3f}** sobre 1.0 indica:

- Cercano a **1.0**: el modelo discrimina perfectamente.
- Cercano a **0.5**: el modelo no sabe distinguir mejor que el azar.
- Nuestro modelo con **{auc_val:.3f}** tiene una capacidad de discriminación
  {"excelente" if auc_val > 0.9 else "buena" if auc_val > 0.8 else "moderada" if auc_val > 0.7 else "limitada"}.
""")

    baseline_obj = models.get("Baseline")
    if baseline_obj is not None:
        bl_acc = evaluate_classifier(baseline_obj, X_test, y_test)["accuracy"]
        improvement = (acc - bl_acc) * 100
        st.markdown(f"""
**Comparación con el punto de partida (Baseline)**

Un sistema que predice siempre la clase más común acertaría el **{bl_acc:.1%}** de los casos.
Nuestro modelo **{best_model_name}** mejora en **{improvement:+.1f} puntos porcentuales**,
lo que representa un {"beneficio significativo" if improvement > 5 else "beneficio moderado"} en la toma de decisiones.
""")
