"""Página 6 — Clasificación: Árbol de Decisión y Random Forest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import numpy as np
import pandas as pd
import streamlit as st

from src.classification import (
    predict,
    predict_proba,
    predict_single,
    train_decision_tree,
    train_random_forest,
)
from src.evaluator import evaluate_classifier, log_experiment
from src.visualizer import (
    plot_decision_tree,
    plot_feature_importance,
)

st.set_page_config(page_title="Clasificación — CustomerIQ", layout="wide")

st.title("Clasificación Supervisada")

if st.session_state.get("X_train") is None:
    st.warning(
        "Primero debes completar la partición en la página **4 — Partición** "
        "para tener disponibles los conjuntos train/val/test."
    )
    st.stop()

X_train = st.session_state.X_train
X_val = st.session_state.X_val
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_val = st.session_state.y_val
y_test = st.session_state.y_test
feature_names = st.session_state.feature_cols

# Inferir nombres de clases
y_all = pd.concat([y_train, y_val, y_test])
class_names = [str(c) for c in sorted(y_all.unique())]

st.info(
    f"Dataset: **{len(X_train):,}** train | **{len(X_val):,}** val | **{len(X_test):,}** test | "
    f"**{len(feature_names)}** features | **{len(class_names)}** clases"
)

tab_dt, tab_rf = st.tabs(["🌳 Árbol de Decisión", "🌲 Random Forest"])

# ===========================================================================
# ÁRBOL DE DECISIÓN
# ===========================================================================
with tab_dt:
    st.markdown("### Hiperparámetros")
    col1, col2, col3 = st.columns(3)
    with col1:
        dt_max_depth = st.slider("Profundidad máxima", 1, 20, 5, key="dt_depth",
                                  help="Controla la complejidad. Árboles muy profundos tienden a sobreajustar.")
    with col2:
        dt_min_split = st.slider("min_samples_split", 2, 50, 2, key="dt_split",
                                  help="Mínimo de muestras para dividir un nodo.")
    with col3:
        dt_min_leaf = st.slider("min_samples_leaf", 1, 30, 1, key="dt_leaf",
                                 help="Mínimo de muestras en cada hoja.")

    dt_balanced = st.checkbox(
        "Balancear clases (class_weight='balanced')",
        value=False,
        key="dt_balanced",
        help="Útil cuando hay desbalance de clases. Pondera más las clases minoritarias.",
    )

    if st.button("Entrenar Árbol de Decisión", type="primary", key="train_dt"):
        with st.spinner("Entrenando..."):
            model_dt = train_decision_tree(
                X_train, y_train,
                max_depth=dt_max_depth,
                min_samples_split=dt_min_split,
                min_samples_leaf=dt_min_leaf,
                class_weight="balanced" if dt_balanced else None,
            )
            st.session_state.models_trained["Árbol de Decisión"] = model_dt

            metrics_train = evaluate_classifier(model_dt, X_train, y_train)
            metrics_val = evaluate_classifier(model_dt, X_val, y_val)
            metrics_test = evaluate_classifier(model_dt, X_test, y_test)

            log_experiment(
                "Árbol de Decisión",
                {"max_depth": dt_max_depth, "min_samples_split": dt_min_split,
                 "min_samples_leaf": dt_min_leaf, "balanced": dt_balanced},
                metrics_test,
                split="test",
            )

        st.success("Árbol de Decisión entrenado.")

        # Métricas
        st.markdown("#### Métricas de rendimiento")
        tab_m1, tab_m2, tab_m3 = st.tabs(["Train", "Validación", "Test"])
        for tab, mets, split_name in [
            (tab_m1, metrics_train, "Train"),
            (tab_m2, metrics_val, "Validación"),
            (tab_m3, metrics_test, "Test"),
        ]:
            with tab:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{mets['accuracy']:.4f}")
                c2.metric("F1", f"{mets['f1']:.4f}")
                c3.metric("Precision", f"{mets['precision']:.4f}")
                c4.metric("Recall", f"{mets['recall']:.4f}")
                if mets["auc"] is not None:
                    st.metric("AUC-ROC", f"{mets['auc']:.4f}")

        # Importancia de variables
        st.markdown("#### Importancia de variables")
        st.plotly_chart(
            plot_feature_importance(model_dt, feature_names),
            use_container_width=True,
        )

        # Árbol visual
        st.markdown("#### Visualización del árbol")
        depth_viz = st.slider("Profundidad a visualizar", 1, min(dt_max_depth, 6), min(3, dt_max_depth), key="dt_viz_depth")
        fig_tree = plot_decision_tree(model_dt, feature_names, class_names, max_depth=depth_viz)
        st.pyplot(fig_tree, use_container_width=True)

    elif "Árbol de Decisión" in st.session_state.models_trained:
        st.info("Árbol ya entrenado. Modifica los hiperparámetros y vuelve a entrenar para registrar una nueva corrida.")
        model_dt = st.session_state.models_trained["Árbol de Decisión"]
        mets = evaluate_classifier(model_dt, X_test, y_test)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy (test)", f"{mets['accuracy']:.4f}")
        c2.metric("F1 (test)", f"{mets['f1']:.4f}")
        c3.metric("Precision (test)", f"{mets['precision']:.4f}")
        c4.metric("Recall (test)", f"{mets['recall']:.4f}")
        st.plotly_chart(plot_feature_importance(model_dt, feature_names), use_container_width=True)

# ===========================================================================
# RANDOM FOREST
# ===========================================================================
with tab_rf:
    st.markdown("### Hiperparámetros")
    col1, col2, col3 = st.columns(3)
    with col1:
        rf_n_est = st.slider("n_estimators", 10, 500, 100, step=10, key="rf_n",
                              help="Número de árboles en el bosque. Más árboles = mayor estabilidad.")
    with col2:
        rf_max_depth = st.slider("max_depth", 1, 30, 10, key="rf_depth",
                                  help="Profundidad máxima de cada árbol.")
    with col3:
        rf_min_leaf = st.slider("min_samples_leaf", 1, 30, 1, key="rf_leaf",
                                 help="Mínimo de muestras por hoja.")

    col4, col5 = st.columns(2)
    with col4:
        rf_max_feat = st.selectbox(
            "max_features",
            ["sqrt", "log2"],
            help="'sqrt' es el estándar para clasificación.",
            key="rf_feat",
        )
    with col5:
        rf_balanced = st.checkbox(
            "Balancear clases (class_weight='balanced')",
            value=False,
            key="rf_balanced",
        )

    if st.button("Entrenar Random Forest", type="primary", key="train_rf"):
        with st.spinner("Entrenando Random Forest (puede tardar unos segundos)..."):
            model_rf = train_random_forest(
                X_train, y_train,
                n_estimators=rf_n_est,
                max_depth=rf_max_depth,
                min_samples_leaf=rf_min_leaf,
                max_features=rf_max_feat,
                class_weight="balanced" if rf_balanced else None,
            )
            st.session_state.models_trained["Random Forest"] = model_rf

            metrics_train_rf = evaluate_classifier(model_rf, X_train, y_train)
            metrics_val_rf = evaluate_classifier(model_rf, X_val, y_val)
            metrics_test_rf = evaluate_classifier(model_rf, X_test, y_test)

            log_experiment(
                "Random Forest",
                {"n_estimators": rf_n_est, "max_depth": rf_max_depth,
                 "min_samples_leaf": rf_min_leaf, "balanced": rf_balanced},
                metrics_test_rf,
                split="test",
            )

        st.success("Random Forest entrenado.")

        tab_m1, tab_m2, tab_m3 = st.tabs(["Train", "Validación", "Test"])
        for tab, mets in [(tab_m1, metrics_train_rf), (tab_m2, metrics_val_rf), (tab_m3, metrics_test_rf)]:
            with tab:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{mets['accuracy']:.4f}")
                c2.metric("F1", f"{mets['f1']:.4f}")
                c3.metric("Precision", f"{mets['precision']:.4f}")
                c4.metric("Recall", f"{mets['recall']:.4f}")
                if mets["auc"] is not None:
                    c1.metric("AUC-ROC", f"{mets['auc']:.4f}")

        st.plotly_chart(
            plot_feature_importance(model_rf, feature_names),
            use_container_width=True,
        )

    elif "Random Forest" in st.session_state.models_trained:
        st.info("Random Forest ya entrenado. Modifica hiperparámetros y vuelve a entrenar para nueva corrida.")
        model_rf = st.session_state.models_trained["Random Forest"]
        mets = evaluate_classifier(model_rf, X_test, y_test)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy (test)", f"{mets['accuracy']:.4f}")
        c2.metric("F1 (test)", f"{mets['f1']:.4f}")
        c3.metric("Precision (test)", f"{mets['precision']:.4f}")
        c4.metric("Recall (test)", f"{mets['recall']:.4f}")
        st.plotly_chart(plot_feature_importance(model_rf, feature_names), use_container_width=True)

# ===========================================================================
# PREDICCIÓN INDIVIDUAL
# ===========================================================================
st.markdown("---")
st.markdown("## Predicción individual")
st.markdown("Ingresa los datos de un cliente nuevo para obtener la predicción del modelo entrenado.")

trained_models = {k: v for k, v in st.session_state.models_trained.items() if k != "Baseline"}
if not trained_models:
    st.info("Entrena al menos un modelo (Árbol o Random Forest) para usar la predicción individual.")
else:
    model_choice = st.selectbox("Modelo a usar", list(trained_models.keys()))
    chosen_model = trained_models[model_choice]

    input_values = {}
    cols_input = st.columns(3)
    for i, feat in enumerate(feature_names):
        with cols_input[i % 3]:
            input_values[feat] = st.number_input(feat, value=0.0, key=f"pred_{feat}")

    if st.button("Predecir", key="predict_single"):
        pred, proba = predict_single(
            chosen_model,
            scaler=st.session_state.get("scaler"),
            feature_values=input_values,
            feature_names=feature_names,
        )
        st.success(f"Clase predicha: **{pred}**")
        if proba is not None:
            prob_df = pd.DataFrame({"Clase": class_names, "Probabilidad": proba[0].round(4)})
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

# ===========================================================================
# PREDICCIÓN POR LOTE
# ===========================================================================
st.markdown("---")
st.markdown("## Predicción por lote")
batch_file = st.file_uploader(
    "Sube un CSV con el mismo formato que las features de entrenamiento",
    type=["csv"],
    key="batch_upload",
)

if batch_file and trained_models:
    model_batch_choice = st.selectbox("Modelo para lote", list(trained_models.keys()), key="batch_model")
    try:
        df_batch = pd.read_csv(batch_file)
        # Alinear columnas
        missing = set(feature_names) - set(df_batch.columns)
        if missing:
            st.error(f"Faltan columnas: {missing}")
        else:
            X_batch = df_batch[feature_names]
            model_b = trained_models[model_batch_choice]
            preds = model_b.predict(X_batch)
            df_batch["prediccion"] = preds
            if hasattr(model_b, "predict_proba"):
                probas = model_b.predict_proba(X_batch)
                for j, cls in enumerate(class_names):
                    df_batch[f"prob_{cls}"] = probas[:, j].round(4)

            st.success(f"{len(df_batch):,} predicciones generadas.")
            st.dataframe(df_batch.head(20), use_container_width=True)

            csv_out = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar predicciones (CSV)",
                data=csv_out,
                file_name="predicciones_lote.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
