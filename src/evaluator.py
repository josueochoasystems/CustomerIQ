"""Evaluación de modelos de clasificación y registro de experimentos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(
    model: object,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    average: str = "weighted",
) -> dict:
    """Evalúa un clasificador en un conjunto de datos.

    Args:
        model: Modelo entrenado con método predict (y predict_proba opcional).
        X: Features de evaluación.
        y: Etiquetas verdaderas.
        average: Estrategia de promediado para F1, Precision, Recall
                 ('weighted', 'macro', 'binary').

    Returns:
        Dict con: accuracy, f1, precision, recall, auc (si disponible).
    """
    y_pred = model.predict(X)
    y_true = np.array(y)

    metrics: dict = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "auc": None,
    }

    # AUC solo si el modelo soporta probabilidades
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                auc_val = roc_auc_score(y_true, proba[:, 1])
            else:
                auc_val = roc_auc_score(
                    y_true, proba, multi_class="ovr", average="weighted"
                )
            metrics["auc"] = round(float(auc_val), 4)
        except Exception:
            metrics["auc"] = None

    return metrics


def confusion_matrix_data(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> dict:
    """Calcula la matriz de confusión y métricas derivadas.

    Para clasificación binaria calcula TP, TN, FP, FN, sensibilidad
    y especificidad. Para multiclase devuelve solo la matriz.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones del modelo.

    Returns:
        Dict con: matrix, classes, y métricas derivadas (si binario).
    """
    y_true_arr = np.array(y_true)
    classes = np.unique(y_true_arr)
    cm = confusion_matrix(y_true_arr, y_pred, labels=classes)

    result: dict = {
        "matrix": cm,
        "classes": [str(c) for c in classes],
    }

    if len(classes) == 2:
        tn, fp, fn, tp = cm.ravel()
        result.update({
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0,
            "ppv": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
        })

    return result


def roc_curve_data(
    model: object,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
) -> dict | None:
    """Calcula datos para la curva ROC (solo clasificación binaria).

    Args:
        model: Modelo con predict_proba.
        X: Features.
        y: Etiquetas verdaderas.

    Returns:
        Dict con fpr, tpr, thresholds, auc_score. None si no aplica.
    """
    if not hasattr(model, "predict_proba"):
        return None

    y_arr = np.array(y)
    classes = np.unique(y_arr)
    if len(classes) != 2:
        return None

    try:
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_arr, proba)
        auc_score = auc(fpr, tpr)
        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc_score": round(float(auc_score), 4),
        }
    except Exception:
        return None


def log_experiment(
    name: str,
    params: dict,
    metrics: dict,
    split: str = "test",
) -> dict:
    """Crea una entrada de experimento para el historial en session_state.

    Actualiza st.session_state.experiments_log automáticamente.

    Args:
        name: Nombre del experimento / modelo.
        params: Hiperparámetros usados.
        metrics: Métricas obtenidas (dict de evaluate_classifier).
        split: Nombre del conjunto evaluado ('train', 'val', 'test').

    Returns:
        Dict con la entrada del experimento registrada.
    """
    import datetime

    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": name,
        "split": split,
        **{f"param_{k}": v for k, v in params.items()},
        **metrics,
    }

    if "experiments_log" not in st.session_state:
        st.session_state.experiments_log = []
    st.session_state.experiments_log.append(entry)

    return entry


def experiments_to_dataframe() -> pd.DataFrame | None:
    """Convierte el historial de experimentos a DataFrame.

    Returns:
        DataFrame o None si no hay experimentos registrados.
    """
    if "experiments_log" not in st.session_state or not st.session_state.experiments_log:
        return None
    return pd.DataFrame(st.session_state.experiments_log)
