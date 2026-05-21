"""Modelos de clasificación: Baseline, Árbol de Decisión y Random Forest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def train_baseline(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    strategy: str = "most_frequent",
    random_state: int = 42,
) -> DummyClassifier:
    """Entrena un clasificador baseline (DummyClassifier).

    Sirve como referencia mínima de rendimiento. Un modelo útil debe
    superar este baseline en todas las métricas.

    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        strategy: Estrategia dummy ('most_frequent', 'stratified', 'uniform').
        random_state: Semilla aleatoria.

    Returns:
        Modelo DummyClassifier entrenado.
    """
    model = DummyClassifier(strategy=strategy, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    max_depth: int | None = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | None = None,
    random_state: int = 42,
    **kwargs,
) -> DecisionTreeClassifier:
    """Entrena un Árbol de Decisión.

    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        max_depth: Profundidad máxima del árbol (None = sin límite).
        min_samples_split: Mínimo de muestras para dividir un nodo.
        min_samples_leaf: Mínimo de muestras en una hoja.
        class_weight: 'balanced' para manejar desbalance, None para uniforme.
        random_state: Semilla aleatoria.
        **kwargs: Parámetros adicionales para DecisionTreeClassifier.

    Returns:
        Modelo DecisionTreeClassifier entrenado.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    class_weight: str | None = None,
    random_state: int = 42,
    **kwargs,
) -> RandomForestClassifier:
    """Entrena un Random Forest.

    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        n_estimators: Número de árboles.
        max_depth: Profundidad máxima de cada árbol.
        min_samples_split: Mínimo de muestras para dividir un nodo.
        min_samples_leaf: Mínimo de muestras en una hoja.
        max_features: Estrategia para selección de features ('sqrt', 'log2').
        class_weight: 'balanced' para manejar desbalance.
        random_state: Semilla aleatoria.
        **kwargs: Parámetros adicionales para RandomForestClassifier.

    Returns:
        Modelo RandomForestClassifier entrenado.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def predict(
    model: object,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Realiza predicciones de clase.

    Args:
        model: Modelo clasificador entrenado.
        X: Features de entrada.

    Returns:
        Array de predicciones.
    """
    return model.predict(X)


def predict_proba(
    model: object,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray | None:
    """Retorna probabilidades de clase si el modelo las soporta.

    Args:
        model: Modelo clasificador entrenado.
        X: Features de entrada.

    Returns:
        Array de probabilidades o None si el modelo no las soporta.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def predict_single(
    model: object,
    scaler: object | None,
    feature_values: dict,
    feature_names: list[str],
) -> tuple[object, np.ndarray | None]:
    """Predice la clase para un único cliente dado como diccionario.

    Args:
        model: Modelo entrenado.
        scaler: Escalador ajustado (o None si no se usó).
        feature_values: Dict {feature_name: value} para el cliente.
        feature_names: Lista ordenada de nombres de features del modelo.

    Returns:
        Tupla (clase_predicha, probabilidades_o_None).
    """
    row = pd.DataFrame([feature_values])[feature_names]
    if scaler is not None:
        numeric_cols = row.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            row[numeric_cols] = scaler.transform(row[numeric_cols])
    pred = model.predict(row)[0]
    proba = predict_proba(model, row)
    return pred, proba
