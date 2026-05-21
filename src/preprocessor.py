"""Preprocesamiento: encoding, escalado y partición de datos con anti data leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


def encode_categoricals(
    df: pd.DataFrame,
    method: str = "auto",
    low_card_threshold: int = 10,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Codifica columnas categóricas con One-Hot o Label Encoding.

    La decisión automática usa One-Hot para cardinalidad < umbral,
    Label Encoding para el resto (evita explosión dimensional).

    Args:
        df: DataFrame con columnas categóricas y numéricas mezcladas.
        method: 'auto', 'onehot' o 'label'.
        low_card_threshold: Cardinalidad máxima para aplicar One-Hot en modo auto.
        exclude_cols: Columnas a excluir del encoding (ej: variable objetivo).

    Returns:
        Tupla (DataFrame codificado, dict con info de encodings aplicados).
    """
    result = df.copy()
    exclude_cols = exclude_cols or []
    encoding_info: dict = {"onehot": [], "label": {}, "skipped": []}

    cat_cols = result.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in exclude_cols]

    for col in cat_cols:
        n_unique = result[col].nunique()
        apply_method = method

        if method == "auto":
            apply_method = "onehot" if n_unique < low_card_threshold else "label"

        if apply_method == "onehot":
            dummies = pd.get_dummies(result[col], prefix=col, drop_first=False, dtype=int)
            result = pd.concat([result.drop(columns=[col]), dummies], axis=1)
            encoding_info["onehot"].append(col)
        elif apply_method == "label":
            le = LabelEncoder()
            result[col] = le.fit_transform(result[col].astype(str))
            encoding_info["label"][col] = list(le.classes_)
        else:
            encoding_info["skipped"].append(col)

    return result, encoding_info


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Divide datos en train/val/test sin data leakage.

    Args:
        X: Features.
        y: Variable objetivo.
        train_size: Proporción para entrenamiento (0-1).
        val_size: Proporción para validación (0-1).
        test_size: Proporción para test (0-1).
        stratify: Si True, estratifica según y (solo para clasificación).
        random_state: Semilla aleatoria.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Las proporciones deben sumar 1.0.")

    strat_param = y if stratify else None

    # Primera división: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_size + test_size),
        stratify=strat_param,
        random_state=random_state,
    )

    # Segunda división: val vs test
    relative_test = test_size / (val_size + test_size)
    strat_temp = y_temp if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        stratify=strat_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "standard",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """Escala features ajustando SOLO sobre train para prevenir data leakage.

    Args:
        X_train: Conjunto de entrenamiento.
        X_val: Conjunto de validación.
        X_test: Conjunto de prueba.
        method: 'standard' (StandardScaler), 'minmax' o 'robust'.

    Returns:
        Tupla (X_train_scaled, X_val_scaled, X_test_scaled, scaler_fitted).
    """
    scaler_map = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if method not in scaler_map:
        raise ValueError(f"Método no soportado: '{method}'. Use 'standard', 'minmax' o 'robust'.")

    scaler = scaler_map[method]
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return X_train.copy(), X_val.copy(), X_test.copy(), scaler

    X_train_s = X_train.copy()
    X_val_s = X_val.copy()
    X_test_s = X_test.copy()

    # Fit SOLO con train → luego transform
    X_train_s[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val_s[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test_s[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_s, X_val_s, X_test_s, scaler


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features y variable objetivo del DataFrame limpio.

    Args:
        df: DataFrame procesado.
        target_col: Nombre de la columna objetivo.
        feature_cols: Lista de columnas a usar como features.

    Returns:
        Tupla (X, y).
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y
