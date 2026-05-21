"""Limpieza de datos: imputación, duplicados y outliers. Funciones puras."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def handle_missing(df: pd.DataFrame, strategy_dict: dict[str, str | float]) -> pd.DataFrame:
    """Imputa valores faltantes según estrategia por columna.

    Args:
        df: DataFrame original (no se modifica).
        strategy_dict: Mapeo {columna: estrategia}. Estrategias válidas:
            'mean', 'median', 'mode', 'drop', o valor numérico/string constante.

    Returns:
        Nuevo DataFrame con imputaciones aplicadas.
    """
    result = df.copy()

    rows_to_drop: set[int] = set()
    for col, strategy in strategy_dict.items():
        if col not in result.columns:
            continue
        null_mask = result[col].isnull()
        if not null_mask.any():
            continue

        if strategy == "mean":
            fill_value = result[col].mean()
            result[col] = result[col].fillna(fill_value)
        elif strategy == "median":
            fill_value = result[col].median()
            result[col] = result[col].fillna(fill_value)
        elif strategy == "mode":
            mode_vals = result[col].mode()
            if len(mode_vals) > 0:
                result[col] = result[col].fillna(mode_vals.iloc[0])
        elif strategy == "drop":
            rows_to_drop.update(result.index[null_mask].tolist())
        else:
            # Valor constante (numérico o string)
            result[col] = result[col].fillna(strategy)

    if rows_to_drop:
        result = result.drop(index=list(rows_to_drop)).reset_index(drop=True)

    return result


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas duplicadas exactas.

    Args:
        df: DataFrame original.

    Returns:
        DataFrame sin duplicados con índice reiniciado.
    """
    return df.drop_duplicates().reset_index(drop=True)


def detect_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    columns: list[str] | None = None,
    threshold: float = 3.0,
) -> pd.Index:
    """Detecta índices de filas con outliers en columnas numéricas.

    Args:
        df: DataFrame.
        method: 'iqr' o 'zscore'.
        columns: Columnas a analizar. Si None, usa todas las numéricas.
        threshold: Para zscore, valor Z por encima del cual se considera outlier.

    Returns:
        Índice de filas que contienen al menos un outlier.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if columns:
        numeric_cols = [c for c in columns if c in numeric_cols]

    if not numeric_cols:
        return pd.Index([])

    outlier_mask = pd.Series(False, index=df.index)

    if method == "iqr":
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask |= (df[col] < lower) | (df[col] > upper)
    elif method == "zscore":
        for col in numeric_cols:
            col_data = df[col].dropna()
            z_scores = np.abs(stats.zscore(col_data))
            outlier_idx = col_data.index[z_scores > threshold]
            outlier_mask.loc[outlier_idx] = True
    else:
        raise ValueError(f"Método no soportado: '{method}'. Use 'iqr' o 'zscore'.")

    return df.index[outlier_mask]


def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    action: str = "keep",
    columns: list[str] | None = None,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Maneja outliers según la acción especificada.

    Args:
        df: DataFrame original.
        method: Método de detección ('iqr' o 'zscore').
        action: 'keep' (dejar), 'remove' (eliminar filas) o 'cap' (winsorize).
        columns: Columnas a analizar.
        threshold: Umbral para zscore.

    Returns:
        DataFrame procesado.
    """
    result = df.copy()

    if action == "keep":
        return result

    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    if columns:
        numeric_cols = [c for c in columns if c in numeric_cols]

    if action == "remove":
        outlier_idx = detect_outliers(result, method=method, columns=numeric_cols, threshold=threshold)
        result = result.drop(index=outlier_idx).reset_index(drop=True)

    elif action == "cap":
        for col in numeric_cols:
            if method == "iqr":
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
            else:
                mean = result[col].mean()
                std = result[col].std()
                lower = mean - threshold * std
                upper = mean + threshold * std
            result[col] = result[col].clip(lower=lower, upper=upper)
    else:
        raise ValueError(f"Acción no soportada: '{action}'. Use 'keep', 'remove' o 'cap'.")

    return result


def outlier_summary(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    """Genera un resumen del número de outliers por columna numérica.

    Args:
        df: DataFrame.
        method: Método de detección.

    Returns:
        DataFrame con columnas [column, n_outliers, pct_outliers].
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    rows = []
    for col in numeric_cols:
        idx = detect_outliers(df, method=method, columns=[col])
        n = len(idx)
        pct = round(n / len(df) * 100, 2) if len(df) > 0 else 0.0
        rows.append({"column": col, "n_outliers": n, "pct_outliers": pct})
    return pd.DataFrame(rows)
