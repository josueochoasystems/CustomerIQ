"""Perfilamiento automático de datasets: tipos, nulos, cardinalidad y sugerencias."""

from __future__ import annotations

import pandas as pd
import numpy as np


def profile_dataset(df: pd.DataFrame) -> dict:
    """Genera un perfil completo del DataFrame.

    Args:
        df: DataFrame a perfilar.

    Returns:
        Dict con claves: column_types, null_pct, duplicates, cardinality,
        descriptive_stats, suggested_target.
    """
    column_types = _detect_column_types(df)
    null_pct = _null_percentages(df)
    duplicates = _count_duplicates(df)
    cardinality = _compute_cardinality(df, column_types)
    descriptive_stats = _descriptive_statistics(df)
    suggested_target = _suggest_target(df, column_types)

    return {
        "column_types": column_types,
        "null_pct": null_pct,
        "duplicates": duplicates,
        "cardinality": cardinality,
        "descriptive_stats": descriptive_stats,
        "suggested_target": suggested_target,
    }


# ---------------------------------------------------------------------------
# Detección de tipos
# ---------------------------------------------------------------------------

def _detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Clasifica cada columna como numeric, categorical, datetime o text.

    Args:
        df: DataFrame a analizar.

    Returns:
        Dict {columna: tipo}.
    """
    types: dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        else:
            # Intentar parsear como fecha
            sample = df[col].dropna().astype(str).head(20)
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                types[col] = "datetime"
            except (ValueError, TypeError):
                n_unique = df[col].nunique()
                n_total = len(df[col].dropna())
                # Si la mayoría son únicos → texto libre
                if n_total > 0 and n_unique / n_total > 0.8 and n_unique > 20:
                    types[col] = "text"
                else:
                    types[col] = "categorical"
    return types


# ---------------------------------------------------------------------------
# Nulos
# ---------------------------------------------------------------------------

def _null_percentages(df: pd.DataFrame) -> dict[str, float]:
    """Calcula el porcentaje de valores nulos por columna.

    Args:
        df: DataFrame a analizar.

    Returns:
        Dict {columna: pct_nulos}.
    """
    return (df.isnull().mean() * 100).round(2).to_dict()


# ---------------------------------------------------------------------------
# Duplicados
# ---------------------------------------------------------------------------

def _count_duplicates(df: pd.DataFrame) -> dict:
    """Cuenta filas duplicadas.

    Args:
        df: DataFrame a analizar.

    Returns:
        Dict con n_duplicates y pct_duplicates.
    """
    n_dup = int(df.duplicated().sum())
    pct_dup = round(n_dup / len(df) * 100, 2) if len(df) > 0 else 0.0
    return {"n_duplicates": n_dup, "pct_duplicates": pct_dup}


# ---------------------------------------------------------------------------
# Cardinalidad
# ---------------------------------------------------------------------------

def _compute_cardinality(
    df: pd.DataFrame, column_types: dict[str, str]
) -> dict[str, int]:
    """Número de valores únicos por columna categórica.

    Args:
        df: DataFrame.
        column_types: Dict de tipos por columna.

    Returns:
        Dict {columna: n_unique} solo para columnas categóricas.
    """
    result = {}
    for col, ctype in column_types.items():
        if ctype in ("categorical", "text"):
            result[col] = int(df[col].nunique())
    return result


# ---------------------------------------------------------------------------
# Estadísticas descriptivas
# ---------------------------------------------------------------------------

def _descriptive_statistics(df: pd.DataFrame) -> dict:
    """Estadísticas descriptivas para columnas numéricas y categóricas.

    Args:
        df: DataFrame.

    Returns:
        Dict con 'numeric' y 'categorical' como sub-dicts.
    """
    numeric_cols = df.select_dtypes(include="number")
    categorical_cols = df.select_dtypes(exclude="number")

    numeric_stats = numeric_cols.describe().round(4).to_dict() if not numeric_cols.empty else {}
    cat_stats: dict = {}
    for col in categorical_cols.columns:
        vc = df[col].value_counts()
        cat_stats[col] = {
            "n_unique": int(df[col].nunique()),
            "top_value": str(vc.index[0]) if len(vc) > 0 else None,
            "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
            "pct_top": round(vc.iloc[0] / len(df) * 100, 2) if len(vc) > 0 else 0.0,
        }

    return {"numeric": numeric_stats, "categorical": cat_stats}


# ---------------------------------------------------------------------------
# Sugerencia de variable objetivo
# ---------------------------------------------------------------------------

def _suggest_target(df: pd.DataFrame, column_types: dict[str, str]) -> list[str]:
    """Sugiere columnas candidatas como variable objetivo.

    Criterios: columnas binarias (2 valores únicos), o categóricas con
    entre 2 y 10 clases únicas. Se excluyen columnas con >30% nulos.

    Args:
        df: DataFrame.
        column_types: Tipos detectados por columna.

    Returns:
        Lista de nombres de columnas sugeridas, ordenadas por prioridad.
    """
    candidates: list[tuple[int, str]] = []
    null_pct = df.isnull().mean()

    for col in df.columns:
        if null_pct[col] > 0.3:
            continue
        n_unique = df[col].nunique()
        col_type = column_types.get(col, "")

        if n_unique == 2:
            # Mejor candidato: binario
            candidates.append((0, col))
        elif col_type == "categorical" and 3 <= n_unique <= 10:
            candidates.append((1, col))
        elif col_type == "numeric" and 2 <= n_unique <= 10:
            candidates.append((2, col))

    candidates.sort(key=lambda x: x[0])
    return [col for _, col in candidates]
