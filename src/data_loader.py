"""Carga de archivos de datos en múltiples formatos con detección automática."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd


def load_file(uploaded_file: Any) -> tuple[pd.DataFrame, dict]:
    """Carga un archivo subido y retorna un DataFrame con metadata.

    Args:
        uploaded_file: Objeto de archivo de st.file_uploader.

    Returns:
        Tupla (DataFrame, metadata) donde metadata incluye filas, columnas y tipos.

    Raises:
        ValueError: Si el formato no es soportado o el archivo no puede leerse.
    """
    name: str = uploaded_file.name.lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = _load_excel(uploaded_file)
    elif name.endswith(".json"):
        df = _load_json(uploaded_file)
    elif name.endswith(".tsv"):
        df = _load_csv(uploaded_file, sep="\t")
    elif name.endswith(".csv"):
        df = _load_csv(uploaded_file, sep=None)
    else:
        raise ValueError(
            f"Formato no soportado: '{uploaded_file.name}'. "
            "Use CSV, XLSX, JSON o TSV."
        )

    metadata = _build_metadata(df)
    return df, metadata


# ---------------------------------------------------------------------------
# Loaders internos
# ---------------------------------------------------------------------------

def _load_csv(uploaded_file: Any, sep: str | None) -> pd.DataFrame:
    """Carga un CSV probando encodings y detectando separador automáticamente."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    raw_bytes = uploaded_file.read()

    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)
            if sep is None:
                # Detectar separador probando los más comunes
                detected_sep = _detect_separator(text)
            else:
                detected_sep = sep
            df = pd.read_csv(io.StringIO(text), sep=detected_sep)
            if df.shape[1] > 1:
                return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    # Último intento con python engine y sep=None (sniffing)
    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()
    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            return df
        except Exception:
            continue

    raise ValueError("No se pudo leer el archivo CSV. Verifica el encoding y separador.")


def _detect_separator(text: str) -> str:
    """Detecta el separador más probable en un texto CSV."""
    sample = "\n".join(text.splitlines()[:5])
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t", "|"]}
    return max(counts, key=lambda k: counts[k])


def _load_excel(uploaded_file: Any) -> pd.DataFrame:
    """Carga un archivo Excel (.xlsx / .xls)."""
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        return df
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo Excel: {e}") from e


def _load_json(uploaded_file: Any) -> pd.DataFrame:
    """Carga un archivo JSON."""
    try:
        df = pd.read_json(uploaded_file)
        return df
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo JSON: {e}") from e


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _build_metadata(df: pd.DataFrame) -> dict:
    """Construye un diccionario con información básica del DataFrame.

    Args:
        df: DataFrame cargado.

    Returns:
        Dict con n_rows, n_cols, dtypes, memory_mb.
    """
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
    }
