"""Algoritmos de clustering: K-Means y jerárquico con métricas de evaluación."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def run_kmeans(
    X: np.ndarray | pd.DataFrame,
    k: int,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray, float]:
    """Ejecuta K-Means con k clústeres.

    Args:
        X: Datos de entrada (ya escalados).
        k: Número de clústeres.
        random_state: Semilla aleatoria.

    Returns:
        Tupla (modelo_entrenado, labels, inercia).
    """
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, labels, float(model.inertia_)


def find_optimal_k(
    X: np.ndarray | pd.DataFrame,
    k_range: tuple[int, int] = (2, 10),
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula inercia y silhouette para un rango de k.

    Args:
        X: Datos de entrada.
        k_range: Tupla (k_min, k_max) inclusive.
        random_state: Semilla aleatoria.

    Returns:
        DataFrame con columnas [k, inertia, silhouette_score].
    """
    results = []
    for k in range(k_range[0], k_range[1] + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        inertia = float(model.inertia_)
        sil = float(silhouette_score(X, labels)) if k >= 2 else np.nan
        results.append({"k": k, "inertia": inertia, "silhouette_score": sil})
    return pd.DataFrame(results)


def run_hierarchical(
    X: np.ndarray | pd.DataFrame,
    n_clusters: int,
    linkage: str = "ward",
) -> tuple[AgglomerativeClustering, np.ndarray]:
    """Ejecuta clustering jerárquico aglomerativo.

    Args:
        X: Datos de entrada.
        n_clusters: Número de clústeres a formar.
        linkage: Método de enlace ('ward', 'complete', 'average', 'single').

    Returns:
        Tupla (modelo_entrenado, labels).
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return model, labels


def compute_silhouette(
    X: np.ndarray | pd.DataFrame,
    labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Calcula el score silhouette global y por muestra.

    Args:
        X: Datos.
        labels: Etiquetas de clúster por muestra.

    Returns:
        Tupla (score_global, scores_por_muestra).
    """
    global_score = float(silhouette_score(X, labels))
    sample_scores = silhouette_samples(X, labels)
    return global_score, sample_scores


def cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    numeric_only: bool = True,
) -> pd.DataFrame:
    """Calcula estadísticas descriptivas por clúster para perfilamiento.

    Args:
        df: DataFrame original (antes o después de escalado, ambos sirven).
        labels: Etiquetas de clúster por fila.
        numeric_only: Si True, solo incluye columnas numéricas.

    Returns:
        DataFrame con media y desviación estándar de cada feature por clúster.
    """
    df_profile = df.copy()
    df_profile["_cluster"] = labels

    if numeric_only:
        num_cols = df_profile.select_dtypes(include="number").columns.tolist()
        num_cols = [c for c in num_cols if c != "_cluster"]
        df_profile = df_profile[num_cols + ["_cluster"]]

    agg = df_profile.groupby("_cluster").agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.index.name = "cluster"
    agg["n_samples"] = df_profile.groupby("_cluster").size()
    return agg.reset_index()


def interpret_clusters(profile_df: pd.DataFrame, feature_cols: list[str]) -> dict[int, str]:
    """Genera una interpretación textual automática de cada clúster.

    Compara la media de cada clúster con la media global y genera
    una descripción de alto nivel orientada a negocio.

    Args:
        profile_df: DataFrame retornado por cluster_profiles().
        feature_cols: Columnas a considerar en la interpretación.

    Returns:
        Dict {cluster_id: texto_descripcion}.
    """
    interpretations: dict[int, str] = {}

    mean_cols = [f"{c}_mean" for c in feature_cols if f"{c}_mean" in profile_df.columns]
    if not mean_cols:
        return interpretations

    global_means = profile_df[mean_cols].mean()

    for _, row in profile_df.iterrows():
        cluster_id = int(row["cluster"])
        traits = []
        for mc in mean_cols:
            feat = mc.replace("_mean", "")
            val = row[mc]
            glob = global_means[mc]
            if glob == 0:
                continue
            ratio = (val - glob) / abs(glob)
            if ratio > 0.15:
                traits.append(f"{feat} alto ({val:.2f})")
            elif ratio < -0.15:
                traits.append(f"{feat} bajo ({val:.2f})")

        n = int(row.get("n_samples", 0))
        desc = f"Clúster {cluster_id} ({n} clientes)"
        if traits:
            desc += ": " + ", ".join(traits[:4])
        else:
            desc += ": perfil similar al promedio general"
        interpretations[cluster_id] = desc

    return interpretations
