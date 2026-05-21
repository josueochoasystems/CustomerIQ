"""Visualizaciones interactivas con Plotly y Matplotlib según corresponda."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """Gráfico de barras con el porcentaje de valores nulos por columna.

    Args:
        df: DataFrame a analizar.

    Returns:
        Figura Plotly.
    """
    null_pct = (df.isnull().mean() * 100).reset_index()
    null_pct.columns = ["columna", "pct_nulos"]
    null_pct = null_pct.sort_values("pct_nulos", ascending=False)

    fig = px.bar(
        null_pct,
        x="columna",
        y="pct_nulos",
        title="Porcentaje de valores nulos por columna",
        labels={"pct_nulos": "% Nulos", "columna": "Columna"},
        color="pct_nulos",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Mapa de calor de correlaciones para columnas numéricas.

    Args:
        df: DataFrame.

    Returns:
        Figura Plotly.
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr().round(3)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Matriz de Correlación",
        xaxis_tickangle=-45,
        height=500,
    )
    return fig


def plot_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """Histograma con curva KDE para una columna.

    Args:
        df: DataFrame.
        column: Nombre de la columna a visualizar.

    Returns:
        Figura Plotly.
    """
    fig = px.histogram(
        df,
        x=column,
        marginal="violin",
        title=f"Distribución de {column}",
        labels={column: column},
        color_discrete_sequence=["#636EFA"],
        histnorm="density",
    )
    return fig


def plot_boxplot(df: pd.DataFrame, column: str) -> go.Figure:
    """Boxplot para una columna numérica.

    Args:
        df: DataFrame.
        column: Columna a visualizar.

    Returns:
        Figura Plotly.
    """
    fig = px.box(
        df,
        y=column,
        title=f"Boxplot de {column}",
        points="outliers",
        color_discrete_sequence=["#EF553B"],
    )
    return fig


def plot_countplot(df: pd.DataFrame, column: str) -> go.Figure:
    """Gráfico de barras de frecuencia para columnas categóricas.

    Args:
        df: DataFrame.
        column: Columna categórica.

    Returns:
        Figura Plotly.
    """
    counts = df[column].value_counts().reset_index()
    counts.columns = ["valor", "frecuencia"]
    fig = px.bar(
        counts,
        x="valor",
        y="frecuencia",
        title=f"Frecuencia de {column}",
        color_discrete_sequence=["#00CC96"],
    )
    return fig


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def plot_elbow(k_range: list[int], inertias: list[float]) -> go.Figure:
    """Gráfico del codo para seleccionar k óptimo en K-Means.

    Args:
        k_range: Lista de valores de k.
        inertias: Inercia correspondiente a cada k.

    Returns:
        Figura Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_range,
        y=inertias,
        mode="lines+markers",
        name="Inercia",
        marker=dict(size=8, color="#636EFA"),
        line=dict(width=2),
    ))
    fig.update_layout(
        title="Método del Codo — K-Means",
        xaxis_title="Número de Clústeres (k)",
        yaxis_title="Inercia (WCSS)",
        xaxis=dict(tickmode="linear"),
    )
    return fig


def plot_silhouette_scores(k_range: list[int], scores: list[float]) -> go.Figure:
    """Gráfico de puntuaciones Silhouette por k.

    Args:
        k_range: Lista de valores de k.
        scores: Silhouette score por k.

    Returns:
        Figura Plotly.
    """
    best_k = k_range[int(np.argmax(scores))]
    colors = ["#EF553B" if k == best_k else "#00CC96" for k in k_range]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=k_range,
        y=scores,
        marker_color=colors,
        name="Silhouette",
    ))
    fig.update_layout(
        title=f"Silhouette Score por k (mejor: k={best_k})",
        xaxis_title="Número de Clústeres (k)",
        yaxis_title="Silhouette Score",
        xaxis=dict(tickmode="linear"),
    )
    return fig


def plot_dendrogram(X: np.ndarray | pd.DataFrame, method: str = "ward") -> plt.Figure:
    """Dendrograma de clustering jerárquico.

    Usa Matplotlib ya que Plotly no tiene soporte nativo para dendrogramas.

    Args:
        X: Datos de entrada.
        method: Método de enlace ('ward', 'complete', 'average', 'single').

    Returns:
        Figura Matplotlib.
    """
    X_arr = np.array(X)
    # Limitar a 200 muestras para performance
    if len(X_arr) > 200:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_arr), 200, replace=False)
        X_arr = X_arr[idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    Z = linkage(X_arr, method=method)
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, leaf_rotation=90)
    ax.set_title(f"Dendrograma — Enlace: {method}")
    ax.set_xlabel("Muestra / Clúster")
    ax.set_ylabel("Distancia")
    plt.tight_layout()
    return fig


def plot_clusters_2d(
    X: np.ndarray | pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
) -> go.Figure:
    """Visualización 2D de clústeres usando PCA.

    Args:
        X: Datos (originalmente n-dimensional).
        labels: Etiquetas de clúster por muestra.
        feature_names: Nombres de features (para tooltip).

    Returns:
        Figura Plotly.
    """
    X_arr = np.array(X)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_arr)
    var_exp = pca.explained_variance_ratio_

    df_plot = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "Clúster": labels.astype(str),
    })

    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="Clúster",
        title="Clústeres en espacio PCA 2D",
        labels={
            "PC1": f"PC1 ({var_exp[0]*100:.1f}% varianza)",
            "PC2": f"PC2 ({var_exp[1]*100:.1f}% varianza)",
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    return fig


def plot_cluster_profiles(profile_df: pd.DataFrame, feature_cols: list[str]) -> go.Figure:
    """Gráfico de radar o barras agrupadas con perfiles de clústeres.

    Args:
        profile_df: DataFrame de cluster_profiles().
        feature_cols: Features a graficar.

    Returns:
        Figura Plotly.
    """
    mean_cols = [f"{c}_mean" for c in feature_cols if f"{c}_mean" in profile_df.columns]
    if not mean_cols:
        return go.Figure()

    fig = go.Figure()
    for _, row in profile_df.iterrows():
        cluster_id = int(row["cluster"])
        values = [row.get(mc, 0) for mc in mean_cols]
        fig.add_trace(go.Bar(
            name=f"Clúster {cluster_id}",
            x=mean_cols,
            y=values,
        ))

    fig.update_layout(
        barmode="group",
        title="Perfil de Clústeres (Media por Feature)",
        xaxis_tickangle=-45,
        xaxis_title="Feature",
        yaxis_title="Valor Medio",
    )
    return fig


# ---------------------------------------------------------------------------
# Clasificación
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str = "Matriz de Confusión",
) -> go.Figure:
    """Heatmap de la matriz de confusión.

    Args:
        cm: Matriz de confusión (numpy array).
        labels: Nombres de las clases.
        title: Título del gráfico.

    Returns:
        Figura Plotly.
    """
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"Real: {l}" for l in labels],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicción",
        yaxis_title="Valor Real",
        height=400,
    )
    return fig


def plot_roc_curves(
    models_dict: dict,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> go.Figure:
    """Curvas ROC superpuestas para múltiples modelos.

    Solo funciona para clasificación binaria.

    Args:
        models_dict: Dict {nombre_modelo: modelo_entrenado}.
        X_test: Features de test.
        y_test: Etiquetas verdaderas de test.

    Returns:
        Figura Plotly con todas las curvas ROC.
    """
    from sklearn.metrics import roc_curve, auc

    y_arr = np.array(y_test)
    classes = np.unique(y_arr)

    fig = go.Figure()

    # Línea de referencia (clasificador aleatorio)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Aleatorio (AUC=0.50)",
    ))

    if len(classes) != 2:
        fig.update_layout(title="ROC no disponible para multiclase en esta vista")
        return fig

    colors = px.colors.qualitative.Bold
    for i, (model_name, model) in enumerate(models_dict.items()):
        if not hasattr(model, "predict_proba"):
            continue
        try:
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_arr, proba)
            auc_score = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{model_name} (AUC={auc_score:.3f})",
                line=dict(width=2, color=colors[i % len(colors)]),
            ))
        except Exception:
            continue

    fig.update_layout(
        title="Curvas ROC — Comparación de Modelos",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.6, y=0.1),
    )
    return fig


def plot_feature_importance(
    model: object,
    feature_names: list[str],
    top_n: int = 20,
) -> go.Figure:
    """Gráfico de importancia de variables.

    Args:
        model: Modelo con atributo feature_importances_.
        feature_names: Nombres de las features.
        top_n: Número de features a mostrar.

    Returns:
        Figura Plotly.
    """
    if not hasattr(model, "feature_importances_"):
        return go.Figure()

    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=True).tail(top_n)

    fig = px.bar(
        df_imp,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Importancia de Variables (top {top_n})",
        color="importance",
        color_continuous_scale="Viridis",
        labels={"importance": "Importancia", "feature": "Variable"},
    )
    fig.update_layout(showlegend=False, yaxis=dict(automargin=True))
    return fig


def plot_decision_tree(
    model: object,
    feature_names: list[str],
    class_names: list[str] | None = None,
    max_depth: int = 3,
) -> plt.Figure:
    """Visualización gráfica del árbol de decisión.

    Usa Matplotlib/sklearn.tree.plot_tree.

    Args:
        model: Modelo DecisionTreeClassifier entrenado.
        feature_names: Nombres de las features.
        class_names: Nombres de las clases.
        max_depth: Profundidad máxima a visualizar.

    Returns:
        Figura Matplotlib.
    """
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        ax=ax,
        fontsize=9,
    )
    ax.set_title("Árbol de Decisión (profundidad máx. visualizada: " + str(max_depth) + ")")
    plt.tight_layout()
    return fig


def plot_metrics_comparison(experiments_df: pd.DataFrame) -> go.Figure:
    """Gráfico de barras agrupadas comparando métricas entre experimentos.

    Args:
        experiments_df: DataFrame del historial de experimentos.

    Returns:
        Figura Plotly.
    """
    metric_cols = [c for c in ["accuracy", "f1", "precision", "recall", "auc"]
                   if c in experiments_df.columns]
    if not metric_cols or "model" not in experiments_df.columns:
        return go.Figure()

    fig = go.Figure()
    colors = px.colors.qualitative.Bold
    for i, metric in enumerate(metric_cols):
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=experiments_df["model"],
            y=experiments_df[metric],
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        barmode="group",
        title="Comparación de Métricas entre Modelos",
        xaxis_title="Modelo",
        yaxis_title="Valor",
        yaxis=dict(range=[0, 1.05]),
        xaxis_tickangle=-30,
    )
    return fig
