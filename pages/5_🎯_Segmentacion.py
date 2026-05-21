"""Página 5 — Segmentación: K-Means y clustering jerárquico."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.clustering import (
    cluster_profiles,
    compute_silhouette,
    find_optimal_k,
    interpret_clusters,
    run_hierarchical,
    run_kmeans,
)
from src.visualizer import (
    plot_cluster_profiles,
    plot_clusters_2d,
    plot_dendrogram,
    plot_elbow,
    plot_silhouette_scores,
)

st.set_page_config(page_title="Segmentación — CustomerIQ", layout="wide")

st.title("Segmentación de Clientes")

if st.session_state.get("df_raw") is None:
    st.warning("Primero debes cargar un dataset en la página **1 — Cargar Datos**.")
    st.stop()

# Usar df_clean si existe, si no df_raw
df_source = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw

# Features numéricas disponibles para clustering
num_cols = df_source.select_dtypes(include="number").columns.tolist()
if not num_cols:
    st.error("No hay columnas numéricas en el dataset. El clustering requiere features numéricas.")
    st.stop()

# ---------------------------------------------------------------------------
# Selector de features
# ---------------------------------------------------------------------------
st.markdown("## Selección de features para clustering")
selected_features = st.multiselect(
    "Features a usar (numéricas)",
    options=num_cols,
    default=num_cols,
    help="Selecciona las columnas numéricas que el algoritmo usará para agrupar clientes.",
)

if not selected_features:
    st.error("Selecciona al menos 2 features.")
    st.stop()

from sklearn.preprocessing import StandardScaler

X_cluster = df_source[selected_features].dropna()
scaler_cl = StandardScaler()
X_scaled = scaler_cl.fit_transform(X_cluster)

st.info(f"Clustering sobre **{len(X_cluster):,}** muestras × **{len(selected_features)}** features.")

# ---------------------------------------------------------------------------
# K-MEANS
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## K-Means")

with st.spinner("Calculando métricas para todos los valores de k..."):
    k_min = st.slider("k mínimo", 2, 5, 2)
    k_max = st.slider("k máximo", 3, 15, 8)
    if k_min >= k_max:
        st.error("k mínimo debe ser menor que k máximo.")
        st.stop()

    metrics_df = find_optimal_k(X_scaled, k_range=(k_min, k_max))

col_elbow, col_sil = st.columns(2)
with col_elbow:
    st.plotly_chart(
        plot_elbow(metrics_df["k"].tolist(), metrics_df["inertia"].tolist()),
        use_container_width=True,
    )
with col_sil:
    st.plotly_chart(
        plot_silhouette_scores(
            metrics_df["k"].tolist(),
            metrics_df["silhouette_score"].tolist(),
        ),
        use_container_width=True,
    )

best_k = int(metrics_df.loc[metrics_df["silhouette_score"].idxmax(), "k"])
k_final = st.slider(
    "Número de clústeres (k final)",
    min_value=k_min,
    max_value=k_max,
    value=best_k,
    help=f"El k sugerido por Silhouette es {best_k}.",
)

if st.button("Ejecutar K-Means", type="primary"):
    with st.spinner("Ejecutando K-Means..."):
        model_km, labels_km, inertia_km = run_kmeans(X_scaled, k=k_final)
        sil_global, sil_samples = compute_silhouette(X_scaled, labels_km)
        profiles_km = cluster_profiles(X_cluster, labels_km)
        interpretations_km = interpret_clusters(profiles_km, selected_features)

        st.session_state.clustering_results["kmeans"] = {
            "model": model_km,
            "labels": labels_km,
            "silhouette": sil_global,
            "k": k_final,
            "profiles": profiles_km,
            "interpretations": interpretations_km,
        }

    st.success(f"K-Means completado — Silhouette Score: **{sil_global:.4f}**")

    # Visualización 2D
    st.plotly_chart(
        plot_clusters_2d(X_scaled, labels_km, feature_names=selected_features),
        use_container_width=True,
    )

    # Perfiles
    st.markdown("### Perfiles por clúster")
    st.dataframe(profiles_km, use_container_width=True)

    st.plotly_chart(
        plot_cluster_profiles(profiles_km, selected_features),
        use_container_width=True,
    )

    # Interpretaciones automáticas
    st.markdown("### Interpretación automática")
    for cluster_id, desc in interpretations_km.items():
        st.markdown(f"- **{desc}**")

elif "kmeans" in st.session_state.clustering_results:
    res = st.session_state.clustering_results["kmeans"]
    st.info(
        f"K-Means ya ejecutado con k={res['k']} — Silhouette: {res['silhouette']:.4f}. "
        "Modifica k y haz clic en 'Ejecutar K-Means' para recalcular."
    )
    st.plotly_chart(
        plot_clusters_2d(X_scaled, res["labels"], feature_names=selected_features),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# CLUSTERING JERÁRQUICO
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Clustering Jerárquico")

col_link, col_nclus = st.columns(2)
with col_link:
    linkage_method = st.selectbox(
        "Método de enlace",
        ["ward", "complete", "average", "single"],
        help="'ward' minimiza la varianza dentro de los clústeres (recomendado).",
    )
with col_nclus:
    n_clusters_hier = st.slider("Número de clústeres", 2, 10, k_final)

st.markdown("### Dendrograma")
with st.spinner("Generando dendrograma..."):
    fig_dend = plot_dendrogram(X_scaled, method=linkage_method)
    st.pyplot(fig_dend, use_container_width=True)

if st.button("Ejecutar Clustering Jerárquico", type="primary"):
    with st.spinner("Ejecutando clustering jerárquico..."):
        model_hier, labels_hier = run_hierarchical(
            X_scaled, n_clusters=n_clusters_hier, linkage=linkage_method
        )
        sil_hier, _ = compute_silhouette(X_scaled, labels_hier)
        profiles_hier = cluster_profiles(X_cluster, labels_hier)
        interpretations_hier = interpret_clusters(profiles_hier, selected_features)

        st.session_state.clustering_results["hierarchical"] = {
            "model": model_hier,
            "labels": labels_hier,
            "silhouette": sil_hier,
            "n_clusters": n_clusters_hier,
            "linkage": linkage_method,
            "profiles": profiles_hier,
            "interpretations": interpretations_hier,
        }

    st.success(f"Clustering jerárquico completado — Silhouette Score: **{sil_hier:.4f}**")

    st.plotly_chart(
        plot_clusters_2d(X_scaled, labels_hier, feature_names=selected_features),
        use_container_width=True,
    )

    st.markdown("### Perfiles por clúster")
    st.dataframe(profiles_hier, use_container_width=True)

    st.markdown("### Interpretación automática")
    for cluster_id, desc in interpretations_hier.items():
        st.markdown(f"- **{desc}**")

elif "hierarchical" in st.session_state.clustering_results:
    res = st.session_state.clustering_results["hierarchical"]
    st.info(
        f"Clustering jerárquico ya ejecutado con {res['n_clusters']} clústeres "
        f"— Silhouette: {res['silhouette']:.4f}."
    )
    st.plotly_chart(
        plot_clusters_2d(X_scaled, res["labels"], feature_names=selected_features),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Comparativa de algoritmos
# ---------------------------------------------------------------------------
if (
    "kmeans" in st.session_state.clustering_results
    and "hierarchical" in st.session_state.clustering_results
):
    st.markdown("---")
    st.markdown("## Comparativa de algoritmos de clustering")

    km = st.session_state.clustering_results["kmeans"]
    hier = st.session_state.clustering_results["hierarchical"]

    comp_data = pd.DataFrame([
        {
            "Algoritmo": "K-Means",
            "Clústeres": km["k"],
            "Silhouette Score": round(km["silhouette"], 4),
        },
        {
            "Algoritmo": "Jerárquico",
            "Clústeres": hier["n_clusters"],
            "Silhouette Score": round(hier["silhouette"], 4),
        },
    ])
    st.dataframe(comp_data, use_container_width=True, hide_index=True)

    mejor = "K-Means" if km["silhouette"] >= hier["silhouette"] else "Jerárquico"
    st.info(
        f"El algoritmo con mayor Silhouette Score es **{mejor}**. "
        "Un Silhouette cercano a 1 indica clústeres bien definidos y separados."
    )
