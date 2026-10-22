from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Segmentación de clientes", page_icon="◈", layout="wide")
ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_assets():
    data_path = ROOT / "Mall_Customers.csv"
    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    data = pd.read_csv(data_path)
    pipeline = joblib.load(ROOT / "modelo_kmeans.joblib")
    features = metadata["features"]
    if list(pipeline.feature_names_in_) != features:
        raise ValueError("Las variables del modelo no coinciden con los metadatos.")
    data["Cluster"] = pipeline.predict(data[features])
    names = {int(k): v for k, v in metadata["profiles"].items()}
    data["Perfil"] = data["Cluster"].map(names)
    metrics = pd.read_csv(ROOT / "metricas_k.csv")
    centers = pipeline["scaler"].inverse_transform(pipeline["kmeans"].cluster_centers_)
    return data, pipeline, metadata, metrics, centers, names


def cluster_plot(data, centers, names, new_customer=None):
    fig, ax = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for cluster, name in sorted(names.items()):
        part = data.loc[data["Cluster"] == cluster]
        ax.scatter(part[features[0]], part[features[1]], s=46, alpha=0.8,
                   color=colors(cluster), label=f"{cluster}: {name}")
    ax.scatter(centers[:, 0], centers[:, 1], color="#182637", marker="X",
               s=180, linewidths=1, edgecolors="white", label="Centroides", zorder=4)
    if new_customer is not None:
        ax.scatter([new_customer[0]], [new_customer[1]], color="#e63946",
                   marker="*", s=360, edgecolors="black", linewidths=0.8,
                   label="Nuevo cliente", zorder=5)
    ax.set(xlabel="Ingreso anual (miles de dólares)", ylabel="Spending Score (1–100)",
           title="Segmentos por ingreso y puntuación de gasto", ylim=(-3, 104))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              fontsize=8, frameon=False)
    return fig


try:
    data, pipeline, metadata, metrics, centers, names = load_assets()
except (OSError, ValueError, KeyError) as exc:
    st.error(f"No se pudo cargar el proyecto: {exc}")
    st.stop()

features = metadata["features"]
selected_k = metadata["selected_k"]
st.title("Segmentación de clientes")
st.write("Explora cinco perfiles de Mall Customers y asigna un nuevo cliente con K-Means.")
a, b, c = st.columns(3)
a.metric("Clientes analizados", len(data))
b.metric("Segmentos", selected_k)
c.metric("Silueta media", f"{metadata['silhouette']:.3f}")
st.caption("Ingreso anual expresado en miles de dólares. Spending Score es una puntuación, no un importe de compra.")

tab_segments, tab_k, tab_new, tab_data = st.tabs(
    ["Segmentos", "Selección de K", "Nuevo cliente", "Datos"]
)

with tab_segments:
    fig = cluster_plot(data, centers, names)
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Perfil de cada grupo")
    summary = pd.read_csv(ROOT / "perfiles.csv")
    st.dataframe(summary.round(2), hide_index=True, width="stretch")
    chosen = st.selectbox("Consultar estrategia", options=sorted(names),
                          format_func=lambda k: f"{k}: {names[k]}")
    st.info(metadata["strategies"][str(chosen)])
    st.caption("Los perfiles se basan en el ingreso y la puntuación de gasto. No indican las ganancias reales de cada cliente.")

with tab_k:
    st.subheader("Comparación de 1 a 10 grupos")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(metrics["K"], metrics["Inercia"], "o-", color="#267789")
    axes[0].set(title="Método del codo", xlabel="K", ylabel="Inercia (WCSS)")
    valid = metrics.dropna(subset=["Silueta"])
    axes[1].plot(valid["K"], valid["Silueta"], "o-", color="#bb624b")
    axes[1].set(title="Coeficiente de silueta", xlabel="K", ylabel="Silueta media")
    for ax in axes:
        ax.axvline(selected_k, color="#182637", linestyle="--", label=f"K elegido = {selected_k}")
        ax.set_xticks(metrics["K"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        ax.grid(alpha=0.15)
    st.pyplot(fig)
    plt.close(fig)
    st.write(f"Se elige K = {selected_k}: coincide el codo estimado con la mayor silueta entre K = 2 y K = 10.")
    st.caption("La inercia disminuye al agregar grupos; minimizarla por sí sola no permite elegir K. La silueta no está definida para K = 1.")
    st.dataframe(metrics.round(4), hide_index=True, width="stretch")

with tab_new:
    st.subheader("Asignar un nuevo cliente")
    st.write("Se utiliza el mismo escalador y el mismo modelo entrenado en el cuaderno.")
    with st.form("new_customer"):
        income = st.number_input("Ingreso anual (miles de dólares)", min_value=0.0,
                                 value=60.0, step=1.0)
        spending = st.number_input("Spending Score (1–100)", min_value=1.0,
                                   max_value=100.0, value=50.0, step=1.0)
        submitted = st.form_submit_button("Clasificar cliente", type="primary")
    if submitted:
        if not np.isfinite([income, spending]).all():
            st.error("Ingresa valores numéricos finitos.")
            st.stop()
        customer = pd.DataFrame([[income, spending]], columns=features)
        cluster = int(pipeline.predict(customer)[0])
        distance = float(pipeline.transform(customer)[0, cluster])
        st.success(f"Cluster {cluster}: {names[cluster]}")
        st.write(metadata["strategies"][str(cluster)])
        outside = [feature for feature in features
                   if not data[feature].min() <= customer.iloc[0][feature] <= data[feature].max()]
        if outside:
            st.warning("El cliente tiene valores fuera del rango de los datos utilizados. Revisa la asignación con cuidado.")
        st.caption(f"Distancia al centroide en el espacio estandarizado: {distance:.3f}. Una distancia menor indica mayor cercanía al centro del grupo; no es un porcentaje de acierto.")
        fig = cluster_plot(data, centers, names, (income, spending))
        st.pyplot(fig)
        plt.close(fig)

with tab_data:
    st.subheader("Clientes y asignaciones")
    selected = st.multiselect("Filtrar segmentos", options=sorted(names), default=sorted(names),
                             format_func=lambda k: names[k])
    filtered = data.loc[data["Cluster"].isin(selected)]
    st.dataframe(filtered, hide_index=True, width="stretch")
    st.download_button("Descargar clientes filtrados", data=filtered.to_csv(index=False).encode("utf-8-sig"),
                       file_name="clientes_segmentados.csv", mime="text/csv")
    st.markdown("[Fuente del dataset: Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)")
    st.caption("Resultados basados en 200 clientes. Las estrategias se deben comprobar con datos de compras y campañas.")
