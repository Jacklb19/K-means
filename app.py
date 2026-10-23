from pathlib import Path
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Segmentación de clientes", page_icon="📊", layout="wide")
ROOT = Path(__file__).resolve().parent
AUTHOR = "Jose Luis Burbano Buchelly · Estudiante"
COLAB_URL = "https://colab.research.google.com/github/Jacklb19/K-means/blob/main/Taller_Final_KMeans.ipynb"
PALETTE = ["#397e85", "#345fba", "#c27b24", "#84569d", "#b85858"]

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2.1rem; padding-bottom: 2rem;}
h1 {font-size: 2.15rem !important; letter-spacing: -.045em;}
h2, h3 {letter-spacing: -.025em;}
[data-testid="stTabs"] [role="tablist"] {gap: 1.2rem; margin-bottom: 1rem;}
[data-testid="stTabs"] [role="tab"] {white-space: nowrap;}
[data-testid="stForm"] {border: 0; padding: 0;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
[data-testid="stCaptionContainer"] {line-height: 1.5;}
@media (max-width: 640px) {
  .block-container {padding: 1.2rem 1rem 2rem;}
  h1 {font-size: 1.75rem !important;}
  [data-testid="stTabs"] [role="tablist"] {gap: .7rem;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    pipeline = joblib.load(ROOT / "modelo_kmeans.joblib")
    features = metadata["features"]
    if list(pipeline.feature_names_in_) != features:
        raise ValueError("Las variables del modelo no coinciden con los datos.")
    data = pd.read_csv(ROOT / "Mall_Customers.csv")
    data["Cluster"] = pipeline.predict(data[features])
    names = {int(k): value for k, value in metadata["profiles"].items()}
    data["Perfil"] = data["Cluster"].map(names)
    data["Ingreso anual (USD)"] = data[features[0]] * 1000
    data["Puntuación de gasto"] = data[features[1]]
    centers = pipeline["scaler"].inverse_transform(pipeline["kmeans"].cluster_centers_)
    return data, pipeline, metadata, names, centers, pd.read_csv(ROOT / "perfiles.csv"), pd.read_csv(ROOT / "metricas_k.csv")


try:
    data, pipeline, metadata, names, centers, summary, metrics = load_assets()
except (OSError, ValueError, KeyError) as exc:
    st.error(f"No se pudieron cargar los datos de la aplicación: {exc}")
    st.stop()

features = metadata["features"]
profile_names = list(names.values())
colors = alt.Scale(domain=profile_names, range=PALETTE)
descriptions = {
    "Promedio": "Este grupo tiene ingresos y una puntuación de gasto cercanos al promedio de la muestra.",
    "Alto valor relativo": "Este grupo combina ingresos altos con una puntuación de gasto alta.",
    "Gasto alto e ingreso bajo": "Este grupo tiene ingresos bajos dentro de la muestra, pero una puntuación de gasto alta.",
    "Potencial de crecimiento": "Este grupo tiene ingresos altos, pero una puntuación de gasto baja. Puede ser una oportunidad para nuevas campañas.",
    "Valor actual bajo": "Este grupo presenta ingresos y una puntuación de gasto bajos dentro de la muestra.",
}
examples = {
    "Sin ejemplo": None,
    "Ingreso y gasto medios": (60000.0, 50),
    "Ingreso alto y gasto alto": (90000.0, 85),
    "Ingreso alto y gasto bajo": (90000.0, 15),
    "Ingreso bajo y gasto alto": (25000.0, 85),
    "Ingreso bajo y gasto bajo": (25000.0, 15),
}
for key, value in {"income_usd": 60000.0, "spending_score": 50, "prediction": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def load_example():
    example = examples[st.session_state["example"]]
    if example:
        st.session_state["income_usd"], st.session_state["spending_score"] = example
    st.session_state["prediction"] = None


def clear_prediction():
    st.session_state["prediction"] = None
    st.session_state["example"] = "Sin ejemplo"


def reset_filters():
    st.session_state["customer_search"] = ""
    st.session_state["data_profiles"] = []


def style_profiles(frame):
    backgrounds = ["#e8f4f4", "#ebf1fc", "#fff3df", "#f4edf7", "#fceeee"]
    foregrounds = ["#24616a", "#284e93", "#875013", "#704185", "#933f3f"]
    styles = {name: f"background-color: {background}; color: {color}; font-weight: 600;"
              for name, background, color in zip(profile_names, backgrounds, foregrounds)}
    return frame.style.map(lambda value: styles.get(value, ""), subset=["Perfil"])


def group_chart(customer=None, focus=None, show_centers=False):
    chart_data = data.copy()
    chart_data["Detalle"] = "Cliente " + chart_data["CustomerID"].astype(str)
    chart_data["Opacidad"] = np.where(chart_data["Cluster"].eq(focus), 0.95, 0.15) if focus is not None else 0.75
    points = alt.Chart(chart_data).mark_circle(size=65).encode(
        x=alt.X("Ingreso anual (USD):Q", title="Ingreso anual (USD)", axis=alt.Axis(format=",.0f", tickCount=5), scale=alt.Scale(zero=True)),
        y=alt.Y("Puntuación de gasto:Q", scale=alt.Scale(domain=[0, 103])),
        color=alt.Color("Perfil:N", scale=colors, legend=alt.Legend(orient="bottom", columns=1, title=None, labelLimit=300)),
        opacity=alt.Opacity("Opacidad:Q", scale=None, legend=None),
        tooltip=["Detalle:N", "Perfil:N", alt.Tooltip("Ingreso anual (USD):Q", format=",.0f"), "Puntuación de gasto:Q"],
    )
    layers = [points]
    if show_centers:
        center_data = pd.DataFrame({"Ingreso anual (USD)": centers[:, 0] * 1000,
                                    "Puntuación de gasto": centers[:, 1], "Perfil": profile_names})
        layers.append(alt.Chart(center_data).mark_point(shape="cross", color="#152e36", size=160, strokeWidth=2).encode(
            x="Ingreso anual (USD):Q", y="Puntuación de gasto:Q", tooltip=["Perfil:N"]))
    if customer:
        point = pd.DataFrame({"Ingreso anual (USD)": [customer["income"]], "Puntuación de gasto": [customer["score"]], "Detalle": ["Nuevo cliente"]})
        layers.append(alt.Chart(point).mark_point(shape="diamond", filled=True, color="#ee6a37",
                                                stroke="#142e36", strokeWidth=2, size=250).encode(
            x="Ingreso anual (USD):Q", y="Puntuación de gasto:Q",
            tooltip=["Detalle:N", alt.Tooltip("Ingreso anual (USD):Q", format=",.0f"), "Puntuación de gasto:Q"]))
    return alt.layer(*layers).properties(height=460 if customer else 420).interactive().configure_view(stroke=None)


st.title("Segmentación de clientes")
st.write("Clasifica un cliente según su ingreso anual y su puntuación de gasto.")
st.caption(AUTHOR)
tab_new, tab_groups, tab_k, tab_data = st.tabs(
    ["Clasificar cliente", "Explorar grupos", "¿Por qué 5 grupos?", "Consultar datos"]
)

with tab_new:
    left, right = st.columns([1, 1.15], gap="large")
    with left, st.container(border=True):
        st.subheader("1. Datos del cliente")
        income = st.number_input("Ingreso anual (USD)", min_value=0.0, step=1000.0, format="%.0f", key="income_usd",
                                 on_change=clear_prediction,
                                 help="Escribe el ingreso anual completo en dólares. Ejemplo: 60000 equivale a 60 mil dólares.")
        st.caption(f"Ingreso que se analizará: USD {income:,.0f} al año.")
        score = st.number_input("Puntuación de gasto (1 a 100)", min_value=1, max_value=100, step=1,
                                key="spending_score", on_change=clear_prediction,
                                help="Usa el Spending Score del dataset. Es un índice de gasto, no un importe de compra.")
        st.progress(score / 100, text=f"{score} de 100 · 1 = poco gasto; 100 = mucho gasto")
        submitted = st.button("Ver perfil del cliente", type="primary", use_container_width=True)
        if submitted:
            if np.isfinite([income, score]).all():
                customer = pd.DataFrame([[income / 1000, score]], columns=features)
                st.session_state["prediction"] = {"income": income, "score": score, "cluster": int(pipeline.predict(customer)[0])}
            else:
                st.error("Ingresa valores numéricos válidos.")
        st.selectbox("Cargar un ejemplo (opcional)", list(examples), key="example", on_change=load_example)
    with right, st.container(border=True):
        st.subheader("2. Perfil del cliente")
        prediction = st.session_state["prediction"]
        if prediction is None:
            st.info("Ingresa los dos datos y pulsa «Ver perfil del cliente».")
            st.write("Aquí verás el grupo asignado, qué lo caracteriza y una acción de marketing sugerida.")
            st.caption("Puedes empezar con uno de los ejemplos y cambiar sus valores.")
        else:
            cluster = prediction["cluster"]
            profile = names[cluster]
            group = summary.loc[summary["Cluster"].eq(cluster)].iloc[0]
            st.success(profile)
            st.write(descriptions[profile])
            st.caption(f"Última clasificación: USD {prediction['income']:,.0f} al año · puntuación {prediction['score']}/100.")
            st.markdown("**Acción sugerida**")
            st.write(metadata["strategies"][str(cluster)])
            st.caption(f"Grupo {cluster} · {int(group['Clientes'])} clientes de la muestra ({group['Porcentaje']:.1f}%).")
            if not data["Ingreso anual (USD)"].min() <= prediction["income"] <= data["Ingreso anual (USD)"].max() or not data[features[1]].min() <= prediction["score"] <= data[features[1]].max():
                st.warning("Este cliente tiene valores fuera del rango de los datos utilizados. Revisa la asignación con cuidado.")
    if prediction is not None:
        with st.expander("Ver al cliente dentro de los grupos", expanded=True):
            st.caption("El rombo naranja representa al nuevo cliente. Acércate a un punto para consultar sus datos.")
            st.altair_chart(group_chart(customer=prediction), use_container_width=True)

with tab_groups:
    st.subheader("Conoce los cinco perfiles")
    a, b, c = st.columns(3)
    a.metric("Clientes analizados", len(data))
    b.metric("Grupos", metadata["selected_k"])
    c.metric("Silueta promedio", f"{metadata['silhouette']:.3f}", help="Mide qué tan separados están los grupos. Cuanto más cerca de 1, mejor separación.")
    focus = st.selectbox("¿Qué grupo quieres explorar?", [None, *names],
                         format_func=lambda value: "Todos los grupos" if value is None else names[value], key="focus")
    show_centers = st.checkbox("Mostrar el centro de cada grupo", help="La cruz marca el ingreso y la puntuación promedio de cada grupo.")
    st.altair_chart(group_chart(focus=focus, show_centers=show_centers), use_container_width=True)
    st.caption("Cada punto es un cliente. Acércate a un punto para ver sus datos; arrastra o usa la rueda para acercar la gráfica.")
    if focus is not None:
        st.markdown(f"**{names[focus]}**")
        st.write(descriptions[names[focus]])
        st.write(metadata["strategies"][str(focus)])
    group_table = summary if focus is None else summary.loc[summary["Cluster"].eq(focus)]
    group_table = group_table.assign(Ingreso_promedio_USD=group_table["Ingreso_medio"] * 1000).rename(columns={
        "Porcentaje": "% de clientes", "Ingreso_promedio_USD": "Ingreso promedio (USD)", "Score_medio": "Puntuación promedio"})
    st.markdown("**Resumen por perfil**")
    st.dataframe(style_profiles(group_table[["Perfil", "Clientes", "% de clientes", "Ingreso promedio (USD)", "Puntuación promedio"]]),
                 hide_index=True, width="stretch", row_height=44, column_config={
                     "Perfil": st.column_config.TextColumn("Perfil de cliente", width="large"),
                     "Clientes": st.column_config.NumberColumn("Clientes", format="%d", width="small"),
                     "% de clientes": st.column_config.ProgressColumn("% de la muestra", min_value=0, max_value=100,
                                                                        format="%.1f%%", width="medium"),
                     "Ingreso promedio (USD)": st.column_config.NumberColumn("Ingreso prom. (USD)", format="dollar", step=1,
                                                                              help="Ingreso anual promedio del perfil, en dólares."),
                     "Puntuación promedio": st.column_config.ProgressColumn("Gasto promedio", min_value=0, max_value=100,
                                                                             format="%.1f / 100", width="medium"),
                 })
    st.caption("Los perfiles describen el ingreso y el gasto de esta muestra; no indican las ganancias reales de cada cliente.")

with tab_k:
    st.subheader("Dos métodos coinciden en cinco grupos")
    st.write("El codo muestra que la mejora se reduce después de cinco grupos. La silueta alcanza su valor más alto con cinco.")
    left_k, right_k = st.columns(2, gap="large")
    for container, column, title, explanation in [
        (left_k, "Inercia", "Método del codo", "Se busca el cambio de pendiente: agregar más grupos aporta cada vez menos mejora."),
        (right_k, "Silueta", "Coeficiente de silueta", "Un valor mayor indica mejor separación. La mayor silueta es 0,555 con cinco grupos."),
    ]:
        with container:
            st.markdown(f"**{title}**")
            values = metrics.dropna(subset=[column])
            chart = alt.Chart(values).mark_line(point=True, color="#397e85").encode(
                x=alt.X("K:O", title="Número de grupos"), y=alt.Y(f"{column}:Q", scale=alt.Scale(zero=True)),
                tooltip=["K:O", alt.Tooltip(f"{column}:Q", format=".4f")])
            chosen = alt.Chart(values.loc[values["K"].eq(5)]).mark_point(size=130, filled=True, color="#ee6a37").encode(x="K:O", y=f"{column}:Q")
            st.altair_chart((chart + chosen).properties(height=250).configure_view(stroke=None), use_container_width=True)
            st.caption(explanation)
    with st.expander("Ver los valores calculados"):
        st.dataframe(metrics[["K", "Inercia", "Silueta"]].round(4), hide_index=True, width="stretch")
        st.caption("La silueta no se calcula para un solo grupo. No es un porcentaje de aciertos.")

with tab_data:
    st.subheader("Consulta los clientes de cada grupo")
    search_column, profiles_column, reset_column = st.columns([1, 2, 0.85], vertical_alignment="bottom")
    with search_column:
        query = st.text_input("Buscar cliente por número", placeholder="Ejemplo: 125", key="customer_search").strip().lstrip("#")
    with profiles_column:
        selected = st.multiselect("Filtrar por perfil", profile_names, default=[], placeholder="Todos los perfiles", key="data_profiles")
    with reset_column:
        st.button("Limpiar filtros", on_click=reset_filters, use_container_width=True)
    filtered = data.loc[data["Perfil"].isin(selected)] if selected else data
    if query:
        if query.isdecimal():
            filtered = filtered.loc[filtered["CustomerID"].eq(int(query))]
        else:
            filtered = filtered.iloc[0:0]
            st.warning("Busca usando un número de cliente, por ejemplo 125.")
    st.caption(f"{len(filtered)} de {len(data)} clientes · Ingresos anuales en USD · Gasto de 1 a 100")
    export = filtered.rename(columns={"CustomerID": "Cliente", "Gender": "Género", "Age": "Edad"})[
        ["Cliente", "Perfil", "Ingreso anual (USD)", "Puntuación de gasto", "Edad", "Género"]].copy()
    export["Género"] = export["Género"].replace({"Male": "Masculino", "Female": "Femenino"})
    if filtered.empty:
        st.info("No hay clientes que coincidan con estos filtros. Puedes cambiar la búsqueda o limpiar los filtros.")
    else:
        st.dataframe(style_profiles(export), hide_index=True, width="stretch", row_height=42,
                     height=min(500, 42 * len(export) + 44), column_config={
                         "Cliente": st.column_config.NumberColumn("Cliente", format="#%03d", pinned=True, width="small"),
                         "Perfil": st.column_config.TextColumn("Perfil de cliente", width="large"),
                         "Ingreso anual (USD)": st.column_config.NumberColumn("Ingreso anual (USD)", format="dollar", step=1, width="medium"),
                         "Puntuación de gasto": st.column_config.ProgressColumn("Puntuación de gasto", min_value=0, max_value=100,
                                                                               format="%d / 100", width="medium"),
                         "Edad": st.column_config.NumberColumn("Edad", format="%d años", width="small"),
                         "Género": st.column_config.TextColumn("Género", width="small"),
                     })
    st.download_button("Descargar estos clientes (.csv)", data=export.to_csv(index=False).encode("utf-8-sig"),
                       file_name="clientes_segmentados.csv", mime="text/csv", disabled=filtered.empty)

st.divider()
st.caption(f"{AUTHOR} · Taller final de Minería de Datos")
st.markdown(f"[Cuaderno en Colab]({COLAB_URL}) · [Dataset en Kaggle]({metadata['dataset_source']})")
