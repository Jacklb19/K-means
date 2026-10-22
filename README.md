# Segmentación de clientes con K-Means

Trabajo del taller final de Minería de Datos.

[Abrir en Colab](https://colab.research.google.com/github/Jacklb19/K-means/blob/main/Taller_Final_KMeans.ipynb)

Modelo: 200 clientes, K=5, silueta=0.5547.
Variables: ingreso anual en miles de dólares y Spending Score (1–100).

## Abrir el cuaderno en Colab

1. Entra en https://colab.research.google.com/ y elige Archivo > Subir cuaderno.
2. Sube Taller_Final_KMeans.ipynb.
3. Usa un entorno nuevo de Python 3 con CPU y ejecuta todas las celdas en orden.
4. La última celda descarga de nuevo este paquete con el modelo de esa ejecución.

## Desplegar en Streamlit Community Cloud

1. Abre el repositorio https://github.com/Jacklb19/K-means.
2. Los archivos de la aplicación están en la raíz. Si partes del ZIP, descomprímelo y sube su contenido, conservando .streamlit/config.toml. El archivo app.py debe quedar en la raíz junto a requirements.txt, Mall_Customers.csv, modelo_kmeans.joblib, metadata.json, metricas_k.csv y perfiles.csv. clientes_segmentados.csv sirve como resultado adicional.
3. Entra en https://share.streamlit.io/ e inicia sesión con GitHub. Conecta tu cuenta si se solicita.
4. Selecciona Create app y elige la opción de una aplicación existente.
5. Indica el repositorio Jacklb19/K-means, la rama main y app.py como archivo principal.
6. En Advanced settings selecciona Python 3.12. No hacen falta secretos ni credenciales de Kaggle.
7. Pulsa Deploy. Cuando termine la instalación, abre el enlace generado.
8. En Nuevo cliente prueba ingreso 90 y score 85: debe asignarse a Alto valor relativo. Prueba también 90 y 15: debe asignarse a Potencial de crecimiento.

Si vuelves a entrenar, reemplaza juntos el modelo, el CSV, metadata.json, metricas_k.csv, perfiles.csv y requirements.txt con los archivos de la nueva exportación.

## Ejecutar localmente

Con Python 3.12, abre una terminal en la carpeta extraída y ejecuta:

```text
python -m venv .venv
```

Activa el entorno: en Windows usa `.venv\Scripts\activate`; en macOS/Linux usa `source .venv/bin/activate`.

```text
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abre la dirección local que indique Streamlit.

## Interpretación

Los grupos se basan en el ingreso y la puntuación de gasto. El score no es un importe de compra, por lo que los perfiles no indican las ganancias reales de cada cliente. La aplicación utiliza el mismo modelo y escalador del cuaderno y avisa si un cliente tiene valores fuera del rango de los datos.

Fuente de datos: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Documentación de despliegue: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

Dependencias: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
