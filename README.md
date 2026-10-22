# Segmentación de clientes con K-Means

Taller final de Minería de Datos con el dataset Mall Customers.

[Abrir el cuaderno en Colab](https://colab.research.google.com/github/Jacklb19/K-means/blob/main/Taller_Final_KMeans.ipynb)

El cuaderno contiene la exploración del dataset, el escalado, el método del codo, la silueta, el modelo y las respuestas del taller. La aplicación de Streamlit está en un archivo aparte.

## Archivos principales

- `Taller_Final_KMeans.ipynb`: análisis y respuestas.
- `Mall_Customers.csv`: dataset.
- `app.py`: aplicación de Streamlit.
- `modelo_kmeans.joblib`: modelo con su escalador.
- `requirements.txt`: bibliotecas necesarias para la aplicación.

Se eligieron cinco grupos y se obtuvo una silueta de aproximadamente 0,555.

## Ejecutar la aplicación localmente

Con Python 3.12, abre una terminal en la carpeta del proyecto:

```text
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Desplegar en Streamlit

1. Entra en https://share.streamlit.io/ e inicia sesión con GitHub.
2. Pulsa Create app.
3. Selecciona el repositorio `Jacklb19/K-means`, la rama `main` y el archivo `app.py`.
4. En Advanced settings selecciona Python 3.12 y pulsa Deploy.
5. Cuando esté lista, prueba ingreso 90 y puntuación 85: debe quedar en Alto valor relativo.

No se necesitan claves de Kaggle. Si se parte del ZIP, se debe subir su contenido a la raíz del repositorio, conservando la carpeta `.streamlit`.

## Fuente

[Mall Customer Segmentation Data en Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

[Documentación de despliegue de Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
