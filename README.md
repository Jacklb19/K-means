# Segmentación de clientes con K-Means

**Jose Luis Burbano Buchelly**
Estudiante

Taller final de Minería de Datos. Se utiliza el dataset Mall Customers para identificar grupos de clientes según su ingreso anual y su puntuación de gasto.

## Aplicación

[Abrir la aplicación en Streamlit](https://k-means-ha6zhom8x3twc4epufwkvp.streamlit.app/)

La aplicación permite clasificar un cliente, explorar los cinco perfiles, consultar las gráficas del codo y de silueta y descargar los clientes filtrados. El ingreso se escribe en dólares completos; por ejemplo, `60000` corresponde a 60 mil dólares anuales.

## Cuaderno

[Abrir el taller en Google Colab](https://colab.research.google.com/github/Jacklb19/K-means/blob/main/Taller_Final_KMeans.ipynb)

Incluye la exploración del dataset, el escalado, la selección del número de grupos, el entrenamiento y las 14 respuestas del taller.

## Resultados

Se analizaron 200 clientes. El método del codo y el coeficiente de silueta respaldan cinco grupos. La silueta promedio fue de aproximadamente 0,555.

## Archivos principales

- `Taller_Final_KMeans.ipynb`: análisis y respuestas.
- `app.py`: aplicación de Streamlit.
- `Mall_Customers.csv`: dataset.
- `modelo_kmeans.joblib`: modelo guardado junto con su escalador.
- `metadata.json`, `metricas_k.csv` y `perfiles.csv`: información del modelo y resultados.
- `requirements.txt`: bibliotecas de la aplicación.

## Dataset

[Mall Customer Segmentation Data en Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
