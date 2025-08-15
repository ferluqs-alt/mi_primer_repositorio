import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Título de la app
st.title("Análisis del California Housing Dataset")

# Cargar datos (forzando la descarga si no existe en cache)
data = fetch_california_housing(as_frame=True, download_if_missing=True)
df = data.frame

# Mostrar estadísticas básicas
st.subheader("Estadísticas descriptivas")
st.write(df.describe())

# Mostrar primeras filas
st.subheader("Primeras filas del dataset")
st.write(df.head())
