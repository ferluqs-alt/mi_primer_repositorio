import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Título de la app
# --- Estimador de precios de viviendas 
#  st.title("DATA SET: CALIFORNIA HOUSING DATASET")
st.title("ESTIMADOR DE PRECIO DE VIVIENDAS")
# Cargar datos (forzando la descarga si no existe en cache)
data = fetch_california_housing(as_frame=True, download_if_missing=True)
df = data.frame
st.subheader("DATA SET: CALIFORNIA HOUSING DATASET")
# Mostrar estadísticas básicas
st.subheader("Estadísticas descriptivas para determinar tendencias")
st.write(df.describe())

# Mostrar primeras filas
st.subheader("Primeras filas del dataset")
st.write(df.head())
