import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Título de la app
# --- Estimador de precios de viviendas 
#  st.title("DATA SET: CALIFORNIA HOUSING DATASET")
st.title("ESTIMADOR DE PRECIO DE VIVIENDAS")
##################################################################################
with st.sidebar:
    st.header("Opciones")
    uploaded_file = st.file_uploader("Cargar archivo", type=["csv", "xlsx"])
    if st.button("Análisis Exploratorio"):
        st.write("Realizando análisis exploratorio...")
    if st.button("Normalización"):
        st.write("Aplicando normalización...")
    if st.button("Otros Análisis"):
        st.write("Procesando otros análisis...")
#####################################################################################
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
