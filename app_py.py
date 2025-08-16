import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Configuración inicial
st.set_page_config(layout="wide", page_title="Estimador de Precios de Viviendas")

# Diccionario de traducciones
tr = {
    "title": "ESTIMADOR DE PRECIOS DE VIVIENDAS",
    "upload": "Sube tu archivo CSV",
    "file_limit": "Límite: 200MB por archivo • CSV",
    "quick_preview": "Vista Previa Rápida",
    "null_analysis": "Análisis de Datos Nulos",
    "duplicates_analysis": "Análisis de Duplicados",
    "outliers_analysis": "Análisis de Outliers",
    "treatment_options": "Opciones de Tratamiento",
    "apply": "Aplicar",
    "reset": "Resetear",
    "report": "Reporte EDA",
    "language": "Idioma"
}

# Sidebar para idioma y selección de análisis
with st.sidebar:
    st.header("Language / idioma / Langue")
    language = st.radio("", ["ESPAÑOL", "ENGLISH", "FRANÇAIS"], label_visibility="collapsed")
    
    st.header("Depuración de dataset")
    analysis_option = st.radio("", [
        tr["null_analysis"],
        tr["duplicates_analysis"],
        tr["outliers_analysis"]
    ])

# Función para cargar datos
@st.cache_data
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

# Función para análisis EDA
def generate_eda_report(df):
    report = StringIO()
    
    # Información básica
    report.write("=== REPORTE DE ANÁLISIS EXPLORATORIO ===\n\n")
    report.write("1. INFORMACIÓN BÁSICA:\n")
    df.info(buf=report)
    report.write("\n\n")
    
    # Estadísticas descriptivas
    report.write("2. ESTADÍSTICAS DESCRIPTIVAS:\n")
    report.write(df.describe().to_string())
    report.write("\n\n")
    
    # Datos nulos
    report.write("3. DATOS NULOS:\n")
    report.write(df.isnull().sum().to_string())
    report.write("\n\n")
    
    # Duplicados
    report.write(f"4. FILAS DUPLICADAS: {df.duplicated().sum()}\n\n")
    
    # Tipos de datos
    report.write("5. TIPOS DE DATOS:\n")
    report.write(df.dtypes.to_string())
    
    return report.getvalue()

# Interfaz principal
st.title(tr["title"])

# Carga de archivo
uploaded_file = st.file_uploader(tr["upload"], type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"**Archivo cargado:** {uploaded_file.name}")
        st.success(f"**Tamaño:** {uploaded_file.size / 1024:.2f} KB")
        
        # Vista previa rápida
        st.subheader(tr["quick_preview"])
        st.dataframe(df.head())
        
        # Análisis seleccionado
        if analysis_option == tr["null_analysis"]:
            st.subheader(tr["null_analysis"])
            
            # Mostrar datos nulos
            null_counts = df.isnull().sum()
            st.write("**Valores nulos por columna:**")
            st.write(null_counts)
            
            # Gráfico de valores nulos
            fig, ax = plt.subplots()
            sns.heatmap(df.isnull(), cbar=False, ax=ax)
            st.pyplot(fig)
            
            # Opciones de tratamiento
            st.subheader(tr["treatment_options"])
            treatment = st.selectbox("Seleccione tratamiento:", [
                "Eliminar filas con nulos",
                "Rellenar con media (numéricos)",
                "Rellenar con mediana (numéricos)",
                "Rellenar con valor específico"
            ])
            
            if treatment == "Rellenar con valor específico":
                fill_value = st.text_input("Valor de relleno:")
            
            if st.button(tr["apply"]):
                if treatment == "Eliminar filas con nulos":
                    df = df.dropna()
                elif treatment == "Rellenar con media (numéricos)":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif treatment == "Rellenar con mediana (numéricos)":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif treatment == "Rellenar con valor específico" and fill_value:
                    try:
                        fill_num = float(fill_value)
                        df = df.fillna(fill_num)
                    except ValueError:
                        df = df.fillna(fill_value)
                
                st.success("Tratamiento aplicado correctamente")
                st.experimental_rerun()
        
        elif analysis_option == tr["duplicates_analysis"]:
            st.subheader(tr["duplicates_analysis"])
            
            duplicates = df.duplicated().sum()
            st.write(f"**Filas duplicadas encontradas:** {duplicates}")
            
            if duplicates > 0:
                st.write("**Filas duplicadas:**")
                st.dataframe(df[df.duplicated(keep=False)])
                
                if st.button("Eliminar duplicados"):
                    df = df.drop_duplicates()
                    st.success(f"Se eliminaron {duplicates} filas duplicadas")
                    st.experimental_rerun()
        
        elif analysis_option == tr["outliers_analysis"]:
            st.subheader(tr["outliers_analysis"])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            selected_col = st.selectbox("Seleccione columna para análisis:", numeric_cols)
            
            # Gráfico de caja
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_col], ax=ax)
            st.pyplot(fig)
            
            # Opciones de tratamiento
            st.subheader(tr["treatment_options"])
            treatment = st.selectbox("Seleccione tratamiento:", [
                "Eliminar outliers",
                "Transformar a valores límite"
            ])
            
            if st.button(tr["apply"]):
                q1 = df[selected_col].quantile(0.25)
                q3 = df[selected_col].quantile(0.75)
                iqr = q3 - q1
                
                if treatment == "Eliminar outliers":
                    df = df[(df[selected_col] >= q1 - 1.5*iqr) & (df[selected_col] <= q3 + 1.5*iqr)]
                else:
                    lower_bound = q1 - 1.5*iqr
                    upper_bound = q3 + 1.5*iqr
                    df[selected_col] = df[selected_col].clip(lower_bound, upper_bound)
                
                st.success("Tratamiento aplicado correctamente")
                st.experimental_rerun()
        
        # Reporte EDA
        st.subheader("Reporte de Análisis Exploratorio")
        if st.button("Generar Reporte EDA"):
            eda_report = generate_eda_report(df)
            st.download_button(
                "Descargar Reporte",
                data=eda_report,
                file_name="reporte_eda.txt",
                mime="text/plain"
            )
            st.text_area("Vista previa del reporte:", eda_report, height=300)
else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis")
