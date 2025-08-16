
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Código Estimador",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar traducciones
def load_translations(lang):
    translations = {
        "es": {
            "title": "🔍 Código Estimador - Análisis de Datos",
            "selector_idioma": "Seleccione el idioma",
            "cargar_datos": "Cargar Archivo CSV",
            "ayuda_carga": "Suba su archivo de datos para comenzar el análisis",
            "archivo_cargado": "Archivo cargado:",
            "tamano_archivo": "Tamaño:",
            "vista_previa": "Vista Previa Rápida",
            "analisis_btn": "Análisis Completo",
            "ayuda_analisis": "Ejecuta un análisis completo del dataset",
            "duplicados_btn": "Análisis de Duplicados",
            "outliers_btn": "Detección de Outliers",
            "tratamiento_nulos_btn": "Tratamiento de Valores Nulos",
            "exportar_btn": "Exportar Resultados",
            "info_basica": "Información Básica",
            "filas": "Número de filas:",
            "columnas": "Número de columnas:",
            "pestana_nulos": "Valores Nulos",
            "pestana_tipos": "Tipos de Datos",
            "pestana_estadisticas": "Estadísticas",
            "pestana_muestra": "Muestra de Datos",
            "titulo_nulos": "Análisis de Valores Nulos",
            "nombre_columna": "Columna",
            "conteo_nulos": "Valores Nulos",
            "porcentaje_nulos": "Porcentaje (%)",
            "advertencia_nulos": "⚠️ El dataset contiene valores nulos",
            "exito_nulos": "✅ No se encontraron valores nulos",
            "titulo_tipos": "Tipos de Datos",
            "tipo_dato": "Tipo de Dato",
            "columnas_numericas": "Columnas numéricas:",
            "columnas_no_numericas": "Columnas no numéricas:",
            "titulo_estadisticas": "Estadísticas Descriptivas",
            "titulo_muestra": "Muestra de Datos (10 primeras filas)",
            "titulo_duplicados": "Análisis de Duplicados",
            "total_duplicados": "Total filas duplicadas:",
            "filas_duplicadas": "Filas duplicadas encontradas:",
            "titulo_outliers": "Detección de Valores Atípicos",
            "seleccion_columna": "Seleccione columna:",
            "outliers_detectados": "Outliers encontrados:",
            "porcentaje_outliers": "Porcentaje de outliers:",
            "titulo_tratamiento": "Tratamiento de Valores Nulos",
            "opcion_tratamiento1": "Eliminar filas con nulos",
            "opcion_tratamiento2": "Rellenar con media (numéricas)",
            "opcion_tratamiento3": "Rellenar con moda (categóricas)",
            "opcion_tratamiento4": "Rellenar con valor específico:",
            "aplicar_tratamiento": "Aplicar Tratamiento",
            "exito_tratamiento": "Tratamiento aplicado con éxito",
            "sin_nulos": "No hay valores nulos para tratar",
            "guardar_dataset": "Guardar Dataset Tratado",
            "nombre_guardado": "Nombre del archivo:",
            "boton_guardar": "Descargar Dataset",
            "exito_guardado": "Dataset guardado con éxito",
            "herramientas_analisis": "Herramientas de Análisis"
        },
        "en": {
            "title": "🔍 Code Estimator - Data Analysis",
            # ... (resto de traducciones en inglés)
            "herramientas_analisis": "Analysis Tools"
        },
        "fr": {
            "title": "🔍 Code Estimator - Analyse de Données",
            # ... (resto de traducciones en francés)
            "herramientas_analisis": "Outils d'Analyse"
        }
    }
    return translations.get(lang, translations["es"])

# ... (resto de las funciones se mantienen igual)

# Interfaz principal de la aplicación
def main():
    # Configuración de idioma
     
    st.sidebar.title("🌍 Language / Idioma / Langue")
    idioma = st.sidebar.radio("", ["Español", "English", "Français"])
    codigo_idioma = {"Español": "es", "English": "en", "Français": "fr"}[idioma]
    tr = cargar_traducciones(codigo_idioma)
    
    st.title(tr["title"])
    
    # Carga de archivo
    archivo = st.file_uploader(
        tr["cargar_datos"], 
        type=['csv'],
        help=tr["ayuda_carga"]
    )
    
    if archivo is not None:
        st.success(f"{tr['archivo_cargado']} {archivo.name}")
        st.write(f"{tr['tamano_archivo']} {archivo.size / 1024:.2f} KB")
        
        df = cargar_dataset(archivo)
        
        if df is not None:
            st.write("### " + tr["vista_previa"])
            st.dataframe(df.head(3))
            
            # SECCIÓN DE HERRAMIENTAS EN EL SIDEBAR
            st.sidebar.title("🛠️ " + tr["herramientas_analisis"])
            
            # Botones de análisis en el sidebar
            if st.sidebar.button("🔍 " + tr["analisis_btn"], help=tr["ayuda_analisis"]):
                mostrar_analisis_completo(df, tr)
            
            if st.sidebar.button("📝 " + tr["duplicados_btn"]):
                analizar_duplicados(df, tr)
            
            if st.sidebar.button("📊 " + tr["outliers_btn"]):
                detectar_outliers(df, tr)
            
            # Botón de tratamiento de nulos en el sidebar
            if st.sidebar.button("🛠️ " + tr["tratamiento_nulos_btn"]):
                df = tratar_valores_nulos(df, tr)
                st.write("### Dataset después del tratamiento")
                st.dataframe(df.head())
                
                # Opción para guardar el dataset tratado
                guardar_dataset(df, tr)
            
            # Opción para exportar resultados (se mantiene en el área principal)
            if st.checkbox(tr["exportar_btn"]):
                nulos = df.isnull().sum().reset_index()
                nulos.columns = ['Columna', 'Valores_Nulos']
                st.download_button(
                    label="📥 Descargar Análisis de Nulos",
                    data=nulos.to_csv(index=False).encode('utf-8'),
                    file_name='analisis_nulos.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()
