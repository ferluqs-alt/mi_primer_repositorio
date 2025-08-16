import streamlit as st
import pandas as pd
import numpy as np
import os

# Configuración de la página
st.set_page_config(
    page_title="Depuración de Dataset",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título de la aplicación
st.title("🔍 Análisis y Depuración de Dataset")

# Función para cargar el archivo CSV
def cargar_dataset(archivo):
    try:
        # Intentar con diferentes codificaciones comunes
        codificaciones = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for codificacion in codificaciones:
            try:
                df = pd.read_csv(archivo, encoding=codificacion)
                return df
            except UnicodeDecodeError:
                continue
        
        # Si ninguna codificación funcionó
        st.error("No se pudo leer el archivo con las codificaciones comunes.")
        return None
        
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

# Función para mostrar el análisis de depuración
def mostrar_depuracion(df):
    st.subheader("📊 Análisis de Depuración")
    
    # Mostrar información básica del dataset
    st.write("### Información Básica")
    st.write(f"**Número de filas:** {df.shape[0]}")
    st.write(f"**Número de columnas:** {df.shape[1]}")
    
    # Crear pestañas para diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs([
        "Valores Nulos", 
        "Tipos de Datos", 
        "Estadísticas", 
        "Muestra de Datos"
    ])
    
    with tab1:
        st.write("### Valores Nulos por Columna")
        nulos = df.isnull().sum()
        nulos_percent = (nulos / len(df)) * 100
        
        # Crear DataFrame para mostrar
        nulos_df = pd.DataFrame({
            'Columna': nulos.index,
            'Valores Nulos': nulos.values,
            'Porcentaje (%)': nulos_percent.values.round(2)
        })
        
        st.dataframe(nulos_df.style.highlight_max(
            axis=0, 
            subset=['Valores Nulos', 'Porcentaje (%)'],
            color='salmon'
        ))
        
        # Gráfico de valores nulos
        st.bar_chart(nulos_percent)
        
        if nulos.sum() > 0:
            st.warning("⚠️ El dataset contiene valores nulos que deben ser tratados")
        else:
            st.success("✅ No se encontraron valores nulos en el dataset")
    
    with tab2:
        st.write("### Tipos de Datos por Columna")
        tipos = df.dtypes.reset_index()
        tipos.columns = ['Columna', 'Tipo de Dato']
        st.dataframe(tipos)
        
        # Verificar tipos numéricos vs no numéricos
        numericas = df.select_dtypes(include=['number']).columns
        no_numericas = df.select_dtypes(exclude=['number']).columns
        
        st.write(f"**Columnas numéricas:** {len(numericas)}")
        st.write(f"**Columnas no numéricas:** {len(no_numericas)}")
    
    with tab3:
        st.write("### Estadísticas Descriptivas")
        st.dataframe(df.describe(include='all').T)
    
    with tab4:
        st.write("### Muestra de Datos (Primeras 10 filas)")
        st.dataframe(df.head(10))

# Interfaz principal de la aplicación
def main():
    # Cargar archivo CSV
    archivo = st.file_uploader(
        "Sube tu archivo CSV", 
        type=['csv'],
        help="Selecciona el dataset que deseas analizar"
    )
    
    if archivo is not None:
        # Mostrar información del archivo
        st.success(f"Archivo cargado: {archivo.name}")
        st.write(f"Tamaño: {archivo.size / 1024:.2f} KB")
        
        # Cargar el dataset
        df = cargar_dataset(archivo)
        
        if df is not None:
            # Mostrar vista previa básica
            st.write("### Vista Previa Rápida")
            st.dataframe(df.head(3))
            
            # Botón de depuración
            if st.button("🔍 Ejecutar Análisis de Depuración", 
                        help="Analiza el dataset para valores nulos y problemas comunes"):
                mostrar_depuracion(df)
            
            # Opción para descargar el análisis
            if st.checkbox("¿Deseas exportar el análisis de valores nulos?"):
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
