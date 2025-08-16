import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Análisis de Calidad de Datos")

# Instrucciones para el usuario
st.info("Sube tu archivo CSV para analizarlo en busca de valores faltantes, datos incorrectos y valores atípicos. Los datos se cargarán en la caché para una mejor experiencia.")

# Carga de archivo
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("¡Archivo cargado exitosamente!")

    # Almacena el DataFrame en la caché de Streamlit
    @st.cache_data
    def get_data(file):
        return pd.read_csv(file)

    data = get_data(uploaded_file)
    st.subheader("Vista previa del Dataset")
    st.dataframe(data.head())

    st.markdown("---")
    
    # 1. Análisis de valores faltantes
    st.header("1. Valores Faltantes 🕵️‍♂️")
    missing_data = data.isnull().sum()
    missing_percentage = (missing_data / len(data)) * 100
    missing_table = pd.DataFrame({'Total Faltantes': missing_data, 'Porcentaje (%)': missing_percentage})
    st.table(missing_table[missing_table['Total Faltantes'] > 0])

    if missing_table[missing_table['Total Faltantes'] > 0].empty:
        st.info("✅ ¡No se encontraron valores faltantes en este dataset!")
    else:
        st.warning("⚠️ Se encontraron valores faltantes. Considera imputarlos o eliminarlos.")

    st.markdown("---")

    # 2. Análisis de datos incorrectos (tipos de datos)
    st.header("2. Datos Incorrectos (Tipos de Datos) 🐛")
    st.write("Verificando los tipos de datos inferidos para cada columna:")
    st.dataframe(data.dtypes)
    
    # Intenta convertir las columnas a tipos numéricos para detectar errores
    incorrect_types = {}
    for column in data.columns:
        # Ignora las columnas que ya son de tipo numérico
        if pd.api.types.is_numeric_dtype(data[column]):
            continue
        
        # Intenta convertir la columna a tipo numérico y detecta errores
        try:
            pd.to_numeric(data[column])
        except ValueError:
            # Si la conversión falla, es un dato incorrecto
            incorrect_types[column] = data[column].dtype
            
    if incorrect_types:
        st.error("❌ ¡Se encontraron datos incorrectos! Algunas columnas tienen valores que no corresponden con su tipo de dato esperado.")
        st.write("Columnas con posibles datos incorrectos:")
        st.table(pd.Series(incorrect_types, name="Tipo de Dato Actual"))
    else:
        st.info("✅ ¡No se encontraron datos incorrectos en las columnas!")

    st.markdown("---")
    
    # 3. Detección de valores atípicos (outliers)
    st.header("3. Valores Atípicos (Outliers) 📊")
    st.write("Los gráficos de caja (box plots) son una excelente manera de visualizar y detectar valores atípicos en variables numéricas.")
    
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("⚠️ El dataset no contiene columnas numéricas para analizar outliers.")
    else:
        st.info(f"Analizando outliers en las columnas numéricas: {', '.join(numeric_cols)}")
        
        for col in numeric_cols:
            fig = px.box(data, y=col, title=f'Box Plot de {col}', points="all")
            st.plotly_chart(fig)
            
        st.markdown(
            """
            * Los puntos fuera de los bigotes en los box plots son los valores atípicos.
            * Puedes pasar el cursor sobre los puntos para ver sus valores.
            """
        )
