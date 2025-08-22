import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================
# CONFIGURACIÓN INICIAL Y TRADUCCIONES
# =============================================

# Configuración de la página
st.set_page_config(
    page_title="Dataset Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar traducciones
def load_translations(lang):
    translations = {
        "es": {
            "title": "🔍 ESTIMADOR DE PRECIOS DE VIVIENDAS",
            "upload_label": "Sube tu archivo (CSV o Excel)",
            "upload_help": "Selecciona el dataset que deseas analizar (CSV o XLSX)",
            "file_loaded": "Archivo cargado:",
            "file_size": "Tamaño:",
            "quick_preview": "Vista Previa Rápida",
            "analysis_btn": "Análisis datos nulos",
            "analysis_help": "Analiza el dataset para valores nulos y problemas comunes",
            "export_label": "¿Deseas exportar el análisis de valores nulos?",
            "export_btn": "Descargar Análisis de Nulos",
            "duplicates_btn": "Análisis de Duplicados",
            "outliers_btn": "Análisis de Outliers",
            "null_treatment_btn": "Tratamiento de Datos Nulos",
            "basic_info": "Información Básica",
            "rows": "Número de filas:",
            "cols": "Número de columnas:",
            "null_tab": "Valores Nulos",
            "types_tab": "Tipos de Datos",
            "stats_tab": "Estadísticas",
            "sample_tab": "Muestra de Datos",
            "null_title": "Valores Nulos por Columna",
            "col_name": "Columna",
            "null_count": "Valores Nulos",
            "null_percent": "Porcentaje (%)",
            "null_warning": "⚠️ El dataset contiene valores nulos que deben ser tratados",
            "null_success": "✅ No se encontraron valores nulos en el dataset",
            "types_title": "Tipos de Datos por Columna",
            "data_type": "Tipo de Dato",
            "numeric_cols": "Columnas numéricas:",
            "non_numeric_cols": "Columnas no numéricas:",
            "stats_title": "Estadísticas Descriptivas",
            "sample_title": "Muestra de Datos (Primeras 10 filas)",
            "duplicates_title": "Análisis de Duplicados",
            "total_duplicates": "Filas duplicadas totales:",
            "duplicate_rows": "Filas duplicadas:",
            "outliers_title": "Análisis de Outliers",
            "outliers_col": "Columna:",
            "outliers_count": "Outliers detectados:",
            "outliers_percent": "Porcentaje de outliers:",
            "treatment_title": "Tratamiento de Datos Nulos",
            "treatment_option1": "Eliminar filas con valores nulos",
            "treatment_option2": "Rellenar con la media (numéricas)",
            "treatment_option3": "Rellenar con la mediana (numéricas)",
            "treatment_option4": "Rellenar con moda (categóricas)",
            "treatment_option5": "Rellenar con valor específico:",
            "apply_treatment": "Aplicar Tratamiento",
            "treatment_success": "Tratamiento aplicado correctamente",
            "no_nulls": "No hay valores nulos para tratar",
            "no_duplicates": "No hay duplicados para mostrar",
            "herramientas_analisis": "Depuración de dataset",
            "selector_idioma": "Seleccione idioma",
            "dataset_tras_tratamiento": "Dataset después del tratamiento",
            "reset_button": "Resetear a datos originales",
            "comparison_title": "Comparación de valores nulos",
            "select_method_label": "Seleccione método de tratamiento",
            "fill_value_prompt": "Ingrese el valor de relleno:",
            "treatment_error": "Error al aplicar tratamiento"
        },
        "en": {
            "title": "🔍 Dataset Analysis and Cleaning",
            "upload_label": "Upload your file (CSV or Excel)",
            "upload_help": "Select the dataset you want to analyze (CSV or XLSX)",
            "file_loaded": "File loaded:",
            "file_size": "Size:",
            "quick_preview": "Quick Preview",
            "analysis_btn": "Run Data Analysis",
            "analysis_help": "Analyze the dataset for null values and common issues",
            "export_label": "Do you want to export the null analysis?",
            "export_btn": "Download Null Analysis",
            "duplicates_btn": "Duplicate Analysis",
            "outliers_btn": "Outliers Analysis",
            "null_treatment_btn": "Null Data Treatment",
            "basic_info": "Basic Information",
            "rows": "Number of rows:",
            "cols": "Number of columns:",
            "null_tab": "Null Values",
            "types_tab": "Data Types",
            "stats_tab": "Statistics",
            "sample_tab": "Data Sample",
            "null_title": "Null Values by Column",
            "col_name": "Column",
            "null_count": "Null Values",
            "null_percent": "Percentage (%)",
            "null_warning": "⚠️ The dataset contains null values that need treatment",
            "null_success": "✅ No null values found in the dataset",
            "types_title": "Data Types by Column",
            "data_type": "Data Type",
            "numeric_cols": "Numeric columns:",
            "non_numeric_cols": "Non-numeric columns:",
            "stats_title": "Descriptive Statistics",
            "sample_title": "Data Sample (First 10 rows)",
            "duplicates_title": "Duplicate Analysis",
            "total_duplicates": "Total duplicate rows:",
            "duplicate_rows": "Duplicate rows:",
            "outliers_title": "Outliers Analysis",
            "outliers_col": "Column:",
            "outliers_count": "Outliers detected:",
            "outliers_percent": "Outliers percentage:",
            "treatment_title": "Null Data Treatment",
            "treatment_option1": "Drop rows with null values",
            "treatment_option2": "Fill with mean (numeric columns)",
            "treatment_option3": "Fill with median (numeric columns)",
            "treatment_option4": "Fill with mode (categorical columns)",
            "treatment_option5": "Fill with specific value:",
            "apply_treatment": "Apply Treatment",
            "treatment_success": "Treatment applied successfully",
            "no_nulls": "No null values to treat",
            "no_duplicates": "No duplicates to show",
            "herramientas_analisis": "Analysis Tools",
            "selector_idioma": "Select language",
            "dataset_tras_tratamiento": "Dataset after treatment",
            "reset_button": "Reset to original data",
            "comparison_title": "Null values comparison",
            "select_method_label": "Select treatment method",
            "fill_value_prompt": "Enter fill value:",
            "treatment_error": "Error applying treatment"
        }
    }
    return translations.get(lang, translations["en"])

# =============================================
# FUNCIONES PARA MANEJO DE DATOS
# =============================================

def load_dataset(file):
    """Carga un archivo CSV o Excel detectando el tipo automáticamente."""
    try:
        if file.name.endswith('.csv'):
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    if df.empty:
                        st.error("The uploaded file is empty.")
                        return None
                    return df
                except UnicodeDecodeError:
                    continue
            st.error("Could not read the CSV file with common encodings.")
            return None
        elif file.name.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(file)
                if df.empty:
                    st.error("The uploaded Excel file is empty.")
                    return None
                return df
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                return None
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# (Las funciones para análisis, visualización, tratamiento de nulos y main() permanecen igual que en tu versión actual)

# =============================================
# INTERFAZ PRINCIPAL
# =============================================

def main():
    # Cargar traducciones por defecto (inglés)
    tr = load_translations("en")

    # Selector de idioma
    st.sidebar.title("🌍 Language / Idioma / Langue")
    language = st.sidebar.radio("", ["Español", "English"], label_visibility="collapsed")
    lang_code = {"Español": "es", "English": "en"}[language]
    tr = load_translations(lang_code)

    st.title(tr["title"])

    # ✅ Acepta CSV y Excel
    file = st.file_uploader(
        tr["upload_label"],
        type=['csv', 'xlsx'],
        help=tr["upload_help"]
    )

    if file is not None:
        st.success(f"{tr['file_loaded']} {file.name}")
        st.write(f"{tr['file_size']} {file.size / 1024:.2f} KB")

        df = load_dataset(file)

        if df is not None:
            st.write("### " + tr["quick_preview"])
            st.dataframe(df.head(3))

            st.sidebar.title("🔧 " + tr["herramientas_analisis"])

            if st.sidebar.button("🔍 " + tr["analysis_btn"], help=tr["analysis_help"]):
                show_analysis(df, tr)

            if st.sidebar.button("📝 " + tr["duplicates_btn"]):
                show_duplicates(df, tr)

            if st.sidebar.button("📊 " + tr["outliers_btn"]):
                show_outliers(df, tr)

            if st.sidebar.button("🛠️ " + tr["null_treatment_btn"]):
                df = null_treatment(df, tr)
                st.write("### " + tr["dataset_tras_tratamiento"])
                st.dataframe(df.head())

            if st.sidebar.checkbox(tr["export_label"]):
                nulls = df.isnull().sum().reset_index()
                nulls.columns = ['Column', 'Null_Values']
                st.sidebar.download_button(
                    label="📥 " + tr["export_btn"],
                    data=nulls.to_csv(index=False).encode('utf-8'),
                    file_name='null_analysis.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()