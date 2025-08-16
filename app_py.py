import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Configuración inicial
st.set_page_config(layout="wide", page_title="Estimador de Precios de Viviendas")

# Diccionarios de traducción
translations = {
    "ESPAÑOL": {
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
        "language": "Idioma",
        "file_loaded": "Archivo cargado",
        "size": "Tamaño",
        "null_values": "Valores nulos por columna",
        "treatment_select": "Seleccione tratamiento",
        "drop_nulls": "Eliminar filas con nulos",
        "fill_mean": "Rellenar con media (numéricos)",
        "fill_median": "Rellenar con mediana (numéricos)",
        "fill_value": "Rellenar con valor específico",
        "fill_input": "Valor de relleno",
        "treatment_success": "Tratamiento aplicado correctamente",
        "duplicates_found": "Filas duplicadas encontradas",
        "duplicates_show": "Filas duplicadas",
        "drop_duplicates": "Eliminar duplicados",
        "outlier_select": "Seleccione columna para análisis",
        "outlier_treatment": "Tratamiento para outliers",
        "remove_outliers": "Eliminar outliers",
        "cap_outliers": "Transformar a valores límite",
        "eda_report": "Reporte de Análisis Exploratorio",
        "generate_report": "Generar Reporte EDA",
        "download_report": "Descargar Reporte",
        "report_preview": "Vista previa del reporte",
        "upload_prompt": "Por favor, sube un archivo CSV para comenzar el análisis"
    },
    "ENGLISH": {
        "title": "HOUSE PRICE ESTIMATOR",
        "upload": "Upload your CSV file",
        "file_limit": "Limit: 200MB per file • CSV",
        "quick_preview": "Quick Preview",
        "null_analysis": "Null Values Analysis",
        "duplicates_analysis": "Duplicates Analysis",
        "outliers_analysis": "Outliers Analysis",
        "treatment_options": "Treatment Options",
        "apply": "Apply",
        "reset": "Reset",
        "report": "EDA Report",
        "language": "Language",
        "file_loaded": "File loaded",
        "size": "Size",
        "null_values": "Null values by column",
        "treatment_select": "Select treatment",
        "drop_nulls": "Drop rows with nulls",
        "fill_mean": "Fill with mean (numeric only)",
        "fill_median": "Fill with median (numeric only)",
        "fill_value": "Fill with specific value",
        "fill_input": "Fill value",
        "treatment_success": "Treatment applied successfully",
        "duplicates_found": "Duplicate rows found",
        "duplicates_show": "Duplicate rows",
        "drop_duplicates": "Remove duplicates",
        "outlier_select": "Select column for analysis",
        "outlier_treatment": "Outlier treatment",
        "remove_outliers": "Remove outliers",
        "cap_outliers": "Cap to threshold values",
        "eda_report": "Exploratory Data Analysis Report",
        "generate_report": "Generate EDA Report",
        "download_report": "Download Report",
        "report_preview": "Report preview",
        "upload_prompt": "Please upload a CSV file to begin analysis"
    },
    "FRANÇAIS": {
        "title": "ESTIMATEUR DE PRIX IMMOBILIERS",
        "upload": "Téléchargez votre fichier CSV",
        "file_limit": "Limite: 200MB par fichier • CSV",
        "quick_preview": "Aperçu Rapide",
        "null_analysis": "Analyse des Valeurs Manquantes",
        "duplicates_analysis": "Analyse des Doublons",
        "outliers_analysis": "Analyse des Valeurs Aberrantes",
        "treatment_options": "Options de Traitement",
        "apply": "Appliquer",
        "reset": "Réinitialiser",
        "report": "Rapport EDA",
        "language": "Langue",
        "file_loaded": "Fichier chargé",
        "size": "Taille",
        "null_values": "Valeurs manquantes par colonne",
        "treatment_select": "Sélectionnez un traitement",
        "drop_nulls": "Supprimer les lignes avec valeurs manquantes",
        "fill_mean": "Remplir avec la moyenne (numériques seulement)",
        "fill_median": "Remplir avec la médiane (numériques seulement)",
        "fill_value": "Remplir avec une valeur spécifique",
        "fill_input": "Valeur de remplissage",
        "treatment_success": "Traitement appliqué avec succès",
        "duplicates_found": "Lignes en double trouvées",
        "duplicates_show": "Lignes en double",
        "drop_duplicates": "Supprimer les doublons",
        "outlier_select": "Sélectionnez une colonne pour analyse",
        "outlier_treatment": "Traitement des valeurs aberrantes",
        "remove_outliers": "Supprimer les valeurs aberrantes",
        "cap_outliers": "Limiter aux valeurs seuils",
        "eda_report": "Rapport d'Analyse Exploratoire",
        "generate_report": "Générer Rapport EDA",
        "download_report": "Télécharger le Rapport",
        "report_preview": "Aperçu du rapport",
        "upload_prompt": "Veuillez télécharger un fichier CSV pour commencer l'analyse"
    }
}

# Sidebar para idioma y selección de análisis
with st.sidebar:
    st.header("Language / idioma / Langue")
    language = st.radio("", ["ESPAÑOL", "ENGLISH", "FRANÇAIS"], label_visibility="collapsed")
    
    tr = translations[language]  # Seleccionar traducciones según idioma
    
    st.header(tr.get("analysis_section", "Depuración de dataset"))
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
def generate_eda_report(df, language):
    report = StringIO()
    
    # Título según idioma
    titles = {
        "ESPAÑOL": "=== REPORTE DE ANÁLISIS EXPLORATORIO ===",
        "ENGLISH": "=== EXPLORATORY DATA ANALYSIS REPORT ===",
        "FRANÇAIS": "=== RAPPORT D'ANALYSE EXPLORATOIRE ==="
    }
    report.write(f"{titles[language]}\n\n")
    
    # Información básica
    section_titles = {
        "ESPAÑOL": ["1. INFORMACIÓN BÁSICA:", "2. ESTADÍSTICAS DESCRIPTIVAS:", 
                   "3. DATOS NULOS:", "4. FILAS DUPLICADAS:", "5. TIPOS DE DATOS:"],
        "ENGLISH": ["1. BASIC INFORMATION:", "2. DESCRIPTIVE STATISTICS:", 
                   "3. NULL VALUES:", "4. DUPLICATE ROWS:", "5. DATA TYPES:"],
        "FRANÇAIS": ["1. INFORMATIONS DE BASE:", "2. STATISTIQUES DESCRIPTIVES:", 
                    "3. VALEURS MANQUANTES:", "4. LIGNES EN DOUBLE:", "5. TYPES DE DONNÉES:"]
    }
    
    report.write(f"{section_titles[language][0]}\n")
    df.info(buf=report)
    report.write("\n\n")
    
    # Estadísticas descriptivas
    report.write(f"{section_titles[language][1]}\n")
    report.write(df.describe().to_string())
    report.write("\n\n")
    
    # Datos nulos
    report.write(f"{section_titles[language][2]}\n")
    report.write(df.isnull().sum().to_string())
    report.write("\n\n")
    
    # Duplicados
    report.write(f"{section_titles[language][3]} {df.duplicated().sum()}\n\n")
    
    # Tipos de datos
    report.write(f"{section_titles[language][4]}\n")
    report.write(df.dtypes.to_string())
    
    return report.getvalue()

# Interfaz principal
st.title(tr["title"])

# Carga de archivo
uploaded_file = st.file_uploader(tr["upload"], type=["csv"], help=tr["file_limit"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"**{tr['file_loaded']}:** {uploaded_file.name}")
        st.success(f"**{tr['size']}:** {uploaded_file.size / 1024:.2f} KB")
        
        # Vista previa rápida
        st.subheader(tr["quick_preview"])
        st.dataframe(df.head())
        
        # Análisis seleccionado
        if analysis_option == tr["null_analysis"]:
            st.subheader(tr["null_analysis"])
            
            # Mostrar datos nulos
            null_counts = df.isnull().sum()
            st.write(f"**{tr['null_values']}:**")
            st.write(null_counts)
            
            # Gráfico de valores nulos
            fig, ax = plt.subplots()
            sns.heatmap(df.isnull(), cbar=False, ax=ax)
            st.pyplot(fig)
            
            # Opciones de tratamiento
            st.subheader(tr["treatment_options"])
            treatment = st.selectbox(tr["treatment_select"], [
                tr["drop_nulls"],
                tr["fill_mean"],
                tr["fill_median"],
                tr["fill_value"]
            ])
            
            if treatment == tr["fill_value"]:
                fill_value = st.text_input(tr["fill_input"])
            
            if st.button(tr["apply"]):
                if treatment == tr["drop_nulls"]:
                    df = df.dropna()
                elif treatment == tr["fill_mean"]:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif treatment == tr["fill_median"]:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif treatment == tr["fill_value"] and fill_value:
                    try:
                        fill_num = float(fill_value)
                        df = df.fillna(fill_num)
                    except ValueError:
                        df = df.fillna(fill_value)
                
                st.success(tr["treatment_success"])
                st.experimental_rerun()
        
        elif analysis_option == tr["duplicates_analysis"]:
            st.subheader(tr["duplicates_analysis"])
            
            duplicates = df.duplicated().sum()
            st.write(f"**{tr['duplicates_found']}:** {duplicates}")
            
            if duplicates > 0:
                st.write(f"**{tr['duplicates_show']}:**")
                st.dataframe(df[df.duplicated(keep=False)])
                
                if st.button(tr["drop_duplicates"]):
                    df = df.drop_duplicates()
                    st.success(f"{tr['treatment_success']} - {duplicates} {tr['duplicates_found'].lower()}")
                    st.experimental_rerun()
        
        elif analysis_option == tr["outliers_analysis"]:
            st.subheader(tr["outliers_analysis"])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            selected_col = st.selectbox(tr["outlier_select"], numeric_cols)
            
            # Gráfico de caja
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_col], ax=ax)
            st.pyplot(fig)
            
            # Opciones de tratamiento
            st.subheader(tr["outlier_treatment"])
            treatment = st.selectbox(tr["treatment_select"], [
                tr["remove_outliers"],
                tr["cap_outliers"]
            ])
            
            if st.button(tr["apply"]):
                q1 = df[selected_col].quantile(0.25)
                q3 = df[selected_col].quantile(0.75)
                iqr = q3 - q1
                
                if treatment == tr["remove_outliers"]:
                    df = df[(df[selected_col] >= q1 - 1.5*iqr) & (df[selected_col] <= q3 + 1.5*iqr)]
                else:
                    lower_bound = q1 - 1.5*iqr
                    upper_bound = q3 + 1.5*iqr
                    df[selected_col] = df[selected_col].clip(lower_bound, upper_bound)
                
                st.success(tr["treatment_success"])
                st.experimental_rerun()
        
        # Reporte EDA
        st.subheader(tr["eda_report"])
        if st.button(tr["generate_report"]):
            eda_report = generate_eda_report(df, language)
            st.download_button(
                tr["download_report"],
                data=eda_report,
                file_name=f"eda_report_{language.lower()}.txt",
                mime="text/plain"
            )
            st.text_area(tr["report_preview"], eda_report, height=300)
else:
    st.info(tr["upload_prompt"])
