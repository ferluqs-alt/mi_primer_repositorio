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
            "upload_label": "Sube tu archivo CSV",
            "upload_help": "Selecciona el dataset que deseas analizar",
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
            "upload_label": "Upload your CSV file",
            "upload_help": "Select the dataset you want to analyze",
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
        },
        "fr": {
            "title": "🔍 Analyse et Nettoyage de Données",
            "upload_label": "Téléchargez votre fichier CSV",
            "upload_help": "Sélectionnez le jeu de données à analyser",
            "file_loaded": "Fichier chargé:",
            "file_size": "Taille:",
            "quick_preview": "Aperçu Rapide",
            "analysis_btn": "Exécuter l'Analyse des Données",
            "analysis_help": "Analyser le jeu de données pour les valeurs nulles et problèmes courants",
            "export_label": "Voulez-vous exporter l'analyse des valeurs nulles?",
            "export_btn": "Télécharger l'Analyse des Nuls",
            "duplicates_btn": "Analyse des Doublons",
            "outliers_btn": "Analyse des Valeurs Aberrantes",
            "null_treatment_btn": "Traitement des Données Nulles",
            "basic_info": "Informations de Base",
            "rows": "Nombre de lignes:",
            "cols": "Nombre de colonnes:",
            "null_tab": "Valeurs Nulles",
            "types_tab": "Types de Données",
            "stats_tab": "Statistiques",
            "sample_tab": "Échantillon de Données",
            "null_title": "Valeurs Nulles par Colonne",
            "col_name": "Colonne",
            "null_count": "Valeurs Nulles",
            "null_percent": "Pourcentage (%)",
            "null_warning": "⚠️ Le jeu de données contient des valeurs nulles à traiter",
            "null_success": "✅ Aucune valeur nulle trouvée dans le jeu de données",
            "types_title": "Types de Données par Colonne",
            "data_type": "Type de Donnée",
            "numeric_cols": "Colonnes numériques:",
            "non_numeric_cols": "Colonnes non numériques:",
            "stats_title": "Statistiques Descriptives",
            "sample_title": "Échantillon de Données (10 premières lignes)",
            "duplicates_title": "Analyse des Doublons",
            "total_duplicates": "Lignes dupliquées totales:",
            "duplicate_rows": "Lignes dupliquées:",
            "outliers_title": "Analyse des Valeurs Aberrantes",
            "outliers_col": "Colonne:",
            "outliers_count": "Valeurs aberrantes détectées:",
            "outliers_percent": "Pourcentage de valeurs aberrantes:",
            "treatment_title": "Traitement des Données Nulles",
            "treatment_option1": "Supprimer les lignes avec valeurs nulles",
            "treatment_option2": "Remplir avec la moyenne (colonnes numériques)",
            "treatment_option3": "Remplir avec la médiane (colonnes numériques)",
            "treatment_option4": "Remplir avec le mode (colonnes catégorielles)",
            "treatment_option5": "Remplir avec une valeur spécifique:",
            "apply_treatment": "Appliquer le Traitement",
            "treatment_success": "Traitement appliqué avec succès",
            "no_nulls": "Aucune valeur nulle à traiter",
            "no_duplicates": "Aucun doublon à afficher",
            "herramientas_analisis": "Outils d'Analyse",
            "selector_idioma": "Choisir la langue",
            "dataset_tras_tratamiento": "Dataset après traitement",
            "reset_button": "Réinitialiser aux données originales",
            "comparison_title": "Comparaison des valeurs nulles",
            "select_method_label": "Sélectionnez la méthode de traitement",
            "fill_value_prompt": "Entrez la valeur de remplissage:",
            "treatment_error": "Erreur lors de l'application du traitement"
        }
    }
    return translations.get(lang, translations["en"])

# =============================================
# FUNCIONES PARA MANEJO DE DATOS
# =============================================

def load_dataset(file):
    """Carga un archivo CSV con manejo de diferentes codificaciones."""
    try:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file, encoding=encoding)
                # Verificar si el DataFrame está vacío
                if df.empty:
                    st.error("The uploaded file is empty.")
                    return None
                return df
            except UnicodeDecodeError:
                continue
        
        st.error("Could not read the file with common encodings.")
        return None
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_null_analysis(df):
    """Genera un análisis de valores nulos."""
    nulls = df.isnull().sum()
    nulls_percent = (nulls / len(df)) * 100
    return nulls, nulls_percent

def get_duplicate_analysis(df):
    """Analiza y devuelve filas duplicadas."""
    total_duplicates = df.duplicated().sum()
    duplicate_rows = df[df.duplicated(keep=False)] if total_duplicates > 0 else None
    return total_duplicates, duplicate_rows

# =============================================
# FUNCIONES PARA VISUALIZACIÓN
# =============================================

def show_analysis(df, tr):
    """Muestra el análisis completo del dataset."""
    st.subheader("📊 " + tr["basic_info"])
    st.write(f"**{tr['rows']}** {df.shape[0]}")
    st.write(f"**{tr['cols']}** {df.shape[1]}")
    
    # Crear pestañas para diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs([
        tr["null_tab"], 
        tr["types_tab"], 
        tr["stats_tab"], 
        tr["sample_tab"]
    ])
    
    with tab1:
        show_null_analysis(df, tr)
    
    with tab2:
        show_data_types(df, tr)
    
    with tab3:
        show_statistics(df, tr)
    
    with tab4:
        show_data_sample(df, tr)

def show_null_analysis(df, tr):
    """Muestra el análisis de valores nulos."""
    st.write("### " + tr["null_title"])
    nulls, nulls_percent = get_null_analysis(df)
    
    nulls_df = pd.DataFrame({
        tr["col_name"]: nulls.index,
        tr["null_count"]: nulls.values,
        tr["null_percent"]: nulls_percent.values.round(2)
    })
    
    st.dataframe(nulls_df.style.highlight_max(
        axis=0, 
        subset=[tr["null_count"], tr["null_percent"]],
        color='salmon'
    ))
    
    st.bar_chart(nulls_percent)
    
    if nulls.sum() > 0:
        st.warning(tr["null_warning"])
    else:
        st.success(tr["null_success"])

def show_data_types(df, tr):
    """Muestra los tipos de datos del dataset."""
    st.write("### " + tr["types_title"])
    types = df.dtypes.reset_index()
    types.columns = [tr["col_name"], tr["data_type"]]
    st.dataframe(types)
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    
    st.write(f"**{tr['numeric_cols']}** {len(numeric_cols)}")
    st.write(f"**{tr['non_numeric_cols']}** {len(non_numeric_cols)}")

def show_statistics(df, tr):
    """Muestra estadísticas descriptivas."""
    st.write("### " + tr["stats_title"])
    try:
        stats = df.describe(include='all').T
        st.dataframe(stats)
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")

def show_data_sample(df, tr):
    """Muestra una muestra de los datos."""
    st.write("### " + tr["sample_title"])
    st.dataframe(df.head(10))

def show_duplicates(df, tr):
    """Muestra el análisis de duplicados."""
    st.subheader("🔍 " + tr["duplicates_title"])
    
    total_duplicates, duplicate_rows = get_duplicate_analysis(df)
    st.write(f"**{tr['total_duplicates']}** {total_duplicates}")
    
    if total_duplicates > 0:
        st.write(f"**{tr['duplicate_rows']}**")
        st.dataframe(duplicate_rows.sort_values(by=list(df.columns)))
    else:
        st.success("✅ " + tr["no_duplicates"])

def show_outliers(df, tr):
    """Muestra el análisis de outliers."""
    st.subheader("📈 " + tr["outliers_title"])
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for outlier detection")
        return
    
    selected_col = st.selectbox(tr["outliers_col"], numeric_cols)
    
    if df[selected_col].notnull().sum() > 0:
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        outliers_count = len(outliers)
        outliers_percent = (outliers_count / len(df)) * 100
        
        st.write(f"**{tr['outliers_count']}** {outliers_count}")
        st.write(f"**{tr['outliers_percent']}** {outliers_percent:.2f}%")
        
        if outliers_count > 0:
            st.dataframe(outliers)
        else:
            st.success("✅ No outliers detected in this column")
    else:
        st.warning("Selected column contains only null values")

# =============================================
# TRATAMIENTO DE VALORES NULOS
# =============================================

def null_treatment(df, tr):
    """Realiza el tratamiento de valores nulos."""
    # Inicialización del estado de sesión
    if 'df_treated' not in st.session_state:
        st.session_state.df_treated = df.copy()
        st.session_state.last_treatment = None
        st.session_state.show_comparison = False

    st.subheader("🛠️ " + tr["treatment_title"])

    # Verificar si ya no hay nulos
    if st.session_state.df_treated.isnull().sum().sum() == 0:
        st.success("✅ " + tr["no_nulls"])
        return st.session_state.df_treated

    # Widgets de selección
    treatment_option = st.radio(
        tr["select_method_label"],
        options=[
            tr["treatment_option1"],
            tr["treatment_option2"],
            tr["treatment_option3"],
            tr["treatment_option4"],
            tr["treatment_option5"]
        ],
        key="treatment_option_radio"
    )

    # Input para valor específico
    fill_value = None
    if treatment_option == tr["treatment_option5"]:
        fill_value = st.text_input(
            tr["fill_value_prompt"],
            key="fill_value_input"
        )

    # Botón de aplicación
    if st.button(tr["apply_treatment"], key="apply_treatment_button"):
        try:
            temp_df = st.session_state.df_treated.copy()
            
            if treatment_option == tr["treatment_option1"]:
                initial_rows = len(temp_df)
                temp_df = temp_df.dropna()
                removed_rows = initial_rows - len(temp_df)
                st.info(f"Removed {removed_rows} rows with null values")
                
            elif treatment_option == tr["treatment_option2"]:
                numeric_cols = temp_df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if temp_df[col].isnull().sum() > 0:
                        mean_val = temp_df[col].mean()
                        temp_df[col] = temp_df[col].fillna(mean_val)
                        
            elif treatment_option == tr["treatment_option3"]:
                numeric_cols = temp_df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if temp_df[col].isnull().sum() > 0:
                        median_val = temp_df[col].median()
                        temp_df[col] = temp_df[col].fillna(median_val)
                        
            elif treatment_option == tr["treatment_option4"]:
                non_numeric_cols = temp_df.select_dtypes(exclude=['number']).columns
                for col in non_numeric_cols:
                    if temp_df[col].isnull().sum() > 0:
                        mode_val = temp_df[col].mode()[0]
                        temp_df[col] = temp_df[col].fillna(mode_val)
            
            elif treatment_option == tr["treatment_option5"] and fill_value:
                try:
                    # Intentar convertir a número
                    fill_value_num = float(fill_value)
                    temp_df = temp_df.fillna(fill_value_num)
                except ValueError:
                    # Si falla, usar como string
                    temp_df = temp_df.fillna(fill_value)
            
            # Actualizar el estado de sesión
            st.session_state.df_treated = temp_df
            st.session_state.last_treatment = treatment_option
            st.session_state.show_comparison = True
            
            st.success("✅ " + tr["treatment_success"])
            
        except Exception as e:
            st.error(f"❌ {tr['treatment_error']}: {str(e)}")

    # Mostrar comparación si se aplicó un tratamiento
    if st.session_state.show_comparison:
        st.subheader(tr["comparison_title"])
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before treatment**")
            st.write(df.isna().sum())
        
        with col2:
            st.markdown("**After treatment**")
            st.write(st.session_state.df_treated.isna().sum())

    # Botón para resetear
    if st.button(tr["reset_button"], key="reset_button"):
        st.session_state.df_treated = df.copy()
        st.session_state.show_comparison = False
        st.experimental_rerun()

    return st.session_state.df_treated

# =============================================
# INTERFAZ PRINCIPAL
# =============================================

def main():
    # Selector de idioma
    st.sidebar.title("🌍 " + tr["selector_idioma"])
    language = st.sidebar.radio("", ["Español", "English", "Français"], label_visibility="collapsed")
    lang_code = {"Español": "es", "English": "en", "Français": "fr"}[language]
    tr = load_translations(lang_code)
    
    st.title(tr["title"])
    
    # Cargar archivo CSV
    file = st.file_uploader(
        tr["upload_label"], 
        type=['csv'],
        help=tr["upload_help"]
    )
    
    if file is not None:
        # Mostrar información del archivo
        st.success(f"{tr['file_loaded']} {file.name}")
        st.write(f"{tr['file_size']} {file.size / 1024:.2f} KB")
        
        # Cargar el dataset
        df = load_dataset(file)
        
        if df is not None:
            # Mostrar vista previa básica
            st.write("### " + tr["quick_preview"])
            st.dataframe(df.head(3))
            
            # Contenedor para los botones de análisis
            st.sidebar.title("🔧 " + tr["herramientas_analisis"])
            
            # Botón de análisis completo
            if st.sidebar.button("🔍 " + tr["analysis_btn"], help=tr["analysis_help"]):
                show_analysis(df, tr)
            
            # Botón de análisis de duplicados
            if st.sidebar.button("📝 " + tr["duplicates_btn"]):
                show_duplicates(df, tr)
            
            # Botón de análisis de outliers
            if st.sidebar.button("📊 " + tr["outliers_btn"]):
                show_outliers(df, tr)
            
            # Botón de tratamiento de nulos
            if st.sidebar.button("🛠️ " + tr["null_treatment_btn"]):
                df = null_treatment(df, tr)
                st.write("### " + tr["dataset_tras_tratamiento"])
                st.dataframe(df.head())
            
            # Opción para descargar el análisis
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
