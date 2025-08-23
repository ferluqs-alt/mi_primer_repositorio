import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import norm, kstest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout
import time
from datetime import datetime
import io
import base64
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="RealEstate Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Internacionalización (español, inglés, francés)
translations = {
    "es": {
        "title": "Estimador de Precios de Viviendas",
        "upload_data": "Cargar Datos",
        "file_types": "Tipos de archivo aceptados: CSV, XLSX",
        "select_target": "Seleccionar variable objetivo",
        "start_analysis": "Iniciar Análisis Exploratorio de Datos (EDA)",
        "data_overview": "Visión General de los Datos",
        "data_shape": "Dimensiones del Dataset",
        "rows": "filas",
        "columns": "columnas",
        "variables": "Variables Disponibles",
        "var_types": "Tipos de Variables",
        "numeric": "Numéricas",
        "categorical": "Categóricas",
        "data_diagnosis": "Diagnóstico de Datos",
        "missing_data": "Datos Faltantes",
        "duplicate_data": "Datos Duplicados",
        "null_data": "Datos Nulos",
        "outliers": "Outliers Detectados",
        "normality_test": "Prueba de Normalidad (Kolmogorov-Smirnov)",
        "normal_dist": "Distribución Normal",
        "not_normal_dist": "No Distribución Normal",
        "numeric_analysis": "Análisis de Variables Numéricas",
        "categorical_analysis": "Análisis de Variables Categóricas",
        "correlation_analysis": "Análisis de Correlación",
        "preprocessing": "Preprocesamiento de Datos",
        "missing_treatment": "Tratamiento de Valores Faltantes",
        "duplicate_treatment": "Tratamiento de Duplicados",
        "outlier_treatment": "Tratamiento de Outliers",
        "encoding": "Codificación de Variables Categóricas",
        "scaling": "Escalado de Variables Numéricas",
        "multicollinearity": "Análisis de Multicolinealidad (VIF)",
        "modeling": "Modelado y Evaluación",
        "model_training": "Entrenamiento de Modelos",
        "model_performance": "Rendimiento de Modelos",
        "advanced_analysis": "Análisis Avanzado",
        "best_model": "Mejor Modelo",
        "generate_report": "Generar Reporte PDF",
        "download_report": "Descargar Reporte",
        "data_loaded": "Datos cargados correctamente",
        "select_preprocessing": "Seleccione las opciones de preprocesamiento",
        "train_models": "Entrenar Modelos",
        "view_results": "Ver Resultados",
        "save_models": "Guardar Modelos",
        "model_comparison": "Comparación de Modelos",
        "real_vs_predicted": "Valores Reales vs. Predichos",
        "confusion_matrix": "Matriz de Confusión",
        "roc_curve": "Curva ROC",
        "feature_importance": "Importancia de Variables",
        "statistical_tests": "Pruebas Estadísticas",
        "summary": "Resumen Ejecutivo"
    },
    "en": {
        "title": "Housing Price Estimator",
        "upload_data": "Upload Data",
        "file_types": "Accepted file types: CSV, XLSX",
        "select_target": "Select target variable",
        "start_analysis": "Start Exploratory Data Analysis (EDA)",
        "data_overview": "Data Overview",
        "data_shape": "Dataset Dimensions",
        "rows": "rows",
        "columns": "columns",
        "variables": "Available Variables",
        "var_types": "Variable Types",
        "numeric": "Numeric",
        "categorical": "Categorical",
        "data_diagnosis": "Data Diagnosis",
        "missing_data": "Missing Data",
        "duplicate_data": "Duplicate Data",
        "null_data": "Null Data",
        "outliers": "Outliers Detected",
        "normality_test": "Normality Test (Kolmogorov-Smirnov)",
        "normal_dist": "Normal Distribution",
        "not_normal_dist": "Not Normal Distribution",
        "numeric_analysis": "Numeric Variables Analysis",
        "categorical_analysis": "Categorical Variables Analysis",
        "correlation_analysis": "Correlation Analysis",
        "preprocessing": "Data Preprocessing",
        "missing_treatment": "Missing Values Treatment",
        "duplicate_treatment": "Duplicate Treatment",
        "outlier_treatment": "Outlier Treatment",
        "encoding": "Categorical Variables Encoding",
        "scaling": "Numeric Variables Scaling",
        "multicollinearity": "Multicollinearity Analysis (VIF)",
        "modeling": "Modeling and Evaluation",
        "model_training": "Model Training",
        "model_performance": "Model Performance",
        "advanced_analysis": "Advanced Analysis",
        "best_model": "Best Model",
        "generate_report": "Generate PDF Report",
        "download_report": "Download Report",
        "data_loaded": "Data loaded successfully",
        "select_preprocessing": "Select preprocessing options",
        "train_models": "Train Models",
        "view_results": "View Results",
        "save_models": "Save Models",
        "model_comparison": "Model Comparison",
        "real_vs_predicted": "Real vs. Predicted Values",
        "confusion_matrix": "Confusion Matrix",
        "roc_curve": "ROC Curve",
        "feature_importance": "Feature Importance",
        "statistical_tests": "Statistical Tests",
        "summary": "Executive Summary"
    },
    "fr": {
        "title": "Estimateur de Prix Immobiliers",
        "upload_data": "Télécharger les Données",
        "file_types": "Types de fichiers acceptés: CSV, XLSX",
        "select_target": "Sélectionner la variable cible",
        "start_analysis": "Démarrer l'Analyse Exploratoire des Données (AED)",
        "data_overview": "Aperçu des Données",
        "data_shape": "Dimensions du Dataset",
        "rows": "lignes",
        "columns": "colonnes",
        "variables": "Variables Disponibles",
        "var_types": "Types de Variables",
        "numeric": "Numériques",
        "categorical": "Catégorielles",
        "data_diagnosis": "Diagnostic des Données",
        "missing_data": "Données Manquantes",
        "duplicate_data": "Données Dupliquées",
        "null_data": "Données Nulles",
        "outliers": "Valeurs Aberrantes Détectées",
        "normality_test": "Test de Normalité (Kolmogorov-Smirnov)",
        "normal_dist": "Distribution Normale",
        "not_normal_dist": "Pas une Distribution Normale",
        "numeric_analysis": "Analyse des Variables Numériques",
        "categorical_analysis": "Analyse des Variables Catégorielles",
        "correlation_analysis": "Analyse de Corrélation",
        "preprocessing": "Prétraitement des Données",
        "missing_treatment": "Traitement des Valeurs Manquantes",
        "duplicate_treatment": "Traitement des Doublons",
        "outlier_treatment": "Traitement des Valeurs Aberrantes",
        "encoding": "Encodage des Variables Catégorielles",
        "scaling": "Mise à l'échelle des Variables Numériques",
        "multicollinearity": "Analyse de Multicollinéarité (VIF)",
        "modeling": "Modélisation et Évaluation",
        "model_training": "Entraînement des Modèles",
        "model_performance": "Performance des Modèles",
        "advanced_analysis": "Analyse Avancée",
        "best_model": "Meilleur Modèle",
        "generate_report": "Générer un Rapport PDF",
        "download_report": "Télécharger le Rapport",
        "data_loaded": "Données chargées avec succès",
        "select_preprocessing": "Sélectionner les options de prétraitement",
        "train_models": "Entraîner les Modèles",
        "view_results": "Voir les Résultats",
        "save_models": "Sauvegarder les Modèles",
        "model_comparison": "Comparaison des Modèles",
        "real_vs_predicted": "Valeurs Réelles vs. Prédites",
        "confusion_matrix": "Matrice de Confusion",
        "roc_curve": "Courbe ROC",
        "feature_importance": "Importance des Variables",
        "statistical_tests": "Tests Statistiques",
        "summary": "Résumé Exécutif"
    }
}

# Estilos CSS personalizados
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0f5b8e;
        color: white;
    }
    .info-box {
        background-color: #e6f2ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f8ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Función para crear PDF
def create_pdf_report(data, lang, analysis_results, models_performance, best_model):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    # Título
    pdf.cell(0, 10, translations[lang]["title"], 0, 1, 'C')
    pdf.ln(10)
    
    # Información general
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, translations[lang]["data_overview"], 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"{translations[lang]['data_shape']}: {data.shape[0]} {translations[lang]['rows']} x {data.shape[1]} {translations[lang]['columns']}", 0, 1)
    
    # Resumen de resultados
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, translations[lang]["summary"], 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"{translations[lang]['best_model']}: {best_model}", 0, 1)
    
    # Resultados de modelos
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, translations[lang]["model_performance"], 0, 1)
    pdf.set_font("Arial", '', 10)
    
    # Crear tabla de resultados
    col_widths = [50, 30, 30, 30, 40]
    headers = ["Model", "MAE", "RMSE", "R²", "Training Time (s)"]
    
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
    pdf.ln()
    
    for model_name, metrics in models_performance.items():
        pdf.cell(col_widths[0], 10, model_name, 1, 0, 'L')
        pdf.cell(col_widths[1], 10, f"${metrics['MAE']:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[2], 10, f"${metrics['RMSE']:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[3], 10, f"{metrics['R2']:.4f}", 1, 0, 'C')
        pdf.cell(col_widths[4], 10, f"{metrics['Training Time']:.2f}", 1, 0, 'C')
        pdf.ln()
    
    # Guardar PDF
    pdf_output = io.BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    pdf_output.seek(0)
    
    return pdf_output

# Funciones para modelos de redes neuronales
def create_sequential_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def create_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def create_hybrid_model(input_dim):
    # Modelo híbrido simplificado (en una implementación real se añadiría lógica difusa)
    model = Sequential()
    model.add(Dense(96, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    return model

# Función principal de la aplicación
def main():
    # Configuración de idioma
    lang = st.sidebar.selectbox("🌐 Idioma / Language / Langue", 
                               options=list(translations.keys()), 
                               format_func=lambda x: {"es": "Español", "en": "English", "fr": "Français"}[x])
    
    # Aplicar estilos CSS
    local_css()
    
    # Título principal
    st.markdown(f'<h1 class="main-header">🏠 {translations[lang]["title"]}</h1>', unsafe_allow_html=True)
    
    # Carga de datos
    st.sidebar.markdown(f"### 📁 {translations[lang]['upload_data']}")
    uploaded_file = st.sidebar.file_uploader(
        translations[lang]["file_types"], 
        type=['csv', 'xlsx'],
        key="file_uploader"
    )
    
    data = None
    if uploaded_file is not None:
        # Leer el archivo
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.sidebar.success(f"✅ {translations[lang]['data_loaded']}: {data.shape[0]} {translations[lang]['rows']} × {data.shape[1]} {translations[lang]['columns']}")
            
            # Selector de variable objetivo
            target_var = st.sidebar.selectbox(
                f"🎯 {translations[lang]['select_target']}", 
                options=data.columns
            )
            
        except Exception as e:
            st.error(f"Error al leer el archivo: {str(e)}")
            return
    
    # Botones principales en sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Acciones de Análisis")
    
    # Inicializar estado de la sesión
    if 'eda_done' not in st.session_state:
        st.session_state.eda_done = False
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    if data is not None:
        # Botón para EDA
        if st.sidebar.button(f"📊 {translations[lang]['start_analysis']}"):
            st.session_state.eda_done = True
            st.session_state.preprocessing_done = False
            st.session_state.models_trained = False
            
        if st.session_state.eda_done:
            # FASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
            st.markdown(f'<h2 class="section-header">📈 Fase 1: {translations[lang]["data_overview"]}</h2>', unsafe_allow_html=True)
            
            # Visión general
            st.markdown(f'<h3 class="subsection-header">🔍 {translations[lang]["data_overview"]}</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**{translations[lang]['data_shape']}:** {data.shape[0]} {translations[lang]['rows']} × {data.shape[1]} {translations[lang]['columns']}")
            
            with col2:
                st.info(f"**{translations[lang]['variables']}:** {', '.join(data.columns)}")
            
            # Clasificación de variables
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            
            st.markdown(f"**{translations[lang]['var_types']}:**")
            type_col1, type_col2 = st.columns(2)
            with type_col1:
                st.write(f"**{translations[lang]['numeric']}:** ({len(numeric_cols)})")
                st.write(", ".join(numeric_cols))
            with type_col2:
                st.write(f"**{translations[lang]['categorical']}:** ({len(categorical_cols)})")
                st.write(", ".join(categorical_cols))
            
            # Diagnóstico de datos
            st.markdown(f'<h3 class="subsection-header">🔎 {translations[lang]["data_diagnosis"]}</h3>', unsafe_allow_html=True)
            
            # Datos faltantes
            missing_data = data.isnull().sum()
            missing_percent = (missing_data / len(data)) * 100
            missing_df = pd.DataFrame({
                'Valores Faltantes': missing_data,
                'Porcentaje': missing_percent
            }).round(2)
            
            st.markdown(f"**{translations[lang]['missing_data']}:**")
            st.dataframe(missing_df[missing_df['Valores Faltantes'] > 0])
            
            # Datos duplicados
            duplicates = data.duplicated().sum()
            st.markdown(f"**{translations[lang]['duplicate_data']}:** {duplicates} ({duplicates/len(data)*100:.2f}%)")
            
            # Outliers usando Z-score
            st.markdown(f"**{translations[lang]['outliers']}:**")
            outliers_df = pd.DataFrame(columns=['Variable', 'Número de Outliers', 'Porcentaje'])
            
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outliers = data[col][z_scores > 3]
                outlier_count = len(outliers)
                outlier_percent = (outlier_count / len(data)) * 100
                
                if outlier_count > 0:
                    new_row = pd.DataFrame({
                        'Variable': [col],
                        'Número de Outliers': [outlier_count],
                        'Porcentaje': [f"{outlier_percent:.2f}%"]
                    })
                    outliers_df = pd.concat([outliers_df, new_row], ignore_index=True)
            
            if len(outliers_df) > 0:
                st.dataframe(outliers_df)
            else:
                st.write("No se detectaron outliers en las variables numéricas.")
            
            # Prueba de normalidad
            st.markdown(f"**{translations[lang]['normality_test']}:**")
            normality_results = pd.DataFrame(columns=['Variable', 'Estadístico', 'p-valor', 'Distribución Normal'])
            
            for col in numeric_cols:
                if len(data[col].dropna()) > 0:
                    stat, p_value = kstest(data[col].dropna(), 'norm')
                    is_normal = "Sí" if p_value > 0.05 else "No"
                    new_row = pd.DataFrame({
                        'Variable': [col],
                        'Estadístico': [f"{stat:.4f}"],
                        'p-valor': [f"{p_value:.4f}"],
                        'Distribución Normal': [is_normal]
                    })
                    normality_results = pd.concat([normality_results, new_row], ignore_index=True)
            
            st.dataframe(normality_results)
            
            # Análisis de variables numéricas
            st.markdown(f'<h3 class="subsection-header">📊 {translations[lang]["numeric_analysis"]}</h3>', unsafe_allow_html=True)
            
            for col in numeric_cols:
                st.markdown(f"**{col}**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Media", f"${data[col].mean():.2f}")
                with col2:
                    st.metric("Mediana", f"${data[col].median():.2f}")
                with col3:
                    st.metric("Desviación Estándar", f"${data[col].std():.2f}")
                with col4:
                    st.metric("Rango", f"${data[col].max() - data[col].min():.2f}")
                
                # Gráficos de distribución
                fig_col1, fig_col2 = st.columns(2)
                
                with fig_col1:
                    fig = px.histogram(data, x=col, title=f"Histograma de {col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with fig_col2:
                    fig = px.box(data, y=col, title=f"Diagrama de Caja de {col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de variables categóricas
            if categorical_cols:
                st.markdown(f'<h3 class="subsection-header">📋 {translations[lang]["categorical_analysis"]}</h3>', unsafe_allow_html=True)
                
                for col in categorical_cols:
                    st.markdown(f"**{col}**")
                    value_counts = data[col].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(value_counts)
                    
                    with col2:
                        fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                    title=f"Distribución de {col}")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de correlación
            st.markdown(f'<h3 class="subsection-header">🔗 {translations[lang]["correlation_analysis"]}</h3>', unsafe_allow_html=True)
            
            correlation_matrix = data[numeric_cols].corr()
            
            fig = px.imshow(correlation_matrix, 
                           title="Matriz de Correlación",
                           color_continuous_scale='RdBu_r',
                           aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlación con la variable objetivo
            if target_var in numeric_cols:
                st.markdown(f"**Correlación con {target_var}:**")
                target_corr = correlation_matrix[target_var].sort_values(ascending=False)
                st.dataframe(target_corr)
            
            # FASE 2: PREPROCESAMIENTO DE DATOS
            st.markdown(f'<h2 class="section-header">⚙️ Fase 2: {translations[lang]["preprocessing"]}</h2>', unsafe_allow_html=True)
            
            st.sidebar.markdown("### ⚙️ Opciones de Preprocesamiento")
            
            # Tratamiento de valores faltantes
            st.sidebar.markdown(f"**{translations[lang]['missing_treatment']}**")
            for col in data.columns:
                if data[col].isnull().sum() > 0:
                    if col in numeric_cols:
                        imp_method = st.sidebar.selectbox(
                            f"Método para {col}",
                            ["Media", "Mediana", "Eliminar"],
                            key=f"imp_{col}"
                        )
                    else:
                        imp_method = st.sidebar.selectbox(
                            f"Método para {col}",
                            ["Moda", "Eliminar", "Valor constante"],
                            key=f"imp_{col}"
                        )
            
            # Tratamiento de duplicados
            if st.sidebar.button("🗑️ Eliminar duplicados"):
                initial_count = len(data)
                data.drop_duplicates(inplace=True)
                final_count = len(data)
                st.sidebar.success(f"Se eliminaron {initial_count - final_count} filas duplicadas.")
            
            # Tratamiento de outliers
            st.sidebar.markdown(f"**{translations[lang]['outlier_treatment']}**")
            for col in numeric_cols:
                treat_outliers = st.sidebar.checkbox(f"Tratar outliers en {col}", key=f"out_{col}")
                
            # Codificación de variables categóricas
            if categorical_cols:
                st.sidebar.markdown(f"**{translations[lang]['encoding']}**")
                for col in categorical_cols:
                    encoding_method = st.sidebar.selectbox(
                        f"Codificación para {col}",
                        ["One-Hot Encoding", "Label Encoding", "Sin cambios"],
                        key=f"enc_{col}"
                    )
            
            # Escalado de variables numéricas
            st.sidebar.markdown(f"**{translations[lang]['scaling']}**")
            scaling_method = st.sidebar.selectbox(
                "Método de escalado",
                ["Estandarización (StandardScaler)", "Normalización (MinMaxScaler)", "Sin escalado"]
            )
            
            if st.sidebar.button("✅ Aplicar Preprocesamiento"):
                st.session_state.preprocessing_done = True
                st.sidebar.success("Preprocesamiento aplicado correctamente")
            
            if st.session_state.preprocessing_done:
                # FASE 3: MODELADO Y EVALUACIÓN
                st.markdown(f'<h2 class="section-header">🤖 Fase 3: {translations[lang]["modeling"]}</h2>', unsafe_allow_html=True)
                
                # División de datos
                X = data.drop(target_var, axis=1)
                y = data[target_var]
                
                # Convertir variables categóricas si es necesario
                if categorical_cols:
                    X = pd.get_dummies(X, columns=categorical_cols)
                
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
                
                st.success(f"Datos divididos: Entrenamiento (70%): {X_train.shape[0]} muestras, Validación (15%): {X_val.shape[0]} muestras, Prueba (15%): {X_test.shape[0]} muestras")
                
                # Entrenamiento de modelos
                if st.sidebar.button(f"🚀 {translations[lang]['train_models']}"):
                    with st.spinner("Entrenando modelos..."):
                        models = {}
                        training_times = {}
                        predictions = {}
                        
                        # Modelo de regresión lineal
                        start_time = time.time()
                        lr_model = LinearRegression()
                        lr_model.fit(X_train, y_train)
                        training_time = time.time() - start_time
                        
                        models["Regresión Lineal"] = lr_model
                        training_times["Regresión Lineal"] = training_time
                        predictions["Regresión Lineal"] = lr_model.predict(X_test)
                        
                        # Red neuronal secuencial
                        start_time = time.time()
                        sequential_model = create_sequential_model(X_train.shape[1])
                        sequential_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                        training_time = time.time() - start_time
                        
                        models["Red Neuronal Secuencial"] = sequential_model
                        training_times["Red Neuronal Secuencial"] = training_time
                        predictions["Red Neuronal Secuencial"] = sequential_model.predict(X_test).flatten()
                        
                        # Perceptrón multicapa
                        start_time = time.time()
                        mlp_model = create_mlp_model(X_train.shape[1])
                        mlp_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                        training_time = time.time() - start_time
                        
                        models["Perceptrón Multicapa"] = mlp_model
                        training_times["Perceptrón Multicapa"] = training_time
                        predictions["Perceptrón Multicapa"] = mlp_model.predict(X_test).flatten()
                        
                        # Modelo híbrido
                        start_time = time.time()
                        hybrid_model = create_hybrid_model(X_train.shape[1])
                        hybrid_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                        training_time = time.time() - start_time
                        
                        models["Modelo Híbrido"] = hybrid_model
                        training_times["Modelo Híbrido"] = training_time
                        predictions["Modelo Híbrido"] = hybrid_model.predict(X_test).flatten()
                        
                        # Guardar modelos
                        for model_name, model in models.items():
                            if hasattr(model, 'save'):
                                model.save(f"{model_name.replace(' ', '_').lower()}.h5")
                        
                        # Calcular métricas
                        models_performance = {}
                        for model_name, y_pred in predictions.items():
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            r2 = r2_score(y_test, y_pred)
                            
                            models_performance[model_name] = {
                                'MAE': mae,
                                'RMSE': rmse,
                                'R2': r2,
                                'Training Time': training_times[model_name]
                            }
                        
                        # Identificar el mejor modelo
                        best_model_name = min(models_performance.keys(), 
                                            key=lambda x: models_performance[x]['MAE'])
                        
                        st.session_state.models_trained = True
                        st.session_state.models_performance = models_performance
                        st.session_state.best_model = best_model_name
                        st.session_state.predictions = predictions
                        st.session_state.y_test = y_test
                
                if st.session_state.models_trained:
                    # Mostrar resultados
                    st.markdown(f'<h3 class="subsection-header">📋 {translations[lang]["model_performance"]}</h3>', unsafe_allow_html=True)
                    
                    performance_df = pd.DataFrame.from_dict(st.session_state.models_performance, orient='index')
                    st.dataframe(performance_df)
                    
                    # Mejor modelo
                    st.markdown(f'<div class="success-box"><h4>🎉 {translations[lang]["best_model"]}: {st.session_state.best_model}</h4></div>', unsafe_allow_html=True)
                    
                    # FASE 4: ANÁLISIS AVANZADO
                    st.markdown(f'<h2 class="section-header">📊 Fase 4: {translations[lang]["advanced_analysis"]}</h2>', unsafe_allow_html=True)
                    
                    # Gráfico de valores reales vs predichos
                    st.markdown(f'<h3 class="subsection-header">📈 {translations[lang]["real_vs_predicted"]}</h3>', unsafe_allow_html=True)
                    
                    best_pred = st.session_state.predictions[st.session_state.best_model]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=st.session_state.y_test, y=best_pred, mode='markers',
                        name='Predicciones', marker=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=[st.session_state.y_test.min(), st.session_state.y_test.max()], 
                        y=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                        mode='lines', name='Línea Perfecta', line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title=f"Valores Reales vs Predichos - {st.session_state.best_model}",
                        xaxis_title="Valores Reales ($)",
                        yaxis_title="Valores Predichos ($)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Generar reporte PDF
                    if st.sidebar.button(f"📄 {translations[lang]['generate_report']}"):
                        pdf_report = create_pdf_report(
                            data, lang, 
                            st.session_state.models_performance, 
                            st.session_state.models_performance,
                            st.session_state.best_model
                        )
                        
                        st.sidebar.download_button(
                            label=translations[lang]["download_report"],
                            data=pdf_report,
                            file_name=f"reporte_estimacion_precios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )

if __name__ == "__main__":
    main()