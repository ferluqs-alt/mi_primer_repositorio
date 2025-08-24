# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import io
import base64
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kstest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Intentar importar scikit-fuzzy, pero continuar si no está disponible
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="Estimador de Precios de Viviendas",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Internacionalización (i18n)
language_dict = {
    "ES": {
        "title": "Estimador de Precios de Viviendas",
        "welcome": "Bienvenido al estimador de precios de viviendas",
        "upload_data": "Cargar datos",
        "file_types": "Tipos de archivo admitidos: CSV, XLSX",
        "select_file": "Seleccione un archivo",
        "eda": "Análisis Exploratorio (EDA)",
        "correlations": "Análisis de Correlaciones",
        "preprocessing": "Limpieza y Preprocesamiento",
        "scaling": "Escalado y División",
        "modeling": "Modelado",
        "validation": "Entrenamiento y Validación",
        "timeseries": "Pronósticos con Series Temporales",
        "report": "Generar Reporte PDF",
        "records": "registros",
        "variables": "variables",
        "numerical": "Numéricas",
        "categorical": "Categóricas",
        "missing_data": "Datos faltantes",
        "duplicates": "Datos duplicados",
        "descriptive_stats": "Estadísticas Descriptivas",
        "normality_test": "Prueba de Normalidad",
        "visualization": "Visualización",
        "correlation_matrix": "Matriz de Correlación",
        "correlation_heatmap": "Mapa de Calor de Correlaciones",
        "strongest_correlations": "Correlaciones más Fuertes con el Precio",
        "multicolinearity": "Análisis de Multicolinealidad",
        "vif": "Factor de Inflación de Varianza (VIF)",
        "preprocessing_options": "Opciones de Preprocesamiento",
        "imputation": "Imputación de Valores Faltantes",
        "outlier_treatment": "Tratamiento de Outliers",
        "encoding": "Codificación de Variables Categóricas",
        "scaling_options": "Opciones de Escalado",
        "train_val_test_split": "División Entrenamiento/Validación/Prueba",
        "cross_validation": "Validación Cruzada",
        "model_training": "Entrenamiento de Modelos",
        "model_performance": "Rendimiento de Modelos",
        "timeseries_forecasting": "Pronóstico de Series Temporales",
        "download_report": "Descargar Reporte PDF",
        "linear_regression": "Regresión Lineal",
        "mlp": "Perceptrón Multicapa (MLP)",
        "sequential_nn": "Red Neuronal Secuencial",
        "hybrid_model": "Modelo Híbrido (Red Neuronal + Lógica Difusa)",
        "train_models": "Entrenar Modelos",
        "generate_report": "Generar Reporte",
        "imputation_complete": "Imputación completada exitosamente",
        "select_imputation": "Seleccione método de imputación",
        "mean_imputation": "Imputación por Media/Moda",
        "knn_imputation": "Imputación KNN",
        "iterative_imputation": "Imputación Iterativa",
        "no_data_loaded": "No se han cargado datos. Por favor, cargue un dataset primero.",
        "data_loaded": "Datos cargados: {} registros, {} variables",
        "preprocessing_first": "Primero debe realizar el preprocesamiento de datos",
        "select_target": "Seleccione la variable objetivo",
        "no_numerical_vars": "No hay variables numéricas para análisis",
        "no_models_trained": "No hay modelos entrenados. Por favor, entrene algunos modelos primero.",
        "skfuzzy_not_available": "scikit-fuzzy no está disponible. Instálelo con: pip install scikit-fuzzy"
    },
    "EN": {
        "title": "Housing Price Estimator",
        "welcome": "Welcome to the housing price estimator",
        "upload_data": "Upload data",
        "file_types": "Supported file types: CSV, XLSX",
        "select_file": "Select a file",
        "eda": "Exploratory Data Analysis (EDA)",
        "correlations": "Correlation Analysis",
        "preprocessing": "Cleaning and Preprocessing",
        "scaling": "Scaling and Splitting",
        "modeling": "Modeling",
        "validation": "Training and Validation",
        "timeseries": "Time Series Forecasting",
        "report": "Generate PDF Report",
        "records": "records",
        "variables": "variables",
        "numerical": "Numerical",
        "categorical": "Categorical",
        "missing_data": "Missing data",
        "duplicates": "Duplicate data",
        "descriptive_stats": "Descriptive Statistics",
        "normality_test": "Normality Test",
        "visualization": "Visualization",
        "correlation_matrix": "Correlation Matrix",
        "correlation_heatmap": "Correlation Heatmap",
        "strongest_correlations": "Strongest Correlations with Price",
        "multicolinearity": "Multicollinearity Analysis",
        "vif": "Variance Inflation Factor (VIF)",
        "preprocessing_options": "Preprocessing Options",
        "imputation": "Missing Value Imputation",
        "outlier_treatment": "Outlier Treatment",
        "encoding": "Categorical Variable Encoding",
        "scaling_options": "Scaling Options",
        "train_val_test_split": "Train/Validation/Test Split",
        "cross_validation": "Cross Validation",
        "model_training": "Model Training",
        "model_performance": "Model Performance",
        "timeseries_forecasting": "Time Series Forecasting",
        "download_report": "Download PDF Report",
        "linear_regression": "Linear Regression",
        "mlp": "Multilayer Perceptron (MLP)",
        "sequential_nn": "Sequential Neural Network",
        "hybrid_model": "Hybrid Model (Neural Network + Fuzzy Logic)",
        "train_models": "Train Models",
        "generate_report": "Generate Report",
        "imputation_complete": "Imputation completed successfully",
        "select_imputation": "Select imputation method",
        "mean_imputation": "Mean/Mode Imputation",
        "knn_imputation": "KNN Imputation",
        "iterative_imputation": "Iterative Imputation",
        "no_data_loaded": "No data loaded. Please upload a dataset first.",
        "data_loaded": "Data loaded: {} records, {} variables",
        "preprocessing_first": "You must first preprocess the data",
        "select_target": "Select target variable",
        "no_numerical_vars": "No numerical variables for analysis",
        "no_models_trained": "No models trained. Please train some models first.",
        "skfuzzy_not_available": "scikit-fuzzy is not available. Install with: pip install scikit-fuzzy"
    },
    "FR": {
        "title": "Estimateur de Prix Immobiliers",
        "welcome": "Bienvenue dans l'estimateur de prix immobiliers",
        "upload_data": "Télécharger des données",
        "file_types": "Types de fichiers pris en charge: CSV, XLSX",
        "select_file": "Sélectionner un fichier",
        "eda": "Analyse Exploratoire (EDA)",
        "correlations": "Analyse des Corrélations",
        "preprocessing": "Nettoyage et Prétraitement",
        "scaling": "Mise à l'échelle et Division",
        "modeling": "Modélisation",
        "validation": "Entraînement et Validation",
        "timeseries": "Prévisions de Séries Chronologiques",
        "report": "Générer un Rapport PDF",
        "records": "enregistrements",
        "variables": "variables",
        "numerical": "Numériques",
        "categorical": "Catégorielles",
        "missing_data": "Données manquantes",
        "duplicates": "Données dupliquées",
        "descriptive_stats": "Statistiques Descriptives",
        "normality_test": "Test de Normalité",
        "visualization": "Visualisation",
        "correlation_matrix": "Matrice de Corrélation",
        "correlation_heatmap": "Carte de Chaleur des Corrélations",
        "strongest_correlations": "Corrélations les Plus Fortes avec le Prix",
        "multicolinearity": "Analyse de Multicolinéarité",
        "vif": "Facteur d'Inflation de la Variance (VIF)",
        "preprocessing_options": "Options de Prétraitement",
        "imputation": "Imputation des Valeurs Manquantes",
        "outlier_treatment": "Traitement des Valeurs Abérrantes",
        "encoding": "Encodage des Variables Catégorielles",
        "scaling_options": "Options de Mise à l'échelle",
        "train_val_test_split": "Division Entraînement/Validation/Test",
        "cross_validation": "Validation Croisée",
        "model_training": "Entraînement des Modèles",
        "model_performance": "Performance des Modèles",
        "timeseries_forecasting": "Prévision de Séries Chronologiques",
        "download_report": "Télécharger le Rapport PDF",
        "linear_regression": "Régression Linéaire",
        "mlp": "Perceptron Multicouche (MLP)",
        "sequential_nn": "Réseau Neuronal Séquentiel",
        "hybrid_model": "Modèle Hybride (Réseau Neuronal + Logique Floue)",
        "train_models": "Entraîner les Modèles",
        "generate_report": "Générer Rapport",
        "imputation_complete": "Imputation terminée avec succès",
        "select_imputation": "Sélectionnez la méthode d'imputation",
        "mean_imputation": "Imputation par Moyenne/Mode",
        "knn_imputation": "Imputation KNN",
        "iterative_imputation": "Imputation Itérative",
        "no_data_loaded": "Aucune donnée chargée. Veuillez télécharger un jeu de datos d'abord.",
        "data_loaded": "Données chargées: {} enregistrements, {} variables",
        "preprocessing_first": "Vous devez d'abord prétraiter les datos",
        "select_target": "Sélectionnez la variable cible",
        "no_numerical_vars": "Aucune variable numérique pour l'analyse",
        "no_models_trained": "Aucun modèle entraîné. Veuillez d'abord entraîner des modèles.",
        "skfuzzy_not_available": "scikit-fuzzy n'est pas disponible. Installez-le avec: pip install scikit-fuzzy"
    }
}

# Inicialización del estado de la sesión
def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'language' not in st.session_state:
        st.session_state.language = "ES"
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_val' not in st.session_state:
        st.session_state.X_val = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_val' not in st.session_state:
        st.session_state.y_val = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "upload"
    if 'imputation_method' not in st.session_state:
        st.session_state.imputation_method = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'model_comparison_df' not in st.session_state:
        st.session_state.model_comparison_df = None

# Función para obtener texto según el idioma
def get_text(key):
    try:
        return language_dict[st.session_state.language][key]
    except KeyError:
        return f"[{key}]"

# CSS personalizado para el diseño
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #0f5d94;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #e6f2ff;
    }
    .metric-card {
        background-color: #e6f2ff;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border: 1px solid #ffeeba;
    }
    .model-comparison-table {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .best-model {
        background-color: #d4edda !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Función para cargar datos
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de archivo no compatible")
            return None
        
        st.success(get_text("data_loaded").format(df.shape[0], df.shape[1]))
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

# Función para realizar EDA
def perform_eda():
    if st.session_state.data is None:
        st.warning(get_text("no_data_loaded"))
        return
    
    df = st.session_state.data
    st.subheader(get_text("eda"))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(get_text("records"), df.shape[0])
    with col2:
        st.metric(get_text("variables"), df.shape[1])
    with col3:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.metric(get_text("numerical"), len(numerical_cols))
        st.metric(get_text("categorical"), len(categorical_cols))
    
    # Detección de datos nulos y duplicados
    st.subheader(get_text("missing_data"))
    missing_data = pd.DataFrame({
        'Columna': df.columns,
        'Valores nulos': df.isnull().sum().values,
        'Porcentaje nulos': (df.isnull().sum().values / df.shape[0]) * 100
    })
    st.dataframe(missing_data)
    
    st.subheader(get_text("duplicates"))
    st.write(f"Número de registros duplicados: {df.duplicated().sum()}")
    
    # Estadísticas descriptivas
    st.subheader(get_text("descriptive_stats"))
    if len(numerical_cols) > 0:
        st.dataframe(df[numerical_cols].describe())
    else:
        st.warning(get_text("no_numerical_vars"))
    
    # Prueba de normalidad
    st.subheader(get_text("normality_test"))
    if len(numerical_cols) > 0:
        normality_results = []
        for col in numerical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                try:
                    stat, p_value = kstest(col_data, 'norm')
                    normality_results.append({
                        'Variable': col,
                        'Estadístico': stat,
                        'p-valor': p_value,
                        'Normal': p_value > 0.05
                    })
                except:
                    continue
        if normality_results:
            normality_df = pd.DataFrame(normality_results)
            st.dataframe(normality_df)
        else:
            st.warning("No se pudieron realizar pruebas de normalidad")
    
    # Visualización
    st.subheader(get_text("visualization"))
    if numerical_cols:
        plot_type = st.selectbox("Tipo de gráfico", ["Histograma", "Boxplot", "Scatterplot"])
        
        if plot_type == "Histograma":
            col_to_plot = st.selectbox("Seleccione columna", numerical_cols)
            fig = px.histogram(df, x=col_to_plot, title=f"Histograma de {col_to_plot}")
            st.plotly_chart(fig)
        
        elif plot_type == "Boxplot":
            col_to_plot = st.selectbox("Seleccione columna", numerical_cols)
            fig = px.box(df, y=col_to_plot, title=f"Boxplot de {col_to_plot}")
            st.plotly_chart(fig)
        
        elif plot_type == "Scatterplot" and len(numerical_cols) >= 2:
            col_x = st.selectbox("Seleccione variable X", numerical_cols)
            col_y = st.selectbox("Seleccione variable Y", numerical_cols)
            fig = px.scatter(df, x=col_x, y=col_y, title=f"Scatterplot: {col_x} vs {col_y}")
            st.plotly_chart(fig)

# Función para análisis de correlaciones
def perform_correlation_analysis():
    if st.session_state.data is None:
        st.warning(get_text("no_data_loaded"))
        return
    
    df = st.session_state.data
    st.subheader(get_text("correlations"))
    
    numerical_df = df.select_dtypes(include=[np.number])
    if numerical_df.empty:
        st.warning(get_text("no_numerical_vars"))
        return
    
    corr_matrix = numerical_df.corr()
    
    st.subheader(get_text("correlation_matrix"))
    st.dataframe(corr_matrix)
    
    st.subheader(get_text("correlation_heatmap"))
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)
    
    st.subheader(get_text("strongest_correlations"))
    if st.session_state.target_column and st.session_state.target_column in corr_matrix.columns:
        target_correlations = corr_matrix[st.session_state.target_column].sort_values(key=abs, ascending=False)
        st.dataframe(target_correlations)
        
        top_correlated = target_correlations.index[1:4]
        for col in top_correlated:
            if col in df.columns and st.session_state.target_column in df.columns:
                fig = px.scatter(df, x=col, y=st.session_state.target_column, 
                                title=f"{col} vs {st.session_state.target_column}")
                st.plotly_chart(fig)
    
    st.subheader(get_text("multicolinearity"))
    if len(numerical_df.columns) > 1:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = numerical_df.columns
        
        vif_values = []
        for i in range(len(numerical_df.columns)):
            temp_df = numerical_df.dropna()
            if len(temp_df) > 0:
                try:
                    vif = variance_inflation_factor(temp_df.values, i)
                    vif_values.append(vif)
                except:
                    vif_values.append(np.nan)
            else:
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        st.dataframe(vif_data)

# Función para preprocesamiento de datos
def perform_preprocessing():
    if st.session_state.data is None:
        st.warning(get_text("no_data_loaded"))
        return
    
    df = st.session_state.data
    st.subheader(get_text("preprocessing"))
    
    processed_df = df.copy()
    
    # Selección de columna objetivo
    numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        target_col = st.selectbox(get_text("select_target"), numerical_cols)
        st.session_state.target_column = target_col
    
    # Imputación de valores faltantes
    st.subheader(get_text("imputation"))
    st.session_state.imputation_method = st.selectbox(get_text("select_imputation"), 
                                    [get_text("mean_imputation"), 
                                     get_text("knn_imputation"), 
                                     get_text("iterative_imputation")])
    
    if st.button("Aplicar Imputación"):
        with st.spinner("Aplicando imputación..."):
            numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
            
            if st.session_state.imputation_method == get_text("mean_imputation"):
                for col in numerical_cols:
                    if processed_df[col].isnull().sum() > 0:
                        try:
                            mean_val = processed_df[col].mean()
                            processed_df[col].fillna(mean_val, inplace=True)
                        except:
                            st.warning(f"No se pudo calcular la media para {col}")
                
                for col in categorical_cols:
                    if processed_df[col].isnull().sum() > 0:
                        try:
                            mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else "Desconocido"
                            processed_df[col].fillna(mode_val, inplace=True)
                        except:
                            st.warning(f"No se pudo calcular la moda para {col}")
            
            elif st.session_state.imputation_method == get_text("knn_imputation"):
                try:
                    numerical_imputer = KNNImputer(n_neighbors=5)
                    processed_df[numerical_cols] = numerical_imputer.fit_transform(processed_df[numerical_cols])
                    
                    for col in categorical_cols:
                        if processed_df[col].isnull().sum() > 0:
                            try:
                                mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else "Desconocido"
                                processed_df[col].fillna(mode_val, inplace=True)
                            except:
                                st.warning(f"No se pudo calcular la moda para {col}")
                except Exception as e:
                    st.error(f"Error en imputación KNN: {str(e)}")
            
            elif st.session_state.imputation_method == get_text("iterative_imputation"):
                try:
                    numerical_imputer = IterativeImputer(random_state=42)
                    processed_df[numerical_cols] = numerical_imputer.fit_transform(processed_df[numerical_cols])
                    
                    for col in categorical_cols:
                        if processed_df[col].isnull().sum() > 0:
                            try:
                                mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else "Desconocido"
                                processed_df[col].fillna(mode_val, inplace=True)
                            except:
                                st.warning(f"No se pudo calcular la moda para {col}")
                except Exception as e:
                    st.error(f"Error en imputación iterativa: {str(e)}")
            
            st.session_state.processed_data = processed_df
            st.markdown(f'<div class="success-box">{get_text("imputation_complete")}</div>', unsafe_allow_html=True)
            st.dataframe(processed_df.head())
    
    # Tratamiento de outliers (solo si ya se aplicó imputación)
    if st.session_state.processed_data is not None:
        st.subheader(get_text("outlier_treatment"))
        outlier_treatment = st.selectbox("Método de tratamiento de outliers", 
                                        ["Ninguno", "Eliminación", "Transformación"])
        
        if outlier_treatment != "Ninguno" and st.session_state.target_column:
            numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            if st.session_state.target_column in numerical_cols:
                numerical_cols.remove(st.session_state.target_column)
            
            for col in numerical_cols:
                try:
                    Q1 = processed_df[col].quantile(0.25)
                    Q3 = processed_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    if outlier_treatment == "Eliminación":
                        processed_df = processed_df[(processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)]
                    elif outlier_treatment == "Transformación":
                        processed_df[col] = np.log1p(processed_df[col])
                except:
                    st.warning(f"No se pudo procesar outliers para {col}")
            
            st.session_state.processed_data = processed_df
            st.success("Tratamiento de outliers completado")
            st.dataframe(processed_df.head())
    
    # Codificación de variables categóricas
    if st.session_state.processed_data is not None:
        st.subheader(get_text("encoding"))
        encoding_method = st.selectbox("Método de codificación", 
                                      ["Ninguno", "One-Hot Encoding", "Label Encoding"])
        
        if encoding_method != "Ninguno":
            categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
            
            if encoding_method == "One-Hot Encoding":
                try:
                    processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
                except Exception as e:
                    st.error(f"Error en One-Hot Encoding: {str(e)}")
            elif encoding_method == "Label Encoding":
                try:
                    le = LabelEncoder()
                    for col in categorical_cols:
                        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                except Exception as e:
                    st.error(f"Error en Label Encoding: {str(e)}")
            
            st.session_state.processed_data = processed_df
            st.success("Codificación completada")
            st.dataframe(processed_df.head())
    
    # Escalado de datos
    if st.session_state.processed_data is not None:
        st.subheader(get_text("scaling_options"))
        scaling_method = st.selectbox("Método de escalado", ["Ninguno", "StandardScaler"])
        
        if scaling_method != "Ninguno" and st.session_state.target_column:
            numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            if st.session_state.target_column in numerical_cols:
                numerical_cols.remove(st.session_state.target_column)
            
            if scaling_method == "StandardScaler":
                try:
                    scaler = StandardScaler()
                    processed_df[numerical_cols] = scaler.fit_transform(processed_df[numerical_cols])
                    st.session_state.scaler = scaler
                    st.session_state.processed_data = processed_df
                    st.success("Escalado completado")
                    st.dataframe(processed_df.head())
                except Exception as e:
                    st.error(f"Error en escalado: {str(e)}")
    
    # División de datos
    if st.session_state.processed_data is not None and st.session_state.target_column:
        st.subheader(get_text("train_val_test_split"))
        if st.session_state.target_column in processed_df.columns:
            X = processed_df.drop(st.session_state.target_column, axis=1)
            y = processed_df[st.session_state.target_column]
            
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            
            st.write(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
            st.write(f"Conjunto de validación: {X_val.shape[0]} muestras")
            st.write(f"Conjunto de prueba: {X_test.shape[0]} muestras")
            
            st.session_state.X_train = X_train
            st.session_state.X_val = X_val
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_val = y_val
            st.session_state.y_test = y_test
            
            st.subheader(get_text("cross_validation"))
            k_folds = st.slider("Número de folds para validación cruzada", 3, 10, 5)

# Función para preparar datos para modelado - CORREGIDA
def prepare_data_for_modeling(X_data):
    """Prepara los datos eliminando o codificando variables categóricas"""
    if X_data is None or X_data.empty:
        st.error("Datos de entrada vacíos o nulos")
        return pd.DataFrame()
    
    X_clean = X_data.copy()
    
    # Guardar el índice original
    original_index = X_clean.index
    
    # Identificar columnas no numéricas
    non_numeric_cols = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if non_numeric_cols:
        st.warning(f"Columnas no numéricas encontradas: {non_numeric_cols}. Aplicando codificación...")
        
        # Para cada columna no numérica, intentar codificación
        cols_to_drop = []
        for col in non_numeric_cols:
            try:
                # Intentar convertir a numérico primero
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
                
                # Si todavía hay NaN, usar Label Encoding
                if X_clean[col].isnull().any():
                    le = LabelEncoder()
                    # Rellenar NaN con un valor temporal para encoding
                    X_clean[col] = X_clean[col].fillna('Missing')
                    X_clean[col] = le.fit_transform(X_clean[col].astype(str))
            except:
                # Si falla la conversión, usar Label Encoding directamente
                try:
                    le = LabelEncoder()
                    X_clean[col] = X_clean[col].fillna('Missing')
                    X_clean[col] = le.fit_transform(X_clean[col].astype(str))
                except:
                    st.warning(f"No se pudo codificar la columna {col}. Se eliminará.")
                    cols_to_drop.append(col)
        
        # Eliminar columnas problemáticas
        if cols_to_drop:
            X_clean = X_clean.drop(columns=cols_to_drop)
    
    # Verificar y manejar valores NaN - CORREGIDO
    if X_clean.isnull().any().any():
        st.warning("Existen valores NaN. Aplicando imputación...")
        
        # Identificar columnas numéricas después de la codificación
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Crear una copia para la imputación
            X_numeric = X_clean[numeric_cols].copy()
            
            # Aplicar imputación
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X_numeric)
            
            # Crear DataFrame con los datos imputados, asegurando la consistencia de índices
            X_imputed_df = pd.DataFrame(X_imputed, columns=numeric_cols)
            X_imputed_df.index = original_index  # Restaurar el índice original
            
            # Reemplazar las columnas numéricas con los datos imputados
            X_clean[numeric_cols] = X_imputed_df
    
    # Asegurar que el índice sea consistente
    X_clean.index = original_index
    
    # Verificar que no haya valores NaN restantes
    if X_clean.isnull().any().any():
        # Si todavía hay NaN, eliminarlos
        X_clean = X_clean.dropna()
        st.warning("Se eliminaron filas con valores NaN restantes")
    
    return X_clean

# Modelo híbrido de red neuronal con lógica difusa (versión simplificada)
def create_hybrid_model(input_dim):
    # Parte de red neuronal
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='mse', 
                 metrics=['mae'])
    
    return model

# Función para crear modelo de Red Neuronal Profunda
def create_deep_nn_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                 loss='mse', 
                 metrics=['mae'])
    
    return model

# Función para crear modelo de Red Neuronal Regularizada
def create_regularized_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='mse', 
                 metrics=['mae'])
    
    return model

# Función para crear modelo Perceptrón Multicapa
def create_mlp_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', 
                 loss='mse', 
                 metrics=['mae'])
    
    return model

# Función para entrenar modelos - CORREGIDA
def train_models():
    if (st.session_state.X_train is None or st.session_state.y_train is None or 
        st.session_state.X_val is None or st.session_state.y_val is None):
        st.warning(get_text("preprocessing_first"))
        return
    
    st.subheader(get_text("model_training"))
    
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_val = st.session_state.X_val
    y_val = st.session_state.y_val
    
    # Preparar datos para modelado con manejo de errores
    try:
        X_train_clean = prepare_data_for_modeling(X_train)
        X_val_clean = prepare_data_for_modeling(X_val)
        
        # Verificar que tenemos datos después del preprocesamiento
        if X_train_clean.empty or X_val_clean.empty:
            st.error("No hay datos válidos para entrenar después del preprocesamiento.")
            return
            
        # Verificar que las dimensiones coincidan
        if X_train_clean.shape[1] != X_val_clean.shape[1]:
            st.error(f"Dimensiones inconsistentes: X_train tiene {X_train_clean.shape[1]} features, X_val tiene {X_val_clean.shape[1]} features")
            return
            
    except Exception as e:
        st.error(f"Error preparando datos para modelado: {str(e)}")
        return
    
    models = {}
    
    # 1. Regresión Lineal Múltiple
    if st.button("Regresión Lineal Múltiple"):
        with st.spinner("Entrenando Regresión Lineal Múltiple..."):
            try:
                start_time = time.time()
                lr_model = LinearRegression()
                lr_model.fit(X_train_clean, y_train)
                training_time = time.time() - start_time
                
                models['Regresión Lineal Múltiple'] = {
                    'model': lr_model,
                    'training_time': training_time,
                    'feature_names': X_train_clean.columns.tolist()
                }
                
                st.success(f"Regresión Lineal Múltiple entrenada en {training_time:.2f} segundos")
                
                # Mostrar coeficientes
                if hasattr(lr_model, 'coef_'):
                    coef_df = pd.DataFrame({
                        'Variable': X_train_clean.columns,
                        'Coeficiente': lr_model.coef_
                    }).sort_values('Coeficiente', key=abs, ascending=False)
                    
                    st.subheader("Coeficientes del Modelo")
                    st.dataframe(coef_df.head(10))
                
            except Exception as e:
                st.error(f"Error entrenando Regresión Lineal Múltiple: {str(e)}")
    
    # 2. Perceptrón Multicapa (MLP)
    if st.button("Perceptrón Multicapa (MLP)"):
        with st.spinner("Entrenando Perceptrón Multicapa..."):
            try:
                start_time = time.time()
                
                mlp_model = create_mlp_model(X_train_clean.shape[1])
                
                # Verificar que los datos no estén vacíos
                if len(X_train_clean) == 0 or len(y_train) == 0:
                    st.error("Datos de entrenamiento vacíos")
                    return
                
                history = mlp_model.fit(X_train_clean, y_train, 
                                       epochs=100, batch_size=32, 
                                       validation_data=(X_val_clean, y_val),
                                       verbose=0)
                training_time = time.time() - start_time
                
                models['Perceptrón Multicapa (MLP)'] = {
                    'model': mlp_model,
                    'training_time': training_time,
                    'history': history,
                    'feature_names': X_train_clean.columns.tolist()
                }
                
                st.success(f"Perceptrón Multicapa entrenado en {training_time:.2f} segundos")
                
                # Gráfico de pérdida durante el entrenamiento
                fig = px.line(x=range(len(history.history['loss'])), 
                             y=history.history['loss'], 
                             labels={'x': 'Época', 'y': 'Pérdida'}, 
                             title='Pérdida durante el entrenamiento (MLP)')
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error entrenando Perceptrón Multicapa: {str(e)}")
    
    # 3. Red Neuronal Regularizada
    if st.button("Red Neuronal Regularizada"):
        with st.spinner("Entrenando Red Neuronal Regularizada..."):
            try:
                start_time = time.time()
                
                regularized_model = create_regularized_nn_model(X_train_clean.shape[1])
                
                history = regularized_model.fit(X_train_clean, y_train, 
                                              epochs=150, batch_size=32, 
                                              validation_data=(X_val_clean, y_val),
                                              verbose=0)
                training_time = time.time() - start_time
                
                models['Red Neuronal Regularizada'] = {
                    'model': regularized_model,
                    'training_time': training_time,
                    'history': history,
                    'feature_names': X_train_clean.columns.tolist()
                }
                
                st.success(f"Red Neuronal Regularizada entrenada en {training_time:.2f} segundos")
                
                # Gráfico de pérdida durante el entrenamiento
                fig = px.line(x=range(len(history.history['loss'])), 
                             y=history.history['loss'], 
                             labels={'x': 'Época', 'y': 'Pérdida'}, 
                             title='Pérdida durante el entrenamiento (Red Regularizada)')
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error entrenando Red Neuronal Regularizada: {str(e)}")
    
    # 4. Red Neuronal Profunda
    if st.button("Red Neuronal Profunda"):
        with st.spinner("Entrenando Red Neuronal Profunda..."):
            try:
                start_time = time.time()
                
                deep_model = create_deep_nn_model(X_train_clean.shape[1])
                
                history = deep_model.fit(X_train_clean, y_train, 
                                       epochs=200, batch_size=32, 
                                       validation_data=(X_val_clean, y_val),
                                       verbose=0)
                training_time = time.time() - start_time
                
                models['Red Neuronal Profunda'] = {
                    'model': deep_model,
                    'training_time': training_time,
                    'history': history,
                    'feature_names': X_train_clean.columns.tolist()
                }
                
                st.success(f"Red Neuronal Profunda entrenada en {training_time:.2f} segundos")
                
                # Gráfico de pérdida durante el entrenamiento
                fig = px.line(x=range(len(history.history['loss'])), 
                             y=history.history['loss'], 
                             labels={'x': 'Época', 'y': 'Pérdida'}, 
                             title='Pérdida durante el entrenamiento (Red Profunda)')
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error entrenando Red Neuronal Profunda: {str(e)}")
    
    # 5. Modelo Híbrido (Red Neuronal) - Versión simplificada sin scikit-fuzzy
    if st.button(get_text("hybrid_model")):
        if not SKFUZZY_AVAILABLE:
            st.warning(get_text("skfuzzy_not_available"))
            st.info("Usando una red neuronal profunda como alternativa...")
        
        with st.spinner("Entrenando Modelo Híbrido..."):
            try:
                start_time = time.time()
                
                # Verificar que los datos estén disponibles
                if X_train_clean is None or X_train_clean.empty:
                    st.error("Datos de entrenamiento no disponibles")
                    return
                
                # Parte de red neuronal (alternativa cuando scikit-fuzzy no está disponible)
                hybrid_model = create_hybrid_model(X_train_clean.shape[1])
                
                # Verificar que los datos de entrenamiento no estén vacíos
                if len(X_train_clean) == 0 or len(y_train) == 0:
                    st.error("Datos de entrenamiento vacíos")
                    return
                
                history = hybrid_model.fit(X_train_clean, y_train, 
                                          epochs=100, batch_size=32, 
                                          validation_data=(X_val_clean, y_val),
                                          verbose=0)
                training_time = time.time() - start_time
                
                models['Modelo Híbrido'] = {
                    'model': hybrid_model,
                    'training_time': training_time,
                    'history': history,
                    'feature_names': X_train_clean.columns.tolist()
                }
                
                st.success(f"Modelo Híbrido entrenado en {training_time:.2f} segundos")
                
                # Gráfico de pérdida durante el entrenamiento
                fig = px.line(x=range(len(history.history['loss'])), 
                             y=history.history['loss'], 
                             labels={'x': 'Época', 'y': 'Pérdida'}, 
                             title='Pérdida durante el entrenamiento (Modelo Híbrido)')
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error entrenando Modelo Híbrido: {str(e)}")
    
    # Actualizar modelos en el estado de la sesión
    if models:
        st.session_state.models.update(models)
        st.success(f"{len(models)} modelo(s) entrenado(s) exitosamente")

# Función para crear tabla comparativa de modelos
def create_model_comparison_table():
    if not st.session_state.models:
        st.warning("No hay modelos entrenados para comparar")
        return None
    
    if st.session_state.X_val is None or st.session_state.y_val is None:
        st.warning("No hay datos de validación disponibles")
        return None
    
    # Preparar datos de validación
    try:
        X_val_clean = prepare_data_for_modeling(st.session_state.X_val)
        y_val = st.session_state.y_val
        
        if X_val_clean.empty:
            st.error("Datos de validación vacíos después del preprocesamiento")
            return None
            
    except Exception as e:
        st.error(f"Error preparando datos de validación: {str(e)}")
        return None
    
    results = []
    for name, model_info in st.session_state.models.items():
        model = model_info['model']
        training_time = model_info['training_time']
        
        if hasattr(model, 'predict'):
            try:
                y_pred = model.predict(X_val_clean)
                
                if len(y_pred.shape) > 1:
                    y_pred = y_pred.flatten()
                
                # Asegurar que las longitudes coincidan
                if len(y_pred) != len(y_val):
                    min_length = min(len(y_pred), len(y_val))
                    y_pred = y_pred[:min_length]
                    y_val_subset = y_val.iloc[:min_length] if hasattr(y_val, 'iloc') else y_val[:min_length]
                else:
                    y_val_subset = y_val
                
                mae = mean_absolute_error(y_val_subset, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val_subset, y_pred))
                r2 = r2_score(y_val_subset, y_pred)
                
                results.append({
                    'Modelo': name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R²': r2,
                    'Tiempo de Entrenamiento (s)': training_time
                })
            except Exception as e:
                st.warning(f"No se pudieron calcular métricas para {name}: {str(e)}")
    
    if results:
        # Crear DataFrame con los resultados
        results_df = pd.DataFrame(results)
        
        # Ordenar por R² de mayor a menor
        results_df = results_df.sort_values('R²', ascending=False)
        
        return results_df
    else:
        st.warning("No se pudieron calcular métricas para los modelos")
        return None

# Función para mostrar tabla comparativa de modelos
def show_model_comparison():
    st.subheader("📊 Tabla Comparativa de Modelos")
    
    # Crear o actualizar la tabla comparativa
    if st.button("🔄 Actualizar Tabla Comparativa"):
        st.session_state.model_comparison_df = create_model_comparison_table()
    
    if st.session_state.model_comparison_df is not None:
        # Formatear la tabla para mejor visualización
        display_df = st.session_state.model_comparison_df.copy()
        display_df['MAE'] = display_df['MAE'].map('{:,.4f}'.format)
        display_df['RMSE'] = display_df['RMSE'].map('{:,.4f}'.format)
        display_df['R²'] = display_df['R²'].map('{:,.4f}'.format)
        display_df['Tiempo de Entrenamiento (s)'] = display_df['Tiempo de Entrenamiento (s)'].map('{:,.2f}'.format)
        
        # Mostrar la tabla con estilo
        st.markdown("""
        <div class='model-comparison-table'>
            <h3>Comparación del Rendimiento de Modelos</h3>
            <p>Tabla ordenada por R² (coeficiente de determinación) de mayor a menor</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Crear una copia para aplicar estilos
        styled_df = st.session_state.model_comparison_df.copy()
        
        # Aplicar estilo a la tabla
        def highlight_best(s):
            is_max = s == s.max()
            is_min = s == s.min()
            return ['background-color: #d4edda' if v else '' for v in is_max] if s.name in ['R²'] else ['background-color: #f8d7da' if v else '' for v in is_min] if s.name in ['MAE', 'RMSE', 'Tiempo de Entrenamiento (s)'] else ['' for _ in s]
        
        # Aplicar estilos
        styled_df = styled_df.style.apply(highlight_best, subset=['MAE', 'RMSE', 'R²', 'Tiempo de Entrenamiento (s)'])
        
        # Mostrar tabla
        st.dataframe(styled_df.format({
            'MAE': '{:,.4f}',
            'RMSE': '{:,.4f}', 
            'R²': '{:,.4f}',
            'Tiempo de Entrenamiento (s)': '{:,.2f}'
        }))
        
        # Mostrar el mejor modelo
        best_model = st.session_state.model_comparison_df.iloc[0]
        st.success(f"🏆 **Mejor modelo: {best_model['Modelo']}** - R²: {best_model['R²']:.4f}")
        
        # Explicación de las métricas
        with st.expander("📖 Explicación de las Métricas"):
            st.write("""
            - **MAE (Error Absoluto Medio)**: Promedio de los errores absolutos. Menor es mejor.
            - **RMSE (Raíz del Error Cuadrático Medio)**: Raíz cuadrada del promedio de errores al cuadrado. Penaliza más los errores grandes. Menor es mejor.
            - **R² (Coeficiente de Determinación)**: Proporción de la varianza explicada por el modelo. Mayor es mejor (máximo 1.0).
            - **Tiempo de Entrenamiento**: Tiempo requerido para entrenar el modelo. Menor es mejor.
            """)
        
        # Gráfico de comparación
        st.subheader("📈 Comparación Visual de Métricas")
        
        fig = go.Figure()
        
        # Normalizar métricas para mejor visualización
        normalized_df = st.session_state.model_comparison_df.copy()
        normalized_df['R² Norm'] = normalized_df['R²'] / normalized_df['R²'].max()
        normalized_df['MAE Norm'] = 1 - (normalized_df['MAE'] / normalized_df['MAE'].max())
        normalized_df['RMSE Norm'] = 1 - (normalized_df['RMSE'] / normalized_df['RMSE'].max())
        normalized_df['Tiempo Norm'] = 1 - (normalized_df['Tiempo de Entrenamiento (s)'] / normalized_df['Tiempo de Entrenamiento (s)'].max())
        
        for metric in ['R² Norm', 'MAE Norm', 'RMSE Norm', 'Tiempo Norm']:
            fig.add_trace(go.Bar(
                x=normalized_df['Modelo'],
                y=normalized_df[metric],
                name=metric.replace(' Norm', ''),
                text=normalized_df[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Comparación Normalizada de Métricas por Modelo',
            xaxis_title='Modelo',
            yaxis_title='Valor Normalizado',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig)
        
    else:
        st.info("Ejecuta 'Actualizar Tabla Comparativa' después de entrenar los modelos para ver los resultados.")

# Función para validación y métricas - MODIFICADA
def perform_validation():
    st.subheader(get_text("model_performance"))
    
    # Mostrar tabla comparativa
    show_model_comparison()
    
    if not st.session_state.models:
        st.warning(get_text("no_models_trained"))
        return
    
    if st.session_state.X_val is None or st.session_state.y_val is None:
        st.warning("No hay datos de validación disponibles")
        return
    
    # Preparar datos de validación
    try:
        X_val_clean = prepare_data_for_modeling(st.session_state.X_val)
        y_val = st.session_state.y_val
        
        # Verificar que los datos estén disponibles
        if X_val_clean.empty:
            st.error("Datos de validación vacíos después del preprocesamiento")
            return
            
    except Exception as e:
        st.error(f"Error preparando datos de validación: {str(e)}")
        return
    
    results = []
    for name, model_info in st.session_state.models.items():
        model = model_info['model']
        training_time = model_info['training_time']
        
        if hasattr(model, 'predict'):
            try:
                y_pred = model.predict(X_val_clean)
                
                if len(y_pred.shape) > 1:
                    y_pred = y_pred.flatten()
                
                # Asegurar que las longitudes coincidan
                if len(y_pred) != len(y_val):
                    min_length = min(len(y_pred), len(y_val))
                    y_pred = y_pred[:min_length]
                    y_val_subset = y_val.iloc[:min_length] if hasattr(y_val, 'iloc') else y_val[:min_length]
                else:
                    y_val_subset = y_val
                
                mae = mean_absolute_error(y_val_subset, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val_subset, y_pred))
                r2 = r2_score(y_val_subset, y_pred)
                
                results.append({
                    'Modelo': name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R²': r2,
                    'Tiempo de entrenamiento (s)': training_time
                })
            except Exception as e:
                st.warning(f"No se pudieron calcular métricas para {name}: {str(e)}")
    
    if results:
        # Crear DataFrame con los resultados
        results_df = pd.DataFrame(results)
        
        # Formatear las métricas para mejor visualización
        display_df = results_df.copy()
        display_df['MAE'] = display_df['MAE'].map('{:.4f}'.format)
        display_df['RMSE'] = display_df['RMSE'].map('{:.4f}'.format)
        display_df['R²'] = display_df['R²'].map('{:.4f}'.format)
        display_df['Tiempo de entrenamiento (s)'] = display_df['Tiempo de entrenamiento (s)'].map('{:.2f}'.format)
        
        # Mostrar tabla consolidada de métricas
        st.subheader("Tabla Consolidada de Métricas de Modelos")
        st.dataframe(display_df)
        
        # Determinar el mejor modelo según los criterios especificados
        if len(results_df) > 0:
            # Primero: menor RMSE (métrica principal)
            best_by_rmse = results_df.loc[results_df['RMSE'].idxmin()]
            
            # Segundo: de los modelos con RMSE similar (dentro del 1%), comparar por MAE
            rmse_threshold = best_by_rmse['RMSE'] * 1.01
            candidates = results_df[results_df['RMSE'] <= rmse_threshold]
            
            if len(candidates) > 1:
                # Entre modelos con RMSE similar, elegir el que tenga menor MAE
                best_model_info = candidates.loc[candidates['MAE'].idxmin()]
            else:
                best_model_info = best_by_rmse
                
            # Tercero: si aún hay empate, usar R² como desempate
            if len(candidates[candidates['MAE'] == best_model_info['MAE']]) > 1:
                best_model_info = candidates.loc[candidates['R²'].idxmax()]
            
            # Mostrar información del mejor modelo
            st.success(f"🏆 **Mejor modelo: {best_model_info['Modelo']}**")
            
            # Crear columnas para mostrar las métricas del mejor modelo
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{best_model_info['RMSE']:.4f}")
            with col2:
                st.metric("MAE", f"{best_model_info['MAE']:.4f}")
            with col3:
                st.metric("R²", f"{best_model_info['R²']:.4f}")
            
            # Explicación de la selección
            with st.expander("🔍 Ver criterios de selección del mejor modelo"):
                st.write("""
                El mejor modelo se seleccionó según los siguientes criterios de prioridad:
                1. **Menor RMSE** (Root Mean Square Error): Métrica principal que penaliza más los errores grandes
                2. **Menor MAE** (Mean Absolute Error): En caso de RMSE similar, se prefiere el modelo con menor error absoluto promedio
                3. **Mayor R²** (Coeficiente de determinación): En caso de empate, se prefiere el modelo que explica mejor la varianza
                """)
        
        # Gráfico de comparación de modelos
        st.subheader("Comparación Visual de Modelos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de comparación de R²
            fig_r2 = px.bar(results_df, x='Modelo', y='R²', title='Comparación de R² entre modelos',
                           color='R²', color_continuous_scale='Viridis')
            fig_r2.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_r2)
        
        with col2:
            # Gráfico de comparación de errores
            fig_errors = go.Figure()
            fig_errors.add_trace(go.Bar(x=results_df['Modelo'], y=results_df['MAE'], name='MAE'))
            fig_errors.add_trace(go.Bar(x=results_df['Modelo'], y=results_df['RMSE'], name='RMSE'))
            fig_errors.update_layout(title='Comparación de Errores entre modelos', 
                                    xaxis_title='Modelo', yaxis_title='Error',
                                    barmode='group')
            st.plotly_chart(fig_errors)
        
        # Gráfico de radar para comparación multidimensional
        st.subheader("Comparación Multidimensional de Modelos")
        
        # Normalizar métricas para el gráfico de radar
        normalized_df = results_df.copy()
        normalized_df['RMSE'] = 1 - (normalized_df['RMSE'] / normalized_df['RMSE'].max())
        normalized_df['MAE'] = 1 - (normalized_df['MAE'] / normalized_df['MAE'].max())
        normalized_df['R²'] = normalized_df['R²'] / normalized_df['R²'].max()
        normalized_df['Eficiencia'] = 1 - (normalized_df['Tiempo de entrenamiento (s)'] / 
                                         normalized_df['Tiempo de entrenamiento (s)'].max())
        
        # Crear gráfico de radar para cada modelo
        fig_radar = go.Figure()
        
        for idx, row in normalized_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['RMSE'], row['MAE'], row['R²'], row['Eficiencia'], row['RMSE']],
                theta=['RMSE', 'MAE', 'R²', 'Eficiencia', 'RMSE'],
                fill='toself',
                name=row['Modelo']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparación de Modelos (Métricas Normalizadas)"
        )
        
        st.plotly_chart(fig_radar)
        
    else:
        st.warning("No se pudieron calcular métricas para los modelos")

# Función para generar reporte
def generate_report():
    st.subheader(get_text("report"))
    st.warning("Esta funcionalidad estará disponible en la próxima versión")
    
    # Aquí se implementaría la generación del reporte PDF
    # con todas las métricas, gráficos y resultados

# Función principal
def main():
    initialize_session_state()
    local_css()
    
    # Sidebar con botones de navegación
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; font-size: 80px;'>🏠</h1>", unsafe_allow_html=True)
        st.title(get_text("title"))
        
        # Selector de idioma
        st.session_state.language = st.selectbox("Idioma/Language/Langue", options=["ES", "EN", "FR"])
        
        # Carga de datos
        st.header(get_text("upload_data"))
        uploaded_file = st.file_uploader(get_text("select_file"), type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None and st.session_state.data is None:
            st.session_state.data = load_data(uploaded_file)
        
        # Navegación con botones
        st.header("Navegación")
        
        if st.button(get_text("eda")):
            st.session_state.current_page = "eda"
        
        if st.button(get_text("correlations")):
            st.session_state.current_page = "correlations"
        
        if st.button(get_text("preprocessing")):
            st.session_state.current_page = "preprocessing"
        
        if st.button(get_text("train_models")):
            st.session_state.current_page = "modeling"
        
        if st.button(get_text("validation")):
            st.session_state.current_page = "validation"
        
        if st.button(get_text("generate_report")):
            st.session_state.current_page = "report"
    
    # Página principal
    st.markdown(f'<h1 class="main-header">{get_text("title")}</h1>', unsafe_allow_html=True)
    
    # Mostrar la página actual según la selección
    if st.session_state.current_page == "eda":
        perform_eda()
    
    elif st.session_state.current_page == "correlations":
        perform_correlation_analysis()
    
    elif st.session_state.current_page == "preprocessing":
        perform_preprocessing()
    
    elif st.session_state.current_page == "modeling":
        train_models()
    
    elif st.session_state.current_page == "validation":
        perform_validation()
    
    elif st.session_state.current_page == "report":
        generate_report()
    
    # Mostrar estado actual de la aplicación
    with st.expander("Estado de la Aplicación"):
        st.write(f"**Datos cargados:** {st.session_state.data is not None}")
        if st.session_state.data is not None:
            st.write(f"**Forma de los datos:** {st.session_state.data.shape}")
        st.write(f"**Datos preprocesados:** {st.session_state.processed_data is not None}")
        st.write(f"**Variable objetivo:** {st.session_state.target_column}")
        st.write(f"**Modelos entrenados:** {len(st.session_state.models)}")
        st.write(f"**Página actual:** {st.session_state.current_page}")

if __name__ == "__main__":
    main()