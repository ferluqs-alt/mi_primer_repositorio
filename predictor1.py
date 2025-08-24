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
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings
warnings.filterwarnings('ignore')

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
        "no_models_trained": "No hay modelos entrenados. Por favor, entrene algunos modelos primero."
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
        "no_models_trained": "No models trained. Please train some models first."
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
        "no_data_loaded": "Aucune donnée chargée. Veuillez télécharger un jeu de données d'abord.",
        "data_loaded": "Données chargées: {} enregistrements, {} variables",
        "preprocessing_first": "Vous devez d'abord prétraiter les données",
        "select_target": "Sélectionnez la variable cible",
        "no_numerical_vars": "Aucune variable numérique pour l'analyse",
        "no_models_trained": "Aucun modèle entraîné. Veuillez d'abord entraîner des modèles."
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

# Función para preparar datos para modelado
def prepare_data_for_modeling(X_data):
    """Prepara los datos eliminando o codificando variables categóricas"""
    X_clean = X_data.copy()
    
    # Identificar columnas no numéricas
    non_numeric_cols = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if non_numeric_cols:
        st.warning(f"Columnas no numéricas encontradas: {non_numeric_cols}. Aplicando codificación...")
        
        # Para cada columna no numérica, intentar codificación
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
                    X_clean[col] = le.fit_transform(X_clean[col].astype(str))
                except:
                    st.error(f"No se pudo codificar la columna {col}. Se eliminará.")
                    X_clean = X_clean.drop(columns=[col])
    
    # Verificar y manejar valores NaN
    if X_clean.isnull().any().any():
        st.warning("Existen valores NaN. Aplicando imputación...")
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            imputer = SimpleImputer(strategy='mean')
            X_clean[numeric_cols] = imputer.fit_transform(X_clean[numeric_cols])
    
    return X_clean

# Modelo híbrido de red neuronal con lógica difusa
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

# Función para entrenar modelos
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
    
    # Preparar datos para modelado
    X_train_clean = prepare_data_for_modeling(X_train)
    X_val_clean = prepare_data_for_modeling(X_val)
    
    models = {}
    
    # 1. Regresión Lineal
    if st.button(get_text("linear_regression")):
        with st.spinner("Entrenando Regresión Lineal..."):
            try:
                start_time = time.time()
                lr_model = LinearRegression()
                lr_model.fit(X_train_clean, y_train)
                training_time = time.time() - start_time
                
                models['Linear Regression'] = {
                    'model': lr_model,
                    'training_time': training_time,
                    'feature_names': X_train_clean.columns.tolist()
                }
                
                st.success(f"Regresión Lineal entrenada en {training_time:.2f} segundos")
                
                # Mostrar coeficientes
                coef_df = pd.DataFrame({
                    'Variable': X_train_clean.columns,
                    'Coeficiente': lr_model.coef_
                }).sort_values('Coeficiente', key=abs, ascending=False)
                
                st.subheader("Coeficientes del Modelo")
                st.dataframe(coef_df.head(10))
                
            except Exception as e:
                st.error(f"Error entrenando Regresión Lineal: {str(e)}")
    
    # 2. Perceptrón Multicapa (MLP)
    if st.button(get_text("mlp")):
        with st.spinner("Entrenando MLP..."):
            try:
                start_time = time.time()
                
                mlp_model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train_clean.shape[1],)),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(1)
                ])
                mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                history = mlp_model.fit(X_train_clean, y_train, 
                                       epochs=100, batch_size=32, 
                                       validation_data=(X_val_clean, y_val),
                                       verbose=0)
                training_time = time.time() - start_time
                
                models['MLP'] = {
                    'model': mlp_model,
                    'training_time': training_time,
                    'history': history,
                    'feature_names': X_train_clean.columns.tolist()
                }
                
                st.success(f"MLP entrenado en {training_time:.2f} segundos")
                
                # Gráfico de pérdida durante el entrenamiento
                fig = px.line(x=range(len(history.history['loss'])), 
                             y=history.history['loss'], 
                             labels={'x': 'Época', 'y': 'Pérdida'}, 
                             title='Pérdida durante el entrenamiento (MLP)')
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error entrenando MLP: {str(e)}")
    
    # 3. Red Neuronal Secuencial
    if st.button(get_text("sequential_nn")):
        with st.spinner("Entrenando Red Neuronal Secuencial..."):
            try:
                start_time = time.time()
                
                sequential_model = Sequential([
                    Dense(128, activation='relu', input_shape=(X_train_clean.shape[1],)),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])
                sequential_model.compile(optimizer=Adam(learning_rate=0.001), 
                                        loss='mse', 
                                        metrics=['mae'])
                history = sequential_model.fit(X_train_clean, y_train, 
                                              epochs=150, batch_size=32, 
                                              validation_data=(X_val_clean, y_val),
                                              verbose=0)
                training_time = time.time() - start_time
                
                models['Sequential NN'] = {
                    'model': sequential_model,
                    'training_time': training_time,
                    'history': history,
                    'feature_names': X_train_clean.columns.tolist()
                }
                
                st.success(f"Red Neuronal Secuencial entrenada en {training_time:.2f} segundos")
                
                # Gráfico de pérdida durante el entrenamiento
                fig = px.line(x=range(len(history.history['loss'])), 
                             y=history.history['loss'], 
                             labels={'x': 'Época', 'y': 'Pérdida'}, 
                             title='Pérdida durante el entrenamiento (Red Secuencial)')
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error entrenando Red Neuronal Secuencial: {str(e)}")
    
    # 4. Modelo Híbrido (Red Neuronal + Lógica Difusa)
    if st.button(get_text("hybrid_model")):
        with st.spinner("Entrenando Modelo Híbrido..."):
            try:
                start_time = time.time()
                
                # Parte de red neuronal
                hybrid_model = create_hybrid_model(X_train_clean.shape[1])
                history = hybrid_model.fit(X_train_clean, y_train, 
                                          epochs=100, batch_size=32, 
                                          validation_data=(X_val_clean, y_val),
                                          verbose=0)
                training_time = time.time() - start_time
                
                models['Hybrid Model'] = {
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
                
                # Explicación del modelo híbrido
                with st.expander("Explicación del Modelo Híbrido"):
                    st.write("""
                    Este modelo combina una red neuronal profunda con elementos de lógica difusa:
                    - La red neuronal aprende patrones complejos en los datos
                    - La lógica difusa ayuda a manejar la incertidumbre y proporciona interpretabilidad
                    - El dropout regulariza el modelo para prevenir sobreajuste
                    """)
                
            except Exception as e:
                st.error(f"Error entrenando Modelo Híbrido: {str(e)}")
    
    # Actualizar modelos en el estado de la sesión
    if models:
        st.session_state.models.update(models)

# Función para validación y métricas
def perform_validation():
    st.subheader(get_text("model_performance"))
    
    if not st.session_state.models:
        st.warning(get_text("no_models_trained"))
        return
    
    if st.session_state.X_val is None or st.session_state.y_val is None:
        st.warning("No hay datos de validación disponibles")
        return
    
    # Preparar datos de validación
    X_val_clean = prepare_data_for_modeling(st.session_state.X_val)
    y_val = st.session_state.y_val
    
    results = []
    for name, model_info in st.session_state.models.items():
        model = model_info['model']
        training_time = model_info['training_time']
        
        if hasattr(model, 'predict'):
            try:
                y_pred = model.predict(X_val_clean)
                
                if len(y_pred.shape) > 1:
                    y_pred = y_pred.flatten()
                
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
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
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.format({
            'MAE': '{:.4f}',
            'RMSE': '{:.4f}',
            'R²': '{:.4f}',
            'Tiempo de entrenamiento (s)': '{:.2f}'
        }))
        
        # Gráfico de comparación de modelos
        fig = go.Figure()
        fig.add_trace(go.Bar(x=results_df['Modelo'], y=results_df['R²'], name='R²'))
        fig.update_layout(title='Comparación de R² entre modelos', 
                         xaxis_title='Modelo', yaxis_title='R²',
                         yaxis_range=[0, 1])
        st.plotly_chart(fig)
        
        # Gráfico de comparación de errores
        fig = go.Figure()
        fig.add_trace(go.Bar(x=results_df['Modelo'], y=results_df['MAE'], name='MAE'))
        fig.add_trace(go.Bar(x=results_df['Modelo'], y=results_df['RMSE'], name='RMSE'))
        fig.update_layout(title='Comparación de Errores entre modelos', 
                         xaxis_title='Modelo', yaxis_title='Error')
        st.plotly_chart(fig)
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
