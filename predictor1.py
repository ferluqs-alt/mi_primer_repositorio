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
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kstest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
        "download_report": "Descargar Reporte PDF"
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
        "download_report": "Download PDF Report"
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
        "download_report": "Télécharger le Rapport PDF"
    }
}

# Inicialización del estado de la sesión
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

# Función para obtener texto según el idioma
def get_text(key):
    return language_dict[st.session_state.language][key]

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
        
        st.success(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

# Función para realizar EDA
def perform_eda(df):
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
    
    # Prueba de normalidad
    st.subheader(get_text("normality_test"))
    if len(numerical_cols) > 0:
        normality_results = []
        for col in numerical_cols:
            # Eliminar valores nulos para la prueba
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stat, p_value = kstest(col_data, 'norm')
                normality_results.append({
                    'Variable': col,
                    'Estadístico': stat,
                    'p-valor': p_value,
                    'Normal': p_value > 0.05
                })
        normality_df = pd.DataFrame(normality_results)
        st.dataframe(normality_df)
    
    # Visualización
    st.subheader(get_text("visualization"))
    plot_type = st.selectbox("Tipo de gráfico", ["Histograma", "Boxplot", "Scatterplot"])
    
    if plot_type == "Histograma":
        col_to_plot = st.selectbox("Seleccione columna", numerical_cols)
        fig = px.histogram(df, x=col_to_plot, title=f"Histograma de {col_to_plot}")
        st.plotly_chart(fig)
    
    elif plot_type == "Boxplot":
        col_to_plot = st.selectbox("Seleccione columna", numerical_cols)
        fig = px.box(df, y=col_to_plot, title=f"Boxplot de {col_to_plot}")
        st.plotly_chart(fig)
    
    elif plot_type == "Scatterplot":
        col_x = st.selectbox("Seleccione variable X", numerical_cols)
        col_y = st.selectbox("Seleccione variable Y", numerical_cols)
        fig = px.scatter(df, x=col_x, y=col_y, title=f"Scatterplot: {col_x} vs {col_y}")
        st.plotly_chart(fig)

# Función para análisis de correlaciones
def perform_correlation_analysis(df):
    st.subheader(get_text("correlations"))
    
    numerical_df = df.select_dtypes(include=[np.number])
    if numerical_df.empty:
        st.warning("No hay variables numéricas para analizar correlaciones")
        return
    
    # Matriz de correlación
    corr_matrix = numerical_df.corr()
    
    st.subheader(get_text("correlation_matrix"))
    st.dataframe(corr_matrix)
    
    # Heatmap de correlaciones
    st.subheader(get_text("correlation_heatmap"))
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)
    
    # Correlaciones más fuertes con la variable objetivo
    st.subheader(get_text("strongest_correlations"))
    if st.session_state.target_column and st.session_state.target_column in corr_matrix.columns:
        target_correlations = corr_matrix[st.session_state.target_column].sort_values(key=abs, ascending=False)
        st.dataframe(target_correlations)
        
        # Scatter plots con las variables más correlacionadas
        top_correlated = target_correlations.index[1:4]  # Excluye la correlación consigo misma
        for col in top_correlated:
            fig = px.scatter(df, x=col, y=st.session_state.target_column, 
                            title=f"{col} vs {st.session_state.target_column}")
            st.plotly_chart(fig)
    
    # Análisis de multicolinealidad (VIF)
    st.subheader(get_text("multicolineality"))
    if len(numerical_df.columns) > 1:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = numerical_df.columns
        
        # Calcular VIF para cada variable
        vif_values = []
        for i in range(len(numerical_df.columns)):
            # Eliminar NaNs para el cálculo de VIF
            temp_df = numerical_df.dropna()
            if len(temp_df) > 0:
                vif = variance_inflation_factor(temp_df.values, i)
                vif_values.append(vif)
            else:
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        st.dataframe(vif_data)

# Función para preprocesamiento de datos
def perform_preprocessing(df):
    st.subheader(get_text("preprocessing"))
    
    processed_df = df.copy()
    
    # Selección de columna objetivo
    numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        target_col = st.selectbox("Seleccione la variable objetivo", numerical_cols)
        st.session_state.target_column = target_col
    
    # Imputación de valores faltantes
    st.subheader(get_text("imputation"))
    imputation_method = st.selectbox("Método de imputación", 
                                    ["Ninguno", "Media/Moda", "KNN"])
    
    if imputation_method != "Ninguno":
        numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
        
        if imputation_method == "Media/Moda":
            for col in numerical_cols:
                if processed_df[col].isnull().sum() > 0:
                    mean_val = processed_df[col].mean()
                    processed_df[col].fillna(mean_val, inplace=True)
            
            for col in categorical_cols:
                if processed_df[col].isnull().sum() > 0:
                    mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else "Desconocido"
                    processed_df[col].fillna(mode_val, inplace=True)
        
        elif imputation_method == "KNN":
            st.warning("La imputación KNN puede ser lenta para conjuntos de datos grandes")
            if st.button("Aplicar imputación KNN"):
                numerical_imputer = KNNImputer(n_neighbors=5)
                processed_df[numerical_cols] = numerical_imputer.fit_transform(processed_df[numerical_cols])
                
                for col in categorical_cols:
                    if processed_df[col].isnull().sum() > 0:
                        mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else "Desconocido"
                        processed_df[col].fillna(mode_val, inplace=True)
    
    # Tratamiento de outliers
    st.subheader(get_text("outlier_treatment"))
    outlier_treatment = st.selectbox("Método de tratamiento de outliers", 
                                    ["Ninguno", "Eliminación", "Transformación"])
    
    if outlier_treatment != "Ninguno" and st.session_state.target_column:
        numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        if st.session_state.target_column in numerical_cols:
            numerical_cols.remove(st.session_state.target_column)
        
        for col in numerical_cols:
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if outlier_treatment == "Eliminación":
                processed_df = processed_df[(processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)]
            elif outlier_treatment == "Transformación":
                processed_df[col] = np.log1p(processed_df[col])
    
    # Codificación de variables categóricas
    st.subheader(get_text("encoding"))
    encoding_method = st.selectbox("Método de codificación", 
                                  ["Ninguno", "One-Hot Encoding", "Label Encoding"])
    
    if encoding_method != "Ninguno":
        categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
        
        if encoding_method == "One-Hot Encoding":
            processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
        elif encoding_method == "Label Encoding":
            le = LabelEncoder()
            for col in categorical_cols:
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
    
    st.session_state.processed_data = processed_df
    st.success("Preprocesamiento completado")
    st.dataframe(processed_df.head())
    
    return processed_df

# Función para escalado y división de datos
def perform_scaling_split(df):
    st.subheader(get_text("scaling"))
    
    if st.session_state.processed_data is None:
        st.warning("Primero debe realizar el preprocesamiento de datos")
        return None, None, None, None, None, None
    
    df = st.session_state.processed_data
    
    # Escalado
    st.subheader(get_text("scaling_options"))
    scaling_method = st.selectbox("Método de escalado", ["Ninguno", "StandardScaler"])
    
    if scaling_method != "Ninguno" and st.session_state.target_column:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if st.session_state.target_column in numerical_cols:
            numerical_cols.remove(st.session_state.target_column)
        
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # División de datos
    st.subheader(get_text("train_val_test_split"))
    if st.session_state.target_column:
        X = df.drop(st.session_state.target_column, axis=1)
        y = df[st.session_state.target_column]
        
        # Primera división: train (70%) y temp (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Segunda división: validation (15%) y test (15%)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        st.write(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
        st.write(f"Conjunto de validación: {X_val.shape[0]} muestras")
        st.write(f"Conjunto de prueba: {X_test.shape[0]} muestras")
        
        # Validación cruzada
        st.subheader(get_text("cross_validation"))
        k_folds = st.slider("Número de folds para validación cruzada", 3, 10, 5)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    return None, None, None, None, None, None

# Función para modelado
def perform_modeling(X_train, y_train):
    st.subheader(get_text("model_training"))
    
    if X_train is None or y_train is None:
        st.warning("Primero debe realizar el escalado y división de datos")
        return
    
    models = {}
    
    # Regresión Lineal
    if st.button("Entrenar Regresión Lineal"):
        with st.spinner("Entrenando Regresión Lineal..."):
            start_time = time.time()
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            models['Linear Regression'] = {
                'model': lr_model,
                'training_time': training_time
            }
            
            st.success(f"Regresión Lineal entrenada en {training_time:.2f} segundos")
    
    # Red Neuronal (MLP)
    if st.button("Entrenar Perceptrón Multicapa (MLP)"):
        with st.spinner("Entrenando MLP..."):
            start_time = time.time()
            
            # Verificar que hay datos para entrenar
            if len(X_train) == 0:
                st.error("No hay datos de entrenamiento disponibles")
                return
            
            mlp_model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            history = mlp_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            training_time = time.time() - start_time
            
            models['MLP'] = {
                'model': mlp_model,
                'training_time': training_time,
                'history': history
            }
            
            st.success(f"MLP entrenado en {training_time:.2f} segundos")
            
            # Gráfico de pérdida durante el entrenamiento
            fig = px.line(x=range(len(history.history['loss'])), y=history.history['loss'], 
                         labels={'x': 'Época', 'y': 'Pérdida'}, title='Pérdida durante el entrenamiento')
            st.plotly_chart(fig)
    
    st.session_state.models = models
    return models

# Función para validación y métricas
def perform_validation(models, X_val, y_val):
    st.subheader(get_text("model_performance"))
    
    if not models:
        st.warning("Primero debe entrenar algunos modelos")
        return
    
    results = []
    for name, model_info in models.items():
        model = model_info['model']
        training_time = model_info['training_time']
        
        # Predecir
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_val)
            
            # Aplanar si es necesario
            if len(y_pred.shape) > 1:
                y_pred = y_pred.flatten()
            
            # Calcular métricas
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
    
    # Mostrar resultados
    if results:
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        # Gráfico de comparación de modelos
        fig = go.Figure()
        fig.add_trace(go.Bar(x=results_df['Modelo'], y=results_df['R²'], name='R²'))
        fig.update_layout(title='Comparación de R² entre modelos', xaxis_title='Modelo', yaxis_title='R²')
        st.plotly_chart(fig)
    else:
        st.warning("No se pudieron calcular métricas para los modelos")

# Función principal
def main():
    # Aplicar CSS personalizado
    local_css()
    
    # Sidebar
    with st.sidebar:
        # Usar markdown para mostrar el emoji en lugar de st.image
        st.markdown("<h1 style='text-align: center; font-size: 80px;'>🏠</h1>", unsafe_allow_html=True)
        st.title(get_text("title"))
        
        # Selector de idioma
        st.session_state.language = st.selectbox("Idioma/Language/Langue", options=["ES", "EN", "FR"])
        
        # Carga de datos
        st.header(get_text("upload_data"))
        uploaded_file = st.file_uploader(get_text("select_file"), type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            if st.session_state.data is None:
                st.session_state.data = load_data(uploaded_file)
            
            if st.session_state.data is not None:
                df = st.session_state.data
                
                # Navegación
                st.header("Navegación")
                options = [
                    get_text("eda"),
                    get_text("correlations"),
                    get_text("preprocessing"),
                    get_text("scaling"),
                    get_text("modeling"),
                    get_text("validation"),
                    get_text("timeseries"),
                    get_text("report")
                ]
                choice = st.radio("Seleccione una opción:", options)
    
    # Página principal
    st.markdown(f'<h1 class="main-header">{get_text("title")}</h1>', unsafe_allow_html=True)
    
    if uploaded_file is None:
        st.info(f"👈 {get_text('upload_data')} - {get_text('file_types')}")
        return
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        if choice == get_text("eda"):
            perform_eda(df)
        
        elif choice == get_text("correlations"):
            perform_correlation_analysis(df)
        
        elif choice == get_text("preprocessing"):
            perform_preprocessing(df)
        
        elif choice == get_text("scaling"):
            X_train, X_val, X_test, y_train, y_val, y_test = perform_scaling_split(df)
            if X_train is not None:
                st.session_state.X_train = X_train
                st.session_state.X_val = X_val
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_val = y_val
                st.session_state.y_test = y_test
        
        elif choice == get_text("modeling"):
            if st.session_state.X_train is not None and st.session_state.y_train is not None:
                models = perform_modeling(st.session_state.X_train, st.session_state.y_train)
                st.session_state.models = models
            else:
                st.warning("Primero debe realizar el escalado y división de datos")
        
        elif choice == get_text("validation"):
            if (st.session_state.X_val is not None and 
                st.session_state.y_val is not None and 
                st.session_state.models):
                perform_validation(st.session_state.models, st.session_state.X_val, st.session_state.y_val)
            else:
                st.warning("Primero debe entrenar algunos modelos y tener conjuntos de validación")
        
        elif choice == get_text("timeseries"):
            st.warning("Funcionalidad de series temporales no implementada en esta versión")
        
        elif choice == get_text("report"):
            st.warning("Generación de reportes PDF no implementada en esta versión")

if __name__ == "__main__":
    main()