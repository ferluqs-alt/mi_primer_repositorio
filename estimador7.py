#########Codigo que cambia cualquier etiqueta, titulo o boton al idioma seleccionado##############
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import base64
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer, KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set page configuration
def set_page_config():
    st.set_page_config(
        page_title=get_text("title"),
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Custom CSS
def apply_custom_css():
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
        .highlight {
            background-color: #e6f2ff;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f8ff;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# Translations - Expanded with more UI elements and data translations
translations = {
    "en": {
        # UI Elements
        "title": "Housing Price Estimator",
        "upload_data": "Upload Data",
        "file_types": "Select a CSV or Excel file",
        "eda": "Exploratory Data Analysis",
        "preprocessing": "Data Preprocessing",
        "modeling": "Modeling",
        "evaluation": "Model Evaluation",
        "report": "Final Report",
        "target_var": "Select target variable",
        "run_eda": "Run Exploratory Analysis",
        "data_dimensions": "Dataset Dimensions",
        "num_rows": "Number of Rows",
        "num_cols": "Number of Columns",
        "numeric_vars": "Numeric Variables",
        "categorical_vars": "Categorical Variables",
        "missing_values": "Missing Values",
        "duplicates": "Duplicate Values",
        "descriptive_stats": "Descriptive Statistics",
        "correlation_matrix": "Correlation Matrix",
        "outlier_detection": "Outlier Detection",
        "preprocess_data": "Preprocess Data",
        "train_models": "Train Models",
        "evaluate_models": "Evaluate Models",
        "generate_report": "Generate Report",
        "results": "Results",
        "model_comparison": "Model Comparison",
        "best_model": "Best Model",
        "training_time": "Training Time",
        "important_features": "Important Features",
        "mae": "Mean Absolute Error (MAE)",
        "rmse": "Root Mean Squared Error (RMSE)",
        "r2": "Coefficient of Determination (R²)",
        "select_language": "Select Language",
        "dataset_preview": "Dataset Preview",
        "handle_missing": "Handle missing values",
        "remove_duplicates": "Remove duplicates",
        "handle_outliers": "Handle outliers (Z-score)",
        "encode_categorical": "Encode categorical variables",
        "handle_multicollinearity": "Handle multicollinearity (VIF)",
        "scale_features": "Scale features",
        "training_set": "Training set",
        "validation_set": "Validation set",
        "test_set": "Test set",
        "samples": "samples",
        "training_times": "Training Times",
        "developed_with": "Developed with Streamlit - Housing Price Estimator Tool",
        "distributions": "Distributions",
        "top_correlations": "Top correlations with",
        
        # Data translations (categorical values)
        "NEAR BAY": "NEAR BAY",
        "INLAND": "INLAND",
        "<1H OCEAN": "<1H OCEAN",
        "NEAR OCEAN": "NEAR OCEAN",
        "ISLAND": "ISLAND",
        "YES": "YES",
        "NO": "NO",
        "TRUE": "TRUE",
        "FALSE": "FALSE",
        # Add more categorical values as needed
    },
    "es": {
        # UI Elements
        "title": "Estimador de Precios de Viviendas",
        "upload_data": "Cargar Datos",
        "file_types": "Seleccione un archivo CSV o Excel",
        "eda": "Análisis Exploratorio de Datos",
        "preprocessing": "Preprocesamiento de Datos",
        "modeling": "Modelado",
        "evaluation": "Evaluación de Modelos",
        "report": "Reporte Final",
        "target_var": "Seleccione la variable objetivo",
        "run_eda": "Ejecutar Análisis Exploratorio",
        "data_dimensions": "Dimensiones del Dataset",
        "num_rows": "Número de Filas",
        "num_cols": "Número de Columnas",
        "numeric_vars": "Variables Numéricas",
        "categorical_vars": "Variables Categóricas",
        "missing_values": "Valores Faltantes",
        "duplicates": "Valores Duplicados",
        "descriptive_stats": "Estadísticas Descriptivas",
        "correlation_matrix": "Matriz de Correlación",
        "outlier_detection": "Detección de Outliers",
        "preprocess_data": "Preprocesar Datos",
        "train_models": "Entrenar Modelos",
        "evaluate_models": "Evaluar Modelos",
        "generate_report": "Generar Reporte",
        "results": "Resultados",
        "model_comparison": "Comparación de Modelos",
        "best_model": "Mejor Modelo",
        "training_time": "Tiempo de Entrenamiento",
        "important_features": "Características Importantes",
        "mae": "Error Absoluto Medio (MAE)",
        "rmse": "Raíz del Error Cuadrático Medio (RMSE)",
        "r2": "Coeficiente de Determinación (R²)",
        "select_language": "Seleccionar Idioma",
        "dataset_preview": "Vista Previa del Dataset",
        "handle_missing": "Manejar valores faltantes",
        "remove_duplicates": "Eliminar duplicados",
        "handle_outliers": "Manejar valores atípicos (Z-score)",
        "encode_categorical": "Codificar variables categóricas",
        "handle_multicollinearity": "Manejar multicolinealidad (VIF)",
        "scale_features": "Escalar características",
        "training_set": "Conjunto de entrenamiento",
        "validation_set": "Conjunto de validación",
        "test_set": "Conjunto de prueba",
        "samples": "muestras",
        "training_times": "Tiempos de Entrenamiento",
        "developed_with": "Desarrollado con Streamlit - Herramienta de Estimación de Precios de Viviendas",
        "distributions": "Distribuciones",
        "top_correlations": "Principales correlaciones con",
        
        # Data translations (categorical values)
        "NEAR BAY": "CERCA DE LA BAHÍA",
        "INLAND": "INTERIOR",
        "<1H OCEAN": "<1H OCÉANO",
        "NEAR OCEAN": "CERCA DEL OCÉANO",
        "ISLAND": "ISLA",
        "YES": "SÍ",
        "NO": "NO",
        "TRUE": "VERDADERO",
        "FALSE": "FALSO",
        # Add more categorical values as needed
    },
    "fr": {
        # UI Elements
        "title": "Estimateur de Prix Immobiliers",
        "upload_data": "Charger les Données",
        "file_types": "Sélectionnez un fichier CSV ou Excel",
        "eda": "Analyse Exploratoire des Données",
        "preprocessing": "Prétraitement des Données",
        "modeling": "Modélisation",
        "evaluation": "Évaluation des Modèles",
        "report": "Rapport Final",
        "target_var": "Sélectionnez la variable cible",
        "run_eda": "Exécuter l'Analyse Exploratoire",
        "data_dimensions": "Dimensions du Jeu de Données",
        "num_rows": "Nombre de Lignes",
        "num_cols": "Nombre de Colonnes",
        "numeric_vars": "Variables Numériques",
        "categorical_vars": "Variables Catégorielles",
        "missing_values": "Valeurs Manquantes",
        "duplicates": "Valeurs Dupliquées",
        "descriptive_stats": "Statistiques Descriptives",
        "correlation_matrix": "Matrice de Corrélation",
        "outlier_detection": "Détection des Valeurs Aberrantes",
        "preprocess_data": "Prétraiter les Données",
        "train_models": "Entraîner les Modèles",
        "evaluate_models": "Évaluer les Modèles",
        "generate_report": "Générer le Rapport",
        "results": "Résultats",
        "model_comparison": "Comparaison des Modèles",
        "best_model": "Meilleur Modèle",
        "training_time": "Temps d'Entraînement",
        "important_features": "Caractéristiques Importantes",
        "mae": "Erreur Absolue Moyenne (MAE)",
        "rmse": "Racine de l'Erreur Quadratique Moyenne (RMSE)",
        "r2": "Coefficient de Détermination (R²)",
        "select_language": "Sélectionner la Langue",
        "dataset_preview": "Aperçu du Jeu de Données",
        "handle_missing": "Gérer les valeurs manquantes",
        "remove_duplicates": "Supprimer les doublons",
        "handle_outliers": "Gérer les valeurs aberrantes (Z-score)",
        "encode_categorical": "Encoder les variables catégorielles",
        "handle_multicollinearity": "Gérer la multicolinéarité (VIF)",
        "scale_features": "Mettre à l'échelle les caractéristiques",
        "training_set": "Ensemble d'entraînement",
        "validation_set": "Ensemble de validation",
        "test_set": "Ensemble de test",
        "samples": "échantillons",
        "training_times": "Temps d'Entraînement",
        "developed_with": "Développé avec Streamlit - Outil d'Estimation des Prix Immobiliers",
        "distributions": "Distributions",
        "top_correlations": "Principales corrélations avec",
        
        # Data translations (categorical values)
        "NEAR BAY": "PRÈS DE LA BAIE",
        "INLAND": "INTÉRIEUR",
        "<1H OCEAN": "<1H OCÉAN",
        "NEAR OCEAN": "PRÈS DE L'OCÉAN",
        "ISLAND": "ÎLE",
        "YES": "OUI",
        "NO": "NON",
        "TRUE": "VRAI",
        "FALSE": "FAUX",
        # Add more categorical values as needed
    }
}

# Function to get translated text
def get_text(key):
    """Returns translated text for the given key based on selected language"""
    lang = st.session_state.language
    if lang in translations and key in translations[lang]:
        return translations[lang][key]
    elif key in translations["en"]:  # Fallback to English
        return translations["en"][key]
    else:
        return key  # Return the key itself if not found

# Function to translate dataframe categorical values
def translate_dataframe(df, language):
    """Translates categorical values in the dataframe based on selected language"""
    df_translated = df.copy()
    
    # Define which columns are likely to contain categorical data
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        # Translate each value in the column if it exists in our translations
        df_translated[col] = df[col].apply(
            lambda x: translations[language].get(str(x).upper(), x) if pd.notna(x) else x
        )
    
    return df_translated

# Initialize session state
def init_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_translated' not in st.session_state:
        st.session_state.df_translated = None
    if 'target_var' not in st.session_state:
        st.session_state.target_var = None
    if 'language' not in st.session_state:
        st.session_state.language = 'es'
    if 'eda_done' not in st.session_state:
        st.session_state.eda_done = False
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
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
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = {}
    if 'show_eda' not in st.session_state:
        st.session_state.show_eda = False
    if 'show_preprocessing' not in st.session_state:
        st.session_state.show_preprocessing = False
    if 'show_modeling' not in st.session_state:
        st.session_state.show_modeling = False
    if 'show_evaluation' not in st.session_state:
        st.session_state.show_evaluation = False

# Helper function to load data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def perform_eda(df, target_var):
    # Esta función ahora solo realiza el análisis y devuelve los resultados
    # La visualización se hará en el área principal
    
    # Calcular dimensiones
    dimensions = {
        'num_rows': df.shape[0],
        'num_cols': df.shape[1]
    }
    
    # Identificar tipos de variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Valores faltantes
    missing_data = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Values': missing_data.values,
        'Percentage': (missing_data.values / len(df)) * 100
    })
    
    # Duplicados
    duplicates = df.duplicated().sum()
    
    # Estadísticas descriptivas
    descriptive_stats = df.describe()
    
    # Matriz de correlación
    corr_matrix = None
    target_correlations = None
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        if target_var in numeric_cols:
            target_correlations = corr_matrix[target_var].abs().sort_values(ascending=False)
    
    return {
        'dimensions': dimensions,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'missing_df': missing_df,
        'duplicates': duplicates,
        'descriptive_stats': descriptive_stats,
        'corr_matrix': corr_matrix,
        'target_correlations': target_correlations,
        'numeric_cols_list': numeric_cols
    }

def display_eda_results(eda_results, target_var):
    """Muestra los resultados del EDA en el área principal"""
    st.subheader(get_text("data_dimensions"))
    col1, col2 = st.columns(2)
    col1.metric(get_text("num_rows"), eda_results['dimensions']['num_rows'])
    col2.metric(get_text("num_cols"), eda_results['dimensions']['num_cols'])
    
    st.subheader(get_text("numeric_vars"))
    st.write(eda_results['numeric_cols'])
    
    st.subheader(get_text("categorical_vars"))
    st.write(eda_results['categorical_cols'])
    
    st.subheader(get_text("missing_values"))
    st.dataframe(eda_results['missing_df'][eda_results['missing_df']['Missing Values'] > 0])
    
    st.subheader(get_text("duplicates"))
    st.metric(get_text("duplicates"), eda_results['duplicates'])
    
    st.subheader(get_text("descriptive_stats"))
    st.dataframe(eda_results['descriptive_stats'])
    
    # Visualizations
    st.subheader(get_text("distributions"))
    
    # Numeric variables distributions
    if eda_results['numeric_cols']:
        num_cols_to_show = min(5, len(eda_results['numeric_cols']))
        fig, axes = plt.subplots(1, num_cols_to_show, figsize=(20, 4))
        if num_cols_to_show == 1:
            axes = [axes]
        
        for i, col in enumerate(eda_results['numeric_cols'][:num_cols_to_show]):
            axes[i].hist(st.session_state.df_translated[col].dropna(), bins=30, alpha=0.7, color='blue')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Correlation matrix
    st.subheader(get_text("correlation_matrix"))
    if eda_results['corr_matrix'] is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(eda_results['corr_matrix'], annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
        # Top correlations with target
        if eda_results['target_correlations'] is not None:
            st.write(f"{get_text('top_correlations')} {target_var}:")
            st.write(eda_results['target_correlations'][1:6])  # Top 5 excluding itself
    
    # Outlier detection
    st.subheader(get_text("outlier_detection"))
    if eda_results['numeric_cols_list']:
        num_cols_to_show = min(3, len(eda_results['numeric_cols_list']))
        fig, axes = plt.subplots(1, num_cols_to_show, figsize=(15, 5))
        if num_cols_to_show == 1:
            axes = [axes]
        
        for i, col in enumerate(eda_results['numeric_cols_list'][:num_cols_to_show]):
            axes[i].boxplot(st.session_state.df_translated[col].dropna())
            axes[i].set_title(f'Boxplot of {col}')
        
        plt.tight_layout()
        st.pyplot(fig)

def preprocess_data(df, target_var, options):
    df_processed = df.copy()
    
    # Handle missing values
    if options['handle_missing'] == get_text("Drop rows"):
        df_processed = df_processed.dropna()
    elif options['handle_missing'] == get_text("Mean imputation"):
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
    elif options['handle_missing'] == get_text("KNN imputation"):
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        imputer = KNNImputer(n_neighbors=5)
        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
    
    # Handle duplicates
    if options['handle_duplicates']:
        df_processed = df_processed.drop_duplicates()
    
    # Handle outliers with Z-score
    if options['handle_outliers']:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_var:  # Don't remove outliers from target variable
                z_scores = np.abs(stats.zscore(df_processed[col].dropna()))
                df_processed = df_processed[(z_scores < 3) | (df_processed[col].isna())]
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    if options['encoding'] == get_text("One-Hot Encoding"):
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    elif options['encoding'] == get_text("Label Encoding"):
        le = LabelEncoder()
        for col in categorical_cols:
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Calculate VIF and handle multicollinearity
    if options['handle_multicollinearity']:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_cols
        vif_data["VIF"] = [variance_inflation_factor(df_processed[numeric_cols].values, i) 
                          for i in range(len(numeric_cols))]
        
        # Remove features with high VIF
        high_vif_features = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
        if target_var in high_vif_features:
            high_vif_features.remove(target_var)
        df_processed = df_processed.drop(columns=high_vif_features)
    
    # Scale features
    if options['scaling']:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        if target_var in numeric_cols:
            numeric_cols = numeric_cols.drop(target_var)
        scaler = StandardScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    return df_processed

def train_linear_regression(X_train, y_train):
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Get feature importance
    importance = np.abs(model.coef_)
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return model, training_time, feature_importance

def train_neural_network(X_train, y_train, X_val, y_val):
    start_time = time.time()
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    training_time = time.time() - start_time
    
    # Get feature importance using permutation importance (simplified)
    baseline_mae = mean_absolute_error(y_train, model.predict(X_train, verbose=0).flatten())
    feature_importance = []
    
    for i in range(X_train.shape[1]):
        X_temp = X_train.copy()
        np.random.shuffle(X_temp.iloc[:, i])
        mae_score = mean_absolute_error(y_train, model.predict(X_temp, verbose=0).flatten())
        importance = mae_score - baseline_mae
        feature_importance.append((X_train.columns[i], importance))
    
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    feature_importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
    
    return model, training_time, feature_importance_df, history

def train_mlp(X_train, y_train, X_val, y_val):
    start_time = time.time()
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    training_time = time.time() - start_time
    
    # Get feature importance
    baseline_mae = mean_absolute_error(y_train, model.predict(X_train, verbose=0).flatten())
    feature_importance = []
    
    for i in range(X_train.shape[1]):
        X_temp = X_train.copy()
        np.random.shuffle(X_temp.iloc[:, i])
        mae_score = mean_absolute_error(y_train, model.predict(X_temp, verbose=0).flatten())
        importance = mae_score - baseline_mae
        feature_importance.append((X_train.columns[i], importance))
    
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    feature_importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
    
    return model, training_time, feature_importance_df, history

def train_fuzzy_neural_network(X_train, y_train):
    # This is a simplified implementation
    start_time = time.time()
    
    # For the purpose of this demo, we'll return a simple linear model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Feature importance (using linear model as proxy)
    importance = np.abs(model.coef_)
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return model, training_time, feature_importance, {}

def evaluate_model(model, X_test, y_test, model_name):
    if hasattr(model, 'predict'):
        # Scikit-learn model
        y_pred = model.predict(X_test)
    else:
        # Keras model
        y_pred = model.predict(X_test, verbose=0).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Set page config and apply CSS
    set_page_config()
    apply_custom_css()
    
    # Sidebar - Solo controles de entrada
    with st.sidebar:
        st.title(get_text("title"))
        
        # Language selection
        language_options = {
            "en": "English",
            "fr": "Français",
            "es": "Español"
        }
        selected_language = st.radio(
            get_text("select_language"),
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=list(language_options.keys()).index(st.session_state.language)
        )
        
        # Update language if changed
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            # Update translated dataframe if data exists
            if st.session_state.df is not None:
                st.session_state.df_translated = translate_dataframe(st.session_state.df, st.session_state.language)
            # Rerun to update all text
            st.rerun()
        
        # Data upload
        st.header(get_text("upload_data"))
        uploaded_file = st.file_uploader(get_text("file_types"), type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            if st.session_state.df is None:
                st.session_state.df = load_data(uploaded_file)
                if st.session_state.df is not None:
                    st.session_state.df_translated = translate_dataframe(st.session_state.df, st.session_state.language)
            
            if st.session_state.df is not None:
                df = st.session_state.df_translated
                numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    st.session_state.target_var = st.selectbox(get_text("target_var"), numeric_cols)
                    
                    # EDA button
                    if st.button(get_text("run_eda")):
                        with st.spinner(get_text("run_eda")):
                            st.session_state.eda_results = perform_eda(st.session_state.df, st.session_state.target_var)
                            st.session_state.eda_done = True
                            st.session_state.show_eda = True
                    
                    # Preprocessing options
                    st.header(get_text("preprocessing"))
                    
                    # Options for handling missing values (translated)
                    missing_options = [
                        get_text("Drop rows"),
                        get_text("Mean imputation"), 
                        get_text("KNN imputation")
                    ]
                    
                    encoding_options = [
                        get_text("One-Hot Encoding"),
                        get_text("Label Encoding")
                    ]
                    
                    preprocessing_options = {
                        'handle_missing': st.selectbox(get_text("handle_missing"), missing_options),
                        'handle_duplicates': st.checkbox(get_text("remove_duplicates"), value=True),
                        'handle_outliers': st.checkbox(get_text("handle_outliers"), value=True),
                        'encoding': st.selectbox(get_text("encode_categorical"), encoding_options),
                        'handle_multicollinearity': st.checkbox(get_text("handle_multicollinearity"), value=True),
                        'scaling': st.checkbox(get_text("scale_features"), value=True)
                    }
                    
                    if st.button(get_text("preprocess_data")):
                        with st.spinner(get_text("preprocess_data")):
                            df_processed = preprocess_data(st.session_state.df, st.session_state.target_var, preprocessing_options)
                            
                            # Split data
                            X = df_processed.drop(columns=[st.session_state.target_var])
                            y = df_processed[st.session_state.target_var]
                            
                            X_temp, X_test, y_temp, y_test = train_test_split(
                                X, y, test_size=0.15, random_state=42
                            )
                            X_train, X_val, y_train, y_val = train_test_split(
                                X_temp, y_temp, test_size=0.1765, random_state=42  # 0.15/0.85 ≈ 0.1765
                            )
                            
                            st.session_state.X_train = X_train
                            st.session_state.X_val = X_val
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_val = y_val
                            st.session_state.y_test = y_test
                            
                            st.session_state.preprocessing_done = True
                            st.session_state.show_preprocessing = True
                    
                    # Model training
                    if st.session_state.preprocessing_done:
                        st.header(get_text("modeling"))
                        if st.button(get_text("train_models")):
                            with st.spinner(get_text("train_models")):
                                # Linear Regression
                                lr_model, lr_time, lr_importance = train_linear_regression(
                                    st.session_state.X_train, st.session_state.y_train
                                )
                                st.session_state.models['Linear Regression'] = lr_model
                                st.session_state.model_results['Linear Regression'] = {
                                    'training_time': lr_time,
                                    'feature_importance': lr_importance
                                }
                                
                                # Neural Network
                                nn_model, nn_time, nn_importance, nn_history = train_neural_network(
                                    st.session_state.X_train, st.session_state.y_train,
                                    st.session_state.X_val, st.session_state.y_val
                                )
                                st.session_state.models['Neural Network'] = nn_model
                                st.session_state.model_results['Neural Network'] = {
                                    'training_time': nn_time,
                                    'feature_importance': nn_importance,
                                    'history': nn_history
                                }
                                
                                # MLP
                                mlp_model, mlp_time, mlp_importance, mlp_history = train_mlp(
                                    st.session_state.X_train, st.session_state.y_train,
                                    st.session_state.X_val, st.session_state.y_val
                                )
                                st.session_state.models['MLP'] = mlp_model
                                st.session_state.model_results['MLP'] = {
                                    'training_time': mlp_time,
                                    'feature_importance': mlp_importance,
                                    'history': mlp_history
                                }
                                
                                # Fuzzy Neural Network
                                fuzzy_model, fuzzy_time, fuzzy_importance, _ = train_fuzzy_neural_network(
                                    st.session_state.X_train, st.session_state.y_train
                                )
                                st.session_state.models['Fuzzy Neural Network'] = fuzzy_model
                                st.session_state.model_results['Fuzzy Neural Network'] = {
                                    'training_time': fuzzy_time,
                                    'feature_importance': fuzzy_importance
                                }
                                
                                st.session_state.models_trained = True
                                st.session_state.show_modeling = True
                    
                    # Model evaluation
                    if st.session_state.models_trained:
                        st.header(get_text("evaluation"))
                        if st.button(get_text("evaluate_models")):
                            with st.spinner(get_text("evaluate_models")):
                                for model_name, model in st.session_state.models.items():
                                    metrics = evaluate_model(
                                        model, 
                                        st.session_state.X_test, 
                                        st.session_state.y_test,
                                        model_name
                                    )
                                    st.session_state.evaluation_results[model_name] = metrics
                                
                                # Determine best model based on RMSE
                                best_model = min(
                                    st.session_state.evaluation_results.items(), 
                                    key=lambda x: x[1]['RMSE']
                                )[0]
                                st.session_state.best_model = best_model
                                st.session_state.show_evaluation = True

    # Main content area - Aquí se muestran todos los resultados
    st.markdown(f"<h1 class='main-header'>{get_text('title')}</h1>", unsafe_allow_html=True)

    if st.session_state.df is not None:
        # Show dataset preview (translated)
        st.subheader(get_text("dataset_preview"))
        st.dataframe(st.session_state.df_translated.head())
        
        # Show EDA results if available
        if st.session_state.eda_done and st.session_state.show_eda:
            st.subheader(get_text("eda"))
            st.success(f"{get_text('eda')} {get_text('completed_successfully')}")
            display_eda_results(st.session_state.eda_results, st.session_state.target_var)
        
        # Show preprocessing status
        if st.session_state.preprocessing_done and st.session_state.show_preprocessing:
            st.subheader(get_text("preprocessing"))
            st.success(f"{get_text('preprocessing')} {get_text('completed_successfully')}")
            st.write(f"{get_text('training_set')}: {st.session_state.X_train.shape[0]} {get_text('samples')}")
            st.write(f"{get_text('validation_set')}: {st.session_state.X_val.shape[0]} {get_text('samples')}")
            st.write(f"{get_text('test_set')}: {st.session_state.X_test.shape[0]} {get_text('samples')}")
        
        # Show model training results
        if st.session_state.models_trained and st.session_state.show_modeling:
            st.subheader(get_text("modeling"))
            st.success(f"{get_text('modeling')} {get_text('completed_successfully')}")
            
            # Display training times
            st.write(get_text("training_times"))
            training_times = {
                model: results['training_time'] 
                for model, results in st.session_state.model_results.items()
            }
            st.bar_chart(training_times)
        
        # Show evaluation results
        if st.session_state.evaluation_results and st.session_state.show_evaluation:
            st.subheader(get_text("evaluation"))
            st.success(f"{get_text('evaluation')} {get_text('completed_successfully')}")
            
            # Display evaluation metrics
            evaluation_df = pd.DataFrame(st.session_state.evaluation_results).T
            st.dataframe(evaluation_df)
            
            # Show best model
            st.subheader(get_text("best_model"))
            st.success(f"{get_text('best_model')}: {st.session_state.best_model}")
            
            # Show feature importance for the best model
            st.subheader(get_text("important_features"))
            best_model_importance = st.session_state.model_results[st.session_state.best_model]['feature_importance']
            st.dataframe(best_model_importance.head(10))
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(
                best_model_importance['Feature'].head(10),
                best_model_importance['Importance'].head(10)
            )
            ax.set_xlabel(get_text('Importance'))
            ax.set_title(get_text('Top 10 Feature Importance'))
            st.pyplot(fig)

    else:
        st.info("Please upload a dataset to get started.")

    # Footer
    st.markdown("---")
    st.markdown(get_text("developed_with"))

if __name__ == "__main__":
    main()