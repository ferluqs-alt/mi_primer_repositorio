# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import ks_2samp
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.contingency_tables import mcnemar

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Estimador de Precios de Viviendas",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .sidebar .sidebar-content {
        background-color: #f0f8ff;
    }
</style>
""", unsafe_allow_html=True)

# Translations
translations = {
    "es": {
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
        "completed_successfully": "completado exitosamente!",
        "Drop rows": "Eliminar filas",
        "Mean imputation": "Imputación por media",
        "KNN imputation": "Imputación KNN",
        "Iterative imputation": "Imputación Iterativa",
        "One-Hot Encoding": "Codificación One-Hot",
        "Label Encoding": "Codificación de Etiquetas",
        "no_categorical_vars": "No se encontraron variables categóricas",
        "no_numeric_vars": "No se encontraron variables numéricas",
        "normality_test": "Prueba de Normalidad",
        "ks_test": "Prueba de Kolmogorov-Smirnov",
        "statistic": "Estadístico",
        "p_value": "Valor p",
        "normal": "Normal",
        "not_normal": "No Normal",
        "requires_normalization": "Requiere normalización",
        "correlation_analysis": "Análisis de Correlaciones",
        "pearson_correlation": "Correlación de Pearson",
        "top_correlated": "Variables más correlacionadas",
        "vif_analysis": "Análisis de Factor de Inflación de Varianza (VIF)",
        "feature": "Variable",
        "vif_score": "Puntuación VIF",
        "multicollinearity": "Multicolinealidad",
        "high_multicollinearity": "Alta multicolinealidad (VIF > 10)",
        "moderate_multicollinearity": "Multicolinealidad moderada (5 < VIF ≤ 10)",
        "low_multicollinearity": "Baja multicolinealidad (VIF ≤ 5)",
        "cross_validation": "Validación Cruzada",
        "k_folds": "Número de folds",
        "cv_scores": "Puntuaciones de Validación Cruzada",
        "mean_score": "Puntuación Media",
        "std_dev": "Desviación Estándar",
        "confusion_matrix": "Matriz de Confusión",
        "mcc": "Coeficiente de Correlación de Matthews (MCC)",
        "mcnemar_test": "Prueba de McNemar",
        "statistic": "Estadístico",
        "p_value": "Valor p",
        "significant": "Significativo (p < 0.05)",
        "not_significant": "No significativo (p ≥ 0.05)",
        "time_series_analysis": "Análisis de Series Temporales",
        "arima_model": "Modelo ARIMA",
        "holt_winters": "Modelo Holt-Winters",
        "theil_u": "Estadístico U de Theil",
        "diebold_mariano": "Prueba de Diebold-Mariano",
        "forecast": "Pronóstico",
        "actual": "Real",
        "predicted": "Predicho",
        "download_pdf": "Descargar Reporte PDF",
        "linear_regression": "Regresión Lineal",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "mlp": "Perceptrón Multicapa",
        "model_type": "Tipo de Modelo",
        "model_performance": "Rendimiento del Modelo",
        "model_comparison": "Comparación de Modelos",
        "summary": "Resumen",
        "conclusions": "Conclusiones",
        "recommendations": "Recomendaciones",
        "usd_format": "${:,.2f} USD",
        "null_handling_header": "Manejo de Valores Nulos",
        "show_null_summary": "Mostrar resumen de valores nulos",
        "null_summary_title": "Resumen de Valores Nulos",
        "select_null_strategy": "Seleccionar Estrategia para Valores Nulos",
        "strategy_for": "Estrategia para",
        "drop_na": "Eliminar filas con nulos",
        "mean_imputation": "Rellenar con la media",
        "median_imputation": "Rellenar con la mediana",
        "mode_imputation": "Rellenar con la moda",
        "constant_imputation": "Rellenar con valor constante",
        "unknown_imputation": "Rellenar con 'Desconocido'",
        "interpolation": "Interpolar",
        "constant_value_for": "Valor constante para",
        "apply_null_strategies": "Aplicar Estrategias para Nulos",
        "nulls_removed_success": "Se eliminaron exitosamente",
        "null_values": "valores nulos",
        "no_null_values": "No se encontraron valores nulos en el dataset.",
        "unknown_value": "Desconocido"
    },
    "en": {
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
        "completed_successfully": "completed successfully!",
        "Drop rows": "Drop rows",
        "Mean imputation": "Mean imputation",
        "KNN imputation": "KNN imputation",
        "Iterative imputation": "Iterative imputation",
        "One-Hot Encoding": "One-Hot Encoding",
        "Label Encoding": "Label Encoding",
        "no_categorical_vars": "No categorical variables found",
        "no_numeric_vars": "No numeric variables found",
        "normality_test": "Normality Test",
        "ks_test": "Kolmogorov-Smirnov Test",
        "statistic": "Statistic",
        "p_value": "p-value",
        "normal": "Normal",
        "not_normal": "Not Normal",
        "requires_normalization": "Requires normalization",
        "correlation_analysis": "Correlation Analysis",
        "pearson_correlation": "Pearson Correlation",
        "top_correlated": "Top Correlated Variables",
        "vif_analysis": "Variance Inflation Factor (VIF) Analysis",
        "feature": "Feature",
        "vif_score": "VIF Score",
        "multicollinearity": "Multicollinearity",
        "high_multicollinearity": "High multicollinearity (VIF > 10)",
        "moderate_multicollinearity": "Moderate multicollinearity (5 < VIF ≤ 10)",
        "low_multicollinearity": "Low multicollinearity (VIF ≤ 5)",
        "cross_validation": "Cross Validation",
        "k_folds": "Number of folds",
        "cv_scores": "Cross Validation Scores",
        "mean_score": "Mean Score",
        "std_dev": "Standard Deviation",
        "confusion_matrix": "Confusion Matrix",
        "mcc": "Matthews Correlation Coefficient (MCC)",
        "mcnemar_test": "McNemar Test",
        "statistic": "Statistic",
        "p_value": "p-value",
        "significant": "Significant (p < 0.05)",
        "not_significant": "Not significant (p ≥ 0.05)",
        "time_series_analysis": "Time Series Analysis",
        "arima_model": "ARIMA Model",
        "holt_winters": "Holt-Winters Model",
        "theil_u": "Theil's U Statistic",
        "diebold_mariano": "Diebold-Mariano Test",
        "forecast": "Forecast",
        "actual": "Actual",
        "predicted": "Predicted",
        "download_pdf": "Download PDF Report",
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "mlp": "Multi-Layer Perceptron",
        "model_type": "Model Type",
        "model_performance": "Model Performance",
        "model_comparison": "Model Comparison",
        "summary": "Summary",
        "conclusions": "Conclusions",
        "recommendations": "Recommendations",
        "usd_format": "${:,.2f} USD",
        "null_handling_header": "Null Values Handling",
        "show_null_summary": "Show null values summary",
        "null_summary_title": "Null Values Summary",
        "select_null_strategy": "Select Strategy for Null Values",
        "strategy_for": "Strategy for",
        "drop_na": "Drop rows with nulls",
        "mean_imputation": "Fill with mean",
        "median_imputation": "Fill with median",
        "mode_imputation": "Fill with mode",
        "constant_imputation": "Fill with constant value",
        "unknown_imputation": "Fill with 'Unknown'",
        "interpolation": "Interpolate",
        "constant_value_for": "Constant value for",
        "apply_null_strategies": "Apply Null Strategies",
        "nulls_removed_success": "Successfully removed",
        "null_values": "null values",
        "no_null_values": "No null values found in the dataset.",
        "unknown_value": "Unknown"
    },
    "fr": {
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
        "completed_successfully": "terminé avec succès!",
        "Drop rows": "Supprimer les lignes",
        "Mean imputation": "Imputation par la moyenne",
        "KNN imputation": "Imputation KNN",
        "Iterative imputation": "Imputation Itérative",
        "One-Hot Encoding": "Encodage One-Hot",
        "Label Encoding": "Encodage d'étiquettes",
        "no_categorical_vars": "Aucune variable catégorielle trouvée",
        "no_numeric_vars": "Aucune variable numérique trouvée",
        "normality_test": "Test de Normalité",
        "ks_test": "Test de Kolmogorov-Smirnov",
        "statistic": "Statistique",
        "p_value": "Valeur p",
        "normal": "Normal",
        "not_normal": "Non Normal",
        "requires_normalization": "Nécessite une normalisation",
        "correlation_analysis": "Analyse de Corrélation",
        "pearson_correlation": "Corrélation de Pearson",
        "top_correlated": "Variables les Plus Corrélées",
        "vif_analysis": "Analyse du Facteur d'Inflation de la Variance (VIF)",
        "feature": "Variable",
        "vif_score": "Score VIF",
        "multicollinearity": "Multicolinéarité",
        "high_multicollinearity": "Multicolinéarité élevée (VIF > 10)",
        "moderate_multicollinearity": "Multicolinéarité modérée (5 < VIF ≤ 10)",
        "low_multicollinearity": "Faible multicolinéarité (VIF ≤ 5)",
        "cross_validation": "Validation Croisée",
        "k_folds": "Nombre de plis",
        "cv_scores": "Scores de Validation Croisée",
        "mean_score": "Score Moyen",
        "std_dev": "Écart-Type",
        "confusion_matrix": "Matrice de Confusion",
        "mcc": "Coefficient de Corrélation de Matthews (MCC)",
        "mcnemar_test": "Test de McNemar",
        "statistic": "Statistique",
        "p_value": "Valeur p",
        "significant": "Significatif (p < 0.05)",
        "not_significant": "Non significatif (p ≥ 0.05)",
        "time_series_analysis": "Analyse de Série Temporelle",
        "arima_model": "Modèle ARIMA",
        "holt_winters": "Modèle Holt-Winters",
        "theil_u": "Statistique U de Theil",
        "diebold_mariano": "Test de Diebold-Mariano",
        "forecast": "Prévision",
        "actual": "Réel",
        "predicted": "Prédit",
        "download_pdf": "Télécharger le Rapport PDF",
        "linear_regression": "Régression Linéaire",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "mlp": "Perceptron Multicouche",
        "model_type": "Type de Modèle",
        "model_performance": "Performance du Modèle",
        "model_comparison": "Comparaison de Modèles",
        "summary": "Résumé",
        "conclusions": "Conclusions",
        "recommendations": "Recommandations",
        "usd_format": "${:,.2f} USD",
        "null_handling_header": "Gestion des Valeurs Manquantes",
        "show_null_summary": "Afficher le résumé des valeurs manquantes",
        "null_summary_title": "Résumé des Valeurs Manquantes",
        "select_null_strategy": "Sélectionner une Stratégie pour les Valeurs Manquantes",
        "strategy_for": "Stratégie pour",
        "drop_na": "Supprimer les lignes avec des valeurs manquantes",
        "mean_imputation": "Remplir avec la moyenne",
        "median_imputation": "Remplir avec la médiane",
        "mode_imputation": "Remplir avec le mode",
        "constant_imputation": "Remplir avec une valeur constante",
        "unknown_imputation": "Remplir avec 'Inconnu'",
        "interpolation": "Interpolation",
        "constant_value_for": "Valeur constante pour",
        "apply_null_strategies": "Appliquer les Stratégies pour Valeurs Manquantes",
        "nulls_removed_success": "Supprimé avec succès",
        "null_values": "valeurs manquantes",
        "no_null_values": "Aucune valeur manquante trouvée dans le dataset.",
        "unknown_value": "Inconnu"
    }
}

# Function to get translated text
def get_text(key):
    """Returns translated text for the given key based on selected language"""
    lang = st.session_state.language
    if lang in translations and key in translations[lang]:
        return translations[lang][key]
    elif "en" in translations and key in translations["en"]:  # Fallback to English
        return translations["en"][key]
    else:
        return key  # Return the key itself if not found

# Function to format currency in USD
def format_usd(value):
    """Formats a number as USD currency"""
    try:
        return get_text("usd_format").format(float(value))
    except (ValueError, TypeError):
        return str(value)

# Function to translate dataframe categorical values and column names
def translate_dataframe(df, language):
    """Translates categorical values and column names in the dataframe based on selected language"""
    if df is None:
        return None
        
    df_translated = df.copy()
    
    # Dictionary for column name translations
    column_translations = {
        "en": {
            "longitude": "longitude",
            "latitude": "latitude",
            "housing_median_age": "housing_median_age",
            "total_rooms": "total_rooms",
            "total_bedrooms": "total_bedrooms",
            "population": "population",
            "households": "households",
            "median_income": "median_income",
            "median_house_value": "median_house_value",
            "ocean_proximity": "ocean_proximity"
        },
        "es": {
            "longitude": "longitud",
            "latitude": "latitud",
            "housing_median_age": "edad_media_vivienda",
            "total_rooms": "total_habitaciones",
            "total_bedrooms": "total_dormitorios",
            "population": "población",
            "households": "hogares",
            "median_income": "ingreso_medio",
            "median_house_value": "valor_medio_vivienda",
            "ocean_proximity": "proximidad_océano"
        },
        "fr": {
            "longitude": "longitude",
            "latitude": "latitude",
            "housing_median_age": "âge_médian_logement",
            "total_rooms": "total_pièces",
            "total_bedrooms": "total_chambres",
            "population": "population",
            "households": "ménages",
            "median_income": "revenu_médian",
            "median_house_value": "valeur_médiane_logement",
            "ocean_proximity": "proximité_océan"
        }
    }
    
    # Translate column names
    if language in column_translations:
        # Create a mapping of original to translated column names
        column_mapping = {}
        for original_col in df.columns:
            if original_col in column_translations[language]:
                column_mapping[original_col] = column_translations[language][original_col]
            else:
                column_mapping[original_col] = original_col  # Keep original if no translation
        
        df_translated = df_translated.rename(columns=column_mapping)
    
    # Define which columns are likely to contain categorical data
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        # Translate each value in the column if it exists in our translations
        df_translated[col] = df[col].apply(
            lambda x: translations[language].get(str(x).upper(), x) if pd.notna(x) else x
        )
    
    return df_translated

# Function to handle null values interactively
def handle_null_values(df):
    """Interfaz para manejar valores nulos de forma interactiva"""
    if df is None:
        return df
        
    st.header(get_text("null_handling_header"))
    
    # Mostrar resumen de valores nulos
    if st.checkbox(get_text("show_null_summary"), value=True):
        null_summary = df.isnull().sum()
        null_summary = null_summary[null_summary > 0]
        
        if len(null_summary) > 0:
            st.subheader(get_text("null_summary_title"))
            st.dataframe(pd.DataFrame({
                'Column': null_summary.index,
                'Null Count': null_summary.values,
                'Percentage': (null_summary.values / len(df) * 100).round(2)
            }))
            
            # Selector de estrategia por columna
            st.subheader(get_text("select_null_strategy"))
            strategies = {}
            
            for col in null_summary.index:
                st.markdown(f"**{col}** ({null_summary[col]} nulos)")
                
                col_type = df[col].dtype
                if np.issubdtype(col_type, np.number):
                    # Columnas numéricas
                    strategy = st.selectbox(
                        f"{get_text('strategy_for')} {col}",
                        options=[
                            get_text("drop_na"),
                            get_text("mean_imputation"),
                            get_text("median_imputation"),
                            get_text("constant_imputation"),
                            get_text("interpolation"),
                            get_text("KNN imputation"),
                            get_text("Iterative imputation")
                        ],
                        key=f"strategy_{col}"
                    )
                    
                    strategies[col] = {
                        'type': 'numeric',
                        'strategy': strategy
                    }
                    
                    if strategy == get_text("constant_imputation"):
                        constant_value = st.number_input(
                            f"{get_text('constant_value_for')} {col}",
                            value=0.0,
                            key=f"constant_{col}"
                        )
                        strategies[col]['constant'] = constant_value
                else:
                    # Columnas categóricas
                    strategy = st.selectbox(
                        f"{get_text('strategy_for')} {col}",
                        options=[
                            get_text("drop_na"),
                            get_text("mode_imputation"),
                            get_text("constant_imputation"),
                            get_text("unknown_imputation")
                        ],
                        key=f"strategy_{col}"
                    )
                    
                    strategies[col] = {
                        'type': 'categorical',
                        'strategy': strategy
                    }
                    
                    if strategy == get_text("constant_imputation"):
                        constant_value = st.text_input(
                            f"{get_text('constant_value_for')} {col}",
                            value="",
                            key=f"constant_{col}"
                        )
                        strategies[col]['constant'] = constant_value
            
            # Botón para aplicar las estrategias
            if st.button(get_text("apply_null_strategies")):
                df_processed = df.copy()
                total_nulls_before = df_processed.isnull().sum().sum()
                
                for col, strategy_info in strategies.items():
                    strategy = strategy_info['strategy']
                    
                    if strategy == get_text("drop_na"):
                        df_processed = df_processed.dropna(subset=[col])
                    elif strategy == get_text("mean_imputation"):
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                    elif strategy == get_text("median_imputation"):
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    elif strategy == get_text("mode_imputation"):
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                    elif strategy == get_text("constant_imputation"):
                        df_processed[col] = df_processed[col].fillna(strategy_info.get('constant', 0))
                    elif strategy == get_text("unknown_imputation"):
                        df_processed[col] = df_processed[col].fillna(get_text("unknown_value"))
                    elif strategy == get_text("interpolation"):
                        df_processed[col] = df_processed[col].interpolate()
                    elif strategy == get_text("KNN imputation"):
                        imputer = KNNImputer(n_neighbors=5)
                        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
                    elif strategy == get_text("Iterative imputation"):
                        imputer = IterativeImputer(random_state=42)
                        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
                
                total_nulls_after = df_processed.isnull().sum().sum()
                nulls_removed = total_nulls_before - total_nulls_after
                
                st.success(f"{get_text('nulls_removed_success')} {nulls_removed} {get_text('null_values')}.")
                return df_processed
        else:
            st.info(get_text("no_null_values"))
    
    return df

# Function to perform EDA
def perform_eda(df, target_var):
    """Realiza análisis exploratorio de datos y devuelve resultados estructurados"""
    try:
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
        
        # Pruebas de normalidad
        normality_tests = {}
        for col in numeric_cols:
            if col != target_var:  # Don't test target variable
                stat, p_value = ks_2samp(df[col].dropna(), 'norm')
                normality_tests[col] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'normal': p_value > 0.05
                }
        
        return {
            'dimensions': dimensions,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'missing_df': missing_df,
            'duplicates': duplicates,
            'descriptive_stats': descriptive_stats,
            'corr_matrix': corr_matrix,
            'target_correlations': target_correlations,
            'numeric_cols_list': numeric_cols,
            'normality_tests': normality_tests
        }
    except Exception as e:
        st.error(f"Error performing EDA: {str(e)}")
        # Return empty structure to prevent KeyError
        return {
            'dimensions': {'num_rows': 0, 'num_cols': 0},
            'numeric_cols': [],
            'categorical_cols': [],
            'missing_df': pd.DataFrame(),
            'duplicates': 0,
            'descriptive_stats': pd.DataFrame(),
            'corr_matrix': None,
            'target_correlations': None,
            'numeric_cols_list': [],
            'normality_tests': {}
        }

# Function to display EDA results
def display_eda_results(eda_results, target_var):
    """Muestra los resultados del EDA en el área principal con manejo seguro de errores"""
    if not eda_results:
        st.error("No EDA results available")
        return
    
    # Usar el dataframe original para EDA (no el traducido)
    df = st.session_state.df
    
    st.subheader(get_text("data_dimensions"))
    col1, col2 = st.columns(2)
    col1.metric(get_text("num_rows"), eda_results.get('dimensions', {}).get('num_rows', 0))
    col2.metric(get_text("num_cols"), eda_results.get('dimensions', {}).get('num_cols', 0))
    
    # Numeric variables - with safe access
    st.subheader(get_text("numeric_vars"))
    numeric_cols = eda_results.get('numeric_cols', [])
    if numeric_cols:
        st.write(numeric_cols)
    else:
        st.info(get_text("no_numeric_vars"))
    
    # Categorical variables - with safe access
    st.subheader(get_text("categorical_vars"))
    categorical_cols = eda_results.get('categorical_cols', [])
    if categorical_cols:
        st.write(categorical_cols)
    else:
        st.info(get_text("no_categorical_vars"))
    
    # Missing values
    st.subheader(get_text("missing_values"))
    missing_df = eda_results.get('missing_df', pd.DataFrame())
    if not missing_df.empty:
        missing_data = missing_df[missing_df['Missing Values'] > 0]
        if not missing_data.empty:
            st.dataframe(missing_data)
        else:
            st.info("No missing values found")
    else:
        st.info("No missing values data available")
    
    # Duplicates
    st.subheader(get_text("duplicates"))
    duplicates = eda_results.get('duplicates', 0)
    st.metric(get_text("duplicates"), duplicates)
    
    # Descriptive statistics
    st.subheader(get_text("descriptive_stats"))
    descriptive_stats = eda_results.get('descriptive_stats', pd.DataFrame())
    if not descriptive_stats.empty:
        st.dataframe(descriptive_stats)
    else:
        st.info("No descriptive statistics available")
    
    # Normality tests
    st.subheader(get_text("normality_test"))
    normality_tests = eda_results.get('normality_tests', {})
    if normality_tests:
        normality_df = pd.DataFrame.from_dict(normality_tests, orient='index')
        normality_df['Normal'] = normality_df['p_value'] > 0.05
        normality_df['Requires Normalization'] = ~normality_df['Normal']
        st.dataframe(normality_df)
    else:
        st.info("No normality test results available")
    
    # Visualizations
    st.subheader(get_text("distributions"))
    
    # Numeric variables distributions
    numeric_cols_list = eda_results.get('numeric_cols', [])
    if numeric_cols_list:
        num_cols_to_show = min(5, len(numeric_cols_list))
        fig, axes = plt.subplots(1, num_cols_to_show, figsize=(20, 4))
        if num_cols_to_show == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols_list[:num_cols_to_show]):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color='blue')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No numeric variables to display distributions")
    
    # Correlation matrix
    st.subheader(get_text("correlation_matrix"))
    corr_matrix = eda_results.get('corr_matrix')
    if corr_matrix is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
        # Top correlations with target
        target_correlations = eda_results.get('target_correlations')
        if target_correlations is not None:
            st.write(f"{get_text('top_correlations')} {target_var}:")
            st.write(target_correlations[1:6])  # Top 5 excluding itself
    else:
        st.info("Not enough numeric variables for correlation matrix")
    
    # Outlier detection
    st.subheader(get_text("outlier_detection"))
    numeric_cols_list = eda_results.get('numeric_cols_list', [])
    if numeric_cols_list:
        num_cols_to_show = min(3, len(numeric_cols_list))
        fig, axes = plt.subplots(1, num_cols_to_show, figsize=(15, 5))
        if num_cols_to_show == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols_list[:num_cols_to_show]):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(f'Boxplot of {col}')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No numeric variables for outlier detection")

# Function to calculate VIF
def calculate_vif(df, target_var):
    """Calcula el Factor de Inflación de Varianza (VIF) para las variables numéricas"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_var in numeric_cols:
        numeric_cols.remove(target_var)
    
    if len(numeric_cols) > 1:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_cols
        vif_data["VIF"] = [variance_inflation_factor(df[numeric_cols].values, i) 
                          for i in range(len(numeric_cols))]
        return vif_data
    return pd.DataFrame()

# Function to preprocess data
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
    elif options['handle_missing'] == get_text("Iterative imputation"):
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        imputer = IterativeImputer(random_state=42)
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

# Function to train linear regression
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

# Function to train random forest
def train_random_forest(X_train, y_train):
    start_time = time.time()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return model, training_time, feature_importance

# Function to train gradient boosting
def train_gradient_boosting(X_train, y_train):
    start_time = time.time()
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return model, training_time, feature_importance

# Function to train MLP
def train_mlp(X_train, y_train):
    start_time = time.time()
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # For MLP, we don't have direct feature importance, so we'll use permutation importance
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
    importance = result.importances_mean
    
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return model, training_time, feature_importance

# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    if hasattr(model, 'predict'):
        # Scikit-learn model
        y_pred = model.predict(X_test)
    else:
        # For other models
        y_pred = np.full(len(y_test), y_test.mean())  # Fallback to mean prediction
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'predictions': y_pred
    }

# Function to perform cross validation
def perform_cross_validation(model, X, y, cv=5):
    """Realiza validación cruzada y devuelve los resultados"""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    if hasattr(model, 'predict'):
        # Scikit-learn model
        scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        scores = np.sqrt(-scores)  # Convert to RMSE
    else:
        # For other models, use a simple implementation
        scores = []
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train a simple model for this fold
            fold_model = LinearRegression()
            fold_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = fold_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(rmse)
    
    return {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores)
    }

# Function to perform McNemar test
def perform_mcnemar_test(y_true, y_pred1, y_pred2):
    """Realiza la prueba de McNemar para comparar dos modelos"""
    # Create contingency table
    # For regression, we need to convert to classification
    # We'll use median as threshold
    median_val = np.median(y_true)
    correct1 = (y_pred1 > median_val) == (y_true > median_val)
    correct2 = (y_pred2 > median_val) == (y_true > median_val)
    
    # Create 2x2 contingency table
    both_correct = np.sum(correct1 & correct2)
    only1_correct = np.sum(correct1 & ~correct2)
    only2_correct = np.sum(~correct1 & correct2)
    both_wrong = np.sum(~correct1 & ~correct2)
    
    table = [[both_correct, only1_correct],
             [only2_correct, both_wrong]]
    
    # Perform McNemar test
    result = mcnemar(table, exact=False)
    
    return {
        'table': table,
        'statistic': result.statistic,
        'p_value': result.pvalue,
        'significant': result.pvalue < 0.05
    }

# Function to calculate Matthews Correlation Coefficient
def calculate_mcc(y_true, y_pred):
    """Calcula el coeficiente de correlación de Matthews"""
    # For regression, we need to convert to classification
    # We'll use median as threshold
    median_val = np.median(y_true)
    y_true_bin = (y_true > median_val).astype(int)
    y_pred_bin = (y_pred > median_val).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    
    return mcc

# Function to perform time series analysis
def perform_time_series_analysis(df, date_col, target_col):
    """Realiza análisis de series temporales si hay una columna de fecha"""
    try:
        # Check if date column exists
        if date_col not in df.columns:
            return None
        
        # Convert to datetime and set as index
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.set_index(date_col)
        df_ts = df_ts[[target_col]].resample('M').mean().dropna()
        
        # Split into train and test
        train_size = int(len(df_ts) * 0.8)
        train, test = df_ts[0:train_size], df_ts[train_size:]
        
        # Fit ARIMA model
        try:
            arima_model = ARIMA(train, order=(1,1,1))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=len(test))[0]
        except:
            arima_forecast = np.full(len(test), train[target_col].mean())
        
        # Fit Holt-Winters model
        try:
            hw_model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)
            hw_fit = hw_model.fit()
            hw_forecast = hw_fit.forecast(steps=len(test))
        except:
            hw_forecast = np.full(len(test), train[target_col].mean())
        
        # Calculate Theil's U statistic
        def theils_u(actual, forecast):
            n = len(actual)
            numerator = np.sqrt(np.sum((actual - forecast) ** 2) / n)
            denominator = np.sqrt(np.sum(actual ** 2) / n) + np.sqrt(np.sum(forecast ** 2) / n)
            return numerator / denominator
        
        arima_u = theils_u(test[target_col].values, arima_forecast)
        hw_u = theils_u(test[target_col].values, hw_forecast)
        
        # Calculate Diebold-Mariano test (simplified)
        def diebold_mariano(actual, forecast1, forecast2):
            loss1 = (actual - forecast1) ** 2
            loss2 = (actual - forecast2) ** 2
            d = loss1 - loss2
            dm_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d)))
            p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
            return dm_stat, p_value
        
        dm_stat, dm_pvalue = diebold_mariano(test[target_col].values, arima_forecast, hw_forecast)
        
        return {
            'arima_forecast': arima_forecast,
            'hw_forecast': hw_forecast,
            'actual': test[target_col].values,
            'theil_u_arima': arima_u,
            'theil_u_hw': hw_u,
            'dm_statistic': dm_stat,
            'dm_pvalue': dm_pvalue,
            'dates': test.index
        }
    except Exception as e:
        st.error(f"Error in time series analysis: {str(e)}")
        return None

# Function to generate PDF report
def generate_pdf_report(eda_results, model_results, evaluation_results, ts_results, filename):
    """Genera un reporte PDF con todos los resultados"""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph(get_text("title"), styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # EDA Summary
    story.append(Paragraph(get_text("eda"), styles['Heading2']))
    story.append(Paragraph(f"{get_text('num_rows')}: {eda_results.get('dimensions', {}).get('num_rows', 0)}", styles['Normal']))
    story.append(Paragraph(f"{get_text('num_cols')}: {eda_results.get('dimensions', {}).get('num_cols', 0)}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Model Results
    story.append(Paragraph(get_text("model_comparison"), styles['Heading2']))
    
    # Create table with model results
    model_data = [['Model', 'MAE', 'RMSE', 'R²', 'Training Time']]
    for model_name, results in evaluation_results.items():
        model_data.append([
            model_name,
            f"{results['MAE']:.4f}",
            f"{results['RMSE']:.4f}",
            f"{results['R2']:.4f}",
            f"{model_results[model_name]['training_time']:.2f}s"
        ])
    
    model_table = Table(model_data)
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(model_table)
    story.append(Spacer(1, 12))
    
    # Time Series Results
    if ts_results:
        story.append(Paragraph(get_text("time_series_analysis"), styles['Heading2']))
        story.append(Paragraph(f"{get_text('theil_u')} (ARIMA): {ts_results['theil_u_arima']:.4f}", styles['Normal']))
        story.append(Paragraph(f"{get_text('theil_u')} (Holt-Winters): {ts_results['theil_u_hw']:.4f}", styles['Normal']))
        story.append(Paragraph(f"{get_text('diebold_mariano')}: {ts_results['dm_statistic']:.4f} (p={ts_results['dm_pvalue']:.4f})", styles['Normal']))
        story.append(Spacer(1, 12))
    
    # Conclusions
    story.append(Paragraph(get_text("conclusions"), styles['Heading2']))
    best_model = min(evaluation_results.items(), key=lambda x: x[1]['RMSE'])[0]
    story.append(Paragraph(f"{get_text('best_model')}: {best_model}", styles['Normal']))
    story.append(Paragraph(f"{get_text('rmse')}: {evaluation_results[best_model]['RMSE']:.4f}", styles['Normal']))
    story.append(Paragraph(f"{get_text('r2')}: {evaluation_results[best_model]['R2']:.4f}", styles['Normal']))
    
    # Build PDF
    doc.build(story)

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
    if 'eda_results' not in st.session_state:
        st.session_state.eda_results = None
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    if 'ts_results' not in st.session_state:
        st.session_state.ts_results = None

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

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Sidebar - Controls and navigation
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
            if st.session_state.df is not None:
                st.session_state.df_translated = translate_dataframe(st.session_state.df, st.session_state.language)
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
                    
                    # Null values handling
                    st.session_state.df = handle_null_values(st.session_state.df)
                    if st.session_state.df is not None:
                        st.session_state.df_translated = translate_dataframe(st.session_state.df, st.session_state.language)
                    
                    # Options for handling missing values (translated)
                    missing_options = [
                        get_text("Drop rows"),
                        get_text("Mean imputation"), 
                        get_text("KNN imputation"),
                        get_text("Iterative imputation")
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
                        
                        # Model selection
                        model_options = [
                            get_text("linear_regression"),
                            get_text("random_forest"),
                            get_text("gradient_boosting"),
                            get_text("mlp")
                        ]
                        
                        selected_models = st.multiselect(
                            "Select models to train",
                            options=model_options,
                            default=[get_text("linear_regression"), get_text("random_forest")]
                        )
                        
                        if st.button(get_text("train_models")):
                            with st.spinner(get_text("train_models")):
                                # Clear previous models
                                st.session_state.models = {}
                                st.session_state.model_results = {}
                                
                                # Train selected models
                                if get_text("linear_regression") in selected_models:
                                    lr_model, lr_time, lr_importance = train_linear_regression(
                                        st.session_state.X_train, st.session_state.y_train
                                    )
                                    st.session_state.models[get_text("linear_regression")] = lr_model
                                    st.session_state.model_results[get_text("linear_regression")] = {
                                        'training_time': lr_time,
                                        'feature_importance': lr_importance
                                    }
                                
                                if get_text("random_forest") in selected_models:
                                    rf_model, rf_time, rf_importance = train_random_forest(
                                        st.session_state.X_train, st.session_state.y_train
                                    )
                                    st.session_state.models[get_text("random_forest")] = rf_model
                                    st.session_state.model_results[get_text("random_forest")] = {
                                        'training_time': rf_time,
                                        'feature_importance': rf_importance
                                    }
                                
                                if get_text("gradient_boosting") in selected_models:
                                    gb_model, gb_time, gb_importance = train_gradient_boosting(
                                        st.session_state.X_train, st.session_state.y_train
                                    )
                                    st.session_state.models[get_text("gradient_boosting")] = gb_model
                                    st.session_state.model_results[get_text("gradient_boosting")] = {
                                        'training_time': gb_time,
                                        'feature_importance': gb_importance
                                    }
                                
                                if get_text("mlp") in selected_models:
                                    mlp_model, mlp_time, mlp_importance = train_mlp(
                                        st.session_state.X_train, st.session_state.y_train
                                    )
                                    st.session_state.models[get_text("mlp")] = mlp_model
                                    st.session_state.model_results[get_text("mlp")] = {
                                        'training_time': mlp_time,
                                        'feature_importance': mlp_importance
                                    }
                                
                                st.session_state.models_trained = True
                                st.session_state.show_modeling = True
                    
                    # Model evaluation
                    if st.session_state.models_trained:
                        st.header(get_text("evaluation"))
                        if st.button(get_text("evaluate_models")):
                            with st.spinner(get_text("evaluate_models")):
                                st.session_state.evaluation_results = {}
                                
                                for model_name, model in st.session_state.models.items():
                                    metrics = evaluate_model(
                                        model, 
                                        st.session_state.X_test, 
                                        st.session_state.y_test,
                                        model_name
                                    )
                                    st.session_state.evaluation_results[model_name] = metrics
                                
                                # Calculate additional metrics
                                for model_name, results in st.session_state.evaluation_results.items():
                                    # Calculate MCC
                                    results['MCC'] = calculate_mcc(
                                        st.session_state.y_test, results['predictions']
                                    )
                                
                                # Determine best model based on RMSE
                                best_model = min(
                                    st.session_state.evaluation_results.items(), 
                                    key=lambda x: x[1]['RMSE']
                                )[0]
                                st.session_state.best_model = best_model
                                st.session_state.show_evaluation = True
                                
                                # Perform McNemar tests between models
                                if len(st.session_state.models) > 1:
                                    model_names = list(st.session_state.models.keys())
                                    st.session_state.mcnemar_results = {}
                                    
                                    for i in range(len(model_names)):
                                        for j in range(i+1, len(model_names)):
                                            model1 = model_names[i]
                                            model2 = model_names[j]
                                            
                                            mcnemar_result = perform_mcnemar_test(
                                                st.session_state.y_test,
                                                st.session_state.evaluation_results[model1]['predictions'],
                                                st.session_state.evaluation_results[model2]['predictions']
                                            )
                                            
                                            st.session_state.mcnemar_results[f"{model1}_vs_{model2}"] = mcnemar_result
                        
                        # Time series analysis if date column exists
                        date_cols = st.session_state.df.select_dtypes(include=['datetime64']).columns.tolist()
                        if date_cols:
                            date_col = st.selectbox("Select date column for time series analysis", date_cols)
                            if st.button("Run Time Series Analysis"):
                                with st.spinner("Running time series analysis..."):
                                    st.session_state.ts_results = perform_time_series_analysis(
                                        st.session_state.df, date_col, st.session_state.target_var
                                    )
                        
                        # Generate PDF report
                        if st.button(get_text("generate_report")):
                            with st.spinner("Generating PDF report..."):
                                # Create a temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                                    generate_pdf_report(
                                        st.session_state.eda_results,
                                        st.session_state.model_results,
                                        st.session_state.evaluation_results,
                                        st.session_state.ts_results,
                                        tmp.name
                                    )
                                    
                                    # Read the file and create a download button
                                    with open(tmp.name, "rb") as f:
                                        pdf_data = f.read()
                                    
                                    st.download_button(
                                        label=get_text("download_pdf"),
                                        data=pdf_data,
                                        file_name="housing_price_report.pdf",
                                        mime="application/pdf"
                                    )

    # Main content area - Results and visualizations
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
            evaluation_data = []
            for model_name, results in st.session_state.evaluation_results.items():
                evaluation_data.append({
                    'Model': model_name,
                    'MAE': results['MAE'],
                    'RMSE': results['RMSE'],
                    'R²': results['R2'],
                    'MCC': results.get('MCC', 'N/A')
                })
            
            evaluation_df = pd.DataFrame(evaluation_data)
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
            
            # Show McNemar test results if available
            if hasattr(st.session_state, 'mcnemar_results'):
                st.subheader(get_text("mcnemar_test"))
                for comparison, result in st.session_state.mcnemar_results.items():
                    st.write(f"**{comparison}**")
                    st.write(f"Statistic: {result['statistic']:.4f}, p-value: {result['p_value']:.4f}")
                    st.write(f"Significant: {result['significant']}")
            
            # Show time series results if available
            if st.session_state.ts_results:
                st.subheader(get_text("time_series_analysis"))
                
                # Plot time series forecast
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(st.session_state.ts_results['dates'], st.session_state.ts_results['actual'], label='Actual')
                ax.plot(st.session_state.ts_results['dates'], st.session_state.ts_results['arima_forecast'], label='ARIMA Forecast')
                ax.plot(st.session_state.ts_results['dates'], st.session_state.ts_results['hw_forecast'], label='Holt-Winters Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.set_title('Time Series Forecast')
                ax.legend()
                st.pyplot(fig)
                
                # Display time series metrics
                col1, col2, col3 = st.columns(3)
                col1.metric(get_text("theil_u") + " (ARIMA)", f"{st.session_state.ts_results['theil_u_arima']:.4f}")
                col2.metric(get_text("theil_u") + " (Holt-Winters)", f"{st.session_state.ts_results['theil_u_hw']:.4f}")
                col3.metric(get_text("diebold_mariano") + " p-value", f"{st.session_state.ts_results['dm_pvalue']:.4f}")

    else:
        st.info("Please upload a dataset to get started.")

    # Footer
    st.markdown("---")
    st.markdown(get_text("developed_with"))

if __name__ == "__main__":
    main()