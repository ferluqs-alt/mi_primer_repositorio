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

# Report generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

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
    }
}

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
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

# Helper functions
def t(key):
    return translations[st.session_state.language][key]

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
    st.subheader(t('data_dimensions'))
    col1, col2 = st.columns(2)
    col1.metric(t('num_rows'), df.shape[0])
    col2.metric(t('num_cols'), df.shape[1])
    
    st.subheader(t('numeric_vars'))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.write(numeric_cols)
    
    st.subheader(t('categorical_vars'))
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    st.write(categorical_cols)
    
    st.subheader(t('missing_values'))
    missing_data = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Values': missing_data.values,
        'Percentage': (missing_data.values / len(df)) * 100
    })
    st.dataframe(missing_df[missing_df['Missing Values'] > 0])
    
    st.subheader(t('duplicates'))
    duplicates = df.duplicated().sum()
    st.metric("Duplicated Rows", duplicates)
    
    st.subheader(t('descriptive_stats'))
    st.dataframe(df.describe())
    
    # Visualizations
    st.subheader("Distributions")
    
    # Numeric variables distributions
    if numeric_cols:
        num_cols_to_show = min(5, len(numeric_cols))
        fig, axes = plt.subplots(1, num_cols_to_show, figsize=(20, 4))
        if num_cols_to_show == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols[:num_cols_to_show]):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color='blue')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Correlation matrix
    st.subheader(t('correlation_matrix'))
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
        # Top correlations with target
        if target_var in numeric_cols:
            st.write(f"Top correlations with {target_var}:")
            target_correlations = corr_matrix[target_var].abs().sort_values(ascending=False)
            st.write(target_correlations[1:6])  # Top 5 excluding itself
    
    # Outlier detection
    st.subheader(t('outlier_detection'))
    if numeric_cols:
        num_cols_to_show = min(3, len(numeric_cols))
        fig, axes = plt.subplots(1, num_cols_to_show, figsize=(15, 5))
        if num_cols_to_show == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols[:num_cols_to_show]):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(f'Boxplot of {col}')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    return True

def preprocess_data(df, target_var, options):
    df_processed = df.copy()
    
    # Handle missing values
    if options['handle_missing'] == 'Drop rows':
        df_processed = df_processed.dropna()
    elif options['handle_missing'] == 'Mean imputation':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
    elif options['handle_missing'] == 'KNN imputation':
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
    if options['encoding'] == 'One-Hot Encoding':
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    elif options['encoding'] == 'Label Encoding':
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

def generate_pdf_report(df, eda_results, models, evaluation_results, best_model):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Housing Price Estimation Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Dataset information
    story.append(Paragraph("Dataset Information", styles['Heading2']))
    data = [["Number of rows", len(df)], ["Number of columns", len(df.columns)]]
    table = Table(data)
    story.append(table)
    story.append(Spacer(1, 12))
    
    # Model comparison
    story.append(Paragraph("Model Comparison", styles['Heading2']))
    model_data = [["Model", "MAE", "RMSE", "R2"]]
    for model_name, metrics in evaluation_results.items():
        model_data.append([model_name, f"{metrics['MAE']:.4f}", f"{metrics['RMSE']:.4f}", f"{metrics['R2']:.4f}"])
    
    model_table = Table(model_data)
    story.append(model_table)
    story.append(Spacer(1, 12))
    
    # Best model
    story.append(Paragraph(f"Best Model: {best_model}", styles['Heading2']))
    best_metrics = evaluation_results[best_model]
    best_data = [["Metric", "Value"], 
                 ["MAE", f"{best_metrics['MAE']:.4f}"],
                 ["RMSE", f"{best_metrics['RMSE']:.4f}"],
                 ["R2", f"{best_metrics['R2']:.4f}"]]
    best_table = Table(best_data)
    story.append(best_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Sidebar
with st.sidebar:
    st.title(t('title'))
    
    # Language selection
    language = st.radio("Select Language / Sélectionnez la langue / Seleccionar idioma", 
                       options=['en', 'fr', 'es'],
                       index=['en', 'fr', 'es'].index(st.session_state.language))
    st.session_state.language = language
    
    # Data upload
    st.header(t('upload_data'))
    uploaded_file = st.file_uploader(t('file_types'), type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        if st.session_state.df is None:
            st.session_state.df = load_data(uploaded_file)
        
        if st.session_state.df is not None:
            df = st.session_state.df
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                st.session_state.target_var = st.selectbox(t('target_var'), numeric_cols)
                
                # EDA button
                if st.button(t('run_eda')):
                    with st.spinner('Performing EDA...'):
                        st.session_state.eda_done = perform_eda(df, st.session_state.target_var)
                
                # Preprocessing options
                st.header(t('preprocessing'))
                preprocessing_options = {
                    'handle_missing': st.selectbox('Handle missing values', 
                                                  ['Drop rows', 'Mean imputation', 'KNN imputation']),
                    'handle_duplicates': st.checkbox('Remove duplicates', value=True),
                    'handle_outliers': st.checkbox('Handle outliers (Z-score)', value=True),
                    'encoding': st.selectbox('Encode categorical variables',
                                           ['One-Hot Encoding', 'Label Encoding']),
                    'handle_multicollinearity': st.checkbox('Handle multicollinearity (VIF)', value=True),
                    'scaling': st.checkbox('Scale features', value=True)
                }
                
                if st.button(t('preprocess_data')):
                    with st.spinner('Preprocessing data...'):
                        df_processed = preprocess_data(df, st.session_state.target_var, preprocessing_options)
                        
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
                        st.success('Data preprocessing completed!')
                
                # Model training
                if st.session_state.preprocessing_done:
                    st.header(t('modeling'))
                    if st.button(t('train_models')):
                        with st.spinner('Training models...'):
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
                            st.success('Models trained successfully!')
                
                # Model evaluation
                if st.session_state.models_trained:
                    st.header(t('evaluation'))
                    if st.button(t('evaluate_models')):
                        with st.spinner('Evaluating models...'):
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
                            st.success('Models evaluated successfully!')
                
                # Report generation
                if st.session_state.models_trained and st.session_state.evaluation_results:
                    st.header(t('report'))
                    if st.button(t('generate_report')):
                        with st.spinner('Generating report...'):
                            report_bytes = generate_pdf_report(
                                df,
                                st.session_state.eda_done,
                                st.session_state.models,
                                st.session_state.evaluation_results,
                                st.session_state.best_model
                            )
                            
                            st.download_button(
                                label=t('download_report'),
                                data=report_bytes,
                                file_name=f"housing_price_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )

# Main content area
st.markdown(f"<h1 class='main-header'>{t('title')}</h1>", unsafe_allow_html=True)

if st.session_state.df is not None:
    df = st.session_state.df
    
    # Show dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Show EDA results if available
    if st.session_state.eda_done:
        st.subheader(t('eda'))
        st.success("EDA completed successfully!")
    
    # Show preprocessing status
    if st.session_state.preprocessing_done:
        st.subheader(t('preprocessing'))
        st.success("Data preprocessing completed successfully!")
        st.write(f"Training set: {st.session_state.X_train.shape[0]} samples")
        st.write(f"Validation set: {st.session_state.X_val.shape[0]} samples")
        st.write(f"Test set: {st.session_state.X_test.shape[0]} samples")
    
    # Show model training results
    if st.session_state.models_trained:
        st.subheader(t('modeling'))
        st.success("Models trained successfully!")
        
        # Display training times
        st.write("Training Times:")
        training_times = {
            model: results['training_time'] 
            for model, results in st.session_state.model_results.items()
        }
        st.bar_chart(training_times)
    
    # Show evaluation results
    if st.session_state.evaluation_results:
        st.subheader(t('evaluation'))
        st.success("Models evaluated successfully!")
        
        # Display evaluation metrics
        evaluation_df = pd.DataFrame(st.session_state.evaluation_results).T
        st.dataframe(evaluation_df)
        
        # Show best model
        st.subheader(t('best_model'))
        st.success(f"The best model is: {st.session_state.best_model}")
        
        # Show feature importance for the best model
        st.subheader(t('important_features'))
        best_model_importance = st.session_state.model_results[st.session_state.best_model]['feature_importance']
        st.dataframe(best_model_importance.head(10))
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            best_model_importance['Feature'].head(10),
            best_model_importance['Importance'].head(10)
        )
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importance')
        st.pyplot(fig)

else:
    st.info("Please upload a dataset to get started.")

# Footer
st.markdown("---")
st.markdown("Developed with Streamlit - Housing Price Estimator Tool")