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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
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
        "download_report": "Descargar Reporte"
    },
    "en": {
        "title": "Housing Price Estimator",
        "upload_data": "Upload Data",
        "file_types": "Accepted file types: CSV, XLSX",
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
        "download_report": "Download Report"
    },
    "fr": {
        "title": "Estimateur de Prix Immobiliers",
        "upload_data": "Télécharger les Données",
        "file_types": "Types de fichiers acceptés: CSV, XLSX",
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
        "download_report": "Télécharger le Rapport"
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
    </style>
    """, unsafe_allow_html=True)

# Función para crear PDF
def create_pdf_report(data, lang, analysis_results):
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
    
    # Aquí se agregarían más secciones del reporte
    
    # Guardar PDF
    pdf_output = io.BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    pdf_output.seek(0)
    
    return pdf_output

# Función principal de la aplicación
def main():
    # Configuración de idioma
    lang = st.sidebar.selectbox("🌐 Idioma / Language / Langue", 
                               options=list(translations.keys()), 
                               format_func=lambda x: {"es": "Español", "en": "English", "fr": "Français"}[x])
    
    # Aplicar estilos CSS
    local_css()
    
    # Título principal
    st.markdown(f'<h1 class="main-header">{translations[lang]["title"]}</h1>', unsafe_allow_html=True)
    
    # Carga de datos
    st.sidebar.markdown(f"### 📁 {translations[lang]['upload_data']}")
    uploaded_file = st.sidebar.file_uploader(
        translations[lang]["file_types"], 
        type=['csv', 'xlsx'],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        # Leer el archivo
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error al leer el archivo: {str(e)}")
            return
        
        # Mostrar datos cargados
        st.sidebar.success(f"✅ Datos cargados: {data.shape[0]} filas × {data.shape[1]} columnas")
        
        # Selector de variable objetivo
        target_var = st.sidebar.selectbox("Seleccionar variable objetivo (precio)", 
                                         options=data.columns)
        
        # Botón para iniciar EDA
        if st.sidebar.button(f"🚀 {translations[lang]['start_analysis']}"):
            # FASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
            st.markdown(f'<h2 class="section-header">Fase 1: {translations[lang]["data_overview"]}</h2>', unsafe_allow_html=True)
            
            # Visión general
            st.markdown(f'<h3 class="subsection-header">{translations[lang]["data_overview"]}</h3>', unsafe_allow_html=True)
            
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
            st.markdown(f'<h3 class="subsection-header">{translations[lang]["data_diagnosis"]}</h3>', unsafe_allow_html=True)
            
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
            
            # Outliers usando IQR
            st.markdown(f"**{translations[lang]['outliers']}:**")
            outliers_df = pd.DataFrame(columns=['Variable', 'Número de Outliers', 'Porcentaje'])
            
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
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
            st.markdown(f'<h3 class="subsection-header">{translations[lang]["numeric_analysis"]}</h3>', unsafe_allow_html=True)
            
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
                st.markdown(f'<h3 class="subsection-header">{translations[lang]["categorical_analysis"]}</h3>', unsafe_allow_html=True)
                
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
            st.markdown(f'<h3 class="subsection-header">{translations[lang]["correlation_analysis"]}</h3>', unsafe_allow_html=True)
            
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
            st.markdown(f'<h2 class="section-header">Fase 2: {translations[lang]["preprocessing"]}</h2>', unsafe_allow_html=True)
            
            # Tratamiento de valores faltantes
            st.markdown(f'<h3 class="subsection-header">{translations[lang]["missing_treatment"]}</h3>', unsafe_allow_html=True)
            
            for col in data.columns:
                if data[col].isnull().sum() > 0:
                    st.markdown(f"**{col}**")
                    
                    if col in numeric_cols:
                        imp_method = st.selectbox(
                            f"Método de imputación para {col}",
                            ["Media", "Mediana", "Eliminar"],
                            key=f"imp_{col}"
                        )
                        
                        if imp_method == "Media":
                            data[col].fillna(data[col].mean(), inplace=True)
                        elif imp_method == "Mediana":
                            data[col].fillna(data[col].median(), inplace=True)
                        else:
                            data.dropna(subset=[col], inplace=True)
                    
                    else:  # Variables categóricas
                        imp_method = st.selectbox(
                            f"Método de imputación para {col}",
                            ["Moda", "Eliminar", "Valor constante"],
                            key=f"imp_{col}"
                        )
                        
                        if imp_method == "Moda":
                            data[col].fillna(data[col].mode()[0], inplace=True)
                        elif imp_method == "Eliminar":
                            data.dropna(subset=[col], inplace=True)
                        else:
                            const_val = st.text_input(f"Valor constante para {col}", "Desconocido")
                            data[col].fillna(const_val, inplace=True)
            
            # Tratamiento de duplicados
            st.markdown(f'<h3 class="subsection-header">{translations[lang]["duplicate_treatment"]}</h3>', unsafe_allow_html=True)
            
            if st.button("Eliminar filas duplicadas"):
                initial_count = len(data)
                data.drop_duplicates(inplace=True)
                final_count = len(data)
                st.success(f"Se eliminaron {initial_count - final_count} filas duplicadas.")
            
            # Tratamiento de outliers
            st.markdown(f'<h3 class="subsection-header">{translations[lang]["outlier_treatment"]}</h3>', unsafe_allow_html=True)
            
            for col in numeric_cols:
                st.markdown(f"**{col}**")
                treat_outliers = st.checkbox(f"Tratar outliers en {col}", key=f"out_{col}")
                
                if treat_outliers:
                    method = st.selectbox(
                        f"Método para {col}",
                        ["IQR", "Z-score"],
                        key=f"method_{col}"
                    )
                    
                    if method == "IQR":
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Opciones: eliminar o ajustar
                        action = st.radio(
                            f"Acción para outliers en {col}",
                            ["Eliminar", "Ajustar a límites"],
                            key=f"action_iqr_{col}"
                        )
                        
                        if action == "Eliminar":
                            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                        else:
                            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
                    
                    else:  # Z-score
                        z_threshold = st.slider(
                            f"Umbral Z-score para {col}",
                            min_value=2.0,
                            max_value=5.0,
                            value=3.0,
                            step=0.5,
                            key=f"z_{col}"
                        )
                        
                        z_scores = np.abs(stats.zscore(data[col].dropna()))
                        
                        # Opciones: eliminar o ajustar
                        action = st.radio(
                            f"Acción para outliers en {col}",
                            ["Eliminar", "Ajustar"],
                            key=f"action_z_{col}"
                        )
                        
                        if action == "Eliminar":
                            data = data[(np.abs(stats.zscore(data[col])) < z_threshold)]
                        else:
                            mean_val = data[col].mean()
                            std_val = data[col].std()
                            lower_bound = mean_val - z_threshold * std_val
                            upper_bound = mean_val + z_threshold * std_val
                            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
            
            # Codificación de variables categóricas
            if categorical_cols:
                st.markdown(f'<h3 class="subsection-header">{translations[lang]["encoding"]}</h3>', unsafe_allow_html=True)
                
                for col in categorical_cols:
                    st.markdown(f"**{col}**")
                    encoding_method = st.selectbox(
                        f"Método de codificación para {col}",
                        ["One-Hot Encoding", "Label Encoding", "Sin cambios"],
                        key=f"enc_{col}"
                    )
                    
                    if encoding_method == "One-Hot Encoding":
                        dummies = pd.get_dummies(data[col], prefix=col)
                        data = pd.concat([data, dummies], axis=1)
                        data.drop(col, axis=1, inplace=True)
                    
                    elif encoding_method == "Label Encoding":
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))
            
            # Escalado de variables numéricas
            st.markdown(f'<h3 class="subsection-header">{translations[lang]["scaling"]}</h3>', unsafe_allow_html=True)
            
            scaling_method = st.selectbox(
                "Método de escalado",
                ["Estandarización (StandardScaler)", "Normalización (MinMaxScaler)", "Sin escalado"]
            )
            
            if scaling_method != "Sin escalado":
                scaler = StandardScaler() if scaling_method == "Estandarización (StandardScaler)" else MinMaxScaler()
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            
            # Análisis de multicolinealidad (VIF)
            st.markdown(f'<h3 class="subsection-header">{translations[lang]["multicollinearity"]}</h3>', unsafe_allow_html=True)
            
            # Calcular VIF
            X = data.drop(target_var, axis=1) if target_var in data.columns else data
            X_numeric = X.select_dtypes(include=[np.number])
            
            if not X_numeric.empty:
                X_const = add_constant(X_numeric)
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X_const.columns
                vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
                
                st.dataframe(vif_data)
                
                # Permitir eliminar variables con alta multicolinealidad
                high_vif_vars = vif_data[vif_data["VIF"] > 5]["Variable"].tolist()
                if high_vif_vars:
                    st.warning("Variables con alta multicolinealidad (VIF > 5):")
                    for var in high_vif_vars:
                        if var != "const" and st.checkbox(f"Eliminar {var}", key=f"vif_{var}"):
                            data.drop(var, axis=1, inplace=True)
            
            # FASE 3: MODELADO Y EVALUACIÓN
            st.markdown(f'<h2 class="section-header">Fase 3: {translations[lang]["modeling"]}</h2>', unsafe_allow_html=True)
            
            # División de datos
            X = data.drop(target_var, axis=1)
            y = data[target_var]
            
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            
            st.success(f"Datos divididos: Entrenamiento (70%): {X_train.shape[0]} muestras, Validación (15%): {X_val.shape[0]} muestras, Prueba (15%): {X_test.shape[0]} muestras")
            
            # Entrenamiento de modelos
            st.markdown(f'<h3 class="subsection-header">{translations[lang]["model_training"]}</h3>', unsafe_allow_html=True)
            
            models = {}
            training_times = {}
            predictions = {}
            
            # Modelo de regresión lineal
            if st.button("Entrenar Modelo de Regresión Lineal"):
                with st.spinner("Entrenando modelo de regresión lineal..."):
                    start_time = time.time()
                    lr_model = LinearRegression()
                    lr_model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    models["Regresión Lineal"] = lr_model
                    training_times["Regresión Lineal"] = training_time
                    predictions["Regresión Lineal"] = lr_model.predict(X_test)
                    
                    st.success(f"Modelo de regresión lineal entrenado en {training_time:.2f} segundos.")
            
            # Aquí se agregarían los otros modelos (redes neuronales, etc.)
            
            # Evaluación de modelos
            if models:
                st.markdown(f'<h3 class="subsection-header">{translations[lang]["model_performance"]}</h3>', unsafe_allow_html=True)
                
                performance_df = pd.DataFrame(columns=["Modelo", "MAE", "RMSE", "R²", "Tiempo de Entrenamiento (s)"])
                
                for model_name, model in models.items():
                    y_pred = predictions[model_name]
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    
                    new_row = pd.DataFrame({
                        "Modelo": [model_name],
                        "MAE": [f"${mae:.2f}"],
                        "RMSE": [f"${rmse:.2f}"],
                        "R²": [f"{r2:.4f}"],
                        "Tiempo de Entrenamiento (s)": [f"{training_times[model_name]:.2f}"]
                    })
                    performance_df = pd.concat([performance_df, new_row], ignore_index=True)
                
                st.dataframe(performance_df)
                
                # Identificar el mejor modelo
                best_model_name = min(models.keys(), key=lambda x: mean_absolute_error(y_test, predictions[x]))
                st.markdown(f'<div class="success-box"><h4>Mejor Modelo: {best_model_name}</h4></div>', unsafe_allow_html=True)
                
                # FASE 4: ANÁLISIS AVANZADO
                st.markdown(f'<h2 class="section-header">Fase 4: {translations[lang]["advanced_analysis"]}</h2>', unsafe_allow_html=True)
                
                # Gráfico de valores reales vs predichos
                st.markdown(f'<h3 class="subsection-header">Valores Reales vs Predichos ({best_model_name})</h3>', unsafe_allow_html=True)
                
                best_pred = predictions[best_model_name]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test, y=best_pred, mode='markers',
                    name='Predicciones', marker=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                    mode='lines', name='Línea Perfecta', line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title=f"Valores Reales vs Predichos - {best_model_name}",
                    xaxis_title="Valores Reales",
                    yaxis_title="Valores Predichos"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Generar reporte PDF
                st.markdown(f'<h3 class="subsection-header">{translations[lang]["generate_report"]}</h3>', unsafe_allow_html=True)
                
                if st.button(f"📊 {translations[lang]['download_report']}"):
                    # Recopilar resultados para el reporte
                    analysis_results = {
                        "data_shape": data.shape,
                        "missing_data": missing_df,
                        "outliers": outliers_df,
                        "normality_test": normality_results,
                        "correlation_matrix": correlation_matrix,
                        "best_model": best_model_name,
                        "performance": performance_df
                    }
                    
                    pdf_report = create_pdf_report(data, lang, analysis_results)
                    
                    st.download_button(
                        label="Descargar Reporte PDF",
                        data=pdf_report,
                        file_name=f"reporte_estimacion_precios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )

if __name__ == "__main__":
    main()
