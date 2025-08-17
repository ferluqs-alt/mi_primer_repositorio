import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Configuración de la página
st.set_page_config(
    page_title="Estimador de Precios de Viviendas",
    page_icon="🏠",
    layout="wide"
)

# Internacionalización (inglés, español, francés)
LANGUAGES = {
    "English": {
        "title": "Housing Price Estimator",
        "description": "This application estimates housing prices in California using machine learning models.",
        "data_loading": "Data Loading and Exploration",
        "dataset_info": "Dataset Information",
        "data_source": "Data Source:",
        "original_dataset": "Original Dataset",
        "initial_analysis": "Initial Analysis",
        "perform_analysis": "Perform Initial Analysis",
        "shape": "Dataset Shape:",
        "null_values": "Null Values:",
        "duplicates": "Duplicate Rows:",
        "descriptive_stats": "Descriptive Statistics",
        "show_stats": "Show Descriptive Statistics",
        "correlation_analysis": "Correlation Analysis",
        "show_correlation": "Show Correlation Analysis",
        "correlation_matrix": "Correlation Matrix",
        "feature_selection": "Selected Features (correlation > 0.1 with target):",
        "preprocessing": "Data Preprocessing",
        "outlier_handling": "Outlier Handling",
        "remove_outliers": "Remove Outliers (IQR method)",
        "standardize_data": "Standardize Data (StandardScaler)",
        "data_split": "Data Split (70% train, 30% test)",
        "visualizations": "Data Visualizations",
        "show_visualizations": "Show Visualizations",
        "scatter_plots": "Scatter Plots",
        "model_training": "Model Training",
        "train_models": "Train Models",
        "linear_regression": "Linear Regression",
        "show_linear": "Show Linear Regression Results",
        "coefficients": "Coefficients:",
        "mlp": "Multilayer Perceptron (MLP)",
        "show_mlp": "Show MLP Results",
        "nn_with_reg": "Neural Network with Regularization",
        "show_nn": "Show Regularized NN Results",
        "training_progress": "Training Progress (MSE)",
        "model_comparison": "Model Comparison",
        "compare_models": "Compare Models",
        "rmse_comparison": "RMSE Comparison",
        "training_time": "Training Time (seconds)",
        "prediction_section": "Price Prediction",
        "input_features": "Input Features",
        "predict_button": "Predict Price",
        "prediction_results": "Prediction Results",
        "linear_pred": "Linear Regression:",
        "mlp_pred": "MLP:",
        "nn_pred": "Regularized NN:",
        "best_model": "Best Performing Model:",
        "summary": "Summary",
        "conclusion": "Based on evaluation metrics, the model with lowest RMSE and highest R² performs best."
    },
    "Español": {
        "title": "Estimador de Precios de Viviendas",
        "description": "Esta aplicación estima precios de viviendas en California usando modelos de aprendizaje automático.",
        "data_loading": "Carga y Exploración de Datos",
        "dataset_info": "Información del Dataset",
        "data_source": "Fuente de Datos:",
        "original_dataset": "Dataset Original",
        "initial_analysis": "Análisis Inicial",
        "perform_analysis": "Realizar Análisis Inicial",
        "shape": "Forma del Dataset:",
        "null_values": "Valores Nulos:",
        "duplicates": "Filas Duplicadas:",
        "descriptive_stats": "Estadísticas Descriptivas",
        "show_stats": "Mostrar Estadísticas Descriptivas",
        "correlation_analysis": "Análisis de Correlación",
        "show_correlation": "Mostrar Análisis de Correlación",
        "correlation_matrix": "Matriz de Correlación",
        "feature_selection": "Características Seleccionadas (correlación > 0.1 con objetivo):",
        "preprocessing": "Preprocesamiento de Datos",
        "outlier_handling": "Manejo de Outliers",
        "remove_outliers": "Eliminar Outliers (método IQR)",
        "standardize_data": "Estandarizar Datos (StandardScaler)",
        "data_split": "División de Datos (70% entrenamiento, 30% prueba)",
        "visualizations": "Visualizaciones de Datos",
        "show_visualizations": "Mostrar Visualizaciones",
        "scatter_plots": "Gráficos de Dispersión",
        "model_training": "Entrenamiento de Modelos",
        "train_models": "Entrenar Modelos",
        "linear_regression": "Regresión Lineal",
        "show_linear": "Mostrar Resultados Regresión Lineal",
        "coefficients": "Coeficientes:",
        "mlp": "Perceptrón Multicapa (MLP)",
        "show_mlp": "Mostrar Resultados MLP",
        "nn_with_reg": "Red Neuronal con Regularización",
        "show_nn": "Mostrar Resultados RN Regularizada",
        "training_progress": "Progreso del Entrenamiento (MSE)",
        "model_comparison": "Comparación de Modelos",
        "compare_models": "Comparar Modelos",
        "rmse_comparison": "Comparación de RMSE",
        "training_time": "Tiempo de Entrenamiento (segundos)",
        "prediction_section": "Predicción de Precios",
        "input_features": "Características de Entrada",
        "predict_button": "Predecir Precio",
        "prediction_results": "Resultados de Predicción",
        "linear_pred": "Regresión Lineal:",
        "mlp_pred": "MLP:",
        "nn_pred": "RN Regularizada:",
        "best_model": "Mejor Modelo:",
        "summary": "Resumen",
        "conclusion": "Basado en las métricas de evaluación, el modelo con menor RMSE y mayor R² es el mejor."
    },
    "Français": {
        "title": "Estimateur de Prix Immobiliers",
        "description": "Cette application estime les prix des logements en Californie à l'aide de modèles d'apprentissage automatique.",
        "data_loading": "Chargement et Exploration des Données",
        "dataset_info": "Informations sur le Dataset",
        "data_source": "Source des Données:",
        "original_dataset": "Dataset Original",
        "initial_analysis": "Analyse Initiale",
        "perform_analysis": "Effectuer l'Analyse Initiale",
        "shape": "Forme du Dataset:",
        "null_values": "Valeurs Nulles:",
        "duplicates": "Lignes Dupliquées:",
        "descriptive_stats": "Statistiques Descriptives",
        "show_stats": "Afficher les Statistiques Descriptives",
        "correlation_analysis": "Analyse de Corrélation",
        "show_correlation": "Afficher l'Analyse de Corrélation",
        "correlation_matrix": "Matrice de Corrélation",
        "feature_selection": "Caractéristiques Sélectionnées (corrélation > 0.1 avec la cible):",
        "preprocessing": "Prétraitement des Données",
        "outlier_handling": "Gestion des Valeurs Aberrantes",
        "remove_outliers": "Supprimer les Valeurs Aberrantes (méthode IQR)",
        "standardize_data": "Standardiser les Données (StandardScaler)",
        "data_split": "Division des Données (70% entraînement, 30% test)",
        "visualizations": "Visualisations des Données",
        "show_visualizations": "Afficher les Visualisations",
        "scatter_plots": "Graphiques de Dispersion",
        "model_training": "Entraînement des Modèles",
        "train_models": "Entraîner les Modèles",
        "linear_regression": "Régression Linéaire",
        "show_linear": "Afficher les Résultats de Régression Linéaire",
        "coefficients": "Coefficients:",
        "mlp": "Perceptron Multicouche (MLP)",
        "show_mlp": "Afficher les Résultats MLP",
        "nn_with_reg": "Réseau Neuronal avec Régularisation",
        "show_nn": "Afficher les Résultats RN Régularisée",
        "training_progress": "Progrès de l'Entraînement (MSE)",
        "model_comparison": "Comparaison des Modèles",
        "compare_models": "Comparer les Modèles",
        "rmse_comparison": "Comparaison des RMSE",
        "training_time": "Temps d'Entraînement (secondes)",
        "prediction_section": "Prédiction de Prix",
        "input_features": "Caractéristiques d'Entrée",
        "predict_button": "Prédire le Prix",
        "prediction_results": "Résultats de Prédiction",
        "linear_pred": "Régression Linéaire:",
        "mlp_pred": "MLP:",
        "nn_pred": "RN Régularisée:",
        "best_model": "Meilleur Modèle:",
        "summary": "Résumé",
        "conclusion": "Sur la base des métriques d'évaluation, le modèle avec le RMSE le plus bas et le score R² le plus élevé est le meilleur."
    }
}

# Sidebar para configuración
with st.sidebar:
    st.title("Configuración")
    lang = st.selectbox("Idioma/Language/Langue", list(LANGUAGES.keys()))
    text = LANGUAGES[lang]
    
    st.markdown(f"**{text['data_source']}**")
    st.markdown(f"[{text['original_dataset']}](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)")

# Título principal
st.title(text["title"])
st.markdown(text["description"])

# Carga de datos
@st.cache_data
def load_data():
    california = fetch_california_housing(as_frame=True)
    data = california.frame
    features = california.feature_names
    target = california.target_names[0]
    return data, features, target

data, features, target = load_data()

# Variables de estado para controlar el flujo
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []

# Sección de análisis exploratorio
st.header(text["data_loading"])

# Análisis inicial
with st.expander(text["initial_analysis"]):
    if st.button(text["perform_analysis"]):
        col1, col2, col3 = st.columns(3)
        col1.metric(text["shape"], f"{data.shape[0]} filas, {data.shape[1]} columnas")
        col2.metric(text["null_values"], data.isnull().sum().sum())
        col3.metric(text["duplicates"], data.duplicated().sum())

    if st.button(text["show_stats"]):
        st.subheader(text["descriptive_stats"])
        st.dataframe(data.describe().T.style.format("{:.2f}"))

# Análisis de correlación
with st.expander(text["correlation_analysis"]):
    if st.button(text["show_correlation"]):
        corr_matrix = data.corr()
        target_corr = corr_matrix[target].abs().sort_values(ascending=False)
        
        # Selección de características relevantes
        st.session_state.selected_features = target_corr[target_corr > 0.1].index.tolist()
        st.session_state.selected_features.remove(target)
        
        st.write(text["feature_selection"], st.session_state.selected_features)
        
        # Mapa de calor de correlación
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Preprocesamiento de datos
with st.expander(text["preprocessing"]):
    if len(st.session_state.selected_features) > 0:
        col1, col2 = st.columns(2)
        with col1:
            remove_outliers = st.checkbox(text["remove_outliers"])
        with col2:
            standardize = st.checkbox(text["standardize_data"])
        
        if st.button(text["data_split"]):
            # Copia de los datos para procesamiento
            processed_data = data.copy()
            
            # Eliminar outliers si está seleccionado
            if remove_outliers:
                def remove_outliers_iqr(df, columns):
                    for col in columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
                    return df
                
                processed_data = remove_outliers_iqr(processed_data, st.session_state.selected_features + [target])
                st.success("Outliers eliminados usando método IQR (1.5*IQR)")
            
            # Dividir datos
            X = processed_data[st.session_state.selected_features]
            y = processed_data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Estandarizar si está seleccionado
            if standardize:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                st.success("Datos estandarizados usando StandardScaler")
            else:
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values
                scaler = None
            
            # Guardar en session state
            st.session_state.X_train = X_train_scaled
            st.session_state.X_test = X_test_scaled
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
            st.session_state.data_processed = True
            
            st.success("Datos divididos en 70% entrenamiento y 30% prueba")
    else:
        st.warning("Por favor realice primero el análisis de correlación para seleccionar características")

# Visualizaciones
with st.expander(text["visualizations"]):
    if st.button(text["show_visualizations"]) and st.session_state.data_processed:
        st.subheader(text["scatter_plots"])
        
        # Recuperar datos originales (sin escalar)
        if st.session_state.scaler is not None:
            X_train_original = st.session_state.scaler.inverse_transform(st.session_state.X_train)
        else:
            X_train_original = st.session_state.X_train
        
        # Crear DataFrame para visualización
        viz_df = pd.DataFrame(X_train_original, columns=st.session_state.selected_features)
        viz_df[target] = st.session_state.y_train.values
        
        for feature in st.session_state.selected_features:
            fig = px.scatter(viz_df, x=feature, y=target, trendline="ols",
                             title=f"{target} vs {feature}")
            st.plotly_chart(fig, use_container_width=True)

# Sección de entrenamiento de modelos
st.header(text["model_training"])

if st.session_state.data_processed:
    if st.button(text["train_models"]):
        with st.spinner("Entrenando modelos..."):
            # Inicialización de modelos
            #models = {
             #   text["linear_regression"]: LinearRegression(),
              #  text["mlp"]: MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
               # text["nn_with_reg"]: MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42, alpha=0.01, dropout=0.2)
            #}
            models = {
                text["linear_regression"]: LinearRegression(),
                text["mlp"]: MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
                text["nn_with_reg"]: MLPRegressor(
                hidden_layer_sizes=(50,), 
                max_iter=500, 
                random_state=42, 
                alpha=0.01,  # Regularización L2 (weight decay)
                early_stopping=True  # Parada temprana para evitar sobreajuste
                )
            }

            # Entrenamiento y evaluación
            results = []
            training_times = {}
            loss_curves = {}

            for name, model in models.items():
                start_time = time.time()
                
                model.fit(st.session_state.X_train, st.session_state.y_train)
                y_pred = model.predict(st.session_state.X_test)
                
                r2 = r2_score(st.session_state.y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(st.session_state.y_test, y_pred))
                
                training_time = time.time() - start_time
                training_times[name] = training_time
                
                # Guardar resultados
                results.append({
                    'Model': name,
                    'R²': r2,
                    'RMSE': rmse,
                    'Training Time': training_time
                })
                
                # Guardar coeficientes para regresión lineal
                if name == text["linear_regression"]:
                    coefficients = pd.DataFrame({
                        'Feature': st.session_state.selected_features,
                        'Coefficient': model.coef_
                    }).sort_values('Coefficient', ascending=False)
                
                # Guardar curva de pérdida para redes neuronales
                if hasattr(model, 'loss_curve_'):
                    loss_curves[name] = model.loss_curve_
            
            # Guardar en session state
            st.session_state.models = models
            st.session_state.results = pd.DataFrame(results).set_index('Model')
            st.session_state.coefficients = coefficients
            st.session_state.loss_curves = loss_curves
            st.session_state.models_trained = True
            
        st.success("Modelos entrenados exitosamente!")
else:
    st.warning("Por favor procese los datos primero en la sección de preprocesamiento")

# Resultados de modelos
if st.session_state.models_trained:
    # Regresión Lineal
    with st.expander(text["linear_regression"]):
        if st.button(text["show_linear"]):
            st.write(text["coefficients"])
            st.dataframe(st.session_state.coefficients.style.format({"Coefficient": "{:.4f}"}))
            
            st.write("Métricas de rendimiento:")
            st.dataframe(st.session_state.results.loc[[text["linear_regression"]]], 
                         columns=['R²', 'RMSE', 'Training Time'])
    
    # MLP
    with st.expander(text["mlp"]):
        if st.button(text["show_mlp"]):
            st.write("Métricas de rendimiento:")
            st.dataframe(st.session_state.results.loc[[text["mlp"]]], 
                     columns=['R²', 'RMSE', 'Training Time'])
            
            if text["mlp"] in st.session_state.loss_curves:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(st.session_state.loss_curves[text["mlp"]]))),
                    y=st.session_state.loss_curves[text["mlp"]],
                    mode='lines',
                    name=text["mlp"]
                ))
                fig.update_layout(
                    title=text["training_progress"],
                    xaxis_title='Epoch',
                    yaxis_title='MSE Loss'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Red Neuronal Regularizada
    with st.expander(text["nn_with_reg"]):
        if st.button(text["show_nn"]):
            st.write("Métricas de rendimiento:")
            st.dataframe(st.session_state.results.loc[[text["nn_with_reg"]]], 
                     columns=['R²', 'RMSE', 'Training Time'])
            
            if text["nn_with_reg"] in st.session_state.loss_curves:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(st.session_state.loss_curves[text["nn_with_reg"]]))),
                    y=st.session_state.loss_curves[text["nn_with_reg"]],
                    mode='lines',
                    name=text["nn_with_reg"]
                ))
                fig.update_layout(
                    title=text["training_progress"],
                    xaxis_title='Epoch',
                    yaxis_title='MSE Loss'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Comparación de modelos
    with st.expander(text["model_comparison"]):
        if st.button(text["compare_models"]):
            st.subheader(text["rmse_comparison"])
            fig = px.bar(st.session_state.results.reset_index(), 
                         x='Model', y='RMSE', color='Model')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader(text["training_time"])
            st.dataframe(st.session_state.results['Training Time'].to_frame().style.format("{:.2f}"))
            
            # Determinar el mejor modelo
            best_model = st.session_state.results.loc[st.session_state.results['RMSE'].idxmin()].name
            st.subheader(f"{text['best_model']} {best_model}")
            
            st.write(text["conclusion"])

# Sección de predicción
st.header(text["prediction_section"])

if st.session_state.models_trained and len(st.session_state.selected_features) > 0:
    st.subheader(text["input_features"])
    
    # Crear sliders para cada característica
    input_values = {}
    cols = st.columns(2)
    for i, feature in enumerate(st.session_state.selected_features):
        with cols[i % 2]:
            # Obtener valores mínimos y máximos de los datos de entrenamiento originales
            if st.session_state.scaler is not None:
                # Si los datos fueron escalados, necesitamos invertir la transformación para mostrar valores originales
                X_train_original = st.session_state.scaler.inverse_transform(st.session_state.X_train)
                min_val = float(X_train_original[:, i].min())
                max_val = float(X_train_original[:, i].max())
                default_val = float(X_train_original[:, i].mean())
            else:
                min_val = float(st.session_state.X_train[:, i].min())
                max_val = float(st.session_state.X_train[:, i].max())
                default_val = float(st.session_state.X_train[:, i].mean())
            
            input_values[feature] = st.slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=(max_val - min_val) / 100,
                key=f"slider_{feature}"
            )
    
    # Botón de predicción
    if st.button(text["predict_button"]):
        # Preparar datos de entrada
        input_data = np.array([[input_values[feat] for feat in st.session_state.selected_features]]).reshape(1, -1)
        
        # Escalar si es necesario
        if st.session_state.scaler is not None:
            input_scaled = st.session_state.scaler.transform(input_data)
        else:
            input_scaled = input_data
        
        # Realizar predicciones
        predictions = {}
        for name, model in st.session_state.models.items():
            predictions[name] = model.predict(input_scaled)[0]
        
        # Mostrar resultados
        st.subheader(text["prediction_results"])
        
        cols = st.columns(3)
        cols[0].metric(text["linear_pred"], f"${predictions[text['linear_regression']]*100000:,.2f}")
        cols[1].metric(text["mlp_pred"], f"${predictions[text['mlp']]*100000:,.2f}")
        cols[2].metric(text["nn_pred"], f"${predictions[text['nn_with_reg']]*100000:,.2f}")
else:
    st.warning("Por favor entrene los modelos primero para habilitar las predicciones")

# Resumen final
if st.session_state.models_trained:
    st.header(text["summary"])
    st.dataframe(st.session_state.results.style.format({
        'R²': '{:.3f}',
        'RMSE': '{:.3f}',
        'Training Time': '{:.2f}'
    }))
    
    best_model = st.session_state.results.loc[st.session_state.results['RMSE'].idxmin()].name
    st.subheader(f"{text['best_model']} {best_model}")
    st.write(text["conclusion"])




