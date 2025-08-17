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

# Configuración inicial de la página
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide"
)

# Internacionalización (inglés, español, francés)
LANGUAGES = {
    "English": {
        "title": "California Housing Price Predictor",
        "description": "This app estimates housing prices in California using machine learning models.",
        "data_loading": "Data Loading and Exploration",
        "dataset_info": "Dataset Information",
        "data_source": "Data Source:",
        "original_dataset": "Original Dataset",
        "shape": "Dataset Shape:",
        "null_values": "Null Values:",
        "duplicates": "Duplicate Rows:",
        "descriptive_stats": "Descriptive Statistics",
        "correlation_analysis": "Correlation Analysis",
        "correlation_matrix": "Correlation Matrix",
        "feature_selection": "Selected Features (correlation > 0.1 with target):",
        "outlier_handling": "Outlier Handling",
        "outliers_removed": "Outliers removed using IQR method (1.5*IQR).",
        "data_preprocessing": "Data Preprocessing",
        "data_split": "Data split into 70% training and 30% test sets.",
        "visualizations": "Data Visualizations",
        "scatter_plots": "Scatter Plots of Selected Features vs House Value",
        "model_training": "Model Training and Evaluation",
        "linear_regression": "Linear Regression",
        "coefficients": "Coefficients:",
        "mlp": "Multilayer Perceptron (MLP)",
        "nn_with_reg": "Neural Network with Regularization",
        "training_progress": "Training Progress (MSE)",
        "model_comparison": "Model Comparison",
        "rmse_comparison": "RMSE Comparison",
        "training_time": "Training Time (seconds)",
        "prediction_section": "Make Predictions",
        "input_features": "Input Features",
        "predict_button": "Predict House Value",
        "prediction_results": "Prediction Results",
        "linear_pred": "Linear Regression Prediction:",
        "mlp_pred": "MLP Prediction:",
        "nn_pred": "Regularized NN Prediction:",
        "best_model": "Best Performing Model:",
        "summary": "Summary",
        "conclusion": "Based on the evaluation metrics, the model with the lowest RMSE and highest R² score performs best.",
        "analyze_button": "Analyze Data",
        "train_button": "Train Models"
    },
    "Español": {
        "title": "Predictor de Precios de Viviendas en California",
        "description": "Esta aplicación estima precios de viviendas en California usando modelos de aprendizaje automático.",
        "data_loading": "Carga y Exploración de Datos",
        "dataset_info": "Información del Dataset",
        "data_source": "Fuente de Datos:",
        "original_dataset": "Dataset Original",
        "shape": "Forma del Dataset:",
        "null_values": "Valores Nulos:",
        "duplicates": "Filas Duplicadas:",
        "descriptive_stats": "Estadísticas Descriptivas",
        "correlation_analysis": "Análisis de Correlación",
        "correlation_matrix": "Matriz de Correlación",
        "feature_selection": "Características Seleccionadas (correlación > 0.1 con objetivo):",
        "outlier_handling": "Manejo de Outliers",
        "outliers_removed": "Outliers eliminados usando método IQR (1.5*IQR).",
        "data_preprocessing": "Preprocesamiento de Datos",
        "data_split": "Datos divididos en 70% entrenamiento y 30% prueba.",
        "visualizations": "Visualizaciones de Datos",
        "scatter_plots": "Gráficos de Dispersión de Características vs Valor de Vivienda",
        "model_training": "Entrenamiento y Evaluación de Modelos",
        "linear_regression": "Regresión Lineal",
        "coefficients": "Coeficientes:",
        "mlp": "Perceptrón Multicapa (MLP)",
        "nn_with_reg": "Red Neuronal con Regularización",
        "training_progress": "Progreso del Entrenamiento (MSE)",
        "model_comparison": "Comparación de Modelos",
        "rmse_comparison": "Comparación de RMSE",
        "training_time": "Tiempo de Entrenamiento (segundos)",
        "prediction_section": "Realizar Predicciones",
        "input_features": "Características de Entrada",
        "predict_button": "Predecir Valor de Vivienda",
        "prediction_results": "Resultados de Predicción",
        "linear_pred": "Predicción Regresión Lineal:",
        "mlp_pred": "Predicción MLP:",
        "nn_pred": "Predicción RN Regularizada:",
        "best_model": "Mejor Modelo:",
        "summary": "Resumen",
        "conclusion": "Basado en las métricas de evaluación, el modelo con el RMSE más bajo y el puntaje R² más alto es el mejor.",
        "analyze_button": "Analizar Datos",
        "train_button": "Entrenar Modelos"
    },
    "Français": {
        "title": "Prédicteur de Prix Immobiliers en Californie",
        "description": "Cette application estime les prix des logements en Californie à l'aide de modèles d'apprentissage automatique.",
        "data_loading": "Chargement et Exploration des Données",
        "dataset_info": "Informations sur le Dataset",
        "data_source": "Source des Données:",
        "original_dataset": "Dataset Original",
        "shape": "Forme du Dataset:",
        "null_values": "Valeurs Nulles:",
        "duplicates": "Lignes Dupliquées:",
        "descriptive_stats": "Statistiques Descriptives",
        "correlation_analysis": "Analyse de Corrélation",
        "correlation_matrix": "Matrice de Corrélation",
        "feature_selection": "Caractéristiques Sélectionnées (corrélation > 0.1 avec la cible):",
        "outlier_handling": "Gestion des Valeurs Aberrantes",
        "outliers_removed": "Valeurs aberrantes supprimées en utilisant la méthode IQR (1.5*IQR).",
        "data_preprocessing": "Prétraitement des Données",
        "data_split": "Données divisées en 70% d'entraînement et 30% de test.",
        "visualizations": "Visualisations des Données",
        "scatter_plots": "Graphiques de Dispersion des Caractéristiques vs Valeur Immobilière",
        "model_training": "Entraînement et Évaluation des Modèles",
        "linear_regression": "Régression Linéaire",
        "coefficients": "Coefficients:",
        "mlp": "Perceptron Multicouche (MLP)",
        "nn_with_reg": "Réseau Neuronal avec Régularisation",
        "training_progress": "Progrès de l'Entraînement (MSE)",
        "model_comparison": "Comparaison des Modèles",
        "rmse_comparison": "Comparaison des RMSE",
        "training_time": "Temps d'Entraînement (secondes)",
        "prediction_section": "Faire des Prédictions",
        "input_features": "Caractéristiques d'Entrée",
        "predict_button": "Prédire la Valeur Immobilière",
        "prediction_results": "Résultats de Prédiction",
        "linear_pred": "Prédiction Régression Linéaire:",
        "mlp_pred": "Prédiction MLP:",
        "nn_pred": "Prédiction RN Régularisée:",
        "best_model": "Meilleur Modèle:",
        "summary": "Résumé",
        "conclusion": "Sur la base des métriques d'évaluation, le modèle avec le RMSE le plus bas et le score R² le plus élevé est le meilleur.",
        "analyze_button": "Analyser les Données",
        "train_button": "Entraîner les Modèles"
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

# Sección de análisis exploratorio
st.header(text["data_loading"])

if st.button(text["analyze_button"]):
    st.subheader(text["dataset_info"])
    
    # Información básica del dataset
    col1, col2, col3 = st.columns(3)
    col1.metric(text["shape"], f"{data.shape[0]} filas, {data.shape[1]} columnas")
    col2.metric(text["null_values"], data.isnull().sum().sum())
    col3.metric(text["duplicates"], data.duplicated().sum())

    # Estadísticas descriptivas
    st.subheader(text["descriptive_stats"])
    st.dataframe(data.describe().T.style.format("{:.2f}"))

    # Análisis de correlación
    st.subheader(text["correlation_analysis"])
    corr_matrix = data.corr()
    target_corr = corr_matrix[target].abs().sort_values(ascending=False)
    
    # Selección de características relevantes
    selected_features = target_corr[target_corr > 0.1].index.tolist()
    selected_features.remove(target)
    st.write(text["feature_selection"], selected_features)
    
    # Mapa de calor de correlación
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Manejo de outliers
    st.subheader(text["outlier_handling"])
    st.write(text["outliers_removed"])
    
    def remove_outliers(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
        return df
    
    data_clean = remove_outliers(data, selected_features + [target])

    # Preprocesamiento de datos
    st.subheader(text["data_preprocessing"])
    st.write(text["data_split"])
    
    # División de datos
    X = data_clean[selected_features]
    y = data_clean[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Estandarización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Visualizaciones
    st.subheader(text["visualizations"])
    st.write(text["scatter_plots"])
    
    for feature in selected_features:
        fig = px.scatter(data_clean, x=feature, y=target, trendline="ols",
                         title=f"{target} vs {feature}")
        st.plotly_chart(fig, use_container_width=True)

    # Sección de modelado
    st.header(text["model_training"])
    
    if st.button(text["train_button"]):
        # Inicialización de modelos
        models = {
            text["linear_regression"]: LinearRegression(),
            text["mlp"]: MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
            text["nn_with_reg"]: MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42, alpha=0.01, dropout=0.2)
        }

        # Entrenamiento y evaluación
        results = []
        training_times = {}
        loss_curves = {}

        for name, model in models.items():
            start_time = time.time()
            
            if name == text["linear_regression"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Coeficientes para regresión lineal
                coefficients = pd.DataFrame({
                    'Feature': selected_features,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', ascending=False)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Curva de pérdida para redes neuronales
                if hasattr(model, 'loss_curve_'):
                    loss_curves[name] = model.loss_curve_
            
            training_time = time.time() - start_time
            training_times[name] = training_time
            
            results.append({
                'Model': name,
                'R²': r2,
                'RMSE': rmse,
                'Training Time': training_time
            })

        # Resultados de regresión lineal
        st.subheader(text["linear_regression"])
        st.write(text["coefficients"])
        st.dataframe(coefficients.style.format({"Coefficient": "{:.4f}"}))

        # Progreso del entrenamiento
        if loss_curves:
            st.subheader(text["training_progress"])
            fig = go.Figure()
            for name, loss in loss_curves.items():
                fig.add_trace(go.Scatter(
                    x=list(range(len(loss))),
                    y=loss,
                    name=name,
                    mode='lines'
                ))
            fig.update_layout(
                xaxis_title='Epoch',
                yaxis_title='MSE Loss',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Comparación de modelos
        st.subheader(text["model_comparison"])
        results_df = pd.DataFrame(results).set_index('Model')
        
        # Comparación de RMSE
        st.write(text["rmse_comparison"])
        fig = px.bar(results_df, y='RMSE', color=results_df.index)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tiempos de entrenamiento
        st.write(text["training_time"])
        st.dataframe(results_df['Training Time'].to_frame().style.format("{:.2f}"))

        # Sección de predicción
        st.header(text["prediction_section"])
        st.subheader(text["input_features"])
        
        # Sliders para entrada de características
        input_values = {}
        cols = st.columns(2)
        for i, feature in enumerate(selected_features):
            with cols[i % 2]:
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                default_val = float(X[feature].median())
                input_values[feature] = st.slider(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100
                )

        # Predicción
        if st.button(text["predict_button"]):
            st.subheader(text["prediction_results"])
            input_df = pd.DataFrame([input_values])
            input_scaled = scaler.transform(input_df)
            
            # Predicciones
            lr_pred = models[text["linear_regression"]].predict(input_scaled)[0]
            mlp_pred = models[text["mlp"]].predict(input_scaled)[0]
            nn_pred = models[text["nn_with_reg"]].predict(input_scaled)[0]
            
            col1, col2, col3 = st.columns(3)
            col1.metric(text["linear_pred"], f"${lr_pred*100000:,.2f}")
            col2.metric(text["mlp_pred"], f"${mlp_pred*100000:,.2f}")
            col3.metric(text["nn_pred"], f"${nn_pred*100000:,.2f}")

        # Resumen final
        st.header(text["summary"])
        best_model = results_df.loc[results_df['RMSE'].idxmin()].name
        st.subheader(f"{text['best_model']} {best_model}")
        
        st.dataframe(results_df.style.format({
            'R²': '{:.3f}',
            'RMSE': '{:.3f}',
            'Training Time': '{:.2f}'
        }))
        
        st.write(text["conclusion"])
