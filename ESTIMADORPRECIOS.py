import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Textos Multilenguaje ---
texts = {
    'es': {
        'title': 'Estimador de Precios de Casas en California',
        'intro': 'Bienvenido a la aplicación de estimación de precios de casas.',
        'dataset_info': 'Trabajamos con el California Housing Dataset.',
        'eda_title': 'Análisis Exploratorio de Datos',
        'preprocessing_title': 'Preprocesamiento de Datos',
        'nulls': 'Datos Nulos',
        'duplicates': 'Datos Duplicados',
        'outliers': 'Outliers (valores atípicos)',
        'descriptive_stats': 'Estadísticas Descriptivas',
        'corr_matrix': 'Matriz de Correlación',
        'vif_title': 'Multicolinealidad (VIF)',
        'vif_desc': 'Un VIF > 5-10 indica una multicolinealidad significativa.',
        'histograms': 'Histogramas de Variables',
        'boxplots': 'Diagramas de Caja de Variables',
        'scatter_plots': 'Diagramas de Dispersión',
        'prediction_title': 'Predicción de Precio',
        'input_features': 'Ingrese las características de la propiedad:',
        'predict_button': 'Estimar Precio',
        'model_results': 'Resultados de los Modelos',
        'mse': 'Error Cuadrático Medio (MSE)',
        'r2': 'R² (Coeficiente de Determinación)',
        'reg_pred': 'Precio estimado por Regresión Múltiple',
        'nn_pred': 'Precio estimado por Red Neuronal',
        'model_metrics': 'Métricas de los Modelos',
        'disclaimer': 'Nota: Los precios están en unidades de $100,000.',
        'language': 'Idioma',
        'null_info': 'No se encontraron datos nulos en el dataset.',
        'dup_info': 'No se encontraron filas duplicadas en el dataset.',
        'hist_desc': 'Distribución de las características del dataset.',
        'box_desc': 'Análisis de la dispersión y los outliers.',
        'scatter_desc': 'Relación de cada característica con el precio de la casa.'
    },
    'en': {
        'title': 'California House Price Estimator',
        'intro': 'Welcome to the house price estimation application.',
        'dataset_info': 'We are working with the California Housing Dataset.',
        'eda_title': 'Exploratory Data Analysis',
        'preprocessing_title': 'Data Preprocessing',
        'nulls': 'Null Data',
        'duplicates': 'Duplicate Data',
        'outliers': 'Outliers',
        'descriptive_stats': 'Descriptive Statistics',
        'corr_matrix': 'Correlation Matrix',
        'vif_title': 'Multicollinearity (VIF)',
        'vif_desc': 'A VIF > 5-10 indicates significant multicollinearity.',
        'histograms': 'Histograms of Variables',
        'boxplots': 'Boxplots of Variables',
        'scatter_plots': 'Scatter Plots',
        'prediction_title': 'Price Prediction',
        'input_features': 'Enter the property features:',
        'predict_button': 'Estimate Price',
        'model_results': 'Model Results',
        'mse': 'Mean Squared Error (MSE)',
        'r2': 'R² (Coefficient of Determination)',
        'reg_pred': 'Estimated Price by Multiple Regression',
        'nn_pred': 'Estimated Price by Neural Network',
        'model_metrics': 'Model Metrics',
        'disclaimer': 'Note: Prices are in units of $100,000.',
        'language': 'Language',
        'null_info': 'No null data found in the dataset.',
        'dup_info': 'No duplicate rows found in the dataset.',
        'hist_desc': 'Distribution of the dataset features.',
        'box_desc': 'Analysis of dispersion and outliers.',
        'scatter_desc': 'Relationship of each feature with the house price.'
    },
    'fr': {
        'title': 'Estimateur de Prix de Maisons en Californie',
        'intro': 'Bienvenue dans l\'application d\'estimation de prix de maisons.',
        'dataset_info': 'Nous travaillons avec le California Housing Dataset.',
        'eda_title': 'Analyse Exploratoire des Données',
        'preprocessing_title': 'Prétraitement des Données',
        'nulls': 'Données Nules',
        'duplicates': 'Données Dupliquées',
        'outliers': 'Valeurs Aberrantes',
        'descriptive_stats': 'Statistiques Descriptives',
        'corr_matrix': 'Matrice de Corrélation',
        'vif_title': 'Multicolinéarité (VIF)',
        'vif_desc': 'Un VIF > 5-10 indique une multicolinéarité significative.',
        'histograms': 'Histogrammes des Variables',
        'boxplots': 'Diagrammes de Boîte des Variables',
        'scatter_plots': 'Diagrammes de Dispersion',
        'prediction_title': 'Prédiction de Prix',
        'input_features': 'Entrez les caractéristiques de la propriété:',
        'predict_button': 'Estimer le Prix',
        'model_results': 'Résultats des Modèles',
        'mse': 'Erreur Quadratique Moyenne (MSE)',
        'r2': 'R² (Coefficient de Détermination)',
        'reg_pred': 'Prix estimé par Régression Multiple',
        'nn_pred': 'Prix estimé par Réseau de Neurones',
        'model_metrics': 'Métriques des Modèles',
        'disclaimer': 'Note: Les prix sont en unités de $100,000.',
        'language': 'Langue',
        'null_info': 'Aucune donnée nulle n\'a été trouvée dans le dataset.',
        'dup_info': 'Aucune ligne dupliquée n\'a été trouvée dans le dataset.',
        'hist_desc': 'Distribution des caractéristiques du dataset.',
        'box_desc': 'Analyse de la dispersion et des valeurs aberrantes.',
        'scatter_desc': 'Relation de chaque caractéristique avec le prix de la maison.'
    }
}

# --- Carga y Preprocesamiento de Datos ---
@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df['MedHouseVal'] = housing.target
    return df

df_original = load_data()
df = df_original.copy()

# --- Streamlit UI ---
st.set_page_config(layout="wide")

st.sidebar.title("Configuration")
language = st.sidebar.selectbox(texts['es']['language'], ['Español', 'English', 'Français'])
lang_code = 'es' if language == 'Español' else 'en' if language == 'English' else 'fr'

st.title(texts[lang_code]['title'])
st.write(texts[lang_code]['intro'])
st.write(texts[lang_code]['dataset_info'])

st.header(texts[lang_code]['preprocessing_title'])
with st.expander(texts[lang_code]['nulls']):
    st.write(df.isnull().sum())
    if df.isnull().sum().sum() == 0:
        st.success(texts[lang_code]['null_info'])

with st.expander(texts[lang_code]['duplicates']):
    num_duplicates = df.duplicated().sum()
    st.write(f"Número de filas duplicadas: {num_duplicates}")
    if num_duplicates == 0:
        st.success(texts[lang_code]['dup_info'])

with st.expander(texts[lang_code]['outliers']):
    st.write("Detección y tratamiento de outliers usando el método IQR.")
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    st.info("Outliers tratados mediante la eliminación de los valores fuera del rango IQR.")
    st.write(f"Tamaño del dataset después de la eliminación de outliers: {df.shape[0]} filas.")

# --- Análisis Exploratorio de Datos (EDA) ---
st.header(texts[lang_code]['eda_title'])

with st.expander(texts[lang_code]['descriptive_stats']):
    st.write(df.describe().T)

with st.expander(texts[lang_code]['vif_title']):
    X = df.drop('MedHouseVal', axis=1)
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    st.write(texts[lang_code]['vif_desc'])
    st.table(vif_data.sort_values(by='VIF', ascending=False))

with st.expander(texts[lang_code]['histograms']):
    st.write(texts[lang_code]['hist_desc'])
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    df.hist(ax=axes.flatten())
    plt.tight_layout()
    st.pyplot(fig)

with st.expander(texts[lang_code]['boxplots']):
    st.write(texts[lang_code]['box_desc'])
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, col in enumerate(df.columns):
        sns.boxplot(y=df[col], ax=axes[i//3, i%3])
        axes[i//3, i%3].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

with st.expander(texts[lang_code]['scatter_plots']):
    st.write(texts[lang_code]['scatter_desc'])
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, col in enumerate(df.columns[:-1]):
        sns.scatterplot(x=df[col], y=df['MedHouseVal'], ax=axes[i//3, i%3])
        axes[i//3, i%3].set_title(f'{col} vs MedHouseVal')
    plt.tight_layout()
    st.pyplot(fig)

# --- Modelado ---
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo de Regresión Lineal
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)
reg_preds = reg_model.predict(X_test_scaled)
reg_mse = mean_squared_error(y_test, reg_preds)
reg_r2 = r2_score(y_test, reg_preds)

# Modelo de Red Neuronal
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
nn_preds = nn_model.predict(X_test_scaled).flatten()
nn_mse = mean_squared_error(y_test, nn_preds)
nn_r2 = r2_score(y_test, nn_preds)

# --- Predicción y Resultados ---
st.header(texts[lang_code]['prediction_title'])
st.write(texts[lang_code]['disclaimer'])

col1, col2 = st.columns(2)
with col1:
    with st.form("input_form"):
        inputs = {}
        for col in X.columns:
            inputs[col] = st.number_input(col, value=float(X[col].mean()))
        predict_button = st.form_submit_button(texts[lang_code]['predict_button'])

with col2:
    if predict_button:
        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df)

        reg_price = reg_model.predict(input_scaled)[0]
        nn_price = nn_model.predict(input_scaled).flatten()[0]

        st.subheader(texts[lang_code]['model_results'])
        st.metric(texts[lang_code]['reg_pred'], f'${reg_price*100000:,.2f}')
        st.metric(texts[lang_code]['nn_pred'], f'${nn_price*100000:,.2f}')
    
    with st.expander(texts[lang_code]['model_metrics']):
        metrics_df = pd.DataFrame({
            'Modelo': ['Regresión Múltiple', 'Red Neuronal'],
            'MSE': [reg_mse, nn_mse],
            'R²': [reg_r2, nn_r2]
        }).set_index('Modelo')
        st.write(metrics_df)