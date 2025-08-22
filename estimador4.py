import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import io
import base64

# Configuración de la página
st.set_page_config(
    page_title="Estimador de Precios de Viviendas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título de la aplicación
st.title("🏠 Estimador de Precios de Viviendas")

# =============================================
# FUNCIONES AUXILIARES
# =============================================

def load_data(uploaded_file):
    """Carga datos desde un archivo CSV o Excel"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de archivo no compatible. Por favor, suba un archivo CSV o Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

def detect_outliers_iqr(df, column):
    """Detecta outliers usando el método IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def get_download_link(df, filename="data.csv", text="Descargar CSV"):
    """Genera un enlace para descargar un DataFrame como CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# =============================================
# INTERFAZ PRINCIPAL
# =============================================

# Sidebar para navegación
st.sidebar.title("Navegación")
app_mode = st.sidebar.selectbox("Seleccione una sección", 
    ["Carga de datos", "Análisis inicial", "Limpieza de datos", 
     "Normalización", "Codificación", "EDA", "Correlación", 
     "División de datos", "Modelado", "Resultados"])

# Inicialización de variables en session_state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# Sección 1: Carga de datos
if app_mode == "Carga de datos":
    st.header("📤 Carga de Dataset")
    
    uploaded_file = st.file_uploader("Suba su archivo (CSV o Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.df_clean = df.copy()
            
            st.success("✅ Datos cargados exitosamente!")
            st.write(f"**Forma del dataset:** {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # Selección de variable objetivo
            target_options = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_options:
                st.session_state.target = st.selectbox("Seleccione la variable objetivo (precio)", target_options)
            else:
                st.warning("No se encontraron columnas numéricas en el dataset.")
            
            # Vista previa de los datos
            st.subheader("Vista previa de los datos")
            st.dataframe(df.head(10))

# Si no hay datos cargados, mostrar mensaje
if st.session_state.df is None:
    st.warning("Por favor, cargue un dataset en la sección 'Carga de datos'")
    st.stop()

# Acceso a los datos
df = st.session_state.df
df_clean = st.session_state.df_clean
target = st.session_state.get('target', None)

# Sección 2: Análisis inicial
if app_mode == "Análisis inicial":
    st.header("📊 Análisis Inicial de Datos")
    
    # Información básica
    st.subheader("Información básica")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Número de filas:** {df.shape[0]}")
        st.write(f"**Número de columnas:** {df.shape[1]}")
    with col2:
        st.write(f"**Valores nulos totales:** {df.isnull().sum().sum()}")
        st.write(f"**Registros duplicados:** {df.duplicated().sum()}")
    
    # Valores nulos por columna
    st.subheader("Valores nulos por columna")
    null_counts = df.isnull().sum()
    null_df = pd.DataFrame({
        'Columna': null_counts.index,
        'Valores nulos': null_counts.values,
        'Porcentaje': (null_counts.values / len(df)) * 100
    })
    st.dataframe(null_df[null_df['Valores nulos'] > 0])
    
    # Detección de outliers
    st.subheader("Detección de Outliers (Método IQR)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        selected_col = st.selectbox("Seleccione columna para analizar outliers", numeric_cols)
        
        if selected_col:
            outliers, lower_bound, upper_bound = detect_outliers_iqr(df, selected_col)
            st.write(f"**Límite inferior:** {lower_bound:.2f}")
            st.write(f"**Límite superior:** {upper_bound:.2f}")
            st.write(f"**Número de outliers detectados:** {len(outliers)}")
            
            # Gráfico de boxplot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, y=selected_col, ax=ax)
            ax.set_title(f"Boxplot de {selected_col}")
            st.pyplot(fig)
    else:
        st.warning("No hay columnas numéricas para analizar outliers.")

# Sección 3: Limpieza de datos
if app_mode == "Limpieza de datos":
    st.header("🧹 Limpieza de Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Eliminar Valores Nulos"):
            initial_shape = df_clean.shape
            df_clean.dropna(inplace=True)
            st.session_state.df_clean = df_clean
            st.success(f"Se eliminaron {initial_shape[0] - df_clean.shape[0]} filas con valores nulos")
    
    with col2:
        if st.button("Eliminar Duplicados"):
            initial_shape = df_clean.shape
            df_clean.drop_duplicates(inplace=True)
            st.session_state.df_clean = df_clean
            st.success(f"Se eliminaron {initial_shape[0] - df_clean.shape[0]} registros duplicados")
    
    with col3:
        if st.button("Eliminar Outliers"):
            if target and target in df_clean.columns:
                outliers, _, _ = detect_outliers_iqr(df_clean, target)
                initial_shape = df_clean.shape
                df_clean = df_clean[~df_clean.index.isin(outliers.index)]
                st.session_state.df_clean = df_clean
                st.success(f"Se eliminaron {initial_shape[0] - df_clean.shape[0]} outliers de la variable objetivo")
            else:
                st.warning("Seleccione una variable objetivo primero en la sección de Carga de Datos")
    
    # Mostrar estado actual después de la limpieza
    st.subheader("Estado después de la limpieza")
    st.write(f"**Filas restantes:** {df_clean.shape[0]}")
    st.write(f"**Columnas restantes:** {df_clean.shape[1]}")
    st.write(f"**Valores nulos restantes:** {df_clean.isnull().sum().sum()}")
    
    # Opción para descargar datos limpios
    st.markdown(get_download_link(df_clean, "datos_limpios.csv", "📥 Descargar datos limpios"), unsafe_allow_html=True)

# Sección 4: Normalización
if app_mode == "Normalización":
    st.header("📏 Normalización de Datos")
    
    # Verificar si se necesita normalización
    st.subheader("Verificar necesidad de normalización")
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    if numeric_cols:
        # Mostrar estadísticas antes de la normalización
        st.write("**Estadísticas antes de la normalización:**")
        st.dataframe(df_clean[numeric_cols].describe())
        
        # Seleccionar método de normalización
        norm_method = st.radio("Seleccione método de normalización", 
                              ["Min-Max Scaling", "Standardization (Z-score)"])
        
        if st.button("Aplicar Normalización"):
            df_normalized = df_clean.copy()
            
            if norm_method == "Min-Max Scaling":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
                
            df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
            st.session_state.df_clean = df_normalized
            
            # Mostrar estadísticas después de la normalización
            st.write("**Estadísticas después de la normalización:**")
            st.dataframe(df_normalized[numeric_cols].describe())
            
            st.success("✅ Normalización aplicada correctamente")
    else:
        st.warning("No hay suficientes columnas numéricas para normalizar")

# Sección 5: Codificación de variables categóricas
if app_mode == "Codificación":
    st.header("🔤 Codificación de Variables Categóricas")
    
    # Identificar variables categóricas
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        st.write("**Variables categóricas identificadas:**", categorical_cols)
        
        for col in categorical_cols:
            st.subheader(f"Variable: {col}")
            unique_vals = df_clean[col].nunique()
            st.write(f"**Valores únicos:** {unique_vals}")
            
            # Seleccionar método de codificación
            encoding_method = st.radio(f"Método de codificación para {col}", 
                                      ["One-Hot Encoding", "Label Encoding"], key=col)
            
            if st.button(f"Aplicar codificación a {col}", key=f"btn_{col}"):
                if encoding_method == "One-Hot Encoding":
                    # Aplicar One-Hot Encoding
                    dummies = pd.get_dummies(df_clean[col], prefix=col)
                    df_clean = pd.concat([df_clean, dummies], axis=1)
                    df_clean.drop(col, axis=1, inplace=True)
                else:
                    # Aplicar Label Encoding
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                
                st.session_state.df_clean = df_clean
                st.success(f"✅ Codificación aplicada a {col}")
        
        # Mostrar dataset después de la codificación
        st.subheader("Dataset después de la codificación")
        st.dataframe(df_clean.head())
    else:
        st.info("No se encontraron variables categóricas en el dataset")

# Sección 6: Análisis Exploratorio de Datos (EDA)
if app_mode == "EDA":
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")
    
    if st.button("Realizar Análisis Exploratorio"):
        # Estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(df_clean.describe())
        
        # Histogramas para variables numéricas
        st.subheader("Histogramas")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            cols_per_row = 3
            rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
            
            for i in range(rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j
                    if idx < len(numeric_cols):
                        col = numeric_cols[idx]
                        fig, ax = plt.subplots(figsize=(4, 3))
                        df_clean[col].hist(bins=30, ax=ax)
                        ax.set_title(f'Histograma de {col}')
                        cols[j].pyplot(fig)
        
        # Diagramas de dispersión
        st.subheader("Diagramas de Dispersión")
        if target and target in df_clean.columns:
            numeric_cols_without_target = [col for col in numeric_cols if col != target]
            if numeric_cols_without_target:
                x_col = st.selectbox("Seleccione variable para eje X", numeric_cols_without_target)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(data=df_clean, x=x_col, y=target, ax=ax)
                ax.set_title(f'Relación entre {x_col} y {target}')
                st.pyplot(fig)

# Sección 7: Matriz de Correlación
if app_mode == "Correlación":
    st.header("📈 Matriz de Correlación")
    
    if st.button("Calcular Matriz de Correlación"):
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Calcular matriz de correlación
            corr_matrix = df_clean[numeric_cols].corr()
            
            # Visualizar matriz de correlación
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Matriz de Correlación')
            st.pyplot(fig)
            
            # Detectar alta correlación
            st.subheader("Detección de Alta Correlación")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]
                        ))
            
            if high_corr:
                st.warning("Se detectaron variables con alta correlación (>0.8):")
                for col1, col2, corr_value in high_corr:
                    st.write(f"- {col1} y {col2}: {corr_value:.3f}")
                
                # Sugerir eliminar una variable de cada par altamente correlacionado
                st.info("**Sugerencia:** Considere eliminar una variable de cada par altamente correlacionado para evitar multicolinealidad.")
            else:
                st.success("No se detectaron variables con correlación muy alta (>0.8)")
        else:
            st.warning("Se necesitan al menos 2 variables numéricas para calcular la correlación")

# Sección 8: División de datos
if app_mode == "División de datos":
    st.header("📊 División de Datos")
    
    if target and target in df_clean.columns:
        # Separar características y variable objetivo
        X = df_clean.drop(target, axis=1)
        y = df_clean[target]
        
        # División de datos
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  # 0.15/0.85 ≈ 0.1765
        
        # Guardar en session_state
        st.session_state.X_train = X_train
        st.session_state.X_val = X_val
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_val = y_val
        st.session_state.y_test = y_test
        
        # Mostrar información de la división
        st.success("✅ Datos divididos exitosamente")
        st.write(f"**Conjunto de entrenamiento:** {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
        st.write(f"**Conjunto de validación:** {X_val.shape[0]} muestras ({X_val.shape[0]/len(X)*100:.1f}%)")
        st.write(f"**Conjunto de prueba:** {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")
    else:
        st.error("Variable objetivo no definida o no encontrada en el dataset")

# Sección 9: Modelado
if app_mode == "Modelado":
    st.header("🤖 Modelado de Datos")
    
    # Verificar si los datos están divididos
    if 'X_train' not in st.session_state:
        st.warning("Primero debe dividir los datos en la sección 'División de datos'")
        st.stop()
    
    # Obtener datos
    X_train = st.session_state.X_train
    X_val = st.session_state.X_val
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_val = st.session_state.y_val
    y_test = st.session_state.y_test
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regresión Lineal Múltiple")
        if st.button("Entrenar Modelo de Regresión Lineal"):
            # Entrenar modelo
            model_lr = LinearRegression()
            model_lr.fit(X_train, y_train)
            
            # Predecir
            y_pred_train = model_lr.predict(X_train)
            y_pred_val = model_lr.predict(X_val)
            y_pred_test = model_lr.predict(X_test)
            
            # Calcular métricas
            metrics = {
                'train': {
                    'r2': r2_score(y_train, y_pred_train),
                    'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'mae': mean_absolute_error(y_train, y_pred_train)
                },
                'val': {
                    'r2': r2_score(y_val, y_pred_val),
                    'rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                    'mae': mean_absolute_error(y_val, y_pred_val)
                },
                'test': {
                    'r2': r2_score(y_test, y_pred_test),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'mae': mean_absolute_error(y_test, y_pred_test)
                }
            }
            
            # Guardar modelo y predicciones
            st.session_state.models['linear_regression'] = {
                'model': model_lr,
                'metrics': metrics,
                'predictions': {
                    'train': y_pred_train,
                    'val': y_pred_val,
                    'test': y_pred_test
                }
            }
            
            st.success("✅ Modelo de Regresión Lineal entrenado")
            
            # Mostrar métricas
            st.write("**Métricas de Regresión Lineal:**")
            st.json(metrics)
    
    with col2:
        st.subheader("Red Neuronal (MLP)")
        if st.button("Entrenar Modelo de Red Neuronal"):
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense
                
                # Crear modelo
                model_nn = Sequential()
                model_nn.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
                model_nn.add(Dense(32, activation='relu'))
                model_nn.add(Dense(16, activation='relu'))
                model_nn.add(Dense(1, activation='linear'))
                
                # Compilar modelo
                model_nn.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                # Entrenar modelo
                history = model_nn.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=32,
                    verbose=0
                )
                
                # Predecir
                y_pred_train = model_nn.predict(X_train).flatten()
                y_pred_val = model_nn.predict(X_val).flatten()
                y_pred_test = model_nn.predict(X_test).flatten()
                
                # Calcular métricas
                metrics = {
                    'train': {
                        'r2': r2_score(y_train, y_pred_train),
                        'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                        'mae': mean_absolute_error(y_train, y_pred_train)
                    },
                    'val': {
                        'r2': r2_score(y_val, y_pred_val),
                        'rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                        'mae': mean_absolute_error(y_val, y_pred_val)
                    },
                    'test': {
                        'r2': r2_score(y_test, y_pred_test),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                        'mae': mean_absolute_error(y_test, y_pred_test)
                    }
                }
                
                # Guardar modelo y predicciones
                st.session_state.models['neural_network'] = {
                    'model': model_nn,
                    'metrics': metrics,
                    'predictions': {
                        'train': y_pred_train,
                        'val': y_pred_val,
                        'test': y_pred_test
                    },
                    'history': history.history
                }
                
                st.success("✅ Modelo de Red Neuronal entrenado")
                
                # Mostrar métricas
                st.write("**Métricas de Red Neuronal:**")
                st.json(metrics)
                
                # Gráfico de pérdida durante el entrenamiento
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(history.history['loss'], label='Pérdida entrenamiento')
                ax.plot(history.history['val_loss'], label='Pérdida validación')
                ax.set_title('Pérdida durante el entrenamiento')
                ax.set_xlabel('Época')
                ax.set_ylabel('Pérdida (MSE)')
                ax.legend()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error al entrenar la red neuronal: {str(e)}")

# Sección 10: Resultados
if app_mode == "Resultados":
    st.header("📋 Resultados y Evaluación")
    
    if not st.session_state.models:
        st.warning("No hay modelos entrenados. Vaya a la sección 'Modelado' para entrenar modelos.")
        st.stop()
    
    # Comparar métricas de los modelos
    st.subheader("Comparación de Modelos")
    
    metrics_comparison = []
    for model_name, model_data in st.session_state.models.items():
        metrics_comparison.append({
            'Modelo': model_name,
            'R² (test)': model_data['metrics']['test']['r2'],
            'RMSE (test)': model_data['metrics']['test']['rmse'],
            'MAE (test)': model_data['metrics']['test']['mae']
        })
    
    metrics_df = pd.DataFrame(metrics_comparison)
    st.dataframe(metrics_df)
    
    # Visualización de predicciones vs valores reales
    st.subheader("Predicciones vs Valores Reales")
    
    for model_name, model_data in st.session_state.models.items():
        st.write(f"**Modelo: {model_name}**")
        
        # Scatter plot de predicciones vs valores reales
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        ax[0].scatter(st.session_state.y_test, model_data['predictions']['test'], alpha=0.5)
        ax[0].plot([st.session_state.y_test.min(), st.session_state.y_test.max()], 
                  [st.session_state.y_test.min(), st.session_state.y_test.max()], 'r--')
        ax[0].set_xlabel('Valores Reales')
        ax[0].set_ylabel('Predicciones')
        ax[0].set_title(f'Predicciones vs Reales - {model_name}')
        
        # Histograma de errores
        errors = st.session_state.y_test - model_data['predictions']['test']
        ax[1].hist(errors, bins=30)
        ax[1].axvline(x=0, color='r', linestyle='--')
        ax[1].set_xlabel('Error de Predicción')
        ax[1].set_ylabel('Frecuencia')
        ax[1].set_title(f'Distribución de Errores - {model_name}')
        
        st.pyplot(fig)
    
    # Boxplot de errores entre modelos
    st.subheader("Comparación de Errores entre Modelos")
    
    errors_data = []
    for model_name, model_data in st.session_state.models.items():
        errors = st.session_state.y_test - model_data['predictions']['test']
        errors_data.append(pd.DataFrame({
            'Modelo': model_name,
            'Error': errors
        }))
    
    if errors_data:
        errors_df = pd.concat(errors_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=errors_df, x='Modelo', y='Error', ax=ax)
        ax.set_title('Distribución de Errores por Modelo')
        st.pyplot(fig)