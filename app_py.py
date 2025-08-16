import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from io import BytesIO
from pathlib import Path

# =============================================
# CONFIGURACIÓN INICIAL Y TRADUCCIONES
# =============================================

# Configuración de la página
st.set_page_config(
    page_title="Advanced Dataset Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar traducciones
def load_translations(lang):
    translations = {
        "es": {
            "title": "🔍 ANALISIS AVANZADO DE DATOS",
            "upload_label": "Sube tu archivo CSV",
            "upload_help": "Selecciona el dataset que deseas analizar",
            "file_loaded": "Archivo cargado:",
            "file_size": "Tamaño:",
            "quick_preview": "Vista Previa Rápida",
            "analysis_btn": "Análisis Exploratorio (EDA)",
            "analysis_help": "Analiza el dataset para valores nulos, correlaciones y problemas comunes",
            "export_label": "¿Deseas exportar el análisis?",
            "export_btn": "Descargar Reporte EDA",
            "duplicates_btn": "Análisis de Duplicados",
            "outliers_btn": "Análisis de Outliers",
            "null_treatment_btn": "Tratamiento de Datos",
            "basic_info": "Información Básica",
            "rows": "Número de filas:",
            "cols": "Número de columnas:",
            "null_tab": "Valores Nulos",
            "types_tab": "Tipos de Datos",
            "stats_tab": "Estadísticas",
            "sample_tab": "Muestra de Datos",
            "eda_tab": "Análisis Exploratorio",
            "correlation_tab": "Correlaciones",
            "visualization_tab": "Visualización",
            "null_title": "Valores Nulos por Columna",
            "col_name": "Columna",
            "null_count": "Valores Nulos",
            "null_percent": "Porcentaje (%)",
            "null_warning": "⚠️ El dataset contiene valores nulos que deben ser tratados",
            "null_success": "✅ No se encontraron valores nulos en el dataset",
            "types_title": "Tipos de Datos por Columna",
            "data_type": "Tipo de Dato",
            "numeric_cols": "Columnas numéricas:",
            "non_numeric_cols": "Columnas no numéricas:",
            "stats_title": "Estadísticas Descriptivas",
            "sample_title": "Muestra de Datos (Primeras 10 filas)",
            "duplicates_title": "Análisis de Duplicados",
            "total_duplicates": "Filas duplicadas totales:",
            "duplicate_rows": "Filas duplicadas:",
            "outliers_title": "Análisis de Outliers",
            "outliers_col": "Columna:",
            "outliers_count": "Outliers detectados:",
            "outliers_percent": "Porcentaje de outliers:",
            "treatment_title": "Tratamiento de Datos",
            "treatment_option1": "Eliminar filas con valores nulos",
            "treatment_option2": "Rellenar con la media (numéricas)",
            "treatment_option3": "Rellenar con la mediana (numéricas)",
            "treatment_option4": "Rellenar con moda (categóricas)",
            "treatment_option5": "Rellenar con valor específico:",
            "apply_treatment": "Aplicar Tratamiento",
            "treatment_success": "Tratamiento aplicado correctamente",
            "no_nulls": "No hay valores nulos para tratar",
            "no_duplicates": "No hay duplicados para mostrar",
            "herramientas_analisis": "Herramientas de Análisis",
            "selector_idioma": "Seleccione idioma",
            "dataset_tras_tratamiento": "Dataset después del tratamiento",
            "reset_button": "Resetear a datos originales",
            "comparison_title": "Comparación de valores nulos",
            "select_method_label": "Seleccione método de tratamiento",
            "fill_value_prompt": "Ingrese el valor de relleno:",
            "treatment_error": "Error al aplicar tratamiento",
            "eda_title": "Análisis Exploratorio de Datos (EDA)",
            "missing_data_title": "Datos Faltantes",
            "data_quality_title": "Calidad de los Datos",
            "correlation_title": "Análisis de Correlación",
            "correlation_matrix": "Matriz de Correlación",
            "top_correlations": "Top Correlaciones",
            "r2_score": "Coeficiente de Determinación (R²)",
            "mse": "Error Cuadrático Medio (MSE)",
            "rmse": "Raíz del Error Cuadrático Medio (RMSE)",
            "visualization_title": "Visualización de Datos",
            "histogram_btn": "Histograma",
            "boxplot_btn": "Diagrama de Caja",
            "scatter_btn": "Diagrama de Dispersión",
            "select_column": "Seleccione columna:",
            "select_columns": "Seleccione columnas:",
            "select_x": "Variable X:",
            "select_y": "Variable Y:",
            "no_numeric_cols": "No hay columnas numéricas para visualizar",
            "report_generated": "Reporte EDA generado con éxito",
            "download_report": "Descargar Reporte Completo",
            "data_quality_issue": "Problema de calidad de datos detectado",
            "high_missing": "Alto porcentaje de valores faltantes (>30%)",
            "potential_outliers": "Potenciales outliers detectados",
            "inconsistent_dates": "Fechas inconsistentes detectadas",
            "text_analysis": "Análisis de Texto",
            "unique_values": "Valores únicos:",
            "most_common": "Valores más comunes:",
            "data_cleaning": "Limpieza de Datos",
            "apply_cleaning": "Aplicar Limpieza",
            "cleaning_options": {
                "trim_spaces": "Eliminar espacios en blanco",
                "lowercase": "Convertir a minúsculas",
                "remove_special": "Eliminar caracteres especiales",
                "date_format": "Estandarizar formato de fecha"
            },
            "cleaning_success": "Limpieza aplicada correctamente"
        },
        "en": {
            "title": "🔍 Advanced Data Analysis",
            "upload_label": "Upload your CSV file",
            "upload_help": "Select the dataset you want to analyze",
            "file_loaded": "File loaded:",
            "file_size": "Size:",
            "quick_preview": "Quick Preview",
            "analysis_btn": "Exploratory Analysis (EDA)",
            "analysis_help": "Analyze the dataset for null values, correlations and common issues",
            "export_label": "Do you want to export the analysis?",
            "export_btn": "Download EDA Report",
            "duplicates_btn": "Duplicate Analysis",
            "outliers_btn": "Outliers Analysis",
            "null_treatment_btn": "Data Treatment",
            "basic_info": "Basic Information",
            "rows": "Number of rows:",
            "cols": "Number of columns:",
            "null_tab": "Null Values",
            "types_tab": "Data Types",
            "stats_tab": "Statistics",
            "sample_tab": "Data Sample",
            "eda_tab": "Exploratory Analysis",
            "correlation_tab": "Correlations",
            "visualization_tab": "Visualization",
            "null_title": "Null Values by Column",
            "col_name": "Column",
            "null_count": "Null Values",
            "null_percent": "Percentage (%)",
            "null_warning": "⚠️ The dataset contains null values that need treatment",
            "null_success": "✅ No null values found in the dataset",
            "types_title": "Data Types by Column",
            "data_type": "Data Type",
            "numeric_cols": "Numeric columns:",
            "non_numeric_cols": "Non-numeric columns:",
            "stats_title": "Descriptive Statistics",
            "sample_title": "Data Sample (First 10 rows)",
            "duplicates_title": "Duplicate Analysis",
            "total_duplicates": "Total duplicate rows:",
            "duplicate_rows": "Duplicate rows:",
            "outliers_title": "Outliers Analysis",
            "outliers_col": "Column:",
            "outliers_count": "Outliers detected:",
            "outliers_percent": "Outliers percentage:",
            "treatment_title": "Data Treatment",
            "treatment_option1": "Drop rows with null values",
            "treatment_option2": "Fill with mean (numeric columns)",
            "treatment_option3": "Fill with median (numeric columns)",
            "treatment_option4": "Fill with mode (categorical columns)",
            "treatment_option5": "Fill with specific value:",
            "apply_treatment": "Apply Treatment",
            "treatment_success": "Treatment applied successfully",
            "no_nulls": "No null values to treat",
            "no_duplicates": "No duplicates to show",
            "herramientas_analisis": "Analysis Tools",
            "selector_idioma": "Select language",
            "dataset_tras_tratamiento": "Dataset after treatment",
            "reset_button": "Reset to original data",
            "comparison_title": "Null values comparison",
            "select_method_label": "Select treatment method",
            "fill_value_prompt": "Enter fill value:",
            "treatment_error": "Error applying treatment",
            "eda_title": "Exploratory Data Analysis (EDA)",
            "missing_data_title": "Missing Data",
            "data_quality_title": "Data Quality",
            "correlation_title": "Correlation Analysis",
            "correlation_matrix": "Correlation Matrix",
            "top_correlations": "Top Correlations",
            "r2_score": "Coefficient of Determination (R²)",
            "mse": "Mean Squared Error (MSE)",
            "rmse": "Root Mean Squared Error (RMSE)",
            "visualization_title": "Data Visualization",
            "histogram_btn": "Histogram",
            "boxplot_btn": "Box Plot",
            "scatter_btn": "Scatter Plot",
            "select_column": "Select column:",
            "select_columns": "Select columns:",
            "select_x": "Variable X:",
            "select_y": "Variable Y:",
            "no_numeric_cols": "No numeric columns to visualize",
            "report_generated": "EDA report generated successfully",
            "download_report": "Download Full Report",
            "data_quality_issue": "Data quality issue detected",
            "high_missing": "High percentage of missing values (>30%)",
            "potential_outliers": "Potential outliers detected",
            "inconsistent_dates": "Inconsistent dates detected",
            "text_analysis": "Text Analysis",
            "unique_values": "Unique values:",
            "most_common": "Most common values:",
            "data_cleaning": "Data Cleaning",
            "apply_cleaning": "Apply Cleaning",
            "cleaning_options": {
                "trim_spaces": "Trim whitespace",
                "lowercase": "Convert to lowercase",
                "remove_special": "Remove special characters",
                "date_format": "Standardize date format"
            },
            "cleaning_success": "Cleaning applied successfully"
        }
    }
    return translations.get(lang, translations["en"])

# =============================================
# FUNCIONES PARA MANEJO DE DATOS
# =============================================

def load_dataset(file):
    """Carga un archivo CSV con manejo de diferentes codificaciones."""
    try:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file, encoding=encoding)
                # Verificar si el DataFrame está vacío
                if df.empty:
                    st.error("The uploaded file is empty.")
                    return None
                return df
            except UnicodeDecodeError:
                continue
        
        st.error("Could not read the file with common encodings.")
        return None
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_null_analysis(df):
    """Genera un análisis de valores nulos."""
    nulls = df.isnull().sum()
    nulls_percent = (nulls / len(df)) * 100
    return nulls, nulls_percent

def get_duplicate_analysis(df):
    """Analiza y devuelve filas duplicadas."""
    total_duplicates = df.duplicated().sum()
    duplicate_rows = df[df.duplicated(keep=False)] if total_duplicates > 0 else None
    return total_duplicates, duplicate_rows

def detect_data_quality_issues(df):
    """Detecta problemas comunes de calidad de datos."""
    issues = []
    
    # Detección de valores faltantes
    nulls, nulls_percent = get_null_analysis(df)
    high_missing = nulls_percent[nulls_percent > 30].index.tolist()
    if high_missing:
        issues.append(("high_missing", high_missing))
    
    # Detección de outliers potenciales (para columnas numéricas)
    numeric_cols = df.select_dtypes(include=['number']).columns
    potential_outliers = []
    
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers_count > 0:
                potential_outliers.append(col)
    
    if potential_outliers:
        issues.append(("potential_outliers", potential_outliers))
    
    # Detección de fechas inconsistentes
    date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns
    inconsistent_dates = []
    
    for col in date_cols:
        if df[col].dt.year.max() > 2100 or df[col].dt.year.min() < 1900:
            inconsistent_dates.append(col)
    
    if inconsistent_dates:
        issues.append(("inconsistent_dates", inconsistent_dates))
    
    return issues

def calculate_correlations(df):
    """Calcula correlaciones entre variables numéricas."""
    numeric_df = df.select_dtypes(include=['number'])
    
    if len(numeric_df.columns) < 2:
        return None, None, None
    
    corr_matrix = numeric_df.corr()
    
    # Obtener las top correlaciones (excluyendo la diagonal)
    corr_unstacked = corr_matrix.unstack()
    corr_unstacked = corr_unstacked[corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)]
    top_correlations = corr_unstacked.sort_values(ascending=False).head(10)
    
    # Calcular R², MSE y RMSE para las top correlaciones
    metrics = []
    for (col1, col2), corr in top_correlations.items():
        if col1 != col2:
            valid_rows = df[[col1, col2]].dropna()
            if len(valid_rows) > 1:
                r2 = r2_score(valid_rows[col1], valid_rows[col2])
                mse = mean_squared_error(valid_rows[col1], valid_rows[col2])
                rmse = np.sqrt(mse)
                metrics.append((col1, col2, corr, r2, mse, rmse))
    
    metrics_df = pd.DataFrame(metrics, columns=['Variable 1', 'Variable 2', 'Correlation', 'R²', 'MSE', 'RMSE'])
    
    return corr_matrix, top_correlations, metrics_df

def generate_eda_report(df):
    """Genera un reporte EDA completo."""
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Información básica
        pd.DataFrame({
            'Metric': ['Number of rows', 'Number of columns', 'Total missing values'],
            'Value': [len(df), len(df.columns), df.isnull().sum().sum()]
        }).to_excel(writer, sheet_name='Basic Info', index=False)
        
        # Valores nulos
        nulls, nulls_percent = get_null_analysis(df)
        pd.DataFrame({
            'Column': nulls.index,
            'Null Count': nulls.values,
            'Null Percentage': nulls_percent.values
        }).to_excel(writer, sheet_name='Null Values', index=False)
        
        # Estadísticas descriptivas
        df.describe(include='all').T.to_excel(writer, sheet_name='Descriptive Stats')
        
        # Correlaciones
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 0:
            numeric_df.corr().to_excel(writer, sheet_name='Correlations')
        
        # Tipos de datos
        pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values
        }).to_excel(writer, sheet_name='Data Types', index=False)
    
    buffer.seek(0)
    return buffer

# =============================================
# FUNCIONES PARA VISUALIZACIÓN
# =============================================

def show_analysis(df, tr):
    """Muestra el análisis completo del dataset."""
    st.subheader("📊 " + tr["basic_info"])
    st.write(f"**{tr['rows']}** {df.shape[0]}")
    st.write(f"**{tr['cols']}** {df.shape[1]}")
    
    # Crear pestañas para diferentes análisis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        tr["null_tab"], 
        tr["types_tab"], 
        tr["stats_tab"], 
        tr["sample_tab"],
        tr["eda_tab"],
        tr["correlation_tab"]
    ])
    
    with tab1:
        show_null_analysis(df, tr)
    
    with tab2:
        show_data_types(df, tr)
    
    with tab3:
        show_statistics(df, tr)
    
    with tab4:
        show_data_sample(df, tr)
        
    with tab5:
        show_eda_analysis(df, tr)
        
    with tab6:
        show_correlation_analysis(df, tr)

def show_null_analysis(df, tr):
    """Muestra el análisis de valores nulos."""
    st.write("### " + tr["null_title"])
    nulls, nulls_percent = get_null_analysis(df)
    
    nulls_df = pd.DataFrame({
        tr["col_name"]: nulls.index,
        tr["null_count"]: nulls.values,
        tr["null_percent"]: nulls_percent.values.round(2)
    })
    
    st.dataframe(nulls_df.style.highlight_max(
        axis=0, 
        subset=[tr["null_count"], tr["null_percent"]],
        color='salmon'
    ))
    
    st.bar_chart(nulls_percent)
    
    if nulls.sum() > 0:
        st.warning(tr["null_warning"])
    else:
        st.success(tr["null_success"])

def show_data_types(df, tr):
    """Muestra los tipos de datos del dataset."""
    st.write("### " + tr["types_title"])
    types = df.dtypes.reset_index()
    types.columns = [tr["col_name"], tr["data_type"]]
    st.dataframe(types)
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    
    st.write(f"**{tr['numeric_cols']}** {len(numeric_cols)}")
    st.write(f"**{tr['non_numeric_cols']}** {len(non_numeric_cols)}")

def show_statistics(df, tr):
    """Muestra estadísticas descriptivas."""
    st.write("### " + tr["stats_title"])
    try:
        stats = df.describe(include='all').T
        st.dataframe(stats)
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")

def show_data_sample(df, tr):
    """Muestra una muestra de los datos."""
    st.write("### " + tr["sample_title"])
    st.dataframe(df.head(10))

def show_duplicates(df, tr):
    """Muestra el análisis de duplicados."""
    st.subheader("🔍 " + tr["duplicates_title"])
    
    total_duplicates, duplicate_rows = get_duplicate_analysis(df)
    st.write(f"**{tr['total_duplicates']}** {total_duplicates}")
    
    if total_duplicates > 0:
        st.write(f"**{tr['duplicate_rows']}**")
        st.dataframe(duplicate_rows.sort_values(by=list(df.columns)))
    else:
        st.success("✅ " + tr["no_duplicates"])

def show_outliers(df, tr):
    """Muestra el análisis de outliers."""
    st.subheader("📈 " + tr["outliers_title"])
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) == 0:
        st.warning(tr["no_numeric_cols"])
        return
    
    selected_col = st.selectbox(tr["outliers_col"], numeric_cols)
    
    if df[selected_col].notnull().sum() > 0:
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        outliers_count = len(outliers)
        outliers_percent = (outliers_count / len(df)) * 100
        
        st.write(f"**{tr['outliers_count']}** {outliers_count}")
        st.write(f"**{tr['outliers_percent']}** {outliers_percent:.2f}%")
        
        if outliers_count > 0:
            st.dataframe(outliers)
            
            # Mostrar gráfico de caja
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(x=df[selected_col], ax=ax)
            ax.set_title(f'Boxplot of {selected_col}')
            st.pyplot(fig)
        else:
            st.success("✅ No outliers detected in this column")
    else:
        st.warning("Selected column contains only null values")

def show_eda_analysis(df, tr):
    """Muestra análisis exploratorio avanzado."""
    st.subheader("🔍 " + tr["eda_title"])
    
    # Detección de problemas de calidad de datos
    st.write("### " + tr["data_quality_title"])
    issues = detect_data_quality_issues(df)
    
    if issues:
        for issue_type, cols in issues:
            if issue_type == "high_missing":
                st.warning(f"⚠️ {tr['high_missing']} in columns: {', '.join(cols)}")
            elif issue_type == "potential_outliers":
                st.warning(f"⚠️ {tr['potential_outliers']} in columns: {', '.join(cols)}")
            elif issue_type == "inconsistent_dates":
                st.warning(f"⚠️ {tr['inconsistent_dates']} in columns: {', '.join(cols)}")
    else:
        st.success("✅ No major data quality issues detected")
    
    # Análisis de columnas categóricas/texto
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        st.write("### " + tr["text_analysis"])
        selected_col = st.selectbox(tr["select_column"], non_numeric_cols)
        
        st.write(f"**{tr['unique_values']}** {df[selected_col].nunique()}")
        
        if df[selected_col].nunique() < 20:
            st.write(f"**{tr['most_common']}**")
            st.table(df[selected_col].value_counts().head(10))
        else:
            st.write(f"**{tr['most_common']}**")
            st.table(df[selected_col].value_counts().head(5))
            
            # Mostrar distribución de longitudes para texto
            if df[selected_col].dtype == 'object':
                df['text_length'] = df[selected_col].astype(str).apply(len)
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(df['text_length'], bins=30, ax=ax)
                ax.set_title(f'Distribution of text lengths for {selected_col}')
                st.pyplot(fig)

def show_correlation_analysis(df, tr):
    """Muestra análisis de correlación."""
    st.subheader("📊 " + tr["correlation_title"])
    
    corr_matrix, top_correlations, metrics_df = calculate_correlations(df)
    
    if corr_matrix is not None:
        st.write("### " + tr["correlation_matrix"])
        
        # Mostrar heatmap de correlación
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                   center=0, ax=ax, linewidths=.5)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
        
        # Mostrar top correlaciones
        st.write("### " + tr["top_correlations"])
        st.dataframe(metrics_df)
        
        # Mostrar métricas R², MSE, RMSE
        if not metrics_df.empty:
            selected_pair = st.selectbox(
                "Select variable pair to view metrics:",
                options=[f"{row['Variable 1']} vs {row['Variable 2']}" for _, row in metrics_df.iterrows()]
            )
            
            if selected_pair:
                selected_row = metrics_df[
                    (metrics_df['Variable 1'] == selected_pair.split(" vs ")[0]) & 
                    (metrics_df['Variable 2'] == selected_pair.split(" vs ")[1])
                ].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(tr["r2_score"], f"{selected_row['R²']:.4f}")
                with col2:
                    st.metric(tr["mse"], f"{selected_row['MSE']:.4f}")
                with col3:
                    st.metric(tr["rmse"], f"{selected_row['RMSE']:.4f}")
                
                # Mostrar gráfico de dispersión
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(
                    data=df, 
                    x=selected_row['Variable 1'], 
                    y=selected_row['Variable 2'],
                    ax=ax
                )
                ax.set_title(f"{selected_row['Variable 1']} vs {selected_row['Variable 2']}")
                st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation analysis")

def show_visualization_tools(df, tr):
    """Muestra herramientas de visualización."""
    st.subheader("📊 " + tr["visualization_title"])
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) == 0:
        st.warning(tr["no_numeric_cols"])
        return
    
    # Pestañas para diferentes tipos de visualizaciones
    tab1, tab2, tab3 = st.tabs([
        tr["histogram_btn"], 
        tr["boxplot_btn"], 
        tr["scatter_btn"]
    ])
    
    with tab1:
        # Histograma
        selected_col = st.selectbox(tr["select_column"] + " (Histogram)", numeric_cols)
        bins = st.slider("Number of bins", 5, 100, 20)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_col], bins=bins, kde=True, ax=ax)
        ax.set_title(f'Histogram of {selected_col}')
        st.pyplot(fig)
    
    with tab2:
        # Boxplot
        selected_col = st.selectbox(tr["select_column"] + " (Boxplot)", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=df[selected_col], ax=ax)
        ax.set_title(f'Boxplot of {selected_col}')
        st.pyplot(fig)
    
    with tab3:
        # Scatter plot
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox(tr["select_x"], numeric_cols)
            with col2:
                y_col = st.selectbox(tr["select_y"], numeric_cols, 
                                    index=1 if len(numeric_cols) > 1 else 0)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f'{x_col} vs {y_col}')
            st.pyplot(fig)
        else:
            st.warning("At least 2 numeric columns are required for scatter plot")

def show_data_cleaning(df, tr):
    """Muestra herramientas de limpieza de datos."""
    st.subheader("🧹 " + tr["data_cleaning"])
    
    # Opciones de limpieza
    st.write("### Cleaning Options")
    col1, col2 = st.columns(2)
    
    with col1:
        trim_spaces = st.checkbox(tr["cleaning_options"]["trim_spaces"])
        lowercase = st.checkbox(tr["cleaning_options"]["lowercase"])
    
    with col2:
        remove_special = st.checkbox(tr["cleaning_options"]["remove_special"])
        date_format = st.checkbox(tr["cleaning_options"]["date_format"])
    
    # Aplicar limpieza
    if st.button(tr["apply_cleaning"]):
        cleaned_df = df.copy()
        
        # Limpieza de espacios en blanco para columnas de texto
        if trim_spaces:
            text_cols = cleaned_df.select_dtypes(include=['object']).columns
            for col in text_cols:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
        
        # Convertir a minúsculas
        if lowercase:
            text_cols = cleaned_df.select_dtypes(include=['object']).columns
            for col in text_cols:
                cleaned_df[col] = cleaned_df[col].astype(str).str.lower()
        
        # Eliminar caracteres especiales
        if remove_special:
            text_cols = cleaned_df.select_dtypes(include=['object']).columns
            for col in text_cols:
                cleaned_df[col] = cleaned_df[col].astype(str).str.replace(r'[^\w\s]', '', regex=True)
        
        # Estandarizar formato de fecha
        if date_format:
            date_cols = cleaned_df.select_dtypes(include=['datetime', 'object']).columns
            for col in date_cols:
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                except:
                    pass
        
        st.session_state.df_treated = cleaned_df
        st.success(tr["cleaning_success"])
        st.experimental_rerun()
    
    return df

# =============================================
# TRATAMIENTO DE VALORES NULOS
# =============================================

def null_treatment(df, tr):
    """Realiza el tratamiento de valores nulos."""
    # Inicialización del estado de sesión
    if 'df_treated' not in st.session_state:
        st.session_state.df_treated = df.copy()
        st.session_state.last_treatment = None
        st.session_state.show_comparison = False

    st.subheader("🛠️ " + tr["treatment_title"])

    # Verificar si ya no hay nulos
    if st.session_state.df_treated.isnull().sum().sum() == 0:
        st.success("✅ " + tr["no_nulls"])
        return st.session_state.df_treated

    # Widgets de selección
    treatment_option = st.radio(
        tr["select_method_label"],
        options=[
            tr["treatment_option1"],
            tr["treatment_option2"],
            tr["treatment_option3"],
            tr["treatment_option4"],
            tr["treatment_option5"]
        ],
        key="treatment_option_radio"
    )

    # Input para valor específico
    fill_value = None
    if treatment_option == tr["treatment_option5"]:
        fill_value = st.text_input(
            tr["fill_value_prompt"],
            key="fill_value_input"
        )

    # Botón de aplicación
    if st.button(tr["apply_treatment"], key="apply_treatment_button"):
        try:
            temp_df = st.session_state.df_treated.copy()
            
            if treatment_option == tr["treatment_option1"]:
                initial_rows = len(temp_df)
                temp_df = temp_df.dropna()
                removed_rows = initial_rows - len(temp_df)
                st.info(f"Removed {removed_rows} rows with null values")
                
            elif treatment_option == tr["treatment_option2"]:
                numeric_cols = temp_df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if temp_df[col].isnull().sum() > 0:
                        mean_val = temp_df[col].mean()
                        temp_df[col] = temp_df[col].fillna(mean_val)
                        
            elif treatment_option == tr["treatment_option3"]:
                numeric_cols = temp_df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if temp_df[col].isnull().sum() > 0:
                        median_val = temp_df[col].median()
                        temp_df[col] = temp_df[col].fillna(median_val)
                        
            elif treatment_option == tr["treatment_option4"]:
                non_numeric_cols = temp_df.select_dtypes(exclude=['number']).columns
                for col in non_numeric_cols:
                    if temp_df[col].isnull().sum() > 0:
                        mode_val = temp_df[col].mode()[0]
                        temp_df[col] = temp_df[col].fillna(mode_val)
            
            elif treatment_option == tr["treatment_option5"] and fill_value:
                try:
                    # Intentar convertir a número
                    fill_value_num = float(fill_value)
                    temp_df = temp_df.fillna(fill_value_num)
                except ValueError:
                    # Si falla, usar como string
                    temp_df = temp_df.fillna(fill_value)
            
            # Actualizar el estado de sesión
            st.session_state.df_treated = temp_df
            st.session_state.last_treatment = treatment_option
            st.session_state.show_comparison = True
            
            st.success("✅ " + tr["treatment_success"])
            
        except Exception as e:
            st.error(f"❌ {tr['treatment_error']}: {str(e)}")

    # Mostrar comparación si se aplicó un tratamiento
    if st.session_state.show_comparison:
        st.subheader(tr["comparison_title"])
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before treatment**")
            st.write(df.isna().sum())
        
        with col2:
            st.markdown("**After treatment**")
            st.write(st.session_state.df_treated.isna().sum())

    # Botón para resetear
    if st.button(tr["reset_button"], key="reset_button"):
        st.session_state.df_treated = df.copy()
        st.session_state.show_comparison = False
        st.experimental_rerun()

    return st.session_state.df_treated
