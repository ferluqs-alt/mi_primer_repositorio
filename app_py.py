import pandas as pd
import os
from pathlib import Path

def load_and_preprocess_data(filepath, sample_data=False):
    """
    Carga y preprocesa los datos desde un archivo CSV
    
    Args:
        filepath (str): Ruta al archivo CSV
        sample_data (bool): Si True, carga datos de muestra si el archivo no existe
    
    Returns:
        pd.DataFrame: Datos procesados
    """
    try:
        # Verificar la ruta del archivo
        filepath = Path(filepath)
        
        if not filepath.exists():
            if sample_data:
                st.warning(f"Archivo {filepath} no encontrado. Cargando datos de muestra...")
                return create_sample_data()
            raise FileNotFoundError(f"El archivo {filepath} no existe")
        
        if filepath.stat().st_size == 0:
            raise ValueError(f"El archivo {filepath} está vacío")
        
        # Intentar leer con diferentes codificaciones comunes
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                data = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("No se pudo decodificar el archivo con las codificaciones comunes")
        
        # Verificar si se cargaron datos
        if data.empty:
            raise ValueError("El archivo CSV no contiene datos válidos")
        
        # Preprocesamiento básico
        data = data.dropna()
        
        # Convertir variables categóricas si es necesario
        # data = pd.get_dummies(data, drop_first=True)
        
        return data
    
    except Exception as e:
        raise Exception(f"Error al procesar datos: {str(e)}")

def create_sample_data():
    """Crea datos de muestra para propósitos de demostración"""
    import numpy as np
    
    np.random.seed(42)
    sample_size = 100
    
    data = pd.DataFrame({
        'feature1': np.random.normal(50, 15, sample_size),
        'feature2': np.random.uniform(0, 100, sample_size),
        'feature3': np.random.randint(1, 5, sample_size),
        'price': np.random.normal(200, 50, sample_size)
    })
    
    # Asegurar que no haya valores negativos en ciertas características
    data['feature1'] = data['feature1'].clip(lower=0)
    data['price'] = data['price'].clip(lower=10)
    
    return data
