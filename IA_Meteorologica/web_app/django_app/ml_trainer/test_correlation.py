#!/usr/bin/env python
"""
Script para probar la generación de matriz de correlación
"""

import os
import sys
import django
import json

# Configurar Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.models import Dataset
import pandas as pd
import numpy as np

def test_correlation():
    """Probar el cálculo de correlación directamente"""
    try:
        # Obtener un dataset
        dataset = Dataset.objects.get(pk=24)
        print(f"Dataset: {dataset.name}")
        
        # Cargar datos
        df = pd.read_csv(dataset.file.path)
        print(f"Shape: {df.shape}")
        
        # Seleccionar columnas numéricas
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        print(f"Numeric columns: {numeric_columns}")
        
        # Calcular correlación
        corr_matrix = df[numeric_columns].corr()
        print(f"Correlation matrix shape: {corr_matrix.shape}")
        
        # Ver si hay NaN
        nan_count = corr_matrix.isna().sum().sum()
        print(f"NaN values in correlation matrix: {nan_count}")
        
        # Intentar serializar a JSON
        try:
            # Método 1: directo
            json_str = json.dumps(corr_matrix.to_dict())
            print("✅ Direct JSON serialization: OK")
        except Exception as e:
            print(f"❌ Direct JSON serialization failed: {e}")
        
        # Método 2: con fillna
        try:
            corr_filled = corr_matrix.fillna(0)
            json_str = json.dumps(corr_filled.to_dict())
            print("✅ JSON with fillna: OK")
        except Exception as e:
            print(f"❌ JSON with fillna failed: {e}")
            
        # Método 3: convertir a numpy luego a lista
        try:
            corr_list = corr_matrix.values.tolist()
            json_str = json.dumps(corr_list)
            print("✅ JSON with .tolist(): OK")
        except Exception as e:
            print(f"❌ JSON with .tolist() failed: {e}")
            
        # Ver tipos de datos
        print("\nData types in correlation matrix:")
        for col in corr_matrix.columns[:3]:  # Primeras 3 columnas
            for idx in corr_matrix.index[:3]:  # Primeras 3 filas
                val = corr_matrix.loc[idx, col]
                print(f"  [{idx}, {col}]: {type(val)} = {val}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_correlation()