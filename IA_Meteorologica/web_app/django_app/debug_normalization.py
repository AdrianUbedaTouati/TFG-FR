#!/usr/bin/env python
"""
Script de depuración para entender el comportamiento de la normalización
"""

import pandas as pd

# Simular el comportamiento actual
def test_normalization_behavior():
    # Crear DataFrame de prueba
    df = pd.DataFrame({
        'temperature': [20.5, 21.0, 19.5, 22.0, 20.0],
        'humidity': [65, 70, 60, 75, 68],
        'weather': ['sunny', 'cloudy', 'sunny', 'rainy', 'cloudy']
    })
    
    print("DataFrame original:")
    print(df)
    print(f"Columnas: {list(df.columns)}")
    print()
    
    # Caso 1: Normalización simple con keep_original=False
    print("=== CASO 1: Normalización simple (keep_original=False) ===")
    normalized_df = df.copy()
    
    # Simular MIN_MAX
    col_min = normalized_df['temperature'].min()
    col_max = normalized_df['temperature'].max()
    normalized_df['temperature'] = (normalized_df['temperature'] - col_min) / (col_max - col_min)
    
    print("Resultado:")
    print(normalized_df)
    print(f"Columnas: {list(normalized_df.columns)}")
    print("NOTA: La columna 'temperature' sigue existiendo pero con valores normalizados")
    print()
    
    # Caso 2: Normalización simple con keep_original=True
    print("=== CASO 2: Normalización simple (keep_original=True) ===")
    normalized_df = df.copy()
    
    # Crear nueva columna
    normalized_df['temperature_normalized'] = (normalized_df['temperature'] - col_min) / (col_max - col_min)
    
    print("Resultado:")
    print(normalized_df)
    print(f"Columnas: {list(normalized_df.columns)}")
    print("NOTA: Ahora hay dos columnas: 'temperature' (original) y 'temperature_normalized'")
    print()
    
    # Caso 3: Multi-capa con keep_original=False en todas
    print("=== CASO 3: Multi-capa (2 capas, ambas con keep_original=False) ===")
    normalized_df = df.copy()
    
    # Capa 1: MIN_MAX
    normalized_df['temperature_step1'] = (normalized_df['temperature'] - col_min) / (col_max - col_min)
    
    # Capa 2: Z_SCORE sobre el resultado de capa 1
    mean = normalized_df['temperature_step1'].mean()
    std = normalized_df['temperature_step1'].std()
    normalized_df['temperature_step2'] = (normalized_df['temperature_step1'] - mean) / std
    
    print("Antes de eliminar columna original:")
    print(f"Columnas: {list(normalized_df.columns)}")
    
    # Verificar si hay columnas nuevas
    column = 'temperature'
    new_cols = [col for col in normalized_df.columns if col.startswith(column) and col != column and len(col) > len(column)]
    print(f"Columnas nuevas detectadas: {new_cols}")
    
    if new_cols:
        # Eliminar columna original
        normalized_df = normalized_df.drop(columns=[column])
        print("Columna original eliminada")
    
    print("\nResultado final:")
    print(normalized_df)
    print(f"Columnas: {list(normalized_df.columns)}")
    print()

if __name__ == "__main__":
    test_normalization_behavior()