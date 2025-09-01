"""
Script para preparar datasets corregidos para los especialistas
Los especialistas necesitan ver TODAS las clases, no solo las suyas
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def prepare_specialist_dataset(df_full, target_classes, other_classes, specialist_name):
    """
    Prepara un dataset para un especialista que incluye todas las clases
    pero con etiquetas modificadas para manejar clases fuera de dominio
    
    Args:
        df_full: DataFrame completo con todas las clases
        target_classes: Lista de clases objetivo del especialista (ej: ['Cloudy', 'Sunny'])
        other_classes: Lista de otras clases (ej: ['Rainy', 'Snowy'])
        specialist_name: Nombre del especialista para guardar el archivo
    """
    print(f"\nPreparando dataset para {specialist_name}")
    print(f"  - Clases objetivo: {target_classes}")
    print(f"  - Clases fuera de dominio: {other_classes}")
    
    # Copiar el dataframe
    df = df_full.copy()
    
    # Crear nueva columna de etiquetas con 3 clases:
    # 0: Primera clase objetivo
    # 1: Segunda clase objetivo  
    # 2: Otras (fuera de dominio)
    
    # Primero, crear columna Weather Type si no existe
    if 'Weather Type' not in df.columns:
        # Inferir desde las columnas one-hot
        for col in df.columns:
            if col.startswith('Weather Type_'):
                class_name = col.replace('Weather Type_', '')
                df.loc[df[col] == 1, 'Weather Type'] = class_name
    
    # Mapear las etiquetas
    label_mapping = {}
    for i, target_class in enumerate(target_classes):
        label_mapping[target_class] = i
    
    # Todas las demás clases van a la clase 2 (Otras)
    for other_class in other_classes:
        label_mapping[other_class] = 2
    
    # Aplicar mapeo
    df['Label_Specialist'] = df['Weather Type'].map(label_mapping)
    
    # Crear columnas one-hot para el especialista
    # Eliminar columnas Weather Type_ existentes
    cols_to_drop = [col for col in df.columns if col.startswith('Weather Type_')]
    df = df.drop(columns=cols_to_drop)
    
    # Crear nuevas columnas one-hot
    df[f'Weather Type_{target_classes[0]}'] = (df['Label_Specialist'] == 0).astype(int)
    df[f'Weather Type_{target_classes[1]}'] = (df['Label_Specialist'] == 1).astype(int)
    df['Weather Type_Other'] = (df['Label_Specialist'] == 2).astype(int)
    
    # Eliminar columna temporal
    df = df.drop(columns=['Label_Specialist'])
    
    # Mostrar distribución
    print("\n  Distribución de clases en el dataset:")
    for col in [f'Weather Type_{target_classes[0]}', f'Weather Type_{target_classes[1]}', 'Weather Type_Other']:
        count = df[col].sum()
        pct = count / len(df) * 100
        print(f"    - {col}: {count} ({pct:.1f}%)")
    
    # Verificar balance
    # Queremos que las clases objetivo tengan más peso que "Other"
    # Podemos duplicar algunas muestras de las clases objetivo si es necesario
    target_count = df[f'Weather Type_{target_classes[0]}'].sum() + df[f'Weather Type_{target_classes[1]}'].sum()
    other_count = df['Weather Type_Other'].sum()
    
    if other_count > target_count * 1.5:
        print(f"\n  ⚠️  Desbalance detectado: {other_count} otras vs {target_count} objetivo")
        print(f"  Aplicando submuestreo de clases 'Other'...")
        
        # Índices de cada tipo
        target_indices = df[(df[f'Weather Type_{target_classes[0]}'] == 1) | 
                           (df[f'Weather Type_{target_classes[1]}'] == 1)].index
        other_indices = df[df['Weather Type_Other'] == 1].index
        
        # Submuestrear las clases "Other" para que no superen 1.5x las clases objetivo
        n_other_keep = int(target_count * 1.2)  # Mantener 20% más que las objetivo
        other_indices_keep = np.random.choice(other_indices, size=n_other_keep, replace=False)
        
        # Combinar índices
        final_indices = np.concatenate([target_indices, other_indices_keep])
        df = df.loc[final_indices].reset_index(drop=True)
        
        print(f"  Dataset balanceado: {len(df)} muestras totales")
    
    return df

def main():
    """Prepara los datasets corregidos para todos los especialistas"""
    
    print("="*80)
    print("PREPARACIÓN DE DATASETS CORREGIDOS PARA ESPECIALISTAS")
    print("="*80)
    
    # Cargar dataset completo
    csv_path = "data/weather_classification_normalized.csv"
    print(f"\nCargando dataset original: {csv_path}")
    df_full = pd.read_csv(csv_path)
    print(f"  - Forma: {df_full.shape}")
    
    # Verificar clases disponibles
    weather_cols = [col for col in df_full.columns if col.startswith('Weather Type_') and col != 'Weather Type']
    print(f"  - Clases encontradas: {[col.replace('Weather Type_', '') for col in weather_cols]}")
    
    # Crear directorio de salida
    output_dir = "data_corrected"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preparar dataset para Especialista A (Cloudy/Sunny)
    df_spec_a = prepare_specialist_dataset(
        df_full,
        target_classes=['Cloudy', 'Sunny'],
        other_classes=['Rainy', 'Snowy'],
        specialist_name='Especialista A (Cloudy/Sunny)'
    )
    
    # Guardar
    output_path_a = os.path.join(output_dir, "weather_classification_CloudySunny_corrected.csv")
    df_spec_a.to_csv(output_path_a, index=False)
    print(f"\n✅ Dataset guardado en: {output_path_a}")
    
    # 2. Preparar dataset para Especialista B (Rainy/Snowy)
    df_spec_b = prepare_specialist_dataset(
        df_full,
        target_classes=['Rainy', 'Snowy'],
        other_classes=['Cloudy', 'Sunny'],
        specialist_name='Especialista B (Rainy/Snowy)'
    )
    
    # Guardar
    output_path_b = os.path.join(output_dir, "weather_classification_RainySnowy_corrected.csv")
    df_spec_b.to_csv(output_path_b, index=False)
    print(f"\n✅ Dataset guardado en: {output_path_b}")
    
    # 3. Verificar características
    print("\n" + "="*80)
    print("VERIFICACIÓN DE CARACTERÍSTICAS")
    print("="*80)
    
    # Verificar que ambos datasets tienen las mismas características
    features_a = [col for col in df_spec_a.columns if not col.startswith('Weather Type_') and col != 'Weather Type']
    features_b = [col for col in df_spec_b.columns if not col.startswith('Weather Type_') and col != 'Weather Type']
    
    print(f"\nEspecialista A - Características: {len(features_a)}")
    print(f"Especialista B - Características: {len(features_b)}")
    
    if set(features_a) == set(features_b):
        print("✅ Ambos datasets tienen las mismas características")
    else:
        print("❌ Los datasets tienen características diferentes!")
        diff_a = set(features_a) - set(features_b)
        diff_b = set(features_b) - set(features_a)
        if diff_a:
            print(f"  - Solo en A: {diff_a}")
        if diff_b:
            print(f"  - Solo en B: {diff_b}")
    
    # Mostrar resumen final
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    print("\nDatasets creados exitosamente con la siguiente estructura:")
    print("\n1. Especialista A (Cloudy/Sunny):")
    print("   - Clase 0: Cloudy")
    print("   - Clase 1: Sunny")
    print("   - Clase 2: Other (Rainy + Snowy)")
    print("\n2. Especialista B (Rainy/Snowy):")
    print("   - Clase 0: Rainy")
    print("   - Clase 1: Snowy")
    print("   - Clase 2: Other (Cloudy + Sunny)")
    print("\nLos especialistas ahora pueden manejar muestras fuera de su dominio.")
    
    # Crear archivo de configuración para el entrenamiento
    config = {
        "specialist_a": {
            "data_path": output_path_a,
            "target_classes": ["Cloudy", "Sunny"],
            "class_weights": {
                "0": 1.0,  # Cloudy
                "1": 1.0,  # Sunny
                "2": 0.3   # Other (menor peso)
            }
        },
        "specialist_b": {
            "data_path": output_path_b,
            "target_classes": ["Rainy", "Snowy"],
            "class_weights": {
                "0": 1.0,  # Rainy
                "1": 1.0,  # Snowy
                "2": 0.3   # Other (menor peso)
            }
        }
    }
    
    import json
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Configuración guardada en: {config_path}")

if __name__ == "__main__":
    main()