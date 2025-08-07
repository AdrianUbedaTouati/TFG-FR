#!/usr/bin/env python
"""
Script para probar la nueva implementación de análisis sin errores de NaN
"""

import os
import sys
import django
import requests
import json

# Configurar Django
django_app_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(django_app_path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.models import Dataset

def test_analysis():
    """Probar ambos análisis (correlación y PCA) con la nueva implementación"""
    
    # URL base
    base_url = "http://localhost:8000/api"
    
    # Obtener un dataset
    try:
        dataset = Dataset.objects.get(pk=24)
        print(f"✅ Dataset encontrado: {dataset.name}")
    except Dataset.DoesNotExist:
        print("❌ Dataset con ID 24 no encontrado")
        # Intentar con el primer dataset disponible
        dataset = Dataset.objects.first()
        if dataset:
            print(f"✅ Usando dataset alternativo: {dataset.name} (ID: {dataset.pk})")
        else:
            print("❌ No hay datasets disponibles")
            return
    
    dataset_id = dataset.pk
    
    # Test 1: Matriz de Correlación
    print("\n" + "="*50)
    print("TEST 1: MATRIZ DE CORRELACIÓN")
    print("="*50)
    
    try:
        # Simulación de llamada directa a la vista
        from ml_trainer.analysis_views import dataset_analysis
        from django.test import RequestFactory
        
        factory = RequestFactory()
        
        # Test correlación
        request = factory.get(f'/api/datasets/{dataset_id}/analysis/', {'type': 'correlation'})
        response = dataset_analysis(request, dataset_id)
        
        if response.status_code == 200:
            data = json.loads(response.content)
            print("✅ Análisis de correlación exitoso")
            print(f"   - Tipo: {data.get('analysis_type')}")
            print(f"   - Estado: {data.get('status')}")
            print(f"   - Columnas numéricas: {len(data.get('statistics', {}).get('numeric_columns', []))}")
            print(f"   - Correlaciones fuertes: {len(data.get('statistics', {}).get('strong_correlations', []))}")
        else:
            print(f"❌ Error en correlación: Status {response.status_code}")
            print(f"   - Respuesta: {response.content.decode()}")
    
    except Exception as e:
        print(f"❌ Error en test de correlación: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Análisis PCA
    print("\n" + "="*50)
    print("TEST 2: ANÁLISIS PCA")
    print("="*50)
    
    try:
        # Test PCA
        request = factory.get(f'/api/datasets/{dataset_id}/analysis/', {'type': 'pca'})
        response = dataset_analysis(request, dataset_id)
        
        if response.status_code == 200:
            data = json.loads(response.content)
            print("✅ Análisis PCA exitoso")
            print(f"   - Tipo: {data.get('analysis_type')}")
            print(f"   - Estado: {data.get('status')}")
            print(f"   - Variables totales: {data.get('statistics', {}).get('total_variables')}")
            print(f"   - Componentes para 95% varianza: {data.get('statistics', {}).get('n_components_95_variance')}")
        else:
            print(f"❌ Error en PCA: Status {response.status_code}")
            print(f"   - Respuesta: {response.content.decode()}")
    
    except Exception as e:
        print(f"❌ Error en test de PCA: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Verificar tipos de datos
    print("\n" + "="*50)
    print("TEST 3: VERIFICACIÓN DE TIPOS DE DATOS")
    print("="*50)
    
    try:
        import pandas as pd
        import numpy as np
        
        df = pd.read_csv(dataset.file.path)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        print(f"✅ Dataset cargado correctamente")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columnas numéricas: {len(numeric_columns)}")
        
        # Verificar NaN en los datos
        nan_count = df[numeric_columns].isna().sum().sum()
        print(f"   - Valores NaN en datos: {nan_count}")
        
        # Calcular correlación y verificar
        corr = df[numeric_columns].corr()
        corr_nan = corr.isna().sum().sum()
        print(f"   - Valores NaN en correlación: {corr_nan}")
        
    except Exception as e:
        print(f"❌ Error en verificación: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando pruebas de la nueva implementación de análisis")
    print(f"   Python: {sys.version}")
    print(f"   Django: {django.VERSION}")
    test_analysis()
    print("\n✅ Pruebas completadas")