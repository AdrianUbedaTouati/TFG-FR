#!/usr/bin/env python
"""
Script de prueba final para verificar las mejoras del reporte
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
from ml_trainer.views.report_views import DatasetReportView
from django.test import RequestFactory

def test_report_with_custom_charts():
    """Probar el reporte con gráficos personalizados"""
    try:
        # Obtener un dataset
        dataset = Dataset.objects.first()
        if not dataset:
            print("❌ No se encontraron datasets")
            return
        
        print(f"✅ Probando con dataset: {dataset.name}")
        
        # Crear request factory
        factory = RequestFactory()
        
        # Datos de gráficos personalizados (simulando lo que enviaría el frontend)
        custom_charts_data = {
            'charts': {
                'chart_1': {
                    'variable': 'Temperature (C)',
                    'title': 'Análisis de Outliers',
                    'chartImage': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
                    'outlierInfo': {
                        'outlier_count': 15,
                        'outlier_percentage': 2.5
                    }
                },
                'chart_2': {
                    'variable': 'Humidity',
                    'title': 'Box Plot Detallado',
                    'chartImage': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
                }
            }
        }
        
        # Crear request POST con datos
        request = factory.post(
            f'/api/datasets/{dataset.pk}/report/',
            data=json.dumps(custom_charts_data),
            content_type='application/json'
        )
        
        # Crear vista y generar reporte
        view = DatasetReportView()
        response = view.post(request, pk=dataset.pk)
        
        if response.status_code == 200:
            # Guardar el reporte
            output_file = f"test_report_enhanced_{dataset.name}.html"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✅ Reporte generado exitosamente: {output_file}")
            print(f"   Tamaño: {len(response.content):,} bytes")
            
            # Verificar que contiene los gráficos personalizados
            content = response.content.decode('utf-8')
            if 'Analyses de Distribution Personnalisées' in content:
                print("✅ Sección de gráficos personalizados incluida")
            if 'Análisis de Outliers' in content:
                print("✅ Gráfico de outliers incluido")
            if 'Box Plot Detallado' in content:
                print("✅ Box plot incluido")
                
        else:
            print(f"❌ Error generando reporte: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Prueba Final del Reporte Mejorado ===\n")
    test_report_with_custom_charts()
    print("\n✨ Prueba completada")