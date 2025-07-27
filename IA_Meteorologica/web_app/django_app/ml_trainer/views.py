from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.http import HttpResponse
import pandas as pd
import numpy as np
import json
import mimetypes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from .models import Dataset, TrainingSession, WeatherPrediction, ModelType, NormalizationMethod, MetricType, ModelDefinition
from .serializers import DatasetSerializer, TrainingSessionSerializer, WeatherPredictionSerializer, ModelDefinitionSerializer
from .ml_utils import get_model_config, get_normalization_methods, get_metrics, train_model, make_predictions
try:
    # Intentar importar desde el módulo local primero
    from .normalization_methods import NumNorm, TextNorm, Normalizador
except ImportError:
    # Si falla, intentar desde el archivo principal
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from normalisaciones import NumNorm, TextNorm, Normalizador
    except ImportError as e:
        print(f"Error importing normalisaciones: {e}")
        # Fallback para desarrollo
        NumNorm = None
        TextNorm = None
        Normalizador = None


class DatasetListCreateView(generics.ListCreateAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer


class DatasetDetailView(generics.RetrieveDestroyAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer


class DatasetColumnsView(APIView):
    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        try:
            df = pd.read_csv(dataset.file.path)
            columns = df.columns.tolist()
            dtypes = {col: str(df[col].dtype) for col in columns}
            
            # Análisis estadístico básico
            stats = {}
            for col in columns:
                # Detectar el tipo real de datos
                dtype_str = str(df[col].dtype)
                
                # Intentar detectar tipos específicos si es object
                if dtype_str == 'object':
                    # Tomar muestra para analizar
                    sample = df[col].dropna().head(100)
                    if len(sample) > 0:
                        # Intentar parsear como fecha con formato específico
                        try:
                            # Intentar con formatos comunes de fecha
                            for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
                                try:
                                    date_parsed = pd.to_datetime(sample, format=date_format, errors='coerce')
                                    if date_parsed.notna().sum() / len(sample) > 0.8:
                                        dtype_str = 'datetime'
                                        break
                                except:
                                    continue
                        except:
                            pass
                        
                        # Si no es fecha, verificar si es numérico almacenado como string
                        if dtype_str == 'object':
                            try:
                                numeric_parsed = pd.to_numeric(sample, errors='coerce')
                                if numeric_parsed.notna().sum() / len(sample) > 0.8:
                                    dtype_str = 'numeric (stored as text)'
                            except:
                                pass
                        
                        # Si sigue siendo object, intentar identificar el tipo específico
                        if dtype_str == 'object' and len(sample) > 0:
                            # Obtener el tipo del primer valor no nulo
                            first_val = sample.iloc[0]
                            if hasattr(first_val, '__class__'):
                                python_type = type(first_val).__name__
                                dtype_str = f'object ({python_type})'
                            
                            # Detectar patrones comunes
                            if all(isinstance(v, str) for v in sample):
                                # Verificar si son URLs
                                if sample.str.match(r'^https?://').sum() / len(sample) > 0.8:
                                    dtype_str = 'object (URL)'
                                # Verificar si son emails
                                elif sample.str.contains('@').sum() / len(sample) > 0.8:
                                    dtype_str = 'object (Email)'
                                # Verificar si son códigos/IDs
                                elif sample.str.match(r'^[A-Z0-9\-]+$').sum() / len(sample) > 0.8:
                                    dtype_str = 'object (ID/Code)'
                                # Verificar si son booleanos como texto
                                elif set(sample.str.lower().unique()) <= {'true', 'false', 'yes', 'no', '1', '0'}:
                                    dtype_str = 'object (Boolean text)'
                
                col_stats = {
                    'dtype': dtype_str,
                    'null_count': int(df[col].isnull().sum()),
                    'null_percentage': float(df[col].isnull().sum() / len(df) * 100),
                    'unique_count': int(df[col].nunique()),
                }
                
                # Estadísticas para columnas numéricas
                if df[col].dtype in ['int64', 'float64']:
                    non_null_values = df[col].dropna()
                    
                    if len(non_null_values) > 0:
                        # Análisis de decimales
                        decimals_info = {}
                        if df[col].dtype == 'float64':
                            # Convertir a string para contar decimales
                            str_values = non_null_values.astype(str)
                            decimal_counts = []
                            for val in str_values.head(1000):  # Muestra de 1000 valores
                                if '.' in val and 'e' not in val.lower():  # Evitar notación científica
                                    decimal_part = val.split('.')[1]
                                    decimal_counts.append(len(decimal_part.rstrip('0')))
                            
                            if decimal_counts:
                                decimals_info = {
                                    'max_decimals': max(decimal_counts),
                                    'avg_decimals': sum(decimal_counts) / len(decimal_counts),
                                    'most_common_decimals': max(set(decimal_counts), key=decimal_counts.count)
                                }
                        
                        # Análisis de orden de magnitud
                        abs_values = non_null_values.abs()
                        abs_values = abs_values[abs_values > 0]  # Excluir ceros
                        if len(abs_values) > 0:
                            magnitude_info = {
                                'min_magnitude': int(np.floor(np.log10(abs_values.min()))),
                                'max_magnitude': int(np.floor(np.log10(abs_values.max()))),
                                'avg_magnitude': int(np.floor(np.log10(abs_values.mean())))
                            }
                        else:
                            magnitude_info = None
                        
                        col_stats.update({
                            'mean': float(non_null_values.mean()),
                            'std': float(non_null_values.std()),
                            'min': float(non_null_values.min()),
                            'max': float(non_null_values.max()),
                            'q25': float(non_null_values.quantile(0.25)),
                            'q50': float(non_null_values.quantile(0.50)),
                            'q75': float(non_null_values.quantile(0.75)),
                            'decimals_info': decimals_info,
                            'magnitude_info': magnitude_info
                        })
                    else:
                        col_stats.update({
                            'mean': None,
                            'std': None,
                            'min': None,
                            'max': None,
                            'q25': None,
                            'q50': None,
                            'q75': None,
                            'decimals_info': None,
                            'magnitude_info': None
                        })
                    
                    # Histograma para visualización
                    try:
                        data = df[col].dropna()
                        q1 = data.quantile(0.25)
                        q3 = data.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        # Identificar outliers
                        outliers = data[(data < lower_bound) | (data > upper_bound)]
                        outliers_count = len(outliers)
                        
                        # Generar histograma completo (con outliers)
                        hist_full, bins_full = np.histogram(data, bins=20)
                        
                        # Generar histograma sin outliers
                        data_no_outliers = data[(data >= lower_bound) & (data <= upper_bound)]
                        hist_no_outliers, bins_no_outliers = np.histogram(data_no_outliers, bins=20)
                        
                        col_stats['histogram'] = {
                            'counts': hist_full.tolist(),
                            'bins': bins_full.tolist(),
                            'counts_no_outliers': hist_no_outliers.tolist(),
                            'bins_no_outliers': bins_no_outliers.tolist(),
                            'outliers_info': {
                                'outlier_count': int(outliers_count),
                                'outlier_percentage': float(outliers_count / len(data) * 100) if len(data) > 0 else 0,
                                'lower_bound': float(lower_bound),
                                'upper_bound': float(upper_bound),
                                'q1': float(q1),
                                'q3': float(q3),
                                'iqr': float(iqr)
                            }
                        }
                    except:
                        col_stats['histogram'] = None
                
                # Top valores para columnas categóricas
                elif df[col].dtype == 'object':
                    value_counts = df[col].value_counts().head(10)
                    col_stats['top_values'] = {
                        'values': value_counts.index.tolist(),
                        'counts': value_counts.values.tolist()
                    }
                
                stats[col] = col_stats
            
            # Preview con manejo de errores
            preview_data = {}
            for col in columns:
                try:
                    preview_data[col] = df[col].head(10).fillna('').astype(str).tolist()
                except:
                    preview_data[col] = ['Error'] * min(10, len(df))
            
            return Response({
                'columns': columns,
                'dtypes': dtypes,
                'shape': df.shape,
                'stats': stats,
                'preview': preview_data,
                'total_null_count': int(df.isnull().sum().sum()),
                'memory_usage': float(df.memory_usage(deep=True).sum() / 1024 / 1024)  # MB
            })
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )


class DatasetColumnDetailsView(APIView):
    def get(self, request, pk, column_name):
        dataset = get_object_or_404(Dataset, pk=pk)
        try:
            df = pd.read_csv(dataset.file.path)
            
            if column_name not in df.columns:
                return Response(
                    {'error': f'Column {column_name} not found'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            col_data = df[column_name]
            
            # Get value counts for frequency information, sorted by frequency (descending)
            value_counts = col_data.value_counts(dropna=False, sort=True)
            total_count = len(col_data)
            
            # Create frequency data
            frequency_data = []
            for value, count in value_counts.items():
                # Handle NaN values
                if pd.isna(value):
                    display_value = 'NaN'
                elif col_data.dtype in ['int64', 'float64']:
                    display_value = float(value)
                else:
                    display_value = str(value)
                
                frequency_data.append({
                    'value': display_value,
                    'count': int(count),
                    'percentage': float(count / total_count * 100)
                })
            
            response_data = {
                'column': column_name,
                'dtype': str(col_data.dtype),
                'unique_count': len(col_data.dropna().unique()),
                'frequency_data': frequency_data,
                'null_count': int(col_data.isnull().sum()),
                'total_count': total_count
            }
            
            # For numeric columns, calculate outlier information
            if col_data.dtype in ['int64', 'float64']:
                non_null = col_data.dropna()
                if len(non_null) > 0:
                    q1 = non_null.quantile(0.25)
                    q3 = non_null.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - (1.5 * iqr)
                    upper_bound = q3 + (1.5 * iqr)
                    
                    outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
                    
                    # Convert outliers to list and sort
                    outlier_list = outliers.tolist() if len(outliers) > 0 else []
                    outlier_list.sort()
                    
                    response_data.update({
                        'outlier_info': {
                            'q1': float(q1),
                            'q3': float(q3),
                            'iqr': float(iqr),
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound),
                            'outlier_count': len(outliers),
                            'outlier_percentage': float(len(outliers) / len(non_null) * 100),
                            'outlier_values': outlier_list  # Now returns all outliers sorted
                        }
                    })
            
            return Response(response_data)
            
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )


class DatasetReportView(APIView):
    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        
        try:
            # Leer y analizar el dataset
            df = pd.read_csv(dataset.file.path)
            
            # Generar HTML del reporte
            html_content = self.generate_html_report(dataset, df, None)
            
            # Configurar la respuesta HTTP
            response = HttpResponse(content_type='text/html; charset=utf-8')
            response['Content-Disposition'] = f'attachment; filename="rapport_{dataset.name}_{pd.Timestamp.now().strftime("%Y%m%d")}.html"'
            response.write(html_content.encode('utf-8'))
            
            return response
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"Error en DatasetReportView: {error_detail}")
            # Usar HttpResponse en lugar de Response para errores
            error_response = HttpResponse(
                f'<html><body><h1>Error</h1><p>{str(e)}</p><pre>{error_detail}</pre></body></html>',
                content_type='text/html',
                status=500
            )
            return error_response
    
    def post(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        
        try:
            # Leer y analizar el dataset
            df = pd.read_csv(dataset.file.path)
            
            # Obtener datos de los gráficos del request
            chart_data = request.data.get('charts', {})
            
            # Generar HTML del reporte con los gráficos
            html_content = self.generate_html_report(dataset, df, chart_data)
            
            # Configurar la respuesta HTTP
            response = HttpResponse(content_type='text/html; charset=utf-8')
            response['Content-Disposition'] = f'attachment; filename="rapport_{dataset.name}_{pd.Timestamp.now().strftime("%Y%m%d")}.html"'
            response.write(html_content.encode('utf-8'))
            
            return response
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"Error en DatasetReportView POST: {error_detail}")
            # Usar HttpResponse en lugar de Response para errores
            error_response = HttpResponse(
                f'<html><body><h1>Error</h1><p>{str(e)}</p><pre>{error_detail}</pre></body></html>',
                content_type='text/html',
                status=500
            )
            return error_response
    
    def generate_html_report(self, dataset, df, chart_data=None):
        """Genera un reporte HTML con el análisis del dataset"""
        
        try:
            # Calcular estadísticas básicas
            shape = df.shape
            total_nulls = int(df.isnull().sum().sum())
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            # Generar tabla de estadísticas por columna
            column_stats = []
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                stats = {
                    'column': col,
                    'dtype': dtype_str,
                    'null_count': int(df[col].isnull().sum()),
                    'null_percentage': float(df[col].isnull().sum() / len(df) * 100),
                    'unique_count': int(df[col].nunique())
                }
                
                # Get value counts for all columns
                value_counts = df[col].value_counts(dropna=False).head(10)
                if len(value_counts) > 0:
                    stats['top_values'] = {
                        'values': [str(v) if not pd.isna(v) else 'NaN' for v in value_counts.index.tolist()],
                        'counts': value_counts.values.tolist()
                    }
                
                if df[col].dtype in ['int64', 'float64']:
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        stats.update({
                            'mean': float(non_null.mean()),
                            'std': float(non_null.std()),
                            'min': float(non_null.min()),
                            'max': float(non_null.max()),
                            'q25': float(non_null.quantile(0.25)),
                            'q50': float(non_null.quantile(0.50)),
                            'q75': float(non_null.quantile(0.75))
                        })
                
                column_stats.append(stats)
        
            # Generar HTML
            html = f'''
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rapport d'Analyse - {dataset.name}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
            <style>
                :root {{
                    --primary-color: #00d4ff;
                    --secondary-color: #0099ff;
                    --accent-color: #ff00ff;
                    --dark-color: #0a0e27;
                    --light-color: #f0f9ff;
                    --neon-glow: 0 0 20px rgba(0, 212, 255, 0.8);
                    --neon-glow-intense: 0 0 40px rgba(0, 212, 255, 1);
                }}
                
                * {{
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Poppins', sans-serif;
                    background: #0a0e27;
                    min-height: 100vh;
                    position: relative;
                    overflow-x: hidden;
                    color: var(--light-color);
                    margin: 0;
                    padding: 0;
                }}
                
                body::before {{
                    content: '';
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: 
                        radial-gradient(circle at 20% 50%, rgba(0, 212, 255, 0.3) 0%, transparent 50%),
                        radial-gradient(circle at 80% 50%, rgba(255, 0, 255, 0.2) 0%, transparent 50%),
                        radial-gradient(circle at 50% 100%, rgba(0, 153, 255, 0.2) 0%, transparent 50%);
                    pointer-events: none;
                    animation: backgroundPulse 10s ease-in-out infinite;
                }}
                
                @keyframes backgroundPulse {{
                    0%, 100% {{ opacity: 0.3; }}
                    50% {{ opacity: 0.5; }}
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    position: relative;
                    z-index: 1;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: var(--primary-color);
                }}
                
                /* Header like the page */
                .page-header {{
                    background: rgba(0, 212, 255, 0.05);
                    border: 1px solid rgba(0, 212, 255, 0.3);
                    border-radius: 20px;
                    padding: 30px;
                    margin-bottom: 40px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                }}
                
                .page-header::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(0, 212, 255, 0.1) 0%, transparent 70%);
                    animation: rotate 20s linear infinite;
                }}
                
                @keyframes rotate {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
                
                .page-header h1 {{
                    font-weight: 700;
                    font-size: 3rem;
                    margin-bottom: 10px;
                    text-shadow: var(--neon-glow);
                    position: relative;
                    z-index: 1;
                }}
                
                .page-header p {{
                    color: rgba(240, 249, 255, 0.8);
                    font-size: 1.2rem;
                    margin: 0;
                    position: relative;
                    z-index: 1;
                }}
                /* Stats Grid - Matching page styles exactly */
                .row {{
                    margin-bottom: 30px;
                }}
                
                .stat-card {{
                    background: rgba(0, 212, 255, 0.05);
                    border: 1px solid rgba(0, 212, 255, 0.3);
                    border-radius: 20px;
                    padding: 30px 20px;
                    height: 100%;
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }}
                
                .stat-card::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(0, 212, 255, 0.05) 0%, transparent 70%);
                    transition: all 0.3s ease;
                    opacity: 0;
                }}
                
                .stat-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: var(--neon-glow);
                    border-color: var(--primary-color);
                }}
                
                .stat-card:hover::before {{
                    opacity: 1;
                }}
                
                .stat-card .card-body {{
                    position: relative;
                    z-index: 1;
                }}
                
                .stat-card i {{
                    font-size: 2rem;
                    color: var(--primary-color);
                    text-shadow: var(--neon-glow);
                }}
                
                .stat-card h3 {{
                    font-size: 2rem;
                    font-weight: 700;
                    margin: 15px 0 5px 0;
                    color: white !important;
                }}
                
                .stat-card .text-muted {{
                    color: rgba(240, 249, 255, 0.6) !important;
                    font-size: 0.9rem;
                }}
                /* Tables matching page style */
                .table {{
                    background: transparent !important;
                    color: var(--light-color);
                    margin-top: 20px;
                }}
                
                .table thead th {{
                    background-color: rgba(0, 212, 255, 0.1);
                    color: var(--primary-color);
                    border-color: rgba(0, 212, 255, 0.3);
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    padding: 15px;
                }}
                
                .table tbody tr {{
                    border-color: rgba(0, 212, 255, 0.1);
                    transition: all 0.3s ease;
                }}
                
                .table tbody tr:hover {{
                    background-color: rgba(0, 212, 255, 0.05);
                    transform: translateX(5px);
                }}
                
                .table td {{
                    border-color: rgba(0, 212, 255, 0.1);
                    padding: 12px 15px;
                    vertical-align: middle;
                }}
                
                .numeric {{
                    text-align: right;
                    font-family: 'Monaco', 'Consolas', monospace;
                }}
                .footer {{
                    margin-top: 50px;
                    text-align: center;
                    color: #64748b;
                    font-size: 14px;
                    padding: 20px;
                    border-top: 1px solid rgba(0, 212, 255, 0.2);
                }}
                .preview-section {{
                    margin-top: 30px;
                    background: rgba(30, 41, 59, 0.8);
                    backdrop-filter: blur(10px);
                    padding: 25px;
                    border-radius: 10px;
                    border: 1px solid rgba(0, 212, 255, 0.3);
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                }}
                .preview-section h2 {{
                    margin-top: 0;
                    color: #00d4ff;
                    border-bottom: 2px solid rgba(0, 212, 255, 0.3);
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                .preview-table {{
                    width: 100%;
                    background: rgba(15, 17, 20, 0.5);
                    border-radius: 8px;
                    overflow: hidden;
                }}
                .preview-table th {{
                    background: rgba(0, 212, 255, 0.1);
                    color: #00d4ff;
                }}
                .preview-table td {{
                    padding: 10px;
                    border: 1px solid rgba(0, 212, 255, 0.1);
                }}
                .null-badge {{
                    background-color: #ef4444;
                    color: white;
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-size: 12px;
                }}
                .type-badge {{
                    background-color: #60a5fa;
                    color: white;
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-size: 12px;
                    margin-left: 5px;
                }}
                .chart-container {{
                    background: rgba(30, 41, 59, 0.8);
                    backdrop-filter: blur(10px);
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid rgba(0, 212, 255, 0.3);
                    margin-bottom: 30px;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                }}
                .chart-container h3 {{
                    color: #00d4ff;
                    margin-bottom: 20px;
                    font-size: 1.3em;
                    border-bottom: 2px solid rgba(0, 212, 255, 0.3);
                    padding-bottom: 10px;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
                }}
                .icon {{
                    margin-right: 8px;
                    color: #00d4ff;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="page-header">
                    <h1>Rapport d'Analyse du Dataset</h1>
                    <p>{dataset.name} - Généré le {pd.Timestamp.now().strftime("%d/%m/%Y à %H:%M")}</p>
                </div>
                
                <!-- Descriptions du dataset -->
                <div class="stat-card mb-4">
                    <div class="card-body">
                        <h3><i class="bi bi-info-circle icon"></i>Informations du Dataset</h3>
                        {f'<p><strong>Description courte:</strong> {dataset.short_description}</p>' if hasattr(dataset, 'short_description') and dataset.short_description else ''}
                        {f'<p><strong>Description détaillée:</strong> {dataset.long_description}</p>' if hasattr(dataset, 'long_description') and dataset.long_description else ''}
                        {f'<p><strong>Dataset normalisé depuis:</strong> {dataset.parent_dataset.name}</p>' if hasattr(dataset, 'parent_dataset') and dataset.parent_dataset else ''}
                        <p><strong>Date de création:</strong> {dataset.uploaded_at.strftime("%d/%m/%Y à %H:%M") if dataset.uploaded_at else "N/A"}</p>
                    </div>
                </div>
                
                <!-- Stats Grid like in the page -->
                <div class="row">
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="card-body text-center">
                                <i class="bi bi-table"></i>
                                <h3>{shape[0]:,} × {shape[1]}</h3>
                                <p class="text-muted mb-0">Dimensions</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="card-body text-center">
                                <i class="bi bi-hdd-fill"></i>
                                <h3>{memory_usage:.2f} MB</h3>
                                <p class="text-muted mb-0">Taille en mémoire</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="card-body text-center">
                                <i class="bi bi-exclamation-triangle-fill"></i>
                                <h3>{total_nulls:,}</h3>
                                <p class="text-muted mb-0">Valeurs manquantes</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="card-body text-center">
                                <i class="bi bi-calendar-check-fill"></i>
                                <h3>{dataset.uploaded_at.strftime("%d/%m/%Y") if dataset.uploaded_at else "N/A"}</h3>
                                <p class="text-muted mb-0">Date d'importation</p>
                            </div>
                        </div>
                    </div>
                </div>
            
            <div class="mt-5">
                <h2>Analyse des Variables</h2>
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Type</th>
                            <th>Valeurs uniques</th>
                            <th>Valeurs nulles</th>
                            <th>Statistiques</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
        '''
        
            for stats in column_stats:
                null_badge = f'<span class="null-badge">{stats["null_count"]} ({stats["null_percentage"]:.1f}%)</span>' if stats['null_count'] > 0 else '0'
            
                if 'mean' in stats:
                    stat_info = f'''
                        μ={stats["mean"]:.2f}, σ={stats["std"]:.2f}<br>
                        Min: {stats["min"]:.2f}, Max: {stats["max"]:.2f}<br>
                        Q1: {stats["q25"]:.2f}, Médiane: {stats["q50"]:.2f}, Q3: {stats["q75"]:.2f}
                    '''
                else:
                    stat_info = f'{stats["unique_count"]} valeurs distinctes'
            
                dtype_display = 'Numérique' if stats['dtype'] in ['int64', 'float64'] else 'Texte'
            
                html += f'''
                        <tr>
                            <td><strong>{stats["column"]}</strong></td>
                            <td><span class="type-badge">{dtype_display}</span></td>
                            <td class="numeric">{stats["unique_count"]}</td>
                            <td class="numeric">{null_badge}</td>
                            <td>{stat_info}</td>
                            <td>
                                <button class="btn btn-sm btn-outline-info" onclick="showUniqueValues('{stats["column"]}')" title="Voir valeurs uniques">
                                    <i class="bi bi-list-ul"></i>
                                </button>
                                {f'''
                                <button class="btn btn-sm btn-outline-warning ms-1" onclick="showOutliers('{stats["column"]}')" title="Voir outliers">
                                    <i class="bi bi-exclamation-triangle"></i>
                                </button>
                                ''' if stats['dtype'] in ['int64', 'float64'] else ''}
                            </td>
                        </tr>
                '''
        
            html += '''
                    </tbody>
                </table>
            </div>
            
            <div class="mt-5">
                <div class="preview-section">
                    <h2>Aperçu des Données (10 premières lignes)</h2>
                    <div style="overflow-x: auto;">
            '''
            
            # Ajouter l'aperçu des données
            try:
                preview_html = df.head(10).to_html(classes='table table-hover preview-table', index=False, escape=True)
                html += preview_html
            except Exception as e:
                html += f'<p>Erreur lors de la génération de l\'aperçu: {str(e)}</p>'
            
            html += '''
                    </div>
                </div>
            </div>
            '''
            
            
            # Ajouter les graphiques si disponibles
            if chart_data:
                html += '''
                <div class="preview-section">
                    <h2>Analyses de Distribution</h2>
                '''
                
                for analysis_id, chart_info in chart_data.items():
                    variable_name = chart_info.get('variable', 'Variable')
                    chart_image = chart_info.get('chartImage', '')
                    outlier_info = chart_info.get('outlierInfo', None)
                    
                    outlier_html = ''
                    if outlier_info and outlier_info.get('removed_count', 0) > 0:
                        percentage = (outlier_info['removed_count'] / outlier_info['total_before'] * 100)
                        outlier_html = f'''
                        <div class="alert alert-warning mt-3">
                            <i class="bi bi-exclamation-triangle"></i> 
                            <strong>Outliers exclus:</strong> {outlier_info['removed_count']} valeurs ({percentage:.1f}%) ont été exclues du graphique.<br>
                            <small>Valeurs en dehors de l'intervalle [{outlier_info['lower_bound']:.2f}, {outlier_info['upper_bound']:.2f}] (méthode IQR)</small>
                        </div>
                        '''
                    
                    html += f'''
                    <div class="chart-container">
                        <h3><span class="icon">📊</span>Distribution de {variable_name}</h3>
                        <img src="{chart_image}" alt="Histogramme de {variable_name}" />
                        {outlier_html}
                    </div>
                    '''
                
                html += '''
                </div>
                '''
            
            html += '''
                <div class="footer">
                    <p>Rapport généré automatiquement par IA Météorologique</p>
                    <p>© 2024 - Tous droits réservés</p>
                </div>
            </div>
            
            <!-- Modal containers -->
            <div id="modalContainer"></div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://unpkg.com/@sgratzl/chartjs-chart-boxplot@3"></script>
            <script>
                // Store column statistics data
                const columnStats = ''' + json.dumps(column_stats) + ''';
                
                // Function to show unique values modal
                function showUniqueValues(columnName) {
                    const stats = columnStats.find(s => s.column === columnName);
                    if (!stats) return;
                    
                    let modalContent = `
                        <div class="modal fade" id="uniqueValuesModal" tabindex="-1">
                            <div class="modal-dialog modal-lg">
                                <div class="modal-content" style="background: var(--dark-color); border: 1px solid var(--primary-color);">
                                    <div class="modal-header" style="border-bottom: 1px solid rgba(0, 212, 255, 0.3);">
                                        <h5 class="modal-title" style="color: var(--primary-color);">
                                            <i class="bi bi-list-ul"></i> Analyse des Valeurs - "${columnName}"
                                        </h5>
                                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                                    </div>
                                    <div class="modal-body">
                                        <div class="mb-3">
                                            <span class="badge bg-info">Total: ${stats.unique_count || 0} valeurs uniques</span>
                                            <span class="badge bg-warning ms-2">Nuls: ${stats.null_count || 0}</span>
                                        </div>
                    `;
                    
                    if (stats.top_values) {
                        modalContent += `
                                        <h6 class="text-primary mb-3">Distribution des fréquences (Top 10):</h6>
                                        <table class="table table-sm table-hover">
                                            <thead>
                                                <tr>
                                                    <th>Valeur</th>
                                                    <th class="text-end">Fréquence</th>
                                                    <th class="text-end">Pourcentage</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                        `;
                        
                        const total = stats.top_values.counts.reduce((a, b) => a + b, 0);
                        for (let i = 0; i < stats.top_values.values.length; i++) {
                            const value = stats.top_values.values[i];
                            const count = stats.top_values.counts[i];
                            const percentage = (count / total * 100).toFixed(1);
                            modalContent += `
                                                <tr>
                                                    <td>${value}</td>
                                                    <td class="text-end">${count.toLocaleString('fr-FR')}</td>
                                                    <td class="text-end">${percentage}%</td>
                                                </tr>
                            `;
                        }
                        
                        modalContent += `
                                            </tbody>
                                        </table>
                        `;
                    }
                    
                    modalContent += `
                                    </div>
                                    <div class="modal-footer" style="border-top: 1px solid rgba(0, 212, 255, 0.3);">
                                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                                            <i class="bi bi-x-circle"></i> Fermer
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // Remove existing modal if any
                    const existingModal = document.getElementById('uniqueValuesModal');
                    if (existingModal) existingModal.remove();
                    
                    // Add modal to container
                    document.getElementById('modalContainer').innerHTML = modalContent;
                    
                    // Show modal
                    const modal = new bootstrap.Modal(document.getElementById('uniqueValuesModal'));
                    modal.show();
                }
                
                // Function to show outliers modal
                function showOutliers(columnName) {
                    const stats = columnStats.find(s => s.column === columnName);
                    if (!stats || !stats.q25 || !stats.q75) return;
                    
                    const q1 = stats.q25;
                    const q3 = stats.q75;
                    const iqr = q3 - q1;
                    const lowerBound = q1 - 1.5 * iqr;
                    const upperBound = q3 + 1.5 * iqr;
                    
                    let modalContent = `
                        <div class="modal fade" id="outliersModal" tabindex="-1">
                            <div class="modal-dialog modal-lg">
                                <div class="modal-content" style="background: var(--dark-color); border: 1px solid var(--primary-color);">
                                    <div class="modal-header" style="border-bottom: 1px solid rgba(0, 212, 255, 0.3);">
                                        <h5 class="modal-title" style="color: var(--primary-color);">
                                            <i class="bi bi-exclamation-triangle"></i> Analyse des Outliers - "${columnName}"
                                        </h5>
                                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                                    </div>
                                    <div class="modal-body">
                                        <div class="row mb-4">
                                            <div class="col-md-6">
                                                <div class="card bg-dark border-info">
                                                    <div class="card-body" style="color: var(--light-color);">
                                                        <h6 class="text-info">Méthode IQR (Rango Intercuartílico)</h6>
                                                        <p class="mb-2" style="color: #e2e8f0;"><strong style="color: #60a5fa;">Q1 (25%):</strong> ${q1.toFixed(2)}</p>
                                                        <p class="mb-2" style="color: #e2e8f0;"><strong style="color: #60a5fa;">Q3 (75%):</strong> ${q3.toFixed(2)}</p>
                                                        <p class="mb-2" style="color: #e2e8f0;"><strong style="color: #60a5fa;">IQR:</strong> ${iqr.toFixed(2)}</p>
                                                        <hr style="border-color: rgba(0, 212, 255, 0.3);">
                                                        <p class="mb-2" style="color: #e2e8f0;"><strong style="color: #60a5fa;">Limite inférieure:</strong> ${lowerBound.toFixed(2)}</p>
                                                        <p class="mb-0" style="color: #e2e8f0;"><strong style="color: #60a5fa;">Limite supérieure:</strong> ${upperBound.toFixed(2)}</p>
                                                    </div>
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <div class="card bg-dark border-warning">
                                                    <div class="card-body" style="color: var(--light-color);">
                                                        <h6 class="text-warning">Résumé des Outliers</h6>
                                                        <p class="mb-2" style="color: #e2e8f0;">
                                                            <i class="bi bi-arrow-down-circle" style="color: #fbbf24;"></i> 
                                                            <strong style="color: #fbbf24;">Valeurs en dessous:</strong> 
                                                            ${stats.min < lowerBound ? 
                                                                `<span class="text-danger">Oui (min: ${stats.min.toFixed(2)})</span>` : 
                                                                '<span class="text-success">Non</span>'}
                                                        </p>
                                                        <p class="mb-2" style="color: #e2e8f0;">
                                                            <i class="bi bi-arrow-up-circle" style="color: #fbbf24;"></i> 
                                                            <strong style="color: #fbbf24;">Valeurs au-dessus:</strong> 
                                                            ${stats.max > upperBound ? 
                                                                `<span class="text-danger">Oui (max: ${stats.max.toFixed(2)})</span>` : 
                                                                '<span class="text-success">Non</span>'}
                                                        </p>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <h6 class="text-primary mb-3">Diagrame de Boîte (Box Plot)</h6>
                                        <div class="text-center mb-4">
                                            <canvas id="outlierBoxPlot" height="200"></canvas>
                                        </div>
                                        
                                        <div class="alert alert-info">
                                            <i class="bi bi-info-circle"></i> 
                                            <strong>Intervalle de valeurs normales:</strong> [${lowerBound.toFixed(2)}, ${upperBound.toFixed(2)}]
                                            <br>
                                            <small>Les valeurs en dehors de cet intervalle sont considérées comme des outliers potentiels.</small>
                                        </div>
                                    </div>
                                    <div class="modal-footer" style="border-top: 1px solid rgba(0, 212, 255, 0.3);">
                                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                                            <i class="bi bi-x-circle"></i> Fermer
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // Remove existing modal if any
                    const existingModal = document.getElementById('outliersModal');
                    if (existingModal) existingModal.remove();
                    
                    // Add modal to container
                    document.getElementById('modalContainer').innerHTML = modalContent;
                    
                    // Show modal
                    const modal = new bootstrap.Modal(document.getElementById('outliersModal'));
                    modal.show();
                    
                    // Create box plot after modal is shown
                    setTimeout(() => {
                        const canvas = document.getElementById('outlierBoxPlot');
                        if (canvas) {
                            const ctx = canvas.getContext('2d');
                            
                            // Prepare data for box plot visualization
                            // Use the actual data within IQR bounds for better visualization
                            const median = stats.q50 || ((q1 + q3) / 2);
                            
                            // For whiskers, find actual min/max within bounds
                            // This prevents extreme outliers from compressing the box plot
                            let whiskerMin = q1;
                            let whiskerMax = q3;
                            
                            // Find actual min value that's >= lowerBound
                            if (stats.min >= lowerBound) {
                                whiskerMin = stats.min;
                            } else {
                                // Use lower bound as whisker if there are outliers below
                                whiskerMin = lowerBound;
                            }
                            
                            // Find actual max value that's <= upperBound
                            if (stats.max <= upperBound) {
                                whiskerMax = stats.max;
                            } else {
                                // Use upper bound as whisker if there are outliers above
                                whiskerMax = upperBound;
                            }
                            
                            new Chart(ctx, {
                                type: 'boxplot',
                                data: {
                                    labels: [columnName],
                                    datasets: [{
                                        label: 'Distribution',
                                        backgroundColor: 'rgba(0, 212, 255, 0.3)',
                                        borderColor: 'rgba(0, 212, 255, 1)',
                                        borderWidth: 2,
                                        padding: 10,
                                        itemRadius: 0,
                                        outlierColor: '#ff0000',
                                        data: [[
                                            whiskerMin,
                                            q1,
                                            median,
                                            q3,
                                            whiskerMax
                                        ]]
                                    }]
                                },
                                options: {
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    plugins: {
                                        legend: {
                                            display: false
                                        },
                                        title: {
                                            display: true,
                                            text: 'Distribution sans outliers',
                                            color: '#00d4ff'
                                        }
                                    },
                                    scales: {
                                        y: {
                                            ticks: {
                                                color: '#f0f9ff'
                                            },
                                            grid: {
                                                color: 'rgba(255, 255, 255, 0.1)'
                                            }
                                        },
                                        x: {
                                            ticks: {
                                                color: '#f0f9ff'
                                            },
                                            grid: {
                                                color: 'rgba(255, 255, 255, 0.1)'
                                            }
                                        }
                                    }
                                }
                            });
                        }
                    }, 200);
                }
            </script>
        </body>
        </html>
        '''
        
            return html
        except Exception as e:
            # Si hay un error, devolver un HTML simple con el error
            return f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Erreur dans le rapport</title>
            </head>
            <body>
                <h1>Erreur lors de la génération du rapport</h1>
                <p>{str(e)}</p>
            </body>
            </html>
            '''


class DatasetDownloadView(APIView):
    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        
        try:
            # Leer el archivo CSV
            file_path = dataset.file.path
            
            # Configurar la respuesta HTTP
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{dataset.name}.csv"'
            
            # Leer y escribir el archivo
            with open(file_path, 'rb') as f:
                response.write(f.read())
            
            return response
            
        except Exception as e:
            return Response(
                {'error': f'Error al descargar el archivo: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TrainingSessionListCreateView(generics.ListCreateAPIView):
    queryset = TrainingSession.objects.all()
    serializer_class = TrainingSessionSerializer


class TrainingSessionDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = TrainingSession.objects.all()
    serializer_class = TrainingSessionSerializer


class TrainModelView(APIView):
    def post(self, request, pk):
        session = get_object_or_404(TrainingSession, pk=pk)
        
        try:
            # Start training in background (you might want to use Celery for this)
            train_model(session)
            
            return Response({
                'message': 'Training started successfully',
                'session_id': session.id
            })
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )


class ModelConfigView(APIView):
    def get(self, request):
        configs = {}
        for model_type in ModelType:
            configs[model_type.value] = get_model_config(model_type.value)
        
        return Response(configs)


class NormalizationMethodsView(APIView):
    def get(self, request, model_type):
        methods = get_normalization_methods(model_type)
        return Response(methods)


class MetricsView(APIView):
    def get(self, request, model_type):
        metrics = get_metrics(model_type)
        return Response(metrics)


class PredictionListCreateView(generics.ListCreateAPIView):
    queryset = WeatherPrediction.objects.all()
    serializer_class = WeatherPredictionSerializer


class PredictionMapView(APIView):
    def get(self, request):
        session_id = request.query_params.get('session_id')
        date = request.query_params.get('date')
        
        if not session_id or not date:
            return Response(
                {'error': 'session_id and date are required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        predictions = WeatherPrediction.objects.filter(
            training_session_id=session_id,
            prediction_date=date
        )
        
        serializer = WeatherPredictionSerializer(predictions, many=True)
        return Response(serializer.data)


class PredictView(APIView):
    def post(self, request):
        session_id = request.data.get('session_id')
        input_data = request.FILES.get('input_data')
        
        if not session_id or not input_data:
            return Response(
                {'error': 'session_id and input_data are required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        session = get_object_or_404(TrainingSession, pk=session_id)
        
        try:
            predictions = make_predictions(session, input_data)
            return Response(predictions)
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )


class DatasetVariableAnalysisView(APIView):
    def get(self, request, pk, column_name):
        dataset = get_object_or_404(Dataset, pk=pk)
        analysis_type = request.query_params.get('type', 'histogram')
        
        try:
            df = pd.read_csv(dataset.file.path)
            
            if column_name not in df.columns:
                return Response(
                    {'error': f'Column {column_name} not found'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            col_data = df[column_name].dropna()
            
            if analysis_type == 'outlier_map':
                return self.generate_outlier_map(col_data, column_name)
            elif analysis_type == 'boxplot':
                return self.generate_boxplot(col_data, column_name)
            elif analysis_type == 'scatter':
                # Necesita una segunda variable
                second_column = request.query_params.get('second_column')
                if not second_column or second_column not in df.columns:
                    return Response(
                        {'error': 'second_column is required for scatter plot'}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
                return self.generate_scatter_plot(df, column_name, second_column)
            else:
                return Response(
                    {'error': f'Unknown analysis type: {analysis_type}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
                
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )
    
    def generate_outlier_map(self, data, column_name):
        """Genera un mapa de outliers para datos numéricos"""
        if data.dtype not in ['int64', 'float64']:
            return Response(
                {'error': 'Outlier map only available for numeric columns'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Calcular outliers usando IQR
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Identificar outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        normal_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        # Crear visualización
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Distribución con outliers marcados
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(data)), data, alpha=0.5, s=20)
        outlier_indices = data[(data < lower_bound) | (data > upper_bound)].index
        plt.scatter(outlier_indices, outliers, color='red', s=50, label='Outliers', zorder=5)
        plt.axhline(y=lower_bound, color='orange', linestyle='--', label='Lower bound')
        plt.axhline(y=upper_bound, color='orange', linestyle='--', label='Upper bound')
        plt.xlabel('Index')
        plt.ylabel(column_name)
        plt.title(f'Outlier Map - {column_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Box plot con outliers
        plt.subplot(1, 2, 2)
        box_plot = plt.boxplot([data], labels=[column_name], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][0].set_alpha(0.7)
        
        # Marcar outliers individuales
        for outlier in outliers:
            plt.scatter(1, outlier, color='red', s=50, alpha=0.7)
        
        plt.ylabel(column_name)
        plt.title(f'Box Plot con Outliers - {column_name}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return Response({
            'analysis_type': 'outlier_map',
            'column': column_name,
            'image': f'data:image/png;base64,{image_base64}',
            'statistics': {
                'total_values': len(data),
                'outlier_count': len(outliers),
                'outlier_percentage': float(len(outliers) / len(data) * 100),
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_values': outliers.tolist()[:100]  # Limitar a 100 para no sobrecargar
            }
        })
    
    def generate_boxplot(self, data, column_name):
        """Genera un boxplot detallado"""
        if data.dtype not in ['int64', 'float64']:
            return Response(
                {'error': 'Boxplot only available for numeric columns'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Crear figura
        plt.figure(figsize=(10, 6))
        
        # Crear boxplot con estilo
        bp = plt.boxplot([data], labels=[column_name], patch_artist=True,
                        notch=True, showmeans=True, meanline=True)
        
        # Personalizar colores
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        bp['medians'][0].set_color('red')
        bp['medians'][0].set_linewidth(2)
        bp['means'][0].set_color('green')
        bp['means'][0].set_linewidth(2)
        
        # Agregar anotaciones
        median = data.median()
        mean = data.mean()
        plt.text(1.1, median, f'Median: {median:.2f}', va='center')
        plt.text(1.1, mean, f'Mean: {mean:.2f}', va='center', color='green')
        
        plt.ylabel(column_name)
        plt.title(f'Box Plot Detallado - {column_name}')
        plt.grid(True, alpha=0.3)
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return Response({
            'analysis_type': 'boxplot',
            'column': column_name,
            'image': f'data:image/png;base64,{image_base64}',
            'statistics': {
                'min': float(data.min()),
                'q1': float(data.quantile(0.25)),
                'median': float(median),
                'mean': float(mean),
                'q3': float(data.quantile(0.75)),
                'max': float(data.max()),
                'std': float(data.std()),
                'variance': float(data.var())
            }
        })
    
    def generate_scatter_plot(self, df, column1, column2):
        """Genera un scatter plot entre dos variables"""
        data1 = df[column1].dropna()
        data2 = df[column2].dropna()
        
        # Asegurar que ambas columnas sean numéricas
        if data1.dtype not in ['int64', 'float64'] or data2.dtype not in ['int64', 'float64']:
            return Response(
                {'error': 'Scatter plot only available for numeric columns'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Obtener datos comunes (sin NaN en ninguna)
        common_index = data1.index.intersection(data2.index)
        data1 = data1.loc[common_index]
        data2 = data2.loc[common_index]
        
        # Crear figura
        plt.figure(figsize=(10, 8))
        
        # Scatter plot con regresión
        plt.scatter(data1, data2, alpha=0.5, s=30)
        
        # Añadir línea de regresión
        z = np.polyfit(data1, data2, 1)
        p = np.poly1d(z)
        plt.plot(data1.sort_values(), p(data1.sort_values()), "r--", alpha=0.8, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Calcular correlación
        correlation = data1.corr(data2)
        
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.title(f'Scatter Plot: {column1} vs {column2}\nCorrelación: {correlation:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return Response({
            'analysis_type': 'scatter',
            'columns': [column1, column2],
            'image': f'data:image/png;base64,{image_base64}',
            'statistics': {
                'correlation': float(correlation),
                'data_points': len(data1),
                'regression_slope': float(z[0]),
                'regression_intercept': float(z[1])
            }
        })


class DatasetGeneralAnalysisView(APIView):
    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        analysis_type = request.query_params.get('type', 'correlation')
        
        try:
            df = pd.read_csv(dataset.file.path)
            
            if analysis_type == 'correlation':
                return self.generate_correlation_matrix(df)
            elif analysis_type == 'pca':
                return self.generate_pca_analysis(df)
            elif analysis_type == 'lasso':
                target_column = request.query_params.get('target')
                if not target_column:
                    # Si no se proporciona target, usar la última columna numérica
                    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    if len(numeric_columns) < 2:
                        return Response(
                            {'error': 'At least 2 numeric columns are required for LASSO analysis'}, 
                            status=status.HTTP_400_BAD_REQUEST
                        )
                    # Usar la última columna numérica como target por defecto
                    target_column = numeric_columns[-1]
                return self.generate_lasso_analysis(df, target_column)
            else:
                return Response(
                    {'error': f'Unknown analysis type: {analysis_type}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
                
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )
    
    def generate_correlation_matrix(self, df):
        """Genera una matriz de correlación para variables numéricas"""
        # Seleccionar solo columnas numéricas
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_columns) < 2:
            return Response(
                {'error': 'At least 2 numeric columns are required for correlation matrix'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Calcular matriz de correlación
        corr_matrix = df[numeric_columns].corr()
        
        # Crear figura
        plt.figure(figsize=(12, 10))
        
        # Heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, square=True,
                    linewidths=1, cbar_kws={"shrink": .8})
        
        plt.title('Matriz de Correlación', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Encontrar correlaciones más fuertes
        strong_correlations = []
        for i in range(len(numeric_columns)):
            for j in range(i+1, len(numeric_columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5 and not np.isnan(corr_value):
                    strong_correlations.append({
                        'var1': numeric_columns[i],
                        'var2': numeric_columns[j],
                        'correlation': float(corr_value)
                    })
        
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return Response({
            'analysis_type': 'correlation_matrix',
            'image': f'data:image/png;base64,{image_base64}',
            'statistics': {
                'numeric_columns': numeric_columns,
                'column_count': len(numeric_columns),
                'strong_correlations': strong_correlations[:10]  # Top 10
            }
        })
    
    def generate_pca_analysis(self, df):
        """Genera análisis PCA para reducción de dimensionalidad"""
        # Seleccionar solo columnas numéricas
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_columns) < 3:
            return Response(
                {'error': 'At least 3 numeric columns are required for PCA'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Preparar datos
        X = df[numeric_columns].dropna()
        
        # Escalar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Aplicar PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Crear figuras
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Varianza explicada
        axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_)+1), 
                       pca.explained_variance_ratio_)
        axes[0, 0].set_xlabel('Componente Principal')
        axes[0, 0].set_ylabel('Varianza Explicada')
        axes[0, 0].set_title('Varianza Explicada por Componente')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Varianza acumulada
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        axes[0, 1].plot(range(1, len(cumsum)+1), cumsum, 'bo-')
        axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
        axes[0, 1].set_xlabel('Número de Componentes')
        axes[0, 1].set_ylabel('Varianza Acumulada')
        axes[0, 1].set_title('Varianza Acumulada')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Biplot PC1 vs PC2
        if X_pca.shape[1] >= 2:
            axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
            axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            axes[1, 0].set_title('PCA - Primeras 2 Componentes')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Loadings (contribución de variables)
        loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
        for i, (var, loading) in enumerate(zip(numeric_columns, loadings)):
            axes[1, 1].arrow(0, 0, loading[0], loading[1], 
                           head_width=0.05, head_length=0.05, alpha=0.7)
            axes[1, 1].text(loading[0]*1.1, loading[1]*1.1, var, fontsize=8)
        
        axes[1, 1].set_xlim(-1.2, 1.2)
        axes[1, 1].set_ylim(-1.2, 1.2)
        axes[1, 1].set_xlabel('PC1')
        axes[1, 1].set_ylabel('PC2')
        axes[1, 1].set_title('PCA Loadings')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Componentes necesarios para 95% varianza
        n_components_95 = np.argmax(cumsum >= 0.95) + 1
        
        return Response({
            'analysis_type': 'pca',
            'image': f'data:image/png;base64,{image_base64}',
            'statistics': {
                'total_variables': len(numeric_columns),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': cumsum.tolist(),
                'n_components_95_variance': int(n_components_95),
                'principal_components': {
                    f'PC{i+1}': {
                        'variance_explained': float(var),
                        'top_contributors': sorted(
                            [(col, float(loading)) for col, loading in zip(numeric_columns, pca.components_[i])],
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )[:5]
                    }
                    for i, var in enumerate(pca.explained_variance_ratio_[:3])
                }
            }
        })
    
    def generate_lasso_analysis(self, df, target_column):
        """Genera análisis LASSO para selección de características"""
        if target_column not in df.columns:
            return Response(
                {'error': f'Target column {target_column} not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Verificar que target sea numérico
        if df[target_column].dtype not in ['int64', 'float64']:
            return Response(
                {'error': 'Target column must be numeric for LASSO'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Seleccionar features numéricas (excluyendo target)
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_columns.remove(target_column)
        
        if len(numeric_columns) < 2:
            return Response(
                {'error': 'At least 2 numeric features are required for LASSO'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Preparar datos
        X = df[numeric_columns].dropna()
        y = df.loc[X.index, target_column]
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Escalar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Probar diferentes valores de alpha
        alphas = np.logspace(-4, 1, 50)
        coefs = []
        scores = []
        
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, random_state=42)
            lasso.fit(X_train_scaled, y_train)
            coefs.append(lasso.coef_)
            scores.append(lasso.score(X_test_scaled, y_test))
        
        # Crear visualizaciones
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Trayectoria de coeficientes
        for i, col in enumerate(numeric_columns):
            axes[0].plot(alphas, [coef[i] for coef in coefs], label=col)
        
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Alpha (Regularización)')
        axes[0].set_ylabel('Coeficientes')
        axes[0].set_title(f'LASSO Path - Prediciendo {target_column}')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Score vs Alpha
        axes[1].plot(alphas, scores, 'b-')
        best_alpha_idx = np.argmax(scores)
        axes[1].scatter(alphas[best_alpha_idx], scores[best_alpha_idx], 
                       color='red', s=100, zorder=5)
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Alpha (Regularización)')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('Score vs Alpha')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Entrenar modelo final con mejor alpha
        best_alpha = alphas[best_alpha_idx]
        lasso_final = Lasso(alpha=best_alpha, random_state=42)
        lasso_final.fit(X_train_scaled, y_train)
        
        # Obtener características importantes
        feature_importance = [(col, float(coef)) for col, coef in zip(numeric_columns, lasso_final.coef_) if abs(coef) > 0.01]
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return Response({
            'analysis_type': 'lasso',
            'target_column': target_column,
            'image': f'data:image/png;base64,{image_base64}',
            'auto_selected': request.query_params.get('target') is None,  # Indicar si se seleccionó automáticamente
            'statistics': {
                'best_alpha': float(best_alpha),
                'best_score': float(scores[best_alpha_idx]),
                'n_features_selected': len(feature_importance),
                'total_features': len(numeric_columns),
                'selected_features': feature_importance,
                'feature_elimination_order': [
                    {
                        'alpha': float(alpha),
                        'n_features': int(np.sum(np.abs(coef) > 0.01))
                    }
                    for alpha, coef in zip(alphas[::5], coefs[::5])  # Cada 5 para no sobrecargar
                ]
            }
        })


class DatasetNormalizationView(APIView):
    def get(self, request, pk):
        """Obtiene información de normalización para un dataset"""
        dataset = get_object_or_404(Dataset, pk=pk)
        
        try:
            df = pd.read_csv(dataset.file.path)
            
            # Analizar cada columna para determinar métodos de normalización disponibles
            normalization_info = {}
            
            for col in df.columns:
                col_type = str(df[col].dtype)
                
                if df[col].dtype in ['int64', 'float64']:
                    # Columna numérica
                    normalization_info[col] = {
                        'type': 'numeric',
                        'primary_methods': [
                            {'value': 'MIN_MAX', 'label': 'Min-Max [0, 1]', 'description': 'Escala valores al rango [0, 1]'},
                            {'value': 'Z_SCORE', 'label': 'Z-Score', 'description': 'Estandariza a media 0 y desviación 1'},
                            {'value': 'LSTM_TCN', 'label': 'LSTM/TCN [-1, 1]', 'description': 'Escala al rango [-1, 1] para RNN/TCN'},
                            {'value': 'CNN', 'label': 'CNN (Z-Score)', 'description': 'Normalización Z-Score para CNN'},
                            {'value': 'TRANSFORMER', 'label': 'Transformer (Robust)', 'description': 'RobustScaler resistente a outliers'},
                            {'value': 'TREE', 'label': 'Tree (Sin cambios)', 'description': 'Sin transformación (modelos de árbol)'}
                        ],
                        'secondary_methods': [
                            {'value': 'LOWER', 'label': 'Minúsculas', 'description': 'Convierte a minúsculas (si son texto)'},
                            {'value': 'STRIP', 'label': 'Eliminar espacios', 'description': 'Elimina espacios al inicio/final'},
                            {'value': 'ONE_HOT', 'label': 'One-Hot Encoding', 'description': 'Convierte categorías a códigos numéricos (0, 1, 2...)'}
                        ],
                        'stats': {
                            'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                            'std': float(df[col].std()) if not df[col].isna().all() else None,
                            'min': float(df[col].min()) if not df[col].isna().all() else None,
                            'max': float(df[col].max()) if not df[col].isna().all() else None
                        }
                    }
                elif df[col].dtype == 'object':
                    # Columna de texto o objeto
                    unique_values = df[col].nunique()
                    sample_values = df[col].dropna().head(5).tolist()
                    
                    normalization_info[col] = {
                        'type': 'text',
                        'primary_methods': [
                            {'value': 'LOWER', 'label': 'Minúsculas', 'description': 'Convierte todo a minúsculas'},
                            {'value': 'STRIP', 'label': 'Eliminar espacios', 'description': 'Elimina espacios al inicio/final'},
                            {'value': 'ONE_HOT', 'label': 'One-Hot Encoding', 'description': f'Convierte {unique_values} categorías a códigos numéricos (0, 1, 2...)'}
                        ],
                        'secondary_methods': [
                            {'value': 'MIN_MAX', 'label': 'Min-Max [0, 1]', 'description': 'Escala valores (si son números como texto)'},
                            {'value': 'Z_SCORE', 'label': 'Z-Score', 'description': 'Estandariza (si son números como texto)'},
                            {'value': 'LSTM_TCN', 'label': 'LSTM/TCN [-1, 1]', 'description': 'Escala [-1, 1] (si son números)'},
                            {'value': 'CNN', 'label': 'CNN (Z-Score)', 'description': 'Z-Score (si son números)'},
                            {'value': 'TRANSFORMER', 'label': 'Transformer (Robust)', 'description': 'RobustScaler (si son números)'},
                            {'value': 'TREE', 'label': 'Tree (Sin cambios)', 'description': 'Sin transformación'}
                        ],
                        'stats': {
                            'unique_count': unique_values,
                            'sample_values': sample_values,
                            'most_common': df[col].value_counts().head(3).to_dict() if unique_values < 100 else None
                        }
                    }
                else:
                    # Tipo desconocido o personalizado
                    normalization_info[col] = {
                        'type': 'unknown',
                        'primary_methods': [
                            {'value': 'MIN_MAX', 'label': 'Min-Max [0, 1]', 'description': 'Intenta escalar al rango [0, 1]'},
                            {'value': 'Z_SCORE', 'label': 'Z-Score', 'description': 'Intenta estandarizar'},
                            {'value': 'LSTM_TCN', 'label': 'LSTM/TCN [-1, 1]', 'description': 'Intenta escalar a [-1, 1]'},
                            {'value': 'CNN', 'label': 'CNN (Z-Score)', 'description': 'Intenta Z-Score'},
                            {'value': 'TRANSFORMER', 'label': 'Transformer (Robust)', 'description': 'Intenta RobustScaler'},
                            {'value': 'TREE', 'label': 'Tree (Sin cambios)', 'description': 'Sin transformación'},
                            {'value': 'LOWER', 'label': 'Minúsculas', 'description': 'Intenta convertir a minúsculas'},
                            {'value': 'STRIP', 'label': 'Eliminar espacios', 'description': 'Intenta eliminar espacios'},
                            {'value': 'ONE_HOT', 'label': 'One-Hot Encoding', 'description': 'Convierte categorías a códigos numéricos'}
                        ],
                        'secondary_methods': [],
                        'stats': {
                            'dtype': col_type,
                            'sample_values': df[col].dropna().head(5).astype(str).tolist()
                        }
                    }
            
            # Obtener copias normalizadas existentes
            normalized_copies = Dataset.objects.filter(
                parent_dataset=dataset
            ).order_by('-uploaded_at')
            
            return Response({
                'dataset': {
                    'id': dataset.id,
                    'name': dataset.name,
                    'shape': list(df.shape),  # Convertir tupla a lista
                    'columns': list(df.columns)
                },
                'normalization_info': normalization_info,
                'normalized_copies': [
                    {
                        'id': copy.id,
                        'name': copy.name,
                        'created_at': copy.uploaded_at,
                        'description': f"Dataset normalizado" if '_normalized_' in copy.name else "Dataset"
                    }
                    for copy in normalized_copies
                ]
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )
    
    def post(self, request, pk):
        """Aplica normalización a columnas específicas del dataset"""
        dataset = get_object_or_404(Dataset, pk=pk)
        
        # Check if this is a progress check request
        if request.data.get('check_progress'):
            session_id = request.data.get('session_id')
            # En una implementación real, aquí verificarías el progreso real
            # Por ahora, retornamos un progreso simulado
            return Response({
                'status': 'processing',
                'progress': 50,
                'current_step': 'Normalizando columna X',
                'total_steps': 10,
                'completed_steps': 5
            })
        
        try:
            df = pd.read_csv(dataset.file.path)
            
            # Obtener configuración de normalización
            normalization_config = request.data.get('normalization_config', {})
            create_copy = request.data.get('create_copy', True)
            
            # Generar nombre con el nuevo formato
            # Contar cuántas normalizaciones existen para este dataset
            normalized_count = Dataset.objects.filter(parent_dataset=dataset).count()
            next_number = normalized_count + 1
            
            # Si el usuario proporcionó un nombre, usarlo; si no, generar uno
            user_provided_name = request.data.get('copy_name', '').strip()
            if user_provided_name:
                copy_name = user_provided_name
            else:
                copy_name = f"{dataset.name}_normalizacion_{next_number}"
            
            # Obtener descripciones personalizadas
            copy_short_description = request.data.get('copy_short_description', '').strip()
            copy_long_description = request.data.get('copy_long_description', '').strip()
            
            # Para una implementación real con progreso, aquí deberías:
            # 1. Crear una tarea asíncrona (usando Celery o similar)
            # 2. Retornar un ID de sesión inmediatamente
            # 3. El frontend consultaría el progreso periódicamente
            
            # Por ahora, hacemos el proceso sincrónicamente
            if create_copy:
                # Trabajar con una copia
                df_normalized = df.copy()
            else:
                df_normalized = df
            
            # Aplicar normalizaciones
            applied_normalizations = []
            total_columns = len(normalization_config)
            processed_columns = 0
            
            for column, method in normalization_config.items():
                if column not in df_normalized.columns:
                    continue
                
                try:
                    if method in ['MIN_MAX', 'Z_SCORE', 'LSTM_TCN', 'CNN', 'TRANSFORMER', 'TREE']:
                        # Normalización numérica
                        num_method = NumNorm[method]
                        normalizador = Normalizador(metodo_numerico=num_method)
                        df_result = normalizador.normalizar(df_normalized[[column]])
                        df_normalized[column] = df_result[column]
                        
                        applied_normalizations.append({
                            'column': column,
                            'method': method,
                            'type': 'numeric'
                        })
                    
                    elif method in ['LOWER', 'STRIP', 'ONE_HOT']:
                        # Normalización de texto
                        text_method = TextNorm[method]
                        normalizador = Normalizador(metodo_texto=text_method)
                        df_result = normalizador.normalizar(df_normalized[[column]])
                        df_normalized[column] = df_result[column]
                        
                        applied_normalizations.append({
                            'column': column,
                            'method': method,
                            'type': 'text'
                        })
                    
                    processed_columns += 1
                    
                except Exception as e:
                    print(f"Error normalizando columna {column} con método {method}: {str(e)}")
                    continue
            
            if create_copy:
                # Guardar como nuevo dataset
                from django.core.files.base import ContentFile
                import io
                
                # Convertir DataFrame a CSV
                csv_buffer = io.StringIO()
                df_normalized.to_csv(csv_buffer, index=False)
                csv_content = csv_buffer.getvalue()
                
                # Determinar el root_dataset_id
                if dataset.is_normalized and dataset.root_dataset_id:
                    # Si el padre ya es normalizado, usar su root_dataset_id
                    root_id = dataset.root_dataset_id
                else:
                    # Si el padre es original, él es el root
                    root_id = dataset.id
                
                # Crear nuevo dataset
                new_dataset = Dataset(
                    name=copy_name,
                    short_description=copy_short_description if copy_short_description else f"Normalizado desde {dataset.name}",
                    long_description=copy_long_description if copy_long_description else f"Dataset normalizado desde {dataset.name} aplicando los siguientes métodos: {', '.join(set(normalization_config.values()))}. Procesado el {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}.",
                    is_normalized=True,
                    parent_dataset=dataset,
                    parent_dataset_name=dataset.name,  # Guardar el nombre del dataset padre
                    root_dataset_id=root_id,  # Guardar el ID del dataset raíz
                    normalization_method=str(normalization_config)
                )
                
                # Guardar archivo
                csv_file = ContentFile(csv_content.encode('utf-8'))
                new_dataset.file.save(f"{copy_name}.csv", csv_file)
                new_dataset.save()
                
                return Response({
                    'success': True,
                    'dataset_id': new_dataset.id,
                    'dataset_name': new_dataset.name,
                    'applied_normalizations': applied_normalizations,
                    'new_shape': list(df_normalized.shape),  # Convertir tupla a lista
                    'new_columns': list(df_normalized.columns),
                    'total_processed': processed_columns,
                    'total_columns': total_columns
                })
            else:
                # Sobrescribir dataset original (no recomendado)
                dataset.file.seek(0)
                df_normalized.to_csv(dataset.file.path, index=False)
                
                return Response({
                    'success': True,
                    'dataset_id': dataset.id,
                    'applied_normalizations': applied_normalizations,
                    'new_shape': list(df_normalized.shape),  # Convertir tupla a lista
                    'new_columns': list(df_normalized.columns),
                    'total_processed': processed_columns,
                    'total_columns': total_columns
                })
                
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"Error en normalización: {str(e)}")
            print(f"Traceback completo: {error_detail}")
            print(f"Config recibida: {request.data}")
            
            return Response(
                {
                    'error': str(e),
                    'traceback': error_detail,
                    'received_data': request.data
                }, 
                status=status.HTTP_400_BAD_REQUEST
            )


class DatasetNormalizationPreviewView(APIView):
    def post(self, request, pk):
        """Previsualiza el resultado de normalizar una columna"""
        dataset = get_object_or_404(Dataset, pk=pk)
        
        try:
            df = pd.read_csv(dataset.file.path)
            
            column = request.data.get('column')
            method = request.data.get('method')
            
            if not column or not method:
                return Response(
                    {'error': 'column and method are required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if column not in df.columns:
                return Response(
                    {'error': f'Column {column} not found'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Tomar una muestra para la previsualización
            sample_size = min(100, len(df))
            df_sample = df[[column]].head(sample_size).copy()
            
            try:
                # Verificar si las clases están disponibles
                if NumNorm is None or TextNorm is None or Normalizador is None:
                    return Response(
                        {'error': 'Normalization module not available. Please check server configuration.'}, 
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                
                if method in ['MIN_MAX', 'Z_SCORE', 'LSTM_TCN', 'CNN', 'TRANSFORMER', 'TREE']:
                    num_method = NumNorm[method]
                    normalizador = Normalizador(metodo_numerico=num_method)
                    df_result = normalizador.normalizar(df_sample)
                elif method in ['LOWER', 'STRIP', 'ONE_HOT']:
                    text_method = TextNorm[method]
                    normalizador = Normalizador(metodo_texto=text_method)
                    df_result = normalizador.normalizar(df_sample)
                else:
                    return Response(
                        {'error': f'Unknown method: {method}'}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                # Preparar respuesta con antes y después
                preview_data = {
                    'column': column,
                    'method': method,
                    'sample_size': sample_size,
                    'before': [str(val) if pd.notna(val) else 'NaN' for val in df_sample[column].head(20)],
                    'after': None,
                    'new_columns': None
                }
                
                # Para todos los métodos, incluyendo ONE_HOT
                preview_data['after'] = [str(val) if pd.notna(val) else 'NaN' for val in df_result[column].head(20)]
                
                # Estadísticas del cambio
                if df[column].dtype in ['int64', 'float64'] and method != 'ONE_HOT':
                    original_stats = {
                        'mean': float(df_sample[column].mean()) if not df_sample[column].isna().all() else None,
                        'std': float(df_sample[column].std()) if not df_sample[column].isna().all() else None,
                        'min': float(df_sample[column].min()) if not df_sample[column].isna().all() else None,
                        'max': float(df_sample[column].max()) if not df_sample[column].isna().all() else None
                    }
                    
                    normalized_stats = {
                        'mean': float(df_result[column].mean()) if not df_result[column].isna().all() else None,
                        'std': float(df_result[column].std()) if not df_result[column].isna().all() else None,
                        'min': float(df_result[column].min()) if not df_result[column].isna().all() else None,
                        'max': float(df_result[column].max()) if not df_result[column].isna().all() else None
                    }
                    
                    preview_data['stats'] = {
                        'before': original_stats,
                        'after': normalized_stats
                    }
                
                return Response(preview_data)
                
            except Exception as e:
                import traceback
                return Response(
                    {
                        'error': f'Error applying normalization: {str(e)}',
                        'column': column,
                        'method': method,
                        'traceback': traceback.format_exc(),
                        'debug_info': {
                            'sample_shape': df_sample.shape,
                            'sample_dtype': str(df_sample[column].dtype),
                            'sample_nulls': int(df_sample[column].isna().sum()),
                            'normalization_available': NumNorm is not None
                        }
                    }, 
                    status=status.HTTP_400_BAD_REQUEST
                )
                
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )


# Model Definition Views
class ModelDefinitionListCreateView(generics.ListCreateAPIView):
    queryset = ModelDefinition.objects.all()
    serializer_class = ModelDefinitionSerializer


class ModelDefinitionDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = ModelDefinition.objects.all()
    serializer_class = ModelDefinitionSerializer


class ModelDefinitionTrainingsView(APIView):
    def get(self, request, pk):
        model_def = get_object_or_404(ModelDefinition, pk=pk)
        trainings = model_def.trainingsession_set.all().order_by('-created_at')
        serializer = TrainingSessionSerializer(trainings, many=True)
        return Response(serializer.data)


class CloneModelDefinitionView(APIView):
    def post(self, request, pk):
        original = get_object_or_404(ModelDefinition, pk=pk)
        
        # Crear copia
        new_model = ModelDefinition.objects.create(
            name=f"{original.name} (Copia)",
            description=original.description,
            model_type=original.model_type,
            dataset=original.dataset,
            predictor_columns=original.predictor_columns,
            target_columns=original.target_columns,
            default_config=original.default_config,
            hyperparameters=original.hyperparameters,
            custom_architecture=original.custom_architecture,
            use_custom_architecture=original.use_custom_architecture,
            user=request.user if request.user.is_authenticated else None
        )
        
        serializer = ModelDefinitionSerializer(new_model)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
