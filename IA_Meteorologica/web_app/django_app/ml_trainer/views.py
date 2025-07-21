from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.http import HttpResponse
import pandas as pd
import numpy as np
import json
import mimetypes
from .models import Dataset, TrainingSession, WeatherPrediction, ModelType, NormalizationMethod, MetricType
from .serializers import DatasetSerializer, TrainingSessionSerializer, WeatherPredictionSerializer
from .ml_utils import get_model_config, get_normalization_methods, get_metrics, train_model, make_predictions


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
                        hist, bins = np.histogram(df[col].dropna(), bins=20)
                        col_stats['histogram'] = {
                            'counts': hist.tolist(),
                            'bins': bins.tolist()
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
            html_content = self.generate_html_report(dataset, df)
            
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
    
    def generate_html_report(self, dataset, df):
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
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f4f4f4;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    background-color: #3498db;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    color: white;
                    margin: 0;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .summary-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .summary-card h3 {{
                    margin-top: 0;
                    color: #3498db;
                }}
                .summary-card .value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                    margin-top: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .numeric {{
                    text-align: right;
                }}
                .footer {{
                    margin-top: 50px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 14px;
                }}
                .preview-section {{
                    margin-top: 30px;
                    background: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .null-badge {{
                    background-color: #e74c3c;
                    color: white;
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-size: 12px;
                }}
                .type-badge {{
                    background-color: #95a5a6;
                    color: white;
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-size: 12px;
                    margin-left: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Rapport d'Analyse du Dataset</h1>
                <p>{dataset.name} - Généré le {pd.Timestamp.now().strftime("%d/%m/%Y à %H:%M")}</p>
            </div>
            
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Dimensions</h3>
                    <div class="value">{shape[0]:,} × {shape[1]}</div>
                    <p>{shape[0]:,} lignes, {shape[1]} colonnes</p>
                </div>
                <div class="summary-card">
                    <h3>Taille en mémoire</h3>
                    <div class="value">{memory_usage:.2f} MB</div>
                    <p>Utilisation mémoire totale</p>
                </div>
                <div class="summary-card">
                    <h3>Valeurs manquantes</h3>
                    <div class="value">{total_nulls:,}</div>
                    <p>{(total_nulls / (shape[0] * shape[1]) * 100):.2f}% du total</p>
                </div>
                <div class="summary-card">
                    <h3>Date de création</h3>
                    <div class="value">{dataset.created_at.strftime("%d/%m/%Y") if dataset.created_at else "N/A"}</div>
                    <p>Import initial</p>
                </div>
            </div>
            
            <h2>Analyse des Variables</h2>
            <table>
                <thead>
                    <tr>
                        <th>Variable</th>
                        <th>Type</th>
                        <th>Valeurs uniques</th>
                        <th>Valeurs nulles</th>
                        <th>Statistiques</th>
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
                        </tr>
                '''
        
            html += '''
                    </tbody>
                </table>
                
                <div class="preview-section">
                    <h2>Aperçu des Données (10 premières lignes)</h2>
                    <div style="overflow-x: auto;">
            '''
            
            # Ajouter l'aperçu des données
            try:
                preview_html = df.head(10).to_html(classes='preview-table', index=False, escape=True)
                html += preview_html
            except Exception as e:
                html += f'<p>Error al generar preview: {str(e)}</p>'
            
            html += '''
                    </div>
                </div>
                
                <div class="footer">
                    <p>Rapport généré automatiquement par IA Météorologique</p>
                    <p>© 2024 - Tous droits réservés</p>
                </div>
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
                <title>Error en el reporte</title>
            </head>
            <body>
                <h1>Error al generar el reporte</h1>
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