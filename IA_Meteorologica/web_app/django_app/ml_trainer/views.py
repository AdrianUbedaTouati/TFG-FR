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
                        # Intentar parsear como fecha
                        try:
                            date_parsed = pd.to_datetime(sample, errors='coerce')
                            # Si más del 80% se parsean como fechas, es probable que sea una columna de fechas
                            if date_parsed.notna().sum() / len(sample) > 0.8:
                                dtype_str = 'datetime'
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