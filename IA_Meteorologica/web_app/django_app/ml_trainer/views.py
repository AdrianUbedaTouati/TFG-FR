from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
import pandas as pd
import json
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
            
            return Response({
                'columns': columns,
                'dtypes': dtypes,
                'shape': df.shape,
                'preview': df.head(5).to_dict()
            })
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
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