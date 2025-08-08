"""
Prediction and inference views
"""
from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
import pandas as pd
from datetime import datetime, timedelta

from ..models import WeatherPrediction, TrainingSession
from ..serializers import WeatherPredictionSerializer
from ..ml_utils import make_predictions, generate_weather_map_data
from ..utils import error_response, success_response, load_dataset


class PredictionListCreateView(generics.ListCreateAPIView):
    """List all predictions or create new ones"""
    serializer_class = WeatherPredictionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter predictions by user"""
        if self.request.user.is_staff:
            return WeatherPrediction.objects.all()
        return WeatherPrediction.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        """Add user to prediction"""
        serializer.save(user=self.request.user)


class PredictView(APIView):
    """Make predictions using a trained model"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Make predictions with trained model"""
        session_id = request.data.get('session_id')
        input_data = request.data.get('input_data')
        
        if not session_id:
            return error_response("session_id is required")
        
        # Verificar permisos
        if request.user.is_staff:
            session = get_object_or_404(TrainingSession, pk=session_id)
        else:
            session = get_object_or_404(TrainingSession, pk=session_id, user=request.user)
        
        # Check if model is trained
        if session.status != 'completed':
            return error_response("Model training is not completed")
        
        if not session.model_file:
            return error_response("Model file not found")
        
        try:
            # Handle different input formats
            if isinstance(input_data, str):
                # File path provided
                predictions = make_predictions(session, input_data)
            elif isinstance(input_data, dict):
                # Single prediction
                df = pd.DataFrame([input_data])
                temp_path = f'/tmp/prediction_{session.id}.csv'
                df.to_csv(temp_path, index=False)
                predictions = make_predictions(session, temp_path)
            elif isinstance(input_data, list):
                # Batch prediction
                df = pd.DataFrame(input_data)
                temp_path = f'/tmp/prediction_{session.id}.csv'
                df.to_csv(temp_path, index=False)
                predictions = make_predictions(session, temp_path)
            else:
                return error_response("Invalid input_data format")
            
            return success_response({
                'session_id': session.id,
                'model_type': session.model_type,
                'predictions': predictions
            })
            
        except Exception as e:
            return error_response(f"Prediction failed: {str(e)}")


class PredictionMapView(APIView):
    """Generate prediction data for map visualization"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get prediction data for map"""
        session_id = request.query_params.get('session_id')
        date_str = request.query_params.get('date')
        
        if not session_id:
            return error_response("session_id is required")
        
        # Verificar permisos
        if request.user.is_staff:
            session = get_object_or_404(TrainingSession, pk=session_id)
        else:
            session = get_object_or_404(TrainingSession, pk=session_id, user=request.user)
        
        # Parse date or use today
        if date_str:
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return error_response("Invalid date format. Use YYYY-MM-DD")
        else:
            date = datetime.now().date()
        
        # Check for existing predictions
        existing = WeatherPrediction.objects.filter(
            training_session=session,
            prediction_date=date
        )
        
        if existing.exists():
            # Return existing predictions
            serializer = WeatherPredictionSerializer(existing, many=True)
            return success_response({
                'date': date,
                'predictions': serializer.data
            })
        
        # Generate new predictions
        try:
            predictions = generate_weather_map_data(session, date)
            
            # Save predictions with user
            for prediction in predictions:
                prediction.user = request.user
            WeatherPrediction.objects.bulk_create(predictions)
            
            serializer = WeatherPredictionSerializer(predictions, many=True)
            return success_response({
                'date': date,
                'predictions': serializer.data
            })
            
        except Exception as e:
            return error_response(f"Failed to generate predictions: {str(e)}")