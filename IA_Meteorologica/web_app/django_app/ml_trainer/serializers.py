from rest_framework import serializers
from .models import Dataset, TrainingSession, WeatherPrediction


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'file', 'uploaded_at']
        read_only_fields = ['uploaded_at']


class TrainingSessionSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    
    class Meta:
        model = TrainingSession
        fields = [
            'id', 'dataset', 'dataset_name', 'model_type', 'created_at',
            'predictor_columns', 'target_columns', 'normalization_method',
            'hyperparameters', 'train_split', 'val_split', 'test_split',
            'selected_metrics', 'training_history', 'test_results',
            'status', 'error_message'
        ]
        read_only_fields = ['created_at', 'training_history', 'test_results', 'status', 'error_message']


class WeatherPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = WeatherPrediction
        fields = ['id', 'training_session', 'prediction_date', 'region', 
                  'latitude', 'longitude', 'predictions', 'created_at']
        read_only_fields = ['created_at']