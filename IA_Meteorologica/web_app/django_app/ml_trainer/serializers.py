from rest_framework import serializers
from .models import Dataset, TrainingSession, WeatherPrediction


class DatasetSerializer(serializers.ModelSerializer):
    row_count = serializers.SerializerMethodField()
    column_count = serializers.SerializerMethodField()
    file_size = serializers.SerializerMethodField()
    description = serializers.SerializerMethodField()
    
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'file', 'uploaded_at', 'row_count', 'column_count', 'file_size', 'description']
        read_only_fields = ['uploaded_at']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._df_cache = {}
    
    def _get_dataframe(self, obj):
        if obj.id not in self._df_cache:
            try:
                import pandas as pd
                self._df_cache[obj.id] = pd.read_csv(obj.file.path)
            except:
                self._df_cache[obj.id] = None
        return self._df_cache[obj.id]
    
    def get_row_count(self, obj):
        df = self._get_dataframe(obj)
        return df.shape[0] if df is not None else 0
    
    def get_column_count(self, obj):
        df = self._get_dataframe(obj)
        return df.shape[1] if df is not None else 0
    
    def get_file_size(self, obj):
        try:
            return obj.file.size
        except:
            return 0
    
    def get_description(self, obj):
        # Generar descripción basada en el nombre
        if '_normalized_' in obj.name:
            return f"Dataset normalizado"
        return "Dataset meteorológico"


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