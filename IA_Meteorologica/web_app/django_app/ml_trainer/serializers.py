from rest_framework import serializers
from .models import Dataset, TrainingSession, WeatherPrediction, ModelDefinition, CustomNormalizationFunction


class DatasetSerializer(serializers.ModelSerializer):
    row_count = serializers.SerializerMethodField()
    column_count = serializers.SerializerMethodField()
    file_size = serializers.SerializerMethodField()
    display_description = serializers.SerializerMethodField()
    parent_dataset_name = serializers.SerializerMethodField()
    upload_time = serializers.SerializerMethodField()
    genealogy = serializers.SerializerMethodField()
    
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'short_description', 'long_description', 'file', 'uploaded_at', 
                  'row_count', 'column_count', 'file_size', 'display_description', 
                  'is_normalized', 'parent_dataset', 'parent_dataset_name', 
                  'normalization_method', 'upload_time', 'genealogy', 'root_dataset_id']
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
    
    def get_display_description(self, obj):
        # Si el dataset tiene una descripción corta personalizada, usarla
        if obj.short_description:
            return obj.short_description
        # Si no, generar descripción basada en el nombre y estado
        if obj.is_normalized and obj.parent_dataset:
            return f"Normalizado desde: {obj.parent_dataset.name}"
        elif '_normalized_' in obj.name or '_normalizacion_' in obj.name:
            return f"Dataset normalizado"
        return "Dataset meteorológico"
    
    def get_parent_dataset_name(self, obj):
        # Primero intentar obtener del dataset padre si existe
        if obj.parent_dataset:
            return obj.parent_dataset.name
        # Si no existe el dataset padre pero tenemos el nombre guardado, usarlo
        elif obj.parent_dataset_name:
            return obj.parent_dataset_name
        return None
    
    def get_upload_time(self, obj):
        # Devolver hora y minutos en formato HH:MM
        return obj.uploaded_at.strftime('%H:%M') if obj.uploaded_at else ''
    
    def get_genealogy(self, obj):
        if obj.is_normalized:
            return obj.get_genealogy()
        return []


class ModelDefinitionSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    latest_training = serializers.SerializerMethodField()
    success_rate = serializers.SerializerMethodField()
    
    class Meta:
        model = ModelDefinition
        fields = [
            'id', 'name', 'description', 'model_type', 'created_at', 'updated_at',
            'dataset', 'dataset_name', 'predictor_columns', 'target_columns',
            'default_config', 'hyperparameters', 'custom_architecture', 
            'use_custom_architecture', 'training_count', 'best_score', 
            'last_trained', 'is_active', 'latest_training', 'success_rate'
        ]
    
    def get_latest_training(self, obj):
        latest = obj.get_latest_training()
        if latest:
            return {
                'id': latest.id,
                'status': latest.status,
                'created_at': latest.created_at,
                'metrics': latest.test_results
            }
        return None
    
    def get_success_rate(self, obj):
        total = obj.trainingsession_set.count()
        if total == 0:
            return 0
        completed = obj.trainingsession_set.filter(status='completed').count()
        return round((completed / total) * 100, 1)


class TrainingSessionSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    
    class Meta:
        model = TrainingSession
        fields = [
            'id', 'name', 'dataset', 'dataset_name', 'model_type', 'created_at',
            'predictor_columns', 'target_columns', 'target_column', 'normalization_method',
            'hyperparameters', 'config', 'train_split', 'val_split', 'test_split', 'test_size',
            'selected_metrics', 'training_history', 'test_results',
            'status', 'error_message', 'custom_architecture', 'use_custom_architecture'
        ]
        read_only_fields = ['created_at', 'training_history', 'test_results', 'status', 'error_message']


class WeatherPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = WeatherPrediction
        fields = ['id', 'training_session', 'prediction_date', 'region', 
                  'latitude', 'longitude', 'predictions', 'created_at']
        read_only_fields = ['created_at']


class CustomNormalizationFunctionSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomNormalizationFunction
        fields = ['id', 'name', 'description', 'function_type', 'code', 
                  'created_at', 'updated_at', 'user', 'remove_original_column', 'new_columns']
        read_only_fields = ['created_at', 'updated_at', 'user']