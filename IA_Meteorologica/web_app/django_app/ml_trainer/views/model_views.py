"""
Model definition and management views
"""
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from typing import Dict, Any

from ..models import ModelDefinition, ModelType, FrameworkType
from ..serializers import ModelDefinitionSerializer
from ..ml_utils import get_model_config, get_normalization_methods, get_metrics
from ..utils import error_response, success_response
from ..constants import (
    NEURAL_NETWORK_MODELS, TRADITIONAL_ML_MODELS,
    SUCCESS_MODEL_CREATED, ERROR_INVALID_MODEL_TYPE
)


class ModelDefinitionListCreateView(generics.ListCreateAPIView):
    """List all model definitions or create a new one"""
    serializer_class = ModelDefinitionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter models by user"""
        if self.request.user.is_staff:
            return ModelDefinition.objects.all()
        return ModelDefinition.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        """Add user to model definition"""
        serializer.save(user=self.request.user)


class ModelDefinitionDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update or delete a model definition"""
    serializer_class = ModelDefinitionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter models by user"""
        if self.request.user.is_staff:
            return ModelDefinition.objects.all()
        return ModelDefinition.objects.filter(user=self.request.user)


class ModelDefinitionTrainingsView(APIView):
    """Get all training sessions for a model definition"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            model_def = get_object_or_404(ModelDefinition, pk=pk)
        else:
            model_def = get_object_or_404(ModelDefinition, pk=pk, user=request.user)
        
        # Get all training sessions
        trainings = model_def.trainingsession_set.all().order_by('-created_at')
        
        # Format response
        training_data = []
        for training in trainings:
            training_data.append({
                'id': training.id,
                'name': training.name,
                'status': training.status,
                'created_at': training.created_at,
                'framework': getattr(training, 'framework', 'keras'),
                'metrics': training.test_results,
                'epochs': training.hyperparameters.get('epochs', 0),
                'error_message': training.error_message
            })
        
        return success_response({
            'model_id': model_def.id,
            'model_name': model_def.name,
            'total_trainings': len(training_data),
            'trainings': training_data
        })


class CloneModelDefinitionView(APIView):
    """Clone an existing model definition"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            original = get_object_or_404(ModelDefinition, pk=pk)
        else:
            original = get_object_or_404(ModelDefinition, pk=pk, user=request.user)
        
        # Create clone
        new_model = ModelDefinition.objects.create(
            name=f"{original.name} (Copy)",
            description=original.description,
            model_type=original.model_type,
            dataset=original.dataset,
            predictor_columns=original.predictor_columns,
            target_columns=original.target_columns,
            default_config=original.default_config,
            hyperparameters=original.hyperparameters,
            custom_architecture=original.custom_architecture,
            use_custom_architecture=original.use_custom_architecture,
            framework=original.framework,
            user=request.user
        )
        
        serializer = ModelDefinitionSerializer(new_model)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class ModelConfigView(APIView):
    """Get default configurations for all model types"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        configs = {}
        for model_type in ModelType:
            configs[model_type.value] = get_model_config(model_type.value)
        
        return success_response(configs)


class NormalizationMethodsView(APIView):
    """Get available normalization methods for a model type"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, model_type):
        if model_type not in [mt.value for mt in ModelType]:
            return error_response(ERROR_INVALID_MODEL_TYPE.format(model_type))
        
        methods = get_normalization_methods(model_type)
        return success_response({'methods': methods})


class MetricsView(APIView):
    """Get available metrics for a model type"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, model_type):
        if model_type not in [mt.value for mt in ModelType]:
            return error_response(ERROR_INVALID_MODEL_TYPE.format(model_type))
        
        metrics = get_metrics(model_type)
        return success_response({'metrics': metrics})


class ModelFrameworkView(APIView):
    """Get available frameworks for a model type"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, model_type):
        # Neural networks can use Keras or PyTorch
        if model_type in NEURAL_NETWORK_MODELS:
            frameworks = [
                {'value': 'keras', 'label': 'TensorFlow/Keras'},
                {'value': 'pytorch', 'label': 'PyTorch'}
            ]
        # Traditional ML models use sklearn
        elif model_type in TRADITIONAL_ML_MODELS:
            frameworks = [
                {'value': 'sklearn', 'label': 'Scikit-learn'}
            ]
        else:
            frameworks = []
        
        return success_response({'frameworks': frameworks})


class ImplementedModelsView(APIView):
    """Get list of actually implemented model types"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Return model types that are actually implemented"""
        implemented_models = []
        
        # Check which models are implemented
        # Neural network models (all implemented with Keras/PyTorch)
        neural_models = [
            {'value': 'lstm', 'label': 'LSTM (Deep Learning)', 'category': 'neural', 'implemented': True},
            {'value': 'gru', 'label': 'GRU (Deep Learning)', 'category': 'neural', 'implemented': True},
            {'value': 'cnn', 'label': 'CNN (Deep Learning)', 'category': 'neural', 'implemented': True},
            {'value': 'transformer', 'label': 'Transformer', 'category': 'neural', 'implemented': True}
        ]
        
        # Traditional ML models
        ml_models = [
            {'value': 'random_forest', 'label': 'Random Forest', 'category': 'ml', 'implemented': True},
            {'value': 'xgboost', 'label': 'XGBoost', 'category': 'ml', 'implemented': True},
            {'value': 'decision_tree', 'label': 'Decision Tree', 'category': 'ml', 'implemented': True}
        ]
        
        # Models not yet implemented (for future reference)
        future_models = [
            {'value': 'gradient_boosting', 'label': 'Gradient Boosting', 'category': 'ml', 'implemented': False},
            {'value': 'svm', 'label': 'Support Vector Machine', 'category': 'ml', 'implemented': False}
        ]
        
        # Only return implemented models
        implemented_models = [m for m in (neural_models + ml_models) if m['implemented']]
        
        return success_response({
            'models': implemented_models,
            'total': len(implemented_models)
        })