from django.db import models
from django.contrib.auth.models import User
import json


class ModelType(models.TextChoices):
    LSTM = 'lstm', 'LSTM'
    CNN = 'cnn', 'CNN'
    DECISION_TREE = 'decision_tree', 'Árbol de Decisiones'
    TRANSFORMER = 'transformer', 'Transformer'
    RANDOM_FOREST = 'random_forest', 'Random Forest'
    XGB = 'xgboost', 'XGBoost'


class NormalizationMethod(models.TextChoices):
    MIN_MAX = 'min_max', 'Min-Max'
    STANDARD = 'standard', 'Estandarización'
    ROBUST = 'robust', 'Robust Scaler'
    NONE = 'none', 'Sin normalización'


class MetricType(models.TextChoices):
    MAE = 'mae', 'Mean Absolute Error'
    MSE = 'mse', 'Mean Squared Error'
    RMSE = 'rmse', 'Root Mean Squared Error'
    R2 = 'r2', 'R² Score'
    ACCURACY = 'accuracy', 'Accuracy'
    ROC_AUC = 'roc_auc', 'ROC AUC'
    F1 = 'f1', 'F1 Score'


class Dataset(models.Model):
    name = models.CharField(max_length=255)
    short_description = models.CharField(max_length=80, blank=True, null=True, help_text="Descripción corta (máx. 80 caracteres)")
    long_description = models.TextField(blank=True, null=True, help_text="Descripción detallada del dataset")
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # Campos para normalización
    is_normalized = models.BooleanField(default=False)
    parent_dataset = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='normalized_copies')
    parent_dataset_name = models.CharField(max_length=255, null=True, blank=True, help_text="Nombre del dataset padre (se mantiene aunque se borre el original)")
    root_dataset_id = models.IntegerField(null=True, blank=True, help_text="ID del dataset raíz original (para mantener agrupación)")
    normalization_method = models.CharField(max_length=255, null=True, blank=True)
    
    def __str__(self):
        return self.name
    
    def get_genealogy(self):
        """Obtiene toda la genealogía del dataset (lista de ancestros)"""
        genealogy = []
        current = self
        visited = set()  # Para evitar ciclos infinitos
        
        while current.parent_dataset and current.parent_dataset.id not in visited:
            visited.add(current.parent_dataset.id)
            genealogy.append({
                'id': current.parent_dataset.id,
                'name': current.parent_dataset.name,
                'exists': True
            })
            current = current.parent_dataset
        
        # Si no hay más padres pero tenemos el nombre guardado, agregarlo
        if not current.parent_dataset and current.parent_dataset_name and current.id != self.id:
            genealogy.append({
                'id': None,
                'name': current.parent_dataset_name,
                'exists': False
            })
        
        return list(reversed(genealogy))  # Devolver desde el más antiguo al más reciente


class TrainingSession(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    model_type = models.CharField(max_length=50, choices=ModelType.choices)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # Variable selection
    predictor_columns = models.JSONField(default=list)
    target_columns = models.JSONField(default=list)
    
    # Normalization
    normalization_method = models.CharField(
        max_length=50, 
        choices=NormalizationMethod.choices,
        default=NormalizationMethod.MIN_MAX
    )
    
    # Hyperparameters (stored as JSON for flexibility)
    hyperparameters = models.JSONField(default=dict)
    
    # Train/Val/Test split
    train_split = models.FloatField(default=0.7)
    val_split = models.FloatField(default=0.15)
    test_split = models.FloatField(default=0.15)
    
    # Metrics
    selected_metrics = models.JSONField(default=list)
    
    # Training results
    training_history = models.JSONField(null=True, blank=True)
    test_results = models.JSONField(null=True, blank=True)
    
    # Model file
    model_file = models.FileField(upload_to='models/', null=True, blank=True)
    
    # Status
    status = models.CharField(max_length=50, default='pending')
    error_message = models.TextField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.model_type} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class WeatherPrediction(models.Model):
    training_session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE)
    prediction_date = models.DateField()
    region = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    predictions = models.JSONField()  # Store all predicted variables
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['training_session', 'prediction_date', 'latitude', 'longitude']