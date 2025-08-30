from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json


class ModelType(models.TextChoices):
    LSTM = 'lstm', 'LSTM'
    GRU = 'gru', 'GRU'
    CNN = 'cnn', 'CNN'
    DECISION_TREE = 'decision_tree', 'Árbol de Decisiones'
    TRANSFORMER = 'transformer', 'Transformer'
    RANDOM_FOREST = 'random_forest', 'Random Forest'
    XGB = 'xgboost', 'XGBoost'


class FrameworkType(models.TextChoices):
    KERAS = 'keras', 'TensorFlow/Keras'
    PYTORCH = 'pytorch', 'PyTorch'
    SKLEARN = 'sklearn', 'Scikit-learn'


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


class CustomNormalizationFunction(models.Model):
    """Funciones de normalización personalizadas creadas por el usuario"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    function_type = models.CharField(max_length=20, choices=[('numeric', 'Numérica'), ('text', 'Texto')])
    initialization_code = models.TextField(blank=True, null=True, help_text="Código Python de inicialización que se ejecuta una sola vez para crear variables globales")
    code = models.TextField(help_text="Código Python de la función de normalización")
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='custom_functions')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # Campo para controlar columnas nuevas
    new_columns = models.JSONField(default=list, help_text="Lista de nombres de columnas nuevas que creará la función")
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.function_type})"


# Comentado temporalmente hasta ejecutar migraciones
# class NormalizationChain(models.Model):
#     """Cadena de funciones de normalización aplicadas secuencialmente"""
#     name = models.CharField(max_length=100)
#     description = models.TextField(blank=True)
#     user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='normalization_chains')
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)
#     
#     class Meta:
#         ordering = ['-created_at']
#         unique_together = ['name', 'user']
#     
#     def __str__(self):
#         return self.name


# class NormalizationChainStep(models.Model):
#     """Un paso en la cadena de normalización"""
#     chain = models.ForeignKey(NormalizationChain, on_delete=models.CASCADE, related_name='steps')
#     order = models.PositiveIntegerField()
#     method = models.CharField(max_length=100, help_text="Método de normalización o ID de función personalizada")
#     keep_original = models.BooleanField(default=False, help_text="Mantener la columna original")
#     
#     class Meta:
#         ordering = ['order']
#         unique_together = ['chain', 'order']
#     
#     def __str__(self):
#         return f"{self.chain.name} - Paso {self.order}: {self.method}"


class Dataset(models.Model):
    name = models.CharField(max_length=255)
    short_description = models.CharField(max_length=80, blank=True, null=True, help_text="Descripción corta (máx. 80 caracteres)")
    long_description = models.TextField(blank=True, null=True, help_text="Descripción detallada del dataset")
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='datasets')
    
    # Campos para normalización
    is_normalized = models.BooleanField(default=False)
    parent_dataset = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='normalized_copies')
    parent_dataset_name = models.CharField(max_length=255, null=True, blank=True, help_text="Nombre del dataset padre (se mantiene aunque se borre el original)")
    root_dataset_id = models.IntegerField(null=True, blank=True, help_text="ID del dataset raíz original (para mantener agrupación)")
    normalization_method = models.CharField(max_length=255, null=True, blank=True)
    # normalization_chain = models.ForeignKey(NormalizationChain, on_delete=models.SET_NULL, null=True, blank=True, related_name='datasets')
    
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


class ModelDefinition(models.Model):
    """Definición/Template de un modelo que puede ser entrenado múltiples veces"""
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    model_type = models.CharField(max_length=50, choices=ModelType.choices)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='model_definitions')
    
    # Dataset asociado
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    
    # Configuración del modelo
    predictor_columns = models.JSONField(default=list)
    target_columns = models.JSONField(default=list)
    
    # Configuración de entrenamiento por defecto
    default_config = models.JSONField(default=dict)
    hyperparameters = models.JSONField(default=dict)
    
    # Arquitectura personalizada
    custom_architecture = models.JSONField(null=True, blank=True)
    use_custom_architecture = models.BooleanField(default=False)
    
    # Framework selection
    framework = models.CharField(
        max_length=20,
        choices=FrameworkType.choices,
        default=FrameworkType.KERAS,
        help_text="Deep learning framework to use"
    )
    
    # Code export/import
    exported_code = models.TextField(null=True, blank=True, help_text="Last exported Python code")
    code_version = models.IntegerField(default=1, help_text="Version of the exported code")
    
    # Estadísticas
    training_count = models.IntegerField(default=0)
    best_score = models.FloatField(null=True, blank=True)
    last_trained = models.DateTimeField(null=True, blank=True)
    
    # Estado
    is_active = models.BooleanField(default=True)
    
    # Default data split configuration
    default_split_method = models.CharField(
        max_length=20,
        choices=[
            ('random', 'División Aleatoria'),
            ('stratified', 'División Estratificada'),
            ('group', 'División por Grupos'),
            ('temporal', 'División Temporal'),
        ],
        default='random',
        help_text='Método de división de datos por defecto'
    )
    default_split_config = models.JSONField(
        default=dict,
        blank=True,
        help_text='Configuración de división por defecto'
    )
    
    # Default execution configuration
    default_execution_method = models.CharField(
        max_length=30,
        choices=[
            ('standard', 'Ejecución Estándar'),
            ('kfold', 'K-Fold Cross Validation'),
            ('stratified_kfold', 'Stratified K-Fold CV'),
            ('time_series_split', 'Time Series Split'),
            ('leave_one_out', 'Leave-One-Out CV'),
            ('repeated_kfold', 'Repeated K-Fold CV'),
            ('repeated_stratified_kfold', 'Repeated Stratified K-Fold CV'),
        ],
        default='standard',
        help_text='Método de ejecución por defecto'
    )
    default_execution_config = models.JSONField(
        default=dict,
        blank=True,
        help_text='Configuración de ejecución por defecto'
    )
    
    def __str__(self):
        return f"{self.name} ({self.get_model_type_display()})"
    
    def get_latest_training(self):
        """Get the most recent training session for this model"""
        return self.trainingsession_set.order_by('-created_at').first()
    
    def calculate_best_score(self):
        """Calculate the best score from all completed training sessions"""
        completed_trainings = self.trainingsession_set.filter(status='completed')
        best_score = None
        
        for training in completed_trainings:
            if training.test_results:
                # For classification models, use accuracy as the primary metric
                if self.model_type in ['random_forest', 'neural_network', 'decision_tree', 'xgboost'] and training.hyperparameters.get('problem_type') == 'classification':
                    score = training.test_results.get('accuracy', training.test_results.get('f1_score'))
                # For regression models, use R² as the primary metric (higher is better)
                else:
                    score = training.test_results.get('r2')
                
                if score is not None:
                    if best_score is None or score > best_score:
                        best_score = score
        
        return best_score
    
    def update_best_score(self):
        """Update the best_score field based on existing trainings"""
        self.best_score = self.calculate_best_score()
        self.save(update_fields=['best_score'])


class TrainingSession(models.Model):
    # Referencia al modelo definido
    model_definition = models.ForeignKey(ModelDefinition, on_delete=models.CASCADE, null=True, blank=True)
    name = models.CharField(max_length=200)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    model_type = models.CharField(max_length=50, choices=ModelType.choices)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='training_sessions')
    
    # Variable selection
    predictor_columns = models.JSONField(default=list)
    target_columns = models.JSONField(default=list)
    target_column = models.CharField(max_length=100, default='')
    
    # Normalization
    normalization_method = models.CharField(
        max_length=50, 
        choices=NormalizationMethod.choices,
        default=NormalizationMethod.MIN_MAX
    )
    
    # Hyperparameters (stored as JSON for flexibility)
    hyperparameters = models.JSONField(default=dict)
    config = models.JSONField(default=dict)  # For additional configuration like epochs
    
    # Custom architecture for expert mode
    custom_architecture = models.JSONField(null=True, blank=True)
    use_custom_architecture = models.BooleanField(default=False)
    
    # Framework
    framework = models.CharField(
        max_length=20,
        choices=FrameworkType.choices,
        default=FrameworkType.KERAS
    )
    
    # Train/Val/Test split
    train_split = models.FloatField(default=0.7)
    val_split = models.FloatField(default=0.15)
    test_split = models.FloatField(default=0.15)
    test_size = models.FloatField(default=0.2)  # For simpler train/test split
    
    # Data split configuration
    SPLIT_METHOD_CHOICES = [
        ('random', 'División Aleatoria'),
        ('stratified', 'División Estratificada'),
        ('group', 'División por Grupos'),
        ('temporal', 'División Temporal'),
        ('sequential', 'División Secuencial'),
    ]
    split_method = models.CharField(
        max_length=20,
        choices=SPLIT_METHOD_CHOICES,
        default='random',
        help_text='Método de división de datos'
    )
    split_config = models.JSONField(
        default=dict,
        blank=True,
        help_text='Configuración específica del método de división'
    )
    random_state = models.IntegerField(
        null=True,
        blank=True,
        help_text='Semilla global para reproducibilidad'
    )
    
    # Execution configuration (Module 2)
    EXECUTION_METHOD_CHOICES = [
        ('standard', 'Ejecución Estándar'),
        ('kfold', 'K-Fold Cross Validation'),
        ('stratified_kfold', 'Stratified K-Fold CV'),
        ('time_series_split', 'Time Series Split'),
        ('leave_one_out', 'Leave-One-Out CV'),
        ('repeated_kfold', 'Repeated K-Fold CV'),
        ('repeated_stratified_kfold', 'Repeated Stratified K-Fold CV'),
    ]
    execution_method = models.CharField(
        max_length=30,
        choices=EXECUTION_METHOD_CHOICES,
        default='standard',
        help_text='Método de ejecución del entrenamiento'
    )
    execution_config = models.JSONField(
        default=dict,
        blank=True,
        help_text='Configuración específica del método de ejecución'
    )
    
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
    
    # Real-time progress tracking
    current_epoch = models.IntegerField(default=0)
    total_epochs = models.IntegerField(default=0)
    current_batch = models.IntegerField(default=0)
    total_batches = models.IntegerField(default=0)
    
    # Preprocessing information for sklearn models
    preprocessing_info = models.JSONField(null=True, blank=True, help_text='Information about preprocessing steps applied')
    train_loss = models.FloatField(null=True, blank=True)
    val_loss = models.FloatField(null=True, blank=True)
    train_accuracy = models.FloatField(null=True, blank=True)
    val_accuracy = models.FloatField(null=True, blank=True)
    progress = models.FloatField(default=0.0)  # Overall progress 0-1
    training_logs = models.JSONField(default=list)  # Store training logs
    
    # Cross-validation metrics storage
    cv_scores = models.JSONField(null=True, blank=True, help_text='All fold scores from cross-validation')
    cv_best_score = models.FloatField(null=True, blank=True, help_text='Best score from cross-validation')
    cv_mean_score = models.FloatField(null=True, blank=True, help_text='Average score from cross-validation')
    cv_worst_score = models.FloatField(null=True, blank=True, help_text='Lowest score from cross-validation')
    cv_std_score = models.FloatField(null=True, blank=True, help_text='Standard deviation of CV scores')
    
    def __str__(self):
        return f"{self.model_type} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def delete(self, *args, **kwargs):
        """Override delete to update model definition's best score"""
        model_def = self.model_definition
        super().delete(*args, **kwargs)
        
        # Update best score after deletion
        if model_def:
            model_def.update_best_score()


class WeatherPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='weatherpredictions')
    training_session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE)
    prediction_date = models.DateField()
    region = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    predictions = models.JSONField()  # Store all predicted variables
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['training_session', 'prediction_date', 'latitude', 'longitude']