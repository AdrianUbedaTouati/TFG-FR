"""
Constants and configuration values for ML Trainer module
"""

# Model types that support neural networks
NEURAL_NETWORK_MODELS = ['lstm', 'gru', 'cnn', 'transformer']

# Traditional ML models
TRADITIONAL_ML_MODELS = ['decision_tree', 'random_forest', 'xgboost']

# Default hyperparameters
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VALIDATION_SPLIT = 0.15
DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_DROPOUT_RATE = 0.2

# Data analysis constants
HISTOGRAM_BINS = 20
OUTLIER_THRESHOLD = 3  # Standard deviations for outlier detection
MIN_SAMPLE_SIZE = 100
MAX_CATEGORICAL_VALUES = 50
CORRELATION_THRESHOLD = 0.7

# File limits
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_DATASET_ROWS = 1000000
MAX_DATASET_COLUMNS = 1000

# Visualization settings
FIGURE_DPI = 100
DEFAULT_FIGURE_SIZE = (10, 6)
HEATMAP_FIGURE_SIZE = (12, 10)
MAX_PLOT_POINTS = 10000

# Cache settings
CACHE_TIMEOUT = 900  # 15 minutes
ANALYSIS_CACHE_KEY = 'dataset_analysis_{}'
COLUMNS_CACHE_KEY = 'dataset_columns_{}'

# Date formats to try when detecting datetime columns
DATE_FORMATS = [
    '%Y-%m-%d',
    '%d/%m/%Y',
    '%m/%d/%Y',
    '%Y/%m/%d',
    '%Y-%m-%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S',
    '%m/%d/%Y %H:%M:%S'
]

# Status messages
STATUS_TRAINING = 'training'
STATUS_COMPLETED = 'completed'
STATUS_FAILED = 'failed'
STATUS_PENDING = 'pending'

# Error messages
ERROR_DATASET_NOT_FOUND = "Dataset not found"
ERROR_INVALID_MODEL_TYPE = "Invalid model type: {}"
ERROR_INVALID_FRAMEWORK = "Invalid framework: {}"
ERROR_FILE_TOO_LARGE = "File size exceeds maximum limit of {} MB"
ERROR_PARSING_FAILED = "Failed to parse the uploaded file"
ERROR_TRAINING_FAILED = "Model training failed: {}"
ERROR_NORMALIZATION_FAILED = "Normalization failed: {}"

# Success messages
SUCCESS_MODEL_CREATED = "Model created successfully"
SUCCESS_TRAINING_STARTED = "Training started successfully"
SUCCESS_DATASET_UPLOADED = "Dataset uploaded successfully"
SUCCESS_NORMALIZATION_COMPLETE = "Normalization completed successfully"
SUCCESS_CODE_EXPORTED = "Model code exported successfully"
SUCCESS_CODE_IMPORTED = "Model code imported successfully"