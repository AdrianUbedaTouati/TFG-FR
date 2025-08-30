"""
ML Trainer Views Package

Organized into modules for better maintainability:
- dataset_views: Dataset management and analysis
- model_views: Model definition and management
- training_views: Training session management
- prediction_views: Prediction and inference
- export_views: Code export/import functionality
- normalization_views: Dataset normalization
- report_views: Report generation
"""

# Import all views to maintain backward compatibility
from .dataset_views import *
from .model_views import *
from .training_views import *
from .prediction_views import *
from .export_views import *
from .normalization_views import *
from .report_views import *
from .split_count_view import *
from .split_download_view import *