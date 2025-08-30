"""
Centralized normalization method mappings
This module provides a unified interface for normalization methods across the application.
"""

from .normalization_methods import NumNorm, TextNorm

# Mapping from string representations to enum values
# This allows frontend and database to use simple strings while backend uses enums
NUMERIC_METHOD_MAPPING = {
    # Standard names
    'min_max': NumNorm.MIN_MAX,
    'z_score': NumNorm.Z_SCORE,
    'lstm_tcn': NumNorm.LSTM_TCN,
    'cnn': NumNorm.CNN,
    'transformer': NumNorm.TRANSFORMER,
    'tree': NumNorm.TREE,
    
    # Uppercase versions (from frontend)
    'MIN_MAX': NumNorm.MIN_MAX,
    'Z_SCORE': NumNorm.Z_SCORE,
    'LSTM_TCN': NumNorm.LSTM_TCN,
    'CNN': NumNorm.CNN,
    'TRANSFORMER': NumNorm.TRANSFORMER,
    'TREE': NumNorm.TREE,
    
    # Alternative names for compatibility
    'minmax': NumNorm.MIN_MAX,
    'standard': NumNorm.Z_SCORE,
    'standardization': NumNorm.Z_SCORE,
    'robust': NumNorm.TRANSFORMER,
    'robust_scaler': NumNorm.TRANSFORMER,
    'none': NumNorm.TREE,
    'no_normalization': NumNorm.TREE,
}

TEXT_METHOD_MAPPING = {
    # Standard names
    'lower': TextNorm.LOWER,
    'strip': TextNorm.STRIP,
    'label_encoding': TextNorm.LABEL_ENCODING,
    'one_hot': TextNorm.ONE_HOT,
    
    # Uppercase versions (from frontend)
    'LOWER': TextNorm.LOWER,
    'STRIP': TextNorm.STRIP,
    'LABEL_ENCODING': TextNorm.LABEL_ENCODING,
    'ONE_HOT': TextNorm.ONE_HOT,
    
    # Alternative names for compatibility
    'lowercase': TextNorm.LOWER,
    'label': TextNorm.LABEL_ENCODING,
    'labelencoding': TextNorm.LABEL_ENCODING,
    'onehot': TextNorm.ONE_HOT,
    'one-hot': TextNorm.ONE_HOT,
}

# Reverse mappings (enum to string)
NUMERIC_ENUM_TO_STRING = {
    NumNorm.MIN_MAX: 'min_max',
    NumNorm.Z_SCORE: 'z_score',
    NumNorm.LSTM_TCN: 'lstm_tcn',
    NumNorm.CNN: 'cnn',
    NumNorm.TRANSFORMER: 'transformer',
    NumNorm.TREE: 'tree',
}

TEXT_ENUM_TO_STRING = {
    TextNorm.LOWER: 'lower',
    TextNorm.STRIP: 'strip',
    TextNorm.LABEL_ENCODING: 'label_encoding',
    TextNorm.ONE_HOT: 'one_hot',
}

# Model type to recommended normalization method
MODEL_TYPE_NORMALIZATION = {
    'linear_regression': 'z_score',
    'logistic_regression': 'z_score',
    'svm': 'z_score',
    'neural_network': 'min_max',
    'random_forest': 'tree',
    'gradient_boosting': 'tree',
    'xgboost': 'tree',
    'decision_tree': 'tree',
    'lstm': 'lstm_tcn',
    'tcn': 'lstm_tcn',
    'cnn': 'cnn',
    'transformer': 'transformer',
}

def get_numeric_enum(method_string: str) -> NumNorm:
    """Get numeric normalization enum from string representation"""
    method_lower = method_string.lower().strip()
    return NUMERIC_METHOD_MAPPING.get(method_lower, NumNorm.MIN_MAX)

def get_text_enum(method_string: str) -> TextNorm:
    """Get text normalization enum from string representation"""
    method_lower = method_string.lower().strip()
    return TEXT_METHOD_MAPPING.get(method_lower, TextNorm.LOWER)

def get_recommended_normalization(model_type: str) -> str:
    """Get recommended normalization method for a model type"""
    return MODEL_TYPE_NORMALIZATION.get(model_type.lower(), 'min_max')