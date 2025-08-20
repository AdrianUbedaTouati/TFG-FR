import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import json
import joblib
import os
import requests
from datetime import datetime, timedelta
from django.utils import timezone
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
import torch
import warnings
warnings.filterwarnings('ignore')

# Import PyTorch utilities
from .ml_utils_pytorch import (
    build_pytorch_model, train_pytorch_model, 
    save_pytorch_model, load_pytorch_model
)

# Import training callbacks
from .training_callbacks import SessionProgressCallback, SklearnProgressCallback

# Import sklearn preprocessor
from .sklearn_preprocessor import SklearnPreprocessor
from .sklearn_preprocessing_fix import SklearnPreprocessingPipeline


def get_model_config(model_type):
    """Get default hyperparameters for each model type"""
    configs = {
        'lstm': {
            'units': 50,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'layers': 2,
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'return_sequences': False,
            'optimizer': 'adam',
            'loss': 'mse',
            'sequence_length': 10
        },
        'gru': {
            'units': 50,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'layers': 2,
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'return_sequences': False,
            'optimizer': 'adam',
            'loss': 'mse',
            'sequence_length': 10
        },
        'cnn': {
            'filters': 64,
            'kernel_size': 3,
            'pool_size': 2,
            'dense_units': 100,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'activation': 'relu',
            'optimizer': 'adam',
            'loss': 'mse',
            'sequence_length': 10
        },
        'decision_tree': {
            # Basic parameters
            'preset': 'balanceado',
            'problem_type': 'regression',
            'criterion': 'squared_error',
            'splitter': 'best',
            'max_depth_enabled': True,
            'max_depth': 12,
            'min_samples_leaf': 1,
            'min_samples_leaf_fraction': False,
            'max_features': None,
            
            # Advanced parameters
            'min_samples_split': 2,
            'min_weight_fraction_leaf': 0.0,
            'min_impurity_decrease': 0.0,
            'max_leaf_nodes': None,
            'ccp_alpha': 0.0,
            
            # Advanced criterion
            'criterion_advanced': 'default',
            
            # Classification specific
            'class_weight': None,
            'decision_threshold': 0.5,
            'output_type': 'class',
            
            # Other
            'random_state': None,
            'validation_method': 'holdout'
        },
        'transformer': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'loss': 'mse',
            'sequence_length': 10
        },
        'random_forest': {
            # Basic options with improved defaults
            'preset': 'balanceado',
            'problem_type': 'auto',  # Auto-detect based on target
            'n_estimators': 300,
            'max_depth_enabled': False,
            'max_depth': None,
            'max_features': 'auto',  # Will be set based on problem_type
            'criterion': 'auto',  # Will be set based on problem_type
            'class_weight': None,  # None, 'balanced', or 'balanced_subsample'
            'validation_method': 'holdout',
            'test_size': 20,  # Percentage
            'cv_folds': 5,
            'stratified': True,  # For classification
            'time_series_split': False,
            
            # Advanced options with scikit-learn defaults
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'min_impurity_decrease': 0.0,
            'max_leaf_nodes': None,
            'ccp_alpha': 0.0,
            'bootstrap': True,
            'max_samples': None,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': 0,
            'warm_start': False,
            
            # Classification specific
            'decision_threshold': 0.5,
            'threshold_optimization': 'f1',  # accuracy, f1, precision, recall
            
            # Feature engineering
            'encode_categorical': True,
            'encoding_method': 'onehot',  # onehot, target, ordinal
            'add_cyclic_features': True,
            'cyclic_columns': [],  # Will be auto-detected
            
            # Data quality
            'handle_missing': 'mean',  # mean, median, mode, drop
            'check_data_leakage': True,
            'min_samples_warning': 100
        },
        'xgboost': {
            # Basic parameters
            'preset': 'balanceado',
            'problem_type': 'regression',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'rmse',
            
            # Early stopping
            'early_stopping_enabled': True,
            'early_stopping_rounds': 50,
            
            # Advanced parameters
            'min_child_weight': 1,
            'gamma': 0,
            'reg_lambda': 1,
            'reg_alpha': 0,
            'max_delta_step': 0,
            
            # Column sampling
            'colsample_bylevel': 1.0,
            'colsample_bynode': 1.0,
            
            # Tree method
            'tree_method': 'auto',
            'max_bin': 256,
            'grow_policy': 'depthwise',
            'max_leaves': None,
            
            # Classification specific
            'scale_pos_weight': 1,
            
            # Booster
            'booster': 'gbtree',
            'rate_drop': 0.1,
            'skip_drop': 0.5,
            
            # Execution
            'random_state': None,
            'n_jobs': -1,
            'verbosity': 1,
            'use_gpu': False,
            'device': 'cpu',
            'objective': 'reg:squarederror'
        }
    }
    return configs.get(model_type, {})


def get_normalization_methods(model_type):
    """Get available normalization methods for each model type"""
    neural_methods = ['min_max', 'standard', 'robust']
    tree_methods = ['none', 'min_max', 'standard']
    
    if model_type in ['lstm', 'cnn', 'transformer']:
        return neural_methods
    elif model_type in ['decision_tree', 'random_forest', 'xgboost']:
        return tree_methods
    return ['none']


def get_metrics(model_type):
    """Get available metrics for each model type"""
    regression_metrics = ['mae', 'mse', 'rmse', 'r2']
    classification_metrics = ['accuracy', 'roc_auc', 'f1']
    
    # For weather prediction, we'll mainly use regression metrics
    return regression_metrics


def get_scaler(method):
    """Get the appropriate scaler based on method"""
    if method == 'min_max':
        return MinMaxScaler()
    elif method == 'standard':
        return StandardScaler()
    elif method == 'robust':
        return RobustScaler()
    return None


def create_sequences(X, y, sequence_length):
    """Create sequences for time series models"""
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)


def prepare_data(session, is_neural_network=False):
    """Prepare data for training"""
    df = pd.read_csv(session.dataset.file.path)
    
    print(f"[prepare_data] Dataset shape: {df.shape}")
    print(f"[prepare_data] Predictor columns: {session.predictor_columns}")
    print(f"[prepare_data] Target columns: {session.target_columns}")
    
    # Check if target columns exist
    for col in session.target_columns:
        if col not in df.columns:
            raise ValueError(f"Target column '{col}' not found in dataset. Available columns: {list(df.columns)}")
    
    # Select columns
    X = df[session.predictor_columns].values
    y_raw = df[session.target_columns]
    
    # Handle categorical targets
    target_encoders = {}
    y = y_raw.values.copy()
    
    # For sklearn models, encode categorical targets
    if session.model_type in ['random_forest', 'decision_tree', 'xgboost']:
        for i, col in enumerate(session.target_columns):
            col_data = y_raw[col]
            if col_data.dtype == 'object' or col_data.dtype.name == 'category':
                print(f"[prepare_data] Encoding categorical target column: {col}")
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                y[:, i] = encoder.fit_transform(col_data)
                target_encoders[col] = encoder
                print(f"[prepare_data] Unique classes in {col}: {encoder.classes_}")
                print(f"[prepare_data] Sample encoded values: {y[:5, i]}")
    else:
        y = y_raw.values
    
    print(f"[prepare_data] X shape before reshape: {X.shape}")
    print(f"[prepare_data] y shape before reshape: {y.shape}")
    
    # Ensure X is always 2D
    if len(X.shape) == 1:
        print(f"[prepare_data] X is 1D, reshaping to 2D...")
        X = X.reshape(-1, 1)
    
    # Check if we need to determine problem type based on target data
    if session.model_type in ['random_forest', 'decision_tree', 'xgboost']:
        # Check if target is categorical (for single target)
        if len(session.target_columns) == 1:
            target_col = session.target_columns[0]
            unique_values = df[target_col].nunique()
            print(f"[prepare_data] Target column '{target_col}' has {unique_values} unique values")
            print(f"[prepare_data] Sample target values: {df[target_col].value_counts().head()}")
            
            # If target has few unique values and they are integers, it's likely classification
            if unique_values < 20 and df[target_col].dtype in ['int64', 'int32', 'float64', 'float32']:
                print(f"[prepare_data] Detected categorical target with {unique_values} classes")
    
    # Ensure y is 2D
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    # Get sequence length for neural networks
    sequence_length = session.hyperparameters.get('sequence_length', 10) if is_neural_network else None
    
    # For time series models, we need to create sequences before splitting
    if is_neural_network and session.model_type in ['lstm', 'gru', 'cnn', 'transformer']:
        # Normalize before creating sequences
        scaler_X = get_scaler(session.normalization_method)
        scaler_y = get_scaler(session.normalization_method)
        
        if scaler_X:
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y)
        
        # Create sequences
        X, y = create_sequences(X, y, sequence_length)
        
        # Split after creating sequences
        n = len(X)
        train_end = int(n * session.train_split)
        val_end = int(n * (session.train_split + session.val_split))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]
    else:
        # Regular splitting for non-sequence models
        n = len(X)
        train_end = int(n * session.train_split)
        val_end = int(n * (session.train_split + session.val_split))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        # Normalize if needed
        scaler_X = get_scaler(session.normalization_method)
        scaler_y = get_scaler(session.normalization_method)
        
        if scaler_X:
            print(f"[prepare_data] Normalizing data with method: {session.normalization_method}")
            print(f"[prepare_data] Shapes before normalization - X_train: {X_train.shape}, y_train: {y_train.shape}")
            
            X_train = scaler_X.fit_transform(X_train)
            X_val = scaler_X.transform(X_val)
            X_test = scaler_X.transform(X_test)
            
            # For classification problems, we typically don't normalize y
            # Check if this is likely a classification problem
            is_classification = session.model_type in ['random_forest', 'decision_tree', 'xgboost'] and len(np.unique(y_train)) < 20
            
            if is_classification:
                print(f"[prepare_data] Skipping y normalization for classification problem")
                scaler_y = None  # Set to None for classification
            else:
                y_train = scaler_y.fit_transform(y_train)
                y_val = scaler_y.transform(y_val)
                y_test = scaler_y.transform(y_test)
            
            print(f"[prepare_data] Shapes after normalization - X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test), (scaler_X, scaler_y), target_encoders


def get_optimizer(optimizer_name, learning_rate):
    """Get Keras optimizer from string name"""
    optimizers = {
        'adam': Adam(learning_rate=learning_rate),
        'sgd': SGD(learning_rate=learning_rate),
        'rmsprop': RMSprop(learning_rate=learning_rate),
        'adagrad': Adagrad(learning_rate=learning_rate),
        'adadelta': Adadelta(learning_rate=learning_rate),
        'adamax': Adamax(learning_rate=learning_rate),
        'nadam': Nadam(learning_rate=learning_rate)
    }
    return optimizers.get(optimizer_name.lower(), Adam(learning_rate=learning_rate))


def build_lstm_model(input_shape, output_shape, hyperparams):
    """Build LSTM model"""
    model = models.Sequential()
    
    # Extract hyperparameters
    units = hyperparams.get('units', 50)
    dropout = hyperparams.get('dropout', 0.2)
    num_layers = hyperparams.get('layers', 2)
    activation = hyperparams.get('activation', 'tanh')
    recurrent_activation = hyperparams.get('recurrent_activation', 'sigmoid')
    
    # Add LSTM layers
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        if i == 0:
            model.add(layers.LSTM(
                units, 
                activation=activation,
                recurrent_activation=recurrent_activation,
                return_sequences=return_sequences,
                input_shape=input_shape
            ))
        else:
            model.add(layers.LSTM(
                units, 
                activation=activation,
                recurrent_activation=recurrent_activation,
                return_sequences=return_sequences
            ))
        
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(output_shape[0]))
    
    return model


def build_gru_model(input_shape, output_shape, hyperparams):
    """Build GRU model"""
    model = models.Sequential()
    
    # Extract hyperparameters
    units = hyperparams.get('units', 50)
    dropout = hyperparams.get('dropout', 0.2)
    num_layers = hyperparams.get('layers', 2)
    activation = hyperparams.get('activation', 'tanh')
    recurrent_activation = hyperparams.get('recurrent_activation', 'sigmoid')
    
    # Add GRU layers
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        if i == 0:
            model.add(layers.GRU(
                units, 
                activation=activation,
                recurrent_activation=recurrent_activation,
                return_sequences=return_sequences,
                input_shape=input_shape
            ))
        else:
            model.add(layers.GRU(
                units, 
                activation=activation,
                recurrent_activation=recurrent_activation,
                return_sequences=return_sequences
            ))
        
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(output_shape[0]))
    
    return model


def build_cnn_model(input_shape, output_shape, hyperparams):
    """Build CNN model for time series"""
    model = models.Sequential()
    
    # Extract hyperparameters
    filters = hyperparams.get('filters', 64)
    kernel_size = hyperparams.get('kernel_size', 3)
    activation = hyperparams.get('activation', 'relu')
    pool_size = hyperparams.get('pool_size', 2)
    dense_units = hyperparams.get('dense_units', 100)
    dropout = hyperparams.get('dropout', 0.2)
    
    # CNN layers
    model.add(layers.Conv1D(
        filters=filters, 
        kernel_size=kernel_size, 
        activation=activation,
        input_shape=input_shape
    ))
    model.add(layers.MaxPooling1D(pool_size=pool_size))
    
    if dropout > 0:
        model.add(layers.Dropout(dropout))
    
    # Add second conv layer
    model.add(layers.Conv1D(
        filters=filters*2, 
        kernel_size=kernel_size, 
        activation=activation
    ))
    model.add(layers.GlobalMaxPooling1D())
    
    # Dense layers
    model.add(layers.Dense(dense_units, activation=activation))
    if dropout > 0:
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(output_shape[0]))
    
    return model


def build_transformer_model(input_shape, output_shape, hyperparams):
    """Build Transformer model for time series"""
    # Extract hyperparameters
    d_model = hyperparams.get('d_model', 128)
    num_heads = hyperparams.get('nhead', 8)
    num_layers = hyperparams.get('num_layers', 3)
    dropout = hyperparams.get('dropout', 0.1)
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # Linear projection to d_model dimensions
    x = layers.Dense(d_model)(inputs)
    
    # Positional encoding (simplified)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    positions = tf.expand_dims(positions, 0)
    positions = tf.expand_dims(positions, -1)
    positions = tf.cast(positions, tf.float32)
    position_embedding = layers.Dense(d_model)(positions)
    x = x + position_embedding
    
    # Transformer blocks
    for _ in range(num_layers):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads,
            dropout=dropout
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed forward network
        ffn_output = layers.Dense(d_model * 4, activation='relu')(x)
        ffn_output = layers.Dense(d_model)(ffn_output)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output layer
    outputs = layers.Dense(output_shape[0])(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def build_custom_model(input_shape, output_shape, custom_architecture):
    """Build model from custom architecture JSON"""
    model = models.Sequential()
    
    for i, layer_config in enumerate(custom_architecture):
        layer_type = layer_config.get('type')
        params = layer_config.get('params', {})
        
        # Add input_shape to first layer
        if i == 0 and 'input_shape' not in params:
            params['input_shape'] = input_shape
        
        # Create layer based on type
        if layer_type == 'Dense':
            model.add(layers.Dense(
                units=params.get('units', 32),
                activation=params.get('activation', 'linear'),
                use_bias=params.get('use_bias', True),
                kernel_initializer=params.get('kernel_initializer', 'glorot_uniform')
            ))
        
        elif layer_type == 'LSTM':
            model.add(layers.LSTM(
                units=params.get('units', 50),
                activation=params.get('activation', 'tanh'),
                recurrent_activation=params.get('recurrent_activation', 'sigmoid'),
                return_sequences=params.get('return_sequences', False),
                dropout=params.get('dropout', 0.0),
                recurrent_dropout=params.get('recurrent_dropout', 0.0)
            ))
        
        elif layer_type == 'GRU':
            model.add(layers.GRU(
                units=params.get('units', 50),
                activation=params.get('activation', 'tanh'),
                recurrent_activation=params.get('recurrent_activation', 'sigmoid'),
                return_sequences=params.get('return_sequences', False),
                dropout=params.get('dropout', 0.0),
                recurrent_dropout=params.get('recurrent_dropout', 0.0)
            ))
        
        elif layer_type == 'Conv1D':
            model.add(layers.Conv1D(
                filters=params.get('filters', 32),
                kernel_size=params.get('kernel_size', 3),
                strides=params.get('strides', 1),
                padding=params.get('padding', 'valid'),
                activation=params.get('activation', 'linear')
            ))
        
        elif layer_type == 'MaxPooling1D':
            model.add(layers.MaxPooling1D(
                pool_size=params.get('pool_size', 2),
                strides=params.get('strides', None),
                padding=params.get('padding', 'valid')
            ))
        
        elif layer_type == 'Dropout':
            model.add(layers.Dropout(rate=params.get('rate', 0.5)))
        
        elif layer_type == 'BatchNormalization':
            model.add(layers.BatchNormalization(
                momentum=params.get('momentum', 0.99),
                epsilon=params.get('epsilon', 0.001)
            ))
        
        elif layer_type == 'Flatten':
            model.add(layers.Flatten())
        
        elif layer_type == 'GlobalMaxPooling1D':
            model.add(layers.GlobalMaxPooling1D())
        
        elif layer_type == 'GlobalAveragePooling1D':
            model.add(layers.GlobalAveragePooling1D())
    
    # Ensure output layer matches expected shape
    if len(model.layers) == 0 or model.layers[-1].output_shape[-1] != output_shape[0]:
        model.add(layers.Dense(output_shape[0]))
    
    return model


def train_neural_network(session, model_type, hyperparams, X_train, y_train, X_val, y_val):
    """Train neural network models"""
    # Get input and output shapes
    input_shape = X_train.shape[1:]
    output_shape = (y_train.shape[1] if len(y_train.shape) > 1 else 1,)
    
    # Build model based on type or custom architecture
    if session.use_custom_architecture and session.custom_architecture:
        model = build_custom_model(input_shape, output_shape, session.custom_architecture)
    elif model_type == 'lstm':
        model = build_lstm_model(input_shape, output_shape, hyperparams)
    elif model_type == 'gru':
        model = build_gru_model(input_shape, output_shape, hyperparams)
    elif model_type == 'cnn':
        model = build_cnn_model(input_shape, output_shape, hyperparams)
    elif model_type == 'transformer':
        model = build_transformer_model(input_shape, output_shape, hyperparams)
    else:
        raise ValueError(f"Unsupported neural network type: {model_type}")
    
    # Compile model
    optimizer = get_optimizer(
        hyperparams.get('optimizer', 'adam'),
        hyperparams.get('learning_rate', 0.001)
    )
    loss = hyperparams.get('loss', 'mse')
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mae', 'mse']
    )
    
    # Calculate batches per epoch for progress tracking
    batch_size = hyperparams.get('batch_size', 32)
    total_epochs = hyperparams.get('epochs', 50)
    total_batches_per_epoch = (len(X_train) + batch_size - 1) // batch_size
    
    # Create progress callback
    progress_callback = SessionProgressCallback(
        session=session,
        total_epochs=total_epochs,
        total_batches_per_epoch=total_batches_per_epoch
    )
    
    # Other callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=total_epochs,
        batch_size=batch_size,
        callbacks=[progress_callback, early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history.history


def _prepare_random_forest_params(hyperparams):
    """Prepare Random Forest parameters for scikit-learn"""
    # Start with a copy of hyperparams
    clean_params = hyperparams.copy()
    
    # Remove UI-specific and feature engineering parameters
    ui_only_params = [
        'preset', 'problem_type', 'max_depth_enabled', 'class_weight_balanced', 
        'validation_method', 'criterion_advanced', 'class_weight_custom',
        'decision_threshold', 'output_type', 'max_features_fraction',
        'test_size', 'cv_folds', 'stratified', 'time_series_split',
        'threshold_optimization', 'encode_categorical', 'encoding_method',
        'add_cyclic_features', 'cyclic_columns', 'handle_missing',
        'check_data_leakage', 'min_samples_warning',
        'epochs', 'batch_size', 'learning_rate',  # Neural network params
        'use_custom_architecture', 'custom_architecture'  # Architecture params
    ]
    for param in ui_only_params:
        clean_params.pop(param, None)
    
    # Handle max_depth logic
    if not hyperparams.get('max_depth_enabled', False):
        clean_params['max_depth'] = None
    
    # Handle max_features based on problem type
    max_features = hyperparams.get('max_features', 'auto')
    problem_type = hyperparams.get('problem_type', 'regression')
    
    if max_features == 'auto':
        # Set default based on problem type
        if problem_type == 'classification':
            max_features = 'sqrt'
        else:
            max_features = 1.0
    elif max_features == 'custom':
        max_features = hyperparams.get('max_features_fraction', 0.5)
    elif max_features in ['sqrt', 'log2']:
        # Keep as string
        pass
    else:
        # Try to convert to float
        try:
            max_features = float(max_features)
        except:
            max_features = 'sqrt'  # fallback
    
    clean_params['max_features'] = max_features
    
    # Handle criterion selection
    criterion = hyperparams.get('criterion', 'auto')
    if criterion == 'auto':
        if problem_type == 'classification':
            clean_params['criterion'] = 'gini'
        else:
            clean_params['criterion'] = 'squared_error'
    elif criterion in ['gini', 'entropy', 'log_loss']:  # Classification criteria
        if problem_type == 'classification':
            clean_params['criterion'] = criterion
        else:
            clean_params['criterion'] = 'squared_error'  # fallback for regression
    elif criterion in ['squared_error', 'absolute_error', 'poisson']:  # Regression criteria
        if problem_type == 'regression':
            clean_params['criterion'] = criterion
        else:
            clean_params['criterion'] = 'gini'  # fallback for classification
    
    # Handle class_weight
    class_weight = hyperparams.get('class_weight', None)
    if problem_type == 'classification' and class_weight in ['balanced', 'balanced_subsample']:
        clean_params['class_weight'] = class_weight
    else:
        clean_params.pop('class_weight', None)
    
    # Handle OOB score - only valid if bootstrap is True
    if clean_params.get('bootstrap', True) and hyperparams.get('oob_score', False):
        clean_params['oob_score'] = True
    else:
        clean_params['oob_score'] = False
    
    # Clean up None values
    keys_to_check = list(clean_params.keys())
    for key in keys_to_check:
        value = clean_params[key]
        if value is None or value == 'None':
            if key in ['max_depth', 'max_leaf_nodes', 'random_state', 'max_samples']:
                clean_params[key] = None
            else:
                clean_params.pop(key, None)
    
    return clean_params


def _prepare_decision_tree_params(hyperparams):
    """Prepare Decision Tree parameters for scikit-learn"""
    # Start with a copy of hyperparams
    clean_params = hyperparams.copy()
    
    # Remove UI-specific parameters that don't belong to scikit-learn
    ui_only_params = [
        'preset', 'problem_type', 'max_depth_enabled', 'min_samples_leaf_fraction',
        'validation_method', 'criterion_advanced', 'decision_threshold', 'output_type',
        'epochs', 'batch_size', 'learning_rate',  # Neural network params that don't apply to DT
        'use_custom_architecture', 'custom_architecture'  # Architecture params for neural networks
    ]
    for param in ui_only_params:
        clean_params.pop(param, None)
    
    # Handle max_depth logic
    if not hyperparams.get('max_depth_enabled', True):
        clean_params['max_depth'] = None
    
    # Handle criterion selection
    criterion_advanced = hyperparams.get('criterion_advanced', 'default')
    if criterion_advanced != 'default':
        clean_params['criterion'] = criterion_advanced
    
    # Handle max_features conversion
    if clean_params.get('max_features') == 'None':
        clean_params['max_features'] = None
    
    # Handle class_weight
    if hyperparams.get('problem_type') == 'classification':
        class_weight = hyperparams.get('class_weight', None)
        if class_weight == 'None':
            clean_params['class_weight'] = None
        elif class_weight == 'balanced':
            clean_params['class_weight'] = 'balanced'
        else:
            clean_params.pop('class_weight', None)
    else:
        clean_params.pop('class_weight', None)
    
    # Remove None values where appropriate
    for key in list(clean_params.keys()):
        if clean_params[key] is None and key not in ['max_depth', 'max_leaf_nodes', 'random_state', 'max_features']:
            clean_params.pop(key)
    
    return clean_params


def _prepare_xgboost_params(hyperparams):
    """Prepare XGBoost parameters for training"""
    # Start with a copy of hyperparams
    clean_params = hyperparams.copy()
    
    # Remove UI-specific parameters that don't belong to XGBoost
    ui_only_params = [
        'preset', 'problem_type', 'early_stopping_enabled', 'use_gpu', 
        'max_features_fraction', 'decision_threshold', 'output_type',
        'epochs', 'batch_size',  # Neural network params (XGBoost uses n_estimators instead)
        'use_custom_architecture', 'custom_architecture'  # Architecture params for neural networks
    ]
    for param in ui_only_params:
        clean_params.pop(param, None)
    
    # Handle max_depth (0 means no limit in XGBoost)
    if clean_params.get('max_depth') == 0:
        clean_params.pop('max_depth', None)
    
    # Handle GPU configuration
    if hyperparams.get('use_gpu', False):
        clean_params['device'] = 'cuda'
        if clean_params.get('tree_method') == 'auto':
            clean_params['tree_method'] = 'hist'
    else:
        clean_params['device'] = 'cpu'
    
    # Handle objective based on problem type
    problem_type = hyperparams.get('problem_type', 'regression')
    if problem_type == 'classification':
        # This will be adjusted for multiclass in the training function
        clean_params['objective'] = 'binary:logistic'
    else:
        clean_params['objective'] = 'reg:squarederror'
    
    # Handle eval_metric
    if clean_params.get('eval_metric') == 'auto':
        if problem_type == 'classification':
            clean_params['eval_metric'] = 'logloss'
        else:
            clean_params['eval_metric'] = 'rmse'
    
    # Handle booster-specific parameters
    booster = clean_params.get('booster', 'gbtree')
    if booster != 'dart':
        # Remove DART-specific parameters if not using DART
        clean_params.pop('rate_drop', None)
        clean_params.pop('skip_drop', None)
    
    # Handle grow_policy specific parameters
    if clean_params.get('grow_policy') != 'lossguide':
        clean_params.pop('max_leaves', None)
    
    # Handle classification-specific parameters
    if problem_type != 'classification':
        clean_params.pop('scale_pos_weight', None)
    
    # Remove None values
    clean_params = {k: v for k, v in clean_params.items() if v is not None}
    
    # Handle random_state
    if 'random_state' in clean_params and clean_params['random_state'] is None:
        clean_params.pop('random_state', None)
    
    return clean_params


def train_sklearn_model(session, model_type, hyperparams, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train scikit-learn models with enhanced feature engineering"""
    print(f"[train_sklearn_model] Model type: {model_type}")
    print(f"[train_sklearn_model] Hyperparameters received: {hyperparams}")
    print(f"[train_sklearn_model] Input shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Create progress callback
    progress_callback = SklearnProgressCallback(session, model_type)
    
    # Initialize preprocessing pipeline
    preprocessing_pipeline = None
    
    # Apply feature engineering for Random Forest
    if model_type == 'random_forest' and hyperparams.get('encode_categorical', True):
        # Convert to DataFrame for easier manipulation
        import pandas as pd
        predictor_columns = session.predictor_columns
        X_train_df = pd.DataFrame(X_train, columns=predictor_columns)
        X_val_df = pd.DataFrame(X_val, columns=predictor_columns)
        
        # Auto-detect problem type if needed
        if hyperparams.get('problem_type', 'auto') == 'auto':
            target_col = session.target_columns[0]
            # Create a temporary DataFrame with target to analyze
            temp_df = pd.DataFrame(y_train, columns=[target_col])
            problem_type, subtype = detect_problem_type(target_col, temp_df)
            hyperparams['problem_type'] = problem_type
            print(f"[train_sklearn_model] Auto-detected problem type: {problem_type} ({subtype})")
        
        # Detect categorical columns
        categorical_columns = []
        for col in predictor_columns:
            # Check if column has few unique values or is object type
            if X_train_df[col].dtype == 'object' or X_train_df[col].nunique() < 10:
                categorical_columns.append(col)
        
        # Auto-detect cyclic columns
        cyclic_columns = hyperparams.get('cyclic_columns', [])
        if hyperparams.get('add_cyclic_features', True) and not cyclic_columns:
            for col in predictor_columns:
                if any(term in col.lower() for term in ['hour', 'day', 'month', 'bearing', 'degree', 'angle']):
                    cyclic_columns.append(col)
        
        # Create preprocessing pipeline
        encoding_method = hyperparams.get('encoding_method', 'onehot')
        preprocessing_pipeline = SklearnPreprocessingPipeline(
            predictor_columns=predictor_columns,
            categorical_columns=categorical_columns,
            cyclic_columns=cyclic_columns if hyperparams.get('add_cyclic_features', True) else [],
            encoding_method=encoding_method,
            normalization_method='none'  # We don't normalize here since it's already done
        )
        
        # Fit and transform the data
        print(f"[train_sklearn_model] Fitting preprocessing pipeline...")
        print(f"[train_sklearn_model] Categorical columns: {categorical_columns}")
        print(f"[train_sklearn_model] Cyclic columns: {cyclic_columns}")
        
        X_train = preprocessing_pipeline.fit_transform(X_train_df, y_train)
        X_val = preprocessing_pipeline.transform(X_val_df)
        
        # Process test data if provided
        if X_test is not None:
            X_test_df = pd.DataFrame(X_test, columns=predictor_columns)
            X_test = preprocessing_pipeline.transform(X_test_df)
            print(f"[train_sklearn_model] After preprocessing - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        else:
            print(f"[train_sklearn_model] After preprocessing - X_train: {X_train.shape}, X_val: {X_val.shape}")
        
        # Store preprocessing info in session for later use
        preprocessing_info = preprocessing_pipeline.get_preprocessing_info()
        session.preprocessing_info = preprocessing_info
        session.save()
        
        # Save the preprocessing pipeline with the model
        import os
        import tempfile
        pipeline_path = os.path.join(tempfile.gettempdir(), f'preprocessing_pipeline_{session.id}.pkl')
        preprocessing_pipeline.save(pipeline_path)
        preprocessing_info['pipeline_path'] = pipeline_path
        session.preprocessing_info = preprocessing_info
        session.save()
        
        # Check for data leakage
        if hyperparams.get('check_data_leakage', True):
            feature_names = preprocessing_info['feature_names_after_preprocessing']
            warnings = check_data_leakage(feature_names, session.target_columns[0], pd.DataFrame(X_train, columns=feature_names))
            if warnings:
                print(f"[train_sklearn_model] Data leakage warnings: {warnings}")
    
    if model_type == 'decision_tree':
        # Determine if it's classification or regression
        problem_type = hyperparams.get('problem_type', 'regression')
        clean_params = _prepare_decision_tree_params(hyperparams)
        
        if problem_type == 'classification':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(**clean_params)
        else:
            model = DecisionTreeRegressor(**clean_params)
    elif model_type == 'random_forest':
        # Determine if it's classification or regression
        problem_type = hyperparams.get('problem_type', 'regression')
        clean_params = _prepare_random_forest_params(hyperparams)
        
        print(f"[train_sklearn_model] Problem type: {problem_type}")
        print(f"[train_sklearn_model] Cleaned parameters for RandomForest: {clean_params}")
        
        if problem_type == 'classification':
            model = RandomForestClassifier(**clean_params)
        else:
            model = RandomForestRegressor(**clean_params)
        
        print(f"[train_sklearn_model] Model created successfully: {type(model).__name__}")
    elif model_type == 'xgboost':
        # Prepare XGBoost parameters
        problem_type = hyperparams.get('problem_type', 'regression')
        clean_params = _prepare_xgboost_params(hyperparams)
        
        # Check if multiclass
        is_multiclass = problem_type == 'classification' and len(np.unique(y_train)) > 2
        
        if problem_type == 'classification':
            if is_multiclass:
                clean_params['objective'] = 'multi:softprob'
                clean_params['num_class'] = len(np.unique(y_train))
                if clean_params.get('eval_metric') == 'logloss':
                    clean_params['eval_metric'] = 'mlogloss'
            model = xgb.XGBClassifier(**clean_params)
        else:
            model = xgb.XGBRegressor(**clean_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Start progress tracking
    progress_callback.on_train_begin()
    progress_callback.update_progress(0.1, f"Préparation de l'entraînement {model_type}...")
    
    # Train the model
    print(f"[train_sklearn_model] Training model...")
    print(f"[train_sklearn_model] Final data shapes before fit - X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    try:
        progress_callback.update_progress(0.3, f"Entraînement du modèle {model_type} en cours...")
        
        # Handle early stopping for XGBoost
        if model_type == 'xgboost' and hyperparams.get('early_stopping_enabled', True):
            early_stopping_rounds = hyperparams.get('early_stopping_rounds', 50)
            eval_set = [(X_val, y_val)]
            print(f"[train_sklearn_model] Training XGBoost with early stopping (rounds={early_stopping_rounds})")
            
            # For XGBoost, we can use a custom callback to track progress
            class XGBProgressCallback(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    # Update progress based on number of boosting rounds
                    n_estimators = hyperparams.get('n_estimators', 100)
                    progress = 0.3 + (0.5 * (epoch / n_estimators))
                    progress_callback.update_progress(
                        progress, 
                        f"XGBoost - Round {epoch + 1}/{n_estimators}"
                    )
                    return False
            
            model.fit(X_train, y_train, 
                      eval_set=eval_set, 
                      early_stopping_rounds=early_stopping_rounds,
                      callbacks=[XGBProgressCallback()],
                      verbose=False)
        else:
            print(f"[train_sklearn_model] Training {model_type} model...")
            
            # For Random Forest, we can track progress using warm_start
            if model_type == 'random_forest':
                n_estimators = hyperparams.get('n_estimators', 100)
                print(f"[train_sklearn_model] Training Random Forest with {n_estimators} trees...")
                
                # Enable warm_start to train incrementally
                model.set_params(warm_start=True)
                
                # Train in batches to show progress
                batch_size = max(10, n_estimators // 10)  # Train in 10 steps minimum
                trees_trained = 0
                
                for i in range(0, n_estimators, batch_size):
                    current_batch = min(batch_size, n_estimators - i)
                    model.set_params(n_estimators=trees_trained + current_batch)
                    model.fit(X_train, y_train)
                    trees_trained += current_batch
                    
                    # Update progress
                    progress = 0.3 + (0.5 * (trees_trained / n_estimators))
                    progress_callback.update_progress(
                        progress, 
                        f"Random Forest - {trees_trained}/{n_estimators} arbres entraînés"
                    )
                    
                    # Log OOB score if available
                    if hasattr(model, 'oob_score_') and model.oob_score_:
                        print(f"[train_sklearn_model] OOB Score after {trees_trained} trees: {model.oob_score_:.4f}")
                        progress_callback.update_message(
                            f"Score OOB actuel: {model.oob_score_:.4f}"
                        )
                
                # Disable warm_start after training
                model.set_params(warm_start=False)
                progress_callback.update_progress(0.8, f"Random Forest - {n_estimators} arbres entraînés avec succès")
                
            else:
                # For other models, simple fit
                model.fit(X_train, y_train)
                progress_callback.update_progress(0.8, f"Entraînement {model_type} terminé")
        
        print(f"[train_sklearn_model] Model training completed successfully!")
    except Exception as e:
        print(f"[train_sklearn_model] ERROR during model.fit: {str(e)}")
        print(f"[train_sklearn_model] Error type: {type(e).__name__}")
        import traceback
        print(f"[train_sklearn_model] Traceback: {traceback.format_exc()}")
        raise
    
    # Calculate validation metrics
    print(f"[train_sklearn_model] Calculating validation metrics...")
    progress_callback.update_progress(0.9, "Calcul des métriques de validation...")
    
    try:
        val_pred = model.predict(X_val)
        print(f"[train_sklearn_model] Validation predictions shape: {val_pred.shape}")
    except Exception as e:
        print(f"[train_sklearn_model] ERROR during validation prediction: {str(e)}")
        # If predict fails, it might be due to data shape issues
        if "Expected 2D array, got 1D array" in str(e):
            print(f"[train_sklearn_model] Attempting to reshape validation data...")
            if len(X_val.shape) == 1:
                X_val = X_val.reshape(-1, 1)
            val_pred = model.predict(X_val)
        else:
            raise
    
    # Choose appropriate metrics based on problem type
    if (model_type in ['random_forest', 'xgboost', 'decision_tree'] and 
        hyperparams.get('problem_type') == 'classification'):
        from sklearn.metrics import accuracy_score, log_loss
        try:
            train_pred = model.predict(X_train)
            history = {
                'train_accuracy': [accuracy_score(y_train, train_pred)],
                'val_accuracy': [accuracy_score(y_val, val_pred)],
                'train_loss': [log_loss(y_train, model.predict_proba(X_train))],
                'val_loss': [log_loss(y_val, model.predict_proba(X_val))]
            }
        except Exception as e:
            print(f"[train_sklearn_model] Warning: Could not calculate log_loss: {str(e)}")
            # Fallback to basic accuracy if log_loss fails
            train_pred = model.predict(X_train)
            history = {
                'train_accuracy': [accuracy_score(y_train, train_pred)],
                'val_accuracy': [accuracy_score(y_val, val_pred)]
            }
    else:
        # Regression metrics
        print(f"[train_sklearn_model] Calculating regression metrics...")
        try:
            train_pred = model.predict(X_train)
            history = {
                'train_loss': [mean_squared_error(y_train, train_pred)],
                'val_loss': [mean_squared_error(y_val, val_pred)]
            }
            print(f"[train_sklearn_model] Metrics calculated successfully")
        except Exception as e:
            print(f"[train_sklearn_model] ERROR calculating metrics: {str(e)}")
            raise
    
    # Update session with metrics
    if 'train_loss' in history:
        session.train_loss = history['train_loss'][0]
        session.val_loss = history['val_loss'][0]
    if 'train_accuracy' in history:
        session.train_accuracy = history['train_accuracy'][0]
        session.val_accuracy = history.get('val_accuracy', [0])[0]
    
    session.save()
    
    # Finish progress tracking
    progress_callback.on_train_end()
    
    # Return processed data along with model
    return {
        'model': model,
        'history': history,
        'preprocessing_pipeline': preprocessing_pipeline,
        'X_train_processed': X_train,
        'X_val_processed': X_val,
        'X_test_processed': X_test if X_test is not None else None
    }


def train_model(session):
    """Main training function"""
    try:
        session.status = 'training'
        session.save()
        
        # Log initial info
        print(f"[Training] Starting training for model type: {session.model_type}")
        print(f"[Training] Dataset: {session.dataset.name}")
        print(f"[Training] Predictor columns: {session.predictor_columns}")
        print(f"[Training] Target columns: {session.target_columns}")
        
        # Validate columns
        if len(session.predictor_columns) == 0:
            raise ValueError("No predictor columns selected. Please select at least one predictor column.")
        if len(session.target_columns) == 0:
            raise ValueError("No target columns selected. Please select at least one target column.")
        
        # Check if target column is being used as predictor
        overlap = set(session.predictor_columns) & set(session.target_columns)
        if overlap:
            raise ValueError(f"Columns cannot be both predictor and target: {overlap}")
        
        print(f"[Training] Number of predictors: {len(session.predictor_columns)}")
        print(f"[Training] Number of targets: {len(session.target_columns)}")
        
        # Check if it's a neural network
        is_neural = session.model_type in ['lstm', 'gru', 'cnn', 'transformer']
        
        # Prepare data
        print(f"[Training] Preparing data...")
        data, scalers, target_encoders = prepare_data(session, is_neural_network=is_neural)
        X_train, y_train, X_val, y_val, X_test, y_test = data
        scaler_X, scaler_y = scalers
        
        # Log data shapes
        print(f"[Training] Data shapes after preparation:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Ensure y data is properly shaped for sklearn models
        if session.model_type in ['decision_tree', 'random_forest', 'xgboost']:
            # For sklearn models, if y has only one column, it should be 1D
            if y_train.shape[1] == 1:
                print(f"[Training] Converting y data to 1D for sklearn model")
                y_train = y_train.ravel()
                y_val = y_val.ravel()
                y_test = y_test.ravel()
                print(f"[Training] New y shapes - train: {y_train.shape}, val: {y_val.shape}, test: {y_test.shape}")
        
        # Train model based on type
        print(f"[Training] Starting model training...")
        if session.model_type in ['decision_tree', 'random_forest', 'xgboost']:
            result = train_sklearn_model(
                session,
                session.model_type, 
                session.hyperparameters,
                X_train, y_train, 
                X_val, y_val,
                X_test, y_test
            )
            model = result['model']
            history = result['history']
            preprocessing_pipeline = result.get('preprocessing_pipeline')
            
            # Use processed data for evaluation
            if preprocessing_pipeline:
                X_test = result['X_test_processed']
            
            model_format = 'sklearn'
        else:
            # Train neural network based on framework
            framework = getattr(session, 'framework', 'keras')
            
            if framework == 'pytorch':
                # Build and train PyTorch model
                input_shape = X_train.shape[1:]
                output_shape = (y_train.shape[1] if len(y_train.shape) > 1 else 1,)
                
                model = build_pytorch_model(
                    session.model_type,
                    input_shape,
                    output_shape,
                    session.hyperparameters,
                    session.custom_architecture if session.use_custom_architecture else None
                )
                
                model, history = train_pytorch_model(
                    session,
                    model,
                    X_train, y_train,
                    X_val, y_val,
                    session.hyperparameters
                )
                model_format = 'pytorch'
            else:
                # Train Keras/TensorFlow model
                model, history = train_neural_network(
                    session,
                    session.model_type,
                    session.hyperparameters,
                    X_train, y_train,
                    X_val, y_val
                )
                model_format = 'tensorflow'
        
        # Test evaluation
        if model_format == 'pytorch':
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                y_pred = model(X_test_tensor).numpy()
        else:
            y_pred = model.predict(X_test)
        
        # For sequences, we need to handle the output shape
        if is_neural and len(y_test.shape) > 2:
            y_test = y_test.reshape(-1, y_test.shape[-1])
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        
        # Inverse transform if needed
        if scaler_y:
            y_test_orig = scaler_y.inverse_transform(y_test)
            y_pred_orig = scaler_y.inverse_transform(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred
        
        # Calculate test metrics
        test_results = {}
        
        # Use selected metrics or default ones if none selected
        metrics_to_calculate = session.selected_metrics if session.selected_metrics else ['mae', 'mse', 'rmse', 'r2']
        
        print(f"[Training] Calculating test metrics: {metrics_to_calculate}")
        print(f"[Training] Test data shapes - y_test: {y_test_orig.shape}, y_pred: {y_pred_orig.shape}")
        
        for metric in metrics_to_calculate:
            try:
                if metric == 'mae':
                    test_results[metric] = float(mean_absolute_error(y_test_orig, y_pred_orig))
                elif metric == 'mse':
                    test_results[metric] = float(mean_squared_error(y_test_orig, y_pred_orig))
                elif metric == 'rmse':
                    test_results[metric] = float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)))
                elif metric == 'r2':
                    test_results[metric] = float(r2_score(y_test_orig, y_pred_orig))
                print(f"[Training] {metric}: {test_results[metric]}")
            except Exception as e:
                print(f"[Training] Error calculating {metric}: {str(e)}")
                test_results[metric] = None
        
        # Save model
        # Use absolute path for file operations
        from django.conf import settings
        model_dir_abs = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir_abs, exist_ok=True)
        # But use relative path for saving in database
        model_dir_rel = 'models'
        
        if model_format == 'tensorflow':
            # Save TensorFlow model
            model_path_abs = f"{model_dir_abs}/model_{session.id}"
            model.save(model_path_abs)
            model_path = f"{model_dir_rel}/model_{session.id}"
            
            # Save scalers and metadata separately
            metadata_path_abs = f"{model_dir_abs}/model_{session.id}_metadata.pkl"
            joblib.dump({
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'predictor_columns': session.predictor_columns,
                'target_columns': session.target_columns,
                'sequence_length': session.hyperparameters.get('sequence_length', 10),
                'model_type': session.model_type,
                'framework': 'keras'
            }, metadata_path_abs)
        elif model_format == 'pytorch':
            # Save PyTorch model
            model_path_abs = f"{model_dir_abs}/model_{session.id}.pth"
            save_pytorch_model(model, model_path_abs)
            model_path = f"{model_dir_rel}/model_{session.id}.pth"
            
            # Save scalers and metadata separately
            metadata_path_abs = f"{model_dir_abs}/model_{session.id}_metadata.pkl"
            joblib.dump({
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'predictor_columns': session.predictor_columns,
                'target_columns': session.target_columns,
                'sequence_length': session.hyperparameters.get('sequence_length', 10),
                'model_type': session.model_type,
                'framework': 'pytorch',
                'input_shape': X_train.shape[1:],
                'output_shape': (y_train.shape[1] if len(y_train.shape) > 1 else 1,)
            }, metadata_path_abs)
        else:
            # Save sklearn model
            model_path_abs = f"{model_dir_abs}/model_{session.id}.pkl"
            model_path = f"{model_dir_rel}/model_{session.id}.pkl"
            
            # Load preprocessing pipeline if it exists
            preprocessing_pipeline = None
            if session.preprocessing_info and 'pipeline_path' in session.preprocessing_info:
                pipeline_path = session.preprocessing_info['pipeline_path']
                if os.path.exists(pipeline_path):
                    preprocessing_pipeline = SklearnPreprocessingPipeline.load(pipeline_path)
                    # Clean up temp file
                    os.remove(pipeline_path)
            
            joblib.dump({
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'predictor_columns': session.predictor_columns,
                'target_columns': session.target_columns,
                'target_encoders': target_encoders,  # Add target encoders
                'preprocessing_pipeline': preprocessing_pipeline,
                'preprocessing_info': session.preprocessing_info
            }, model_path_abs)
        
        # Update session
        print(f"[Training] Saving session with test results: {test_results}")
        session.training_history = history
        session.test_results = test_results
        session.model_file = model_path
        session.status = 'completed'
        session.save()
        
        print(f"[Training] Session saved successfully with status: {session.status}")
        
        # Update model definition statistics
        if session.model_definition:
            model_def = session.model_definition
            model_def.training_count += 1
            model_def.last_trained = timezone.now()
            
            # Update best score if this is better
            primary_metric = test_results.get('mae', float('inf'))
            if model_def.best_score is None or primary_metric < model_def.best_score:
                model_def.best_score = primary_metric
            
            model_def.save()
        
    except Exception as e:
        print(f"\n[ERROR] ====== TRAINING FAILED ======")
        print(f"[ERROR] Error type: {type(e).__name__}")
        print(f"[ERROR] Error message: {str(e)}")
        
        # Get detailed traceback
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[ERROR] Full traceback:\n{error_traceback}")
        print(f"[ERROR] =============================\n")
        
        # Save error details to session
        session.status = 'failed'
        session.error_message = f"{type(e).__name__}: {str(e)}"
        
        # You could also save more details if your model has a field for it
        # For example, if you add a 'error_details' or 'logs' field to TrainingSession:
        # session.error_details = error_traceback
        
        session.save()
        
        # Re-raise the exception to ensure execution stops
        raise


def make_predictions(session, input_file):
    """Make predictions using trained model"""
    # Check model type
    is_neural = session.model_type in ['lstm', 'gru', 'cnn', 'transformer']
    
    if is_neural:
        # Load TensorFlow model
        model_path = session.model_file.path
        if model_path.endswith('.pkl'):
            # Old format, need to handle
            model_path = model_path.replace('.pkl', '')
        
        model = tf.keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = f"{model_path}_metadata.pkl"
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            scaler_X = metadata['scaler_X']
            scaler_y = metadata['scaler_y']
            sequence_length = metadata.get('sequence_length', 10)
        else:
            # Fallback for old models
            scaler_X = None
            scaler_y = None
            sequence_length = 10
    else:
        # Load sklearn model
        model_data = joblib.load(session.model_file.path)
        model = model_data['model']
        scaler_X = model_data['scaler_X']
        scaler_y = model_data['scaler_y']
        preprocessing_pipeline = model_data.get('preprocessing_pipeline', None)
        sequence_length = None
    
    # Load input data
    df = pd.read_csv(input_file)
    
    # For sklearn models with preprocessing pipeline
    if not is_neural and preprocessing_pipeline:
        # Use preprocessing pipeline
        X_df = df[session.predictor_columns]
        X = preprocessing_pipeline.transform(X_df)
    else:
        # Traditional approach
        X = df[session.predictor_columns].values
        
        # Scale input if needed
        if scaler_X:
            X = scaler_X.transform(X)
    
    # Create sequences for neural networks
    if is_neural and sequence_length:
        # For prediction, we need to create sequences
        X_sequences = []
        for i in range(len(X) - sequence_length + 1):
            X_sequences.append(X[i:i + sequence_length])
        X = np.array(X_sequences)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Ensure predictions are 2D
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    
    # Inverse transform if needed
    if scaler_y:
        predictions = scaler_y.inverse_transform(predictions)
    
    # Format results
    results = []
    num_predictions = len(predictions)
    
    for i in range(num_predictions):
        # Get corresponding input row
        if is_neural and sequence_length:
            # For sequences, use the last row of the sequence
            input_idx = i + sequence_length - 1
        else:
            input_idx = i
        
        if input_idx < len(df):
            result = {
                'input': df.iloc[input_idx].to_dict(),
                'predictions': {}
            }
            
            # Handle both single and multi-output predictions
            if len(session.target_columns) == 1:
                result['predictions'][session.target_columns[0]] = float(predictions[i, 0])
            else:
                for j, col in enumerate(session.target_columns):
                    if j < predictions.shape[1]:
                        result['predictions'][col] = float(predictions[i, j])
            
            results.append(result)
    
    return results


def generate_weather_map_data(session, date):
    """Generate predictions for weather map visualization using real weather data"""
    import requests
    from datetime import datetime
    import os
    
    # Define grid points for Spain and France
    cities = [
        # Spain major cities
        {'region': 'Madrid', 'latitude': 40.4168, 'longitude': -3.7038, 'country': 'ES'},
        {'region': 'Barcelona', 'latitude': 41.3851, 'longitude': 2.1734, 'country': 'ES'},
        {'region': 'Valencia', 'latitude': 39.4699, 'longitude': -0.3763, 'country': 'ES'},
        {'region': 'Sevilla', 'latitude': 37.3891, 'longitude': -5.9845, 'country': 'ES'},
        {'region': 'Bilbao', 'latitude': 43.2630, 'longitude': -2.9350, 'country': 'ES'},
        # France major cities
        {'region': 'Paris', 'latitude': 48.8566, 'longitude': 2.3522, 'country': 'FR'},
        {'region': 'Lyon', 'latitude': 45.7640, 'longitude': 4.8357, 'country': 'FR'},
        {'region': 'Marseille', 'latitude': 43.2965, 'longitude': 5.3698, 'country': 'FR'},
        {'region': 'Toulouse', 'latitude': 43.6047, 'longitude': 1.4442, 'country': 'FR'},
        {'region': 'Bordeaux', 'latitude': 44.8378, 'longitude': -0.5792, 'country': 'FR'},
    ]
    
    predictions = []
    
    # Get weather data from OpenWeatherMap API (free tier)
    # You need to set OPENWEATHER_API_KEY in environment variables
    api_key = os.environ.get('OPENWEATHER_API_KEY', '')
    
    if not api_key:
        # Fallback to OpenMeteo API (no key required)
        for city in cities:
            try:
                # OpenMeteo API for current weather
                url = f"https://api.open-meteo.com/v1/forecast"
                params = {
                    'latitude': city['latitude'],
                    'longitude': city['longitude'],
                    'current_weather': True,
                    'hourly': 'temperature_2m,relativehumidity_2m,surface_pressure,windspeed_10m',
                    'forecast_days': 1
                }
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    current = data.get('current_weather', {})
                    hourly = data.get('hourly', {})
                    
                    # Get the current hour's data
                    current_hour = datetime.now().hour
                    
                    pred = WeatherPrediction(
                        training_session=session,
                        prediction_date=date,
                        region=city['region'],
                        latitude=city['latitude'],
                        longitude=city['longitude'],
                        predictions={
                            'temperature': current.get('temperature', hourly['temperature_2m'][current_hour] if hourly else 20),
                            'humidity': hourly['relativehumidity_2m'][current_hour] if 'relativehumidity_2m' in hourly else 60,
                            'pressure': hourly['surface_pressure'][current_hour] if 'surface_pressure' in hourly else 1013,
                            'wind_speed': current.get('windspeed', hourly['windspeed_10m'][current_hour] if hourly else 5)
                        }
                    )
                else:
                    # Fallback prediction if API fails
                    pred = _create_fallback_prediction(session, date, city)
                    
            except Exception as e:
                print(f"Error fetching weather data for {city['region']}: {str(e)}")
                pred = _create_fallback_prediction(session, date, city)
                
            predictions.append(pred)
    else:
        # Use OpenWeatherMap API
        for city in cities:
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': city['latitude'],
                    'lon': city['longitude'],
                    'appid': api_key,
                    'units': 'metric'
                }
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    main = data.get('main', {})
                    wind = data.get('wind', {})
                    
                    pred = WeatherPrediction(
                        training_session=session,
                        prediction_date=date,
                        region=city['region'],
                        latitude=city['latitude'],
                        longitude=city['longitude'],
                        predictions={
                            'temperature': main.get('temp', 20),
                            'humidity': main.get('humidity', 60),
                            'pressure': main.get('pressure', 1013),
                            'wind_speed': wind.get('speed', 5) * 3.6  # Convert m/s to km/h
                        }
                    )
                else:
                    pred = _create_fallback_prediction(session, date, city)
                    
            except Exception as e:
                print(f"Error fetching weather data for {city['region']}: {str(e)}")
                pred = _create_fallback_prediction(session, date, city)
                
            predictions.append(pred)
    
    return predictions


def detect_problem_type(target_column, df):
    """
    Automatically detect if the problem is classification or regression
    based on the target column characteristics
    """
    # Get unique values in target column
    unique_values = df[target_column].dropna().unique()
    n_unique = len(unique_values)
    n_samples = len(df[target_column].dropna())
    
    # Heuristics for classification vs regression
    if n_unique <= 2:
        # Binary classification
        return 'classification', 'binary'
    elif n_unique < 20 and n_unique < n_samples * 0.05:
        # Multi-class classification (few unique values relative to samples)
        return 'classification', 'multiclass'
    elif df[target_column].dtype == 'object' or df[target_column].dtype.name == 'category':
        # String/categorical type = classification
        return 'classification', 'multiclass'
    else:
        # Continuous values = regression
        return 'regression', None


def encode_categorical_features(X, categorical_columns, method='onehot', y=None):
    """
    Encode categorical features using specified method
    """
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
    
    X_encoded = X.copy()
    
    if method == 'onehot':
        # One-hot encoding
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        for col in categorical_columns:
            if col in X_encoded.columns:
                encoded = encoder.fit_transform(X_encoded[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_encoded.index)
                X_encoded = pd.concat([X_encoded.drop(col, axis=1), encoded_df], axis=1)
    
    elif method == 'ordinal':
        # Ordinal encoding
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        for col in categorical_columns:
            if col in X_encoded.columns:
                X_encoded[col] = encoder.fit_transform(X_encoded[[col]])
    
    elif method == 'target' and y is not None:
        # Target encoding (mean encoding)
        for col in categorical_columns:
            if col in X_encoded.columns:
                # Calculate mean target value for each category
                mean_target = y.groupby(X[col]).mean()
                X_encoded[col] = X[col].map(mean_target).fillna(mean_target.mean())
    
    return X_encoded


def add_cyclic_features(df, columns):
    """
    Add sine and cosine transformations for cyclic features
    """
    df_encoded = df.copy()
    
    for col in columns:
        if col in df_encoded.columns:
            # Detect the period based on column name or values
            if 'hour' in col.lower():
                period = 24
            elif 'day' in col.lower() and 'week' in col.lower():
                period = 7
            elif 'month' in col.lower():
                period = 12
            elif 'bearing' in col.lower() or 'degree' in col.lower():
                period = 360
            else:
                # Try to infer from data
                period = df_encoded[col].max() - df_encoded[col].min() + 1
            
            # Add sine and cosine features
            df_encoded[f'{col}_sin'] = np.sin(2 * np.pi * df_encoded[col] / period)
            df_encoded[f'{col}_cos'] = np.cos(2 * np.pi * df_encoded[col] / period)
            
            # Optionally remove original column to avoid linear order assumption
            # df_encoded = df_encoded.drop(col, axis=1)
    
    return df_encoded


def check_data_leakage(predictor_columns, target_column, df):
    """
    Check for potential data leakage issues
    """
    warnings = []
    
    # Check for future information in column names
    future_indicators = ['future', 'next', 'forecast', 'prediction', 'will', 'tomorrow']
    for col in predictor_columns:
        if any(indicator in col.lower() for indicator in future_indicators):
            warnings.append(f"Column '{col}' might contain future information")
    
    # Check for perfect correlation with target
    for col in predictor_columns:
        if col in df.columns and target_column in df.columns:
            corr = df[col].corr(df[target_column])
            if abs(corr) > 0.99:
                warnings.append(f"Column '{col}' has perfect correlation ({corr:.3f}) with target")
    
    # Check for derived features from target
    target_variants = [target_column.lower(), target_column.upper(), 
                      target_column.replace('_', ''), target_column.replace('-', '')]
    for col in predictor_columns:
        col_clean = col.lower().replace('_', '').replace('-', '')
        if any(variant in col_clean for variant in target_variants):
            warnings.append(f"Column '{col}' might be derived from target '{target_column}'")
    
    return warnings


def _create_fallback_prediction(session, date, city):
    """Create a fallback prediction with seasonal averages"""
    from datetime import datetime
    
    # Get month to determine season
    month = date.month if hasattr(date, 'month') else datetime.now().month
    
    # Seasonal temperature averages (approximate)
    seasonal_temps = {
        'ES': {  # Spain
            'winter': 12, 'spring': 18, 'summer': 28, 'autumn': 20
        },
        'FR': {  # France
            'winter': 8, 'spring': 15, 'summer': 25, 'autumn': 16
        }
    }
    
    # Determine season
    if month in [12, 1, 2]:
        season = 'winter'
    elif month in [3, 4, 5]:
        season = 'spring'
    elif month in [6, 7, 8]:
        season = 'summer'
    else:
        season = 'autumn'
    
    country = city.get('country', 'ES')
    base_temp = seasonal_temps.get(country, seasonal_temps['ES'])[season]
    
    return WeatherPrediction(
        training_session=session,
        prediction_date=date,
        region=city['region'],
        latitude=city['latitude'],
        longitude=city['longitude'],
        predictions={
            'temperature': base_temp + np.random.uniform(-3, 3),
            'humidity': 60 + np.random.uniform(-15, 15),
            'pressure': 1013 + np.random.uniform(-10, 10),
            'wind_speed': 10 + np.random.uniform(-5, 10)
        }
    )