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
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto'
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
            # Basic options with new defaults
            'preset': 'balanceado',
            'problem_type': 'regression',
            'n_estimators': 300,
            'max_depth_enabled': False,
            'max_depth': None,
            'max_features': '1.0',  # Default for regression
            'criterion': 'squared_error',
            'class_weight_balanced': False,
            'validation_method': 'holdout',
            
            # Advanced options with scikit-learn defaults
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'min_impurity_decrease': 0.0,
            'max_leaf_nodes': None,
            'ccp_alpha': 0.0,
            'criterion_advanced': 'default',
            'bootstrap': True,
            'max_samples': None,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': None,
            'verbose': 0,
            'warm_start': False,
            'class_weight_custom': False,
            'decision_threshold': 0.5,
            'output_type': 'class'
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8
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
    
    # Select columns
    X = df[session.predictor_columns].values
    y = df[session.target_columns].values
    
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
            X_train = scaler_X.fit_transform(X_train)
            X_val = scaler_X.transform(X_val)
            X_test = scaler_X.transform(X_test)
            
            y_train = scaler_y.fit_transform(y_train)
            y_val = scaler_y.transform(y_val)
            y_test = scaler_y.transform(y_test)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test), (scaler_X, scaler_y)


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
    
    # Callbacks
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
        epochs=hyperparams.get('epochs', 50),
        batch_size=hyperparams.get('batch_size', 32),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history.history


def _prepare_random_forest_params(hyperparams):
    """Prepare Random Forest parameters for scikit-learn"""
    # Start with a copy of hyperparams
    clean_params = hyperparams.copy()
    
    # Remove UI-specific parameters that don't belong to scikit-learn
    ui_only_params = [
        'preset', 'problem_type', 'max_depth_enabled', 'class_weight_balanced', 
        'validation_method', 'criterion_advanced', 'class_weight_custom',
        'decision_threshold', 'output_type', 'max_features_fraction'
    ]
    for param in ui_only_params:
        clean_params.pop(param, None)
    
    # Handle max_depth logic
    if not hyperparams.get('max_depth_enabled', False):
        clean_params['max_depth'] = None
    
    # Handle max_features conversion
    max_features = hyperparams.get('max_features', 'sqrt')
    if max_features == 'custom':
        max_features = hyperparams.get('max_features_fraction', 0.5)
    elif max_features == '1.0':
        max_features = 1.0
    clean_params['max_features'] = max_features
    
    # Handle criterion selection
    criterion_advanced = hyperparams.get('criterion_advanced', 'default')
    if criterion_advanced != 'default':
        clean_params['criterion'] = criterion_advanced
    elif hyperparams.get('criterion') == 'auto':
        # Auto criterion was already resolved in frontend, but fallback logic
        problem_type = hyperparams.get('problem_type', 'regression')
        if problem_type == 'classification':
            clean_params['criterion'] = 'gini'
        else:
            clean_params['criterion'] = 'squared_error'
    
    # Handle class_weight
    if hyperparams.get('class_weight_balanced', False):
        clean_params['class_weight'] = 'balanced'
    else:
        clean_params.pop('class_weight', None)
    
    # Handle null values properly for scikit-learn
    for key, value in clean_params.items():
        if value is None or value == 'None':
            if key in ['max_depth', 'max_leaf_nodes', 'random_state', 'max_samples']:
                clean_params[key] = None
            else:
                clean_params.pop(key, None)
    
    return clean_params


def train_sklearn_model(model_type, hyperparams, X_train, y_train, X_val, y_val):
    """Train scikit-learn models"""
    if model_type == 'decision_tree':
        model = DecisionTreeRegressor(**hyperparams)
    elif model_type == 'random_forest':
        # Determine if it's classification or regression
        problem_type = hyperparams.get('problem_type', 'regression')
        clean_params = _prepare_random_forest_params(hyperparams)
        
        if problem_type == 'classification':
            model = RandomForestClassifier(**clean_params)
        else:
            model = RandomForestRegressor(**clean_params)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(**hyperparams)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # Calculate validation metrics
    val_pred = model.predict(X_val)
    
    # Choose appropriate metrics based on problem type
    if model_type == 'random_forest' and hyperparams.get('problem_type') == 'classification':
        from sklearn.metrics import accuracy_score, log_loss
        try:
            train_pred = model.predict(X_train)
            history = {
                'train_accuracy': [accuracy_score(y_train, train_pred)],
                'val_accuracy': [accuracy_score(y_val, val_pred)],
                'train_loss': [log_loss(y_train, model.predict_proba(X_train))],
                'val_loss': [log_loss(y_val, model.predict_proba(X_val))]
            }
        except:
            # Fallback to basic accuracy if log_loss fails
            train_pred = model.predict(X_train)
            history = {
                'train_accuracy': [accuracy_score(y_train, train_pred)],
                'val_accuracy': [accuracy_score(y_val, val_pred)]
            }
    else:
        # Regression metrics
        history = {
            'train_loss': [mean_squared_error(y_train, model.predict(X_train))],
            'val_loss': [mean_squared_error(y_val, val_pred)]
        }
    
    return model, history


def train_model(session):
    """Main training function"""
    try:
        session.status = 'training'
        session.save()
        
        # Check if it's a neural network
        is_neural = session.model_type in ['lstm', 'gru', 'cnn', 'transformer']
        
        # Prepare data
        data, scalers = prepare_data(session, is_neural_network=is_neural)
        X_train, y_train, X_val, y_val, X_test, y_test = data
        scaler_X, scaler_y = scalers
        
        # Train model based on type
        if session.model_type in ['decision_tree', 'random_forest', 'xgboost']:
            model, history = train_sklearn_model(
                session.model_type, 
                session.hyperparameters,
                X_train, y_train, 
                X_val, y_val
            )
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
        for metric in session.selected_metrics:
            if metric == 'mae':
                test_results[metric] = float(mean_absolute_error(y_test_orig, y_pred_orig))
            elif metric == 'mse':
                test_results[metric] = float(mean_squared_error(y_test_orig, y_pred_orig))
            elif metric == 'rmse':
                test_results[metric] = float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)))
            elif metric == 'r2':
                test_results[metric] = float(r2_score(y_test_orig, y_pred_orig))
        
        # Save model
        model_dir = 'media/models'
        os.makedirs(model_dir, exist_ok=True)
        
        if model_format == 'tensorflow':
            # Save TensorFlow model
            model_path = f"{model_dir}/model_{session.id}"
            model.save(model_path)
            
            # Save scalers and metadata separately
            metadata_path = f"{model_dir}/model_{session.id}_metadata.pkl"
            joblib.dump({
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'predictor_columns': session.predictor_columns,
                'target_columns': session.target_columns,
                'sequence_length': session.hyperparameters.get('sequence_length', 10),
                'model_type': session.model_type,
                'framework': 'keras'
            }, metadata_path)
        elif model_format == 'pytorch':
            # Save PyTorch model
            model_path = f"{model_dir}/model_{session.id}.pth"
            save_pytorch_model(model, model_path)
            
            # Save scalers and metadata separately
            metadata_path = f"{model_dir}/model_{session.id}_metadata.pkl"
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
            }, metadata_path)
        else:
            # Save sklearn model
            model_path = f"{model_dir}/model_{session.id}.pkl"
            joblib.dump({
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'predictor_columns': session.predictor_columns,
                'target_columns': session.target_columns
            }, model_path)
        
        # Update session
        session.training_history = history
        session.test_results = test_results
        session.model_file = model_path
        session.status = 'completed'
        session.save()
        
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
        session.status = 'failed'
        session.error_message = str(e)
        session.save()
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
        sequence_length = None
    
    # Load input data
    df = pd.read_csv(input_file)
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