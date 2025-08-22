"""
Neural Network Code Generation and Parsing Module

This module provides functionality to generate executable Keras/TensorFlow and PyTorch code
from model definitions, as well as parse existing code back into architecture definitions.

Main functions:
- generate_keras_code(): Generate Keras/TensorFlow model code
- generate_pytorch_code(): Generate PyTorch model code  
- parse_keras_code(): Parse Keras code back to architecture
- parse_pytorch_code(): Parse PyTorch code back to architecture
- validate_architecture(): Validate architecture definitions
"""
import json
import ast
import re
from typing import Dict, List, Any

# Constants for better maintainability
DEFAULT_LAYER_PARAMS = {
    'DENSE': {'units': 32, 'activation': 'relu', 'use_bias': True, 'kernel_initializer': 'glorot_uniform'},
    'LSTM': {'units': 50, 'activation': 'tanh', 'recurrent_activation': 'sigmoid', 'return_sequences': False, 'dropout': 0.0, 'recurrent_dropout': 0.0},
    'GRU': {'units': 50, 'activation': 'tanh', 'recurrent_activation': 'sigmoid', 'return_sequences': False, 'dropout': 0.0, 'recurrent_dropout': 0.0},
    'CONV1D': {'filters': 32, 'kernel_size': 3, 'strides': 1, 'padding': 'valid', 'activation': 'relu'},
    'DROPOUT': {'rate': 0.5},
    'BATCHNORMALIZATION': {'momentum': 0.99, 'epsilon': 0.001}
}

SUPPORTED_LAYER_TYPES = {'DENSE', 'LSTM', 'GRU', 'CONV1D', 'DROPOUT', 'BATCHNORMALIZATION', 'FLATTEN', 'MAXPOOLING1D', 'GLOBALMAXPOOLING1D', 'GLOBALAVERAGEPOOLING1D'}


def _normalize_layer_type(layer_type: str) -> str:
    """Normalize layer type to uppercase for consistent comparison"""
    return layer_type.upper() if layer_type else None


def _get_layer_params(layer_type_normalized: str, params: Dict) -> Dict:
    """Get layer parameters with defaults applied"""
    defaults = DEFAULT_LAYER_PARAMS.get(layer_type_normalized, {})
    return {key: params.get(key, default_value) for key, default_value in defaults.items()}


def generate_keras_code(model_def) -> str:
    """Generate Keras/TensorFlow code from model definition"""
    
    hyperparams = model_def.hyperparameters or {}
    
    code_lines = []
    
    # Header section
    code_lines.extend([
        '"""',
        'Auto-generated Keras model code',
        f'Model: {model_def.name}',
        f'Type: {model_def.model_type.upper()}',
        f'Generated at: {model_def.updated_at}',
        '',
        'Configuration:',
        f'- Target columns: {model_def.target_columns}',
        f'- Predictor columns: {len(model_def.predictor_columns)} features',
        f'- Loss function: {hyperparams.get("loss_function", "mse")}',
        f'- Optimizer: {hyperparams.get("optimizer", "Adam")}',
        f'- Learning rate: {hyperparams.get("learning_rate", 0.001)}',
        f'- Batch size: {hyperparams.get("batch_size", 32)}',
        f'- Epochs: {hyperparams.get("epochs", 50)}',
        '"""',
        '',
        'import tensorflow as tf',
        'from tensorflow import keras',
        'from tensorflow.keras import layers, models, optimizers, callbacks',
        'import numpy as np',
        'import matplotlib.pyplot as plt',
        'import seaborn as sns',
        'from sklearn.metrics import confusion_matrix, classification_report',
        '',
        'def create_model(input_shape, output_shape):',
        '    """',
        '    Create and return the configured model',
        '    ',
        '    Args:',
        '        input_shape: Tuple defining input shape (e.g., (timesteps, features) for sequences)',
        '        output_shape: Tuple defining output shape (e.g., (num_targets,) for outputs)',
        '    ',
        '    Returns:',
        '        Compiled Keras model ready for training',
        '    """',
        '    print(f"Creating model with input_shape={input_shape} and output_shape={output_shape}")',
        '    '
    ])
    
    if model_def.use_custom_architecture and model_def.custom_architecture:
        # Generate from custom architecture defined by user
        code_lines.extend([
            "    print('Building custom architecture...')",
            "    model = models.Sequential()",
            ""
        ])
        
        for i, layer in enumerate(model_def.custom_architecture):
            layer_type = layer.get('type')
            params = layer.get('params', {})
            layer_name = layer.get('name', f'{layer_type} Layer {i+1}')
            
            # Normalize layer type to uppercase for consistency
            layer_type_normalized = _normalize_layer_type(layer_type)
            
            code_lines.append(f"    # Layer {i+1}: {layer_name}")
            
            if i == 0:
                # First layer needs input_shape
                if layer_type_normalized == 'DENSE':
                    code_lines.extend([
                        "    model.add(layers.Dense(",
                        f"        units={params.get('units', 32)},",
                        f"        activation='{params.get('activation', 'relu')}',",
                        f"        use_bias={params.get('use_bias', True)},",
                        f"        kernel_initializer='{params.get('kernel_initializer', 'glorot_uniform')}',",
                        "        input_shape=input_shape,",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'LSTM':
                    code_lines.extend([
                        "    model.add(layers.LSTM(",
                        f"        units={params.get('units', 50)},",
                        f"        activation='{params.get('activation', 'tanh')}',",
                        f"        recurrent_activation='{params.get('recurrent_activation', 'sigmoid')}',",
                        f"        return_sequences={params.get('return_sequences', True)},",
                        f"        dropout={params.get('dropout', 0.0)},",
                        f"        recurrent_dropout={params.get('recurrent_dropout', 0.0)},",
                        "        input_shape=input_shape,",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'GRU':
                    code_lines.extend([
                        "    model.add(layers.GRU(",
                        f"        units={params.get('units', 50)},",
                        f"        activation='{params.get('activation', 'tanh')}',",
                        f"        recurrent_activation='{params.get('recurrent_activation', 'sigmoid')}',",
                        f"        return_sequences={params.get('return_sequences', True)},",
                        f"        dropout={params.get('dropout', 0.0)},",
                        f"        recurrent_dropout={params.get('recurrent_dropout', 0.0)},",
                        "        input_shape=input_shape,",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'CONV1D':
                    code_lines.extend([
                        "    model.add(layers.Conv1D(",
                        f"        filters={params.get('filters', 32)},",
                        f"        kernel_size={params.get('kernel_size', 3)},",
                        f"        strides={params.get('strides', 1)},",
                        f"        padding='{params.get('padding', 'valid')}',",
                        f"        activation='{params.get('activation', 'relu')}',",
                        "        input_shape=input_shape,",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
            else:
                # Subsequent layers - no input_shape needed
                if layer_type_normalized == 'DENSE':
                    code_lines.extend([
                        "    model.add(layers.Dense(",
                        f"        units={params.get('units', 32)},",
                        f"        activation='{params.get('activation', 'relu')}',",
                        f"        use_bias={params.get('use_bias', True)},",
                        f"        kernel_initializer='{params.get('kernel_initializer', 'glorot_uniform')}',",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'LSTM':
                    code_lines.extend([
                        "    model.add(layers.LSTM(",
                        f"        units={params.get('units', 50)},",
                        f"        activation='{params.get('activation', 'tanh')}',",
                        f"        recurrent_activation='{params.get('recurrent_activation', 'sigmoid')}',",
                        f"        return_sequences={params.get('return_sequences', False)},",
                        f"        dropout={params.get('dropout', 0.0)},",
                        f"        recurrent_dropout={params.get('recurrent_dropout', 0.0)},",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'GRU':
                    code_lines.extend([
                        "    model.add(layers.GRU(",
                        f"        units={params.get('units', 50)},",
                        f"        activation='{params.get('activation', 'tanh')}',",
                        f"        recurrent_activation='{params.get('recurrent_activation', 'sigmoid')}',",
                        f"        return_sequences={params.get('return_sequences', False)},",
                        f"        dropout={params.get('dropout', 0.0)},",
                        f"        recurrent_dropout={params.get('recurrent_dropout', 0.0)},",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'CONV1D':
                    code_lines.extend([
                        "    model.add(layers.Conv1D(",
                        f"        filters={params.get('filters', 32)},",
                        f"        kernel_size={params.get('kernel_size', 3)},",
                        f"        strides={params.get('strides', 1)},",
                        f"        padding='{params.get('padding', 'valid')}',",
                        f"        activation='{params.get('activation', 'relu')}',",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'DROPOUT':
                    code_lines.extend([
                        "    model.add(layers.Dropout(",
                        f"        rate={params.get('rate', 0.5)},",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'BATCHNORMALIZATION':
                    code_lines.extend([
                        "    model.add(layers.BatchNormalization(",
                        f"        momentum={params.get('momentum', 0.99)},",
                        f"        epsilon={params.get('epsilon', 0.001)},",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'FLATTEN':
                    code_lines.extend([
                        "    model.add(layers.Flatten(",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'MAXPOOLING1D':
                    code_lines.extend([
                        "    model.add(layers.MaxPooling1D(",
                        f"        pool_size={params.get('pool_size', 2)},",
                        f"        strides={params.get('strides', None)},",
                        f"        padding='{params.get('padding', 'valid')}',",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'GLOBALMAXPOOLING1D':
                    code_lines.extend([
                        "    model.add(layers.GlobalMaxPooling1D(",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
                    
                elif layer_type_normalized == 'GLOBALAVERAGEPOOLING1D':
                    code_lines.extend([
                        "    model.add(layers.GlobalAveragePooling1D(",
                        f"        name='{layer_name.replace(' ', '_').lower()}'",
                        "    ))",
                        ""
                    ])
        
        # Add final output layer
        output_activation = hyperparams.get('output_activation', 'linear')
        output_units = hyperparams.get('output_units', len(model_def.target_columns))
        code_lines.extend([
            "    # Final Output Layer",
            "    model.add(layers.Dense(",
            f"        units={output_units},",
            f"        activation='{output_activation}',",
            "        name='output_layer'",
            "    ))",
            ""
        ])
        
    else:
        # Generate standard model based on model_type
        model_type = model_def.model_type
        
        code_lines.extend([
            f"    print('Building {model_type.upper()} model...')",
            "    model = models.Sequential()",
            ""
        ])
        
        if model_type == 'lstm':
            num_layers = hyperparams.get('layers', 2)
            units = hyperparams.get('units', 50)
            dropout = hyperparams.get('dropout', 0.2)
            
            for i in range(num_layers):
                return_sequences = i < num_layers - 1  # All except last layer return sequences
                code_lines.append(f"    # LSTM Layer {i+1}")
                lstm_lines = [
                    "    model.add(layers.LSTM(",
                    f"        units={units},",
                    f"        activation='{hyperparams.get('activation', 'tanh')}',",
                    f"        recurrent_activation='{hyperparams.get('recurrent_activation', 'sigmoid')}',",
                    f"        return_sequences={return_sequences},",
                    f"        dropout={dropout},",
                    f"        recurrent_dropout={hyperparams.get('recurrent_dropout', 0.0)},"
                ]
                if i == 0:
                    lstm_lines.append("        input_shape=input_shape,")
                lstm_lines.extend([
                    f"        name='lstm_layer_{i+1}'",
                    "    ))",
                    ""
                ])
                code_lines.extend(lstm_lines)
                
                if dropout > 0 and i < num_layers - 1:
                    code_lines.extend([
                        f"    # Dropout after LSTM Layer {i+1}",
                        f"    model.add(layers.Dropout({dropout}, name='dropout_{i+1}'))",
                        ""
                    ])
            
            # Output layer
            output_units = hyperparams.get('output_units', len(model_def.target_columns))
            output_activation = hyperparams.get('output_activation', 'linear')
            code_lines.extend([
                "    # Output Layer",
                f"    model.add(layers.Dense({output_units}, activation='{output_activation}', name='output'))",
                ""
            ])
            
        elif model_type == 'gru':
            num_layers = hyperparams.get('layers', 2)
            units = hyperparams.get('units', 50)
            dropout = hyperparams.get('dropout', 0.2)
            
            for i in range(num_layers):
                return_sequences = i < num_layers - 1
                code_lines.append(f"    # GRU Layer {i+1}")
                gru_lines = [
                    "    model.add(layers.GRU(",
                    f"        units={units},",
                    f"        activation='{hyperparams.get('activation', 'tanh')}',",
                    f"        recurrent_activation='{hyperparams.get('recurrent_activation', 'sigmoid')}',",
                    f"        return_sequences={return_sequences},",
                    f"        dropout={dropout},",
                    f"        recurrent_dropout={hyperparams.get('recurrent_dropout', 0.0)},"
                ]
                if i == 0:
                    gru_lines.append("        input_shape=input_shape,")
                gru_lines.extend([
                    f"        name='gru_layer_{i+1}'",
                    "    ))",
                    ""
                ])
                code_lines.extend(gru_lines)
                
                if dropout > 0 and i < num_layers - 1:
                    code_lines.extend([
                        f"    # Dropout after GRU Layer {i+1}",
                        f"    model.add(layers.Dropout({dropout}, name='dropout_{i+1}'))",
                        ""
                    ])
            
            # Output layer
            output_units = hyperparams.get('output_units', len(model_def.target_columns))
            output_activation = hyperparams.get('output_activation', 'linear')
            code_lines.extend([
                "    # Output Layer",
                f"    model.add(layers.Dense({output_units}, activation='{output_activation}', name='output'))",
                ""
            ])
            
        elif model_type == 'cnn':
            filters = hyperparams.get('filters', 64)
            kernel_size = hyperparams.get('kernel_size', 3)
            pool_size = hyperparams.get('pool_size', 2)
            dropout = hyperparams.get('dropout', 0.2)
            dense_units = hyperparams.get('dense_units', 100)
            
            # First Conv1D layer
            code_lines.extend([
                "    # First Convolutional Layer",
                "    model.add(layers.Conv1D(",
                f"        filters={filters},",
                f"        kernel_size={kernel_size},",
                f"        activation='{hyperparams.get('activation', 'relu')}',",
                "        input_shape=input_shape,",
                "        name='conv1d_1'",
                "    ))",
                ""
            ])
            
            # Max pooling
            code_lines.extend([
                "    # Max Pooling Layer",
                f"    model.add(layers.MaxPooling1D(pool_size={pool_size}, name='maxpool_1'))",
                ""
            ])
            
            if dropout > 0:
                code_lines.extend([
                    "    # Dropout Layer",
                    f"    model.add(layers.Dropout({dropout}, name='dropout_1'))",
                    ""
                ])
            
            # Second Conv1D layer
            code_lines.extend([
                "    # Second Convolutional Layer",
                "    model.add(layers.Conv1D(",
                f"        filters={filters * 2},",
                f"        kernel_size={kernel_size},",
                f"        activation='{hyperparams.get('activation', 'relu')}',",
                "        name='conv1d_2'",
                "    ))",
                ""
            ])
            
            # Global max pooling
            code_lines.extend([
                "    # Global Max Pooling Layer",
                "    model.add(layers.GlobalMaxPooling1D(name='global_maxpool'))",
                ""
            ])
            
            # Dense layer
            code_lines.extend([
                "    # Dense Layer",
                "    model.add(layers.Dense(",
                f"        units={dense_units},",
                f"        activation='{hyperparams.get('activation', 'relu')}',",
                "        name='dense_1'",
                "    ))",
                ""
            ])
            
            if dropout > 0:
                code_lines.extend([
                    "    # Final Dropout Layer",
                    f"    model.add(layers.Dropout({dropout}, name='dropout_final'))",
                    ""
                ])
            
            # Output layer
            output_units = hyperparams.get('output_units', len(model_def.target_columns))
            output_activation = hyperparams.get('output_activation', 'linear')
            code_lines.extend([
                "    # Output Layer",
                f"    model.add(layers.Dense({output_units}, activation='{output_activation}', name='output'))",
                ""
            ])
    
    # Add compilation with all configured parameters
    optimizer_name = hyperparams.get('optimizer', 'Adam').lower()
    learning_rate = hyperparams.get('learning_rate', 0.001)
    loss_function = hyperparams.get('loss_function', 'mse')
    metrics = hyperparams.get('metrics', ['mae', 'mse'])
    
    code_lines.extend([
        "    # Compile model with configured parameters",
        "    model.compile("
    ])
    
    # Handle different optimizers
    if optimizer_name == 'adam':
        code_lines.append(f"        optimizer=optimizers.Adam(learning_rate={learning_rate}),")
    elif optimizer_name == 'sgd':
        code_lines.append(f"        optimizer=optimizers.SGD(learning_rate={learning_rate}),")
    elif optimizer_name == 'rmsprop':
        code_lines.append(f"        optimizer=optimizers.RMSprop(learning_rate={learning_rate}),")
    else:
        code_lines.append(f"        optimizer=optimizers.Adam(learning_rate={learning_rate}),")
    
    code_lines.extend([
        f"        loss='{loss_function}',",
        f"        metrics={metrics}",
        "    )",
        "",
        "    print('Model compiled successfully!')",
        "    return model",
        "",
        ""
    ])
    
    # Add training function with callbacks
    code_lines.extend([
        "def train_model(model, X_train, y_train, X_val=None, y_val=None):",
        '    """',
        '    Train the model with configured parameters',
        '    """',
        f"    batch_size = {hyperparams.get('batch_size', 32)}",
        f"    epochs = {hyperparams.get('epochs', 50)}",
        ""
    ])
    
    # Add callbacks if configured
    callbacks_config = hyperparams.get('callbacks', {})
    if callbacks_config:
        code_lines.extend([
            "    # Configure callbacks",
            "    callback_list = []",
            ""
        ])
        
        if callbacks_config.get('early_stopping', False):
            code_lines.extend([
                "    # Early stopping",
                "    callback_list.append(callbacks.EarlyStopping(",
                "        monitor='val_loss',",
                "        patience=10,",
                "        restore_best_weights=True",
                "    ))",
                ""
            ])
        
        if callbacks_config.get('reduce_lr', False):
            code_lines.extend([
                "    # Reduce learning rate on plateau",
                "    callback_list.append(callbacks.ReduceLROnPlateau(",
                "        monitor='val_loss',",
                "        factor=0.5,",
                "        patience=5,",
                "        min_lr=1e-7",
                "    ))",
                ""
            ])
        
        if callbacks_config.get('model_checkpoint', False):
            code_lines.extend([
                "    # Model checkpoint",
                "    callback_list.append(callbacks.ModelCheckpoint(",
                "        'best_model.h5',",
                "        monitor='val_loss',",
                "        save_best_only=True",
                "    ))",
                ""
            ])
        
        code_lines.extend([
            "    # Train the model",
            "    history = model.fit(",
            "        X_train, y_train,",
            "        batch_size=batch_size,",
            "        epochs=epochs,",
            "        validation_data=(X_val, y_val) if X_val is not None else None,",
            "        callbacks=callback_list,",
            "        verbose=1",
            "    )",
            ""
        ])
    else:
        code_lines.extend([
            "    # Train the model",
            "    history = model.fit(",
            "        X_train, y_train,",
            "        batch_size=batch_size,",
            "        epochs=epochs,",
            "        validation_data=(X_val, y_val) if X_val is not None else None,",
            "        verbose=1",
            "    )",
            ""
        ])
    
    code_lines.extend([
        "    return history",
        "",
        ""
    ])
    
    # Add data loading and preprocessing functions
    predictor_columns = [str(col) for col in model_def.predictor_columns]
    target_columns = [str(col) for col in model_def.target_columns]
    
    code_lines.extend([
        "def load_and_preprocess_data(dataset_path, test_size=0.2):",
        '    """',
        '    Load dataset and preprocess for training',
        '    """',
        "    import pandas as pd",
        "    from sklearn.model_selection import train_test_split",
        "    from sklearn.preprocessing import StandardScaler, LabelEncoder",
        "    from sklearn.impute import SimpleImputer",
        "",
        "    print(f'Loading dataset from: {dataset_path}')",
        "    df = pd.read_csv(dataset_path)",
        "    print(f'Dataset shape: {df.shape}')",
        "",
        "    # Define columns based on model configuration",
        f"    predictor_columns = {predictor_columns}",
        f"    target_columns = {target_columns}",
        "",
        "    # Validate columns exist in dataset",
        "    missing_predictors = [col for col in predictor_columns if col not in df.columns]",
        "    missing_targets = [col for col in target_columns if col not in df.columns]",
        "    ",
        "    if missing_predictors:",
        "        raise ValueError(f'Missing predictor columns: {missing_predictors}')",
        "    if missing_targets:",
        "        raise ValueError(f'Missing target columns: {missing_targets}')",
        "",
        "    # Extract features and targets",
        "    X = df[predictor_columns].copy()",
        "    y = df[target_columns].copy()",
        "",
        "    # Handle missing values",
        "    print('Handling missing values...')",
        "    # For numeric columns, use mean imputation",
        "    numeric_cols = X.select_dtypes(include=[np.number]).columns",
        "    if len(numeric_cols) > 0:",
        "        imputer_num = SimpleImputer(strategy='mean')",
        "        X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])",
        "",
        "    # For categorical columns, use most frequent imputation and label encoding",
        "    categorical_cols = X.select_dtypes(include=['object']).columns",
        "    label_encoders = {}",
        "    if len(categorical_cols) > 0:",
        "        imputer_cat = SimpleImputer(strategy='most_frequent')",
        "        X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])",
        "        ",
        "        # Label encode categorical variables",
        "        for col in categorical_cols:",
        "            le = LabelEncoder()",
        "            X[col] = le.fit_transform(X[col].astype(str))",
        "            label_encoders[col] = le",
        "",
        "    # Handle target variables",
        "    target_encoders = {}",
        "    for col in target_columns:",
        "        if y[col].dtype == 'object':",
        "            # Categorical target - encode",
        "            le = LabelEncoder()",
        "            y[col] = le.fit_transform(y[col].astype(str))",
        "            target_encoders[col] = le",
        "        else:",
        "            # Numeric target - handle missing values",
        "            y[col] = y[col].fillna(y[col].mean())",
        "",
        "    # Normalize features",
        "    print('Normalizing features...')",
        "    scaler = StandardScaler()",
        "    X_scaled = scaler.fit_transform(X)",
        ""
    ])
    
    # Add sequence creation for RNN models
    if model_def.model_type in ['lstm', 'gru']:
        code_lines.extend([
            "    # Create sequences for RNN models",
            "    def create_sequences(data, target, timesteps=10):",
            "        X_seq, y_seq = [], []",
            "        for i in range(timesteps, len(data)):",
            "            X_seq.append(data[i-timesteps:i])",
            "            y_seq.append(target[i])",
            "        return np.array(X_seq), np.array(y_seq)",
            "",
            "    print('Creating sequences for RNN...')",
            "    timesteps = 10",
            "    X_sequences, y_sequences = create_sequences(X_scaled, y.values, timesteps)",
            "    print(f'Sequence shape: X={X_sequences.shape}, y={y_sequences.shape}')",
            "",
            "    # Split data",
            "    X_train, X_test, y_train, y_test = train_test_split(",
            "        X_sequences, y_sequences, test_size=test_size, random_state=42",
            "    )",
            "",
            "    input_shape = (timesteps, X_scaled.shape[1])"
        ])
    else:
        code_lines.extend([
            "    # Split data",
            "    X_train, X_test, y_train, y_test = train_test_split(",
            "        X_scaled, y.values, test_size=test_size, random_state=42",
            "    )",
            "",
            "    input_shape = (X_scaled.shape[1],)"
        ])
    
    code_lines.extend([
        "    output_shape = (len(target_columns),)",
        "",
        "    print(f'Training set: X={X_train.shape}, y={y_train.shape}')",
        "    print(f'Test set: X={X_test.shape}, y={y_test.shape}')",
        "",
        "    return {",
        "        'X_train': X_train, 'X_test': X_test,",
        "        'y_train': y_train, 'y_test': y_test,",
        "        'input_shape': input_shape, 'output_shape': output_shape,",
        "        'scaler': scaler, 'label_encoders': label_encoders,",
        "        'target_encoders': target_encoders",
        "    }",
        "",
        ""
    ])
    
    # Add model evaluation function
    code_lines.extend([
        "def evaluate_model(model, X_test, y_test, target_encoders=None):",
        '    """',
        '    Evaluate the trained model',
        '    """',
        "    print('Evaluating model...')",
        "    ",
        "    # Make predictions",
        "    y_pred = model.predict(X_test)",
        "    ",
        "    # Calculate metrics",
        "    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
        "    ",
        r"    print('\n=== Model Evaluation Results ===')",
        f"    for i, target_col in enumerate({target_columns}):",
        "        y_true_col = y_test[:, i] if y_test.ndim > 1 else y_test",
        "        y_pred_col = y_pred[:, i] if y_pred.ndim > 1 else y_pred",
        "        ",
        "        mae = mean_absolute_error(y_true_col, y_pred_col)",
        "        mse = mean_squared_error(y_true_col, y_pred_col)",
        "        rmse = np.sqrt(mse)",
        "        r2 = r2_score(y_true_col, y_pred_col)",
        "        ",
        "        print(f'Target: {target_col}')",
        "        print(f'  MAE:  {mae:.4f}')",
        "        print(f'  MSE:  {mse:.4f}')",
        "        print(f'  RMSE: {rmse:.4f}')",
        "        print(f'  R²:   {r2:.4f}')",
        "        print()",
        "",
        "    return y_pred",
        "",
        ""
    ])
    
    # Add analysis visualization functions
    code_lines.extend([
        "def create_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', save_path=None):",
        '    """',
        '    Create and display confusion matrix',
        '    """',
        "    from sklearn.metrics import confusion_matrix",
        "    import matplotlib.pyplot as plt",
        "    import seaborn as sns",
        "    import numpy as np",
        "    ",
        "    # Calculate confusion matrix",
        "    cm = confusion_matrix(y_true, y_pred)",
        "    ",
        "    # Create figure",
        "    plt.figure(figsize=(10, 8))",
        "    ",
        "    # Use seaborn heatmap for better visualization",
        "    if labels is None:",
        "        labels = [f'Class_{i}' for i in range(len(cm))]",
        "    ",
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',",
        "                xticklabels=labels, yticklabels=labels,",
        "                cbar_kws={'label': 'Count'})",
        "    ",
        "    plt.title(title, fontsize=16, fontweight='bold')",
        "    plt.xlabel('Predicted Label', fontsize=12)",
        "    plt.ylabel('True Label', fontsize=12)",
        "    ",
        "    # Rotate labels if they are long",
        "    if any(len(str(label)) > 10 for label in labels):",
        "        plt.xticks(rotation=45, ha='right')",
        "        plt.yticks(rotation=0)",
        "    ",
        "    plt.tight_layout()",
        "    ",
        "    if save_path:",
        "        plt.savefig(save_path, dpi=300, bbox_inches='tight')",
        "        print(f'Confusion matrix saved to: {save_path}')",
        "    ",
        "    plt.show()",
        "    return cm",
        "",
        "",
        "def create_scatter_plot(y_true, y_pred, title='Predictions vs Actual', save_path=None):",
        '    """',
        '    Create scatter plot of predictions vs actual values',
        '    """',
        "    import matplotlib.pyplot as plt",
        "    import numpy as np",
        "    from sklearn.metrics import r2_score",
        "    ",
        "    plt.figure(figsize=(10, 8))",
        "    ",
        "    # Create scatter plot",
        "    plt.scatter(y_true, y_pred, alpha=0.6, s=50)",
        "    ",
        "    # Add perfect prediction line",
        "    min_val = min(min(y_true), min(y_pred))",
        "    max_val = max(max(y_true), max(y_pred))",
        "    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')",
        "    ",
        "    # Calculate and display R² score",
        "    r2 = r2_score(y_true, y_pred)",
        "    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,",
        "             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),",
        "             fontsize=12, verticalalignment='top')",
        "    ",
        "    plt.xlabel('Actual Values', fontsize=12)",
        "    plt.ylabel('Predicted Values', fontsize=12)",
        "    plt.title(title, fontsize=16, fontweight='bold')",
        "    plt.legend()",
        "    plt.grid(True, alpha=0.3)",
        "    plt.tight_layout()",
        "    ",
        "    if save_path:",
        "        plt.savefig(save_path, dpi=300, bbox_inches='tight')",
        "        print(f'Scatter plot saved to: {save_path}')",
        "    ",
        "    plt.show()",
        "    return r2",
        "",
        "",
        "def create_residuals_plot(y_true, y_pred, title='Residuals Analysis', save_path=None):",
        '    """',
        '    Create residuals plot for regression analysis',
        '    """',
        "    import matplotlib.pyplot as plt",
        "    import numpy as np",
        "    ",
        "    residuals = y_true - y_pred",
        "    ",
        "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))",
        "    ",
        "    # Residuals vs Predicted",
        "    ax1.scatter(y_pred, residuals, alpha=0.6)",
        "    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)",
        "    ax1.set_xlabel('Predicted Values')",
        "    ax1.set_ylabel('Residuals')",
        "    ax1.set_title('Residuals vs Predicted')",
        "    ax1.grid(True, alpha=0.3)",
        "    ",
        "    # Histogram of residuals",
        "    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')",
        "    ax2.set_xlabel('Residuals')",
        "    ax2.set_ylabel('Frequency')",
        "    ax2.set_title('Distribution of Residuals')",
        "    ax2.grid(True, alpha=0.3)",
        "    ",
        "    # Add statistics",
        "    mean_residuals = np.mean(residuals)",
        "    std_residuals = np.std(residuals)",
        "    ax2.axvline(mean_residuals, color='red', linestyle='--', ",
        "               label=f'Mean: {mean_residuals:.4f}')",
        "    ax2.legend()",
        "    ",
        "    plt.suptitle(title, fontsize=16, fontweight='bold')",
        "    plt.tight_layout()",
        "    ",
        "    if save_path:",
        "        plt.savefig(save_path, dpi=300, bbox_inches='tight')",
        "        print(f'Residuals plot saved to: {save_path}')",
        "    ",
        "    plt.show()",
        "    ",
        "    return {'mean': mean_residuals, 'std': std_residuals}",
        "",
        "",
        "def analyze_model_performance(model, X_test, y_test, target_columns, ",
        "                             model_name='Model', save_plots=True):",
        '    """',
        '    Comprehensive model performance analysis with visualizations',
        '    """',
        "    print('Analyzing model performance...')",
        "    print('=' * 50)",
        "    ",
        "    # Make predictions",
        "    y_pred = model.predict(X_test)",
        "    ",
        "    # Ensure predictions are properly shaped",
        "    # Handle both numpy arrays and pandas Series/DataFrames",
        "    if hasattr(y_pred, 'values'):",
        "        y_pred = y_pred.values",
        "    if hasattr(y_test, 'values'):",
        "        y_test = y_test.values",
        "    ",
        "    if len(y_pred.shape) == 1:",
        "        y_pred = y_pred.reshape(-1, 1)",
        "    if len(y_test.shape) == 1:",
        "        y_test = y_test.reshape(-1, 1)",
        "    ",
        "    analysis_results = {}",
        "    ",
        "    # Analyze each target column",
        "    for i, target_col in enumerate(target_columns):",
        "        print(f'\\nAnalyzing target: {target_col}')",
        "        print('-' * 30)",
        "        ",
        "        y_true_col = y_test[:, i] if y_test.shape[1] > 1 else y_test.flatten()",
        "        y_pred_col = y_pred[:, i] if y_pred.shape[1] > 1 else y_pred.flatten()",
        "        ",
        "        # Determine if classification or regression",
        "        unique_values = len(np.unique(y_true_col))",
        "        is_classification = unique_values <= 20  # Threshold for classification",
        "        ",
        "        if is_classification:",
        "            print(f'Classification Analysis for {target_col}')",
        "            ",
        "            # Classification metrics",
        "            from sklearn.metrics import accuracy_score, classification_report",
        "            ",
        "            # Round predictions for classification",
        "            y_pred_rounded = np.round(y_pred_col).astype(int)",
        "            y_true_int = y_true_col.astype(int)",
        "            ",
        "            accuracy = accuracy_score(y_true_int, y_pred_rounded)",
        "            print(f'Accuracy: {accuracy:.4f}')",
        "            ",
        "            # Create confusion matrix",
        "            save_path = f'{model_name}_{target_col}_confusion_matrix.png' if save_plots else None",
        "            cm = create_confusion_matrix(",
        "                y_true_int, y_pred_rounded,",
        "                title=f'Confusion Matrix - {target_col}',",
        "                save_path=save_path",
        "            )",
        "            ",
        "            analysis_results[target_col] = {",
        "                'type': 'classification',",
        "                'accuracy': accuracy,",
        "                'confusion_matrix': cm.tolist()",
        "            }",
        "        ",
        "        else:",
        "            print(f'Regression Analysis for {target_col}')",
        "            ",
        "            # Regression metrics",
        "            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
        "            ",
        "            mae = mean_absolute_error(y_true_col, y_pred_col)",
        "            mse = mean_squared_error(y_true_col, y_pred_col)",
        "            rmse = np.sqrt(mse)",
        "            r2 = r2_score(y_true_col, y_pred_col)",
        "            ",
        "            print(f'MAE:  {mae:.4f}')",
        "            print(f'MSE:  {mse:.4f}')",
        "            print(f'RMSE: {rmse:.4f}')",
        "            print(f'R²:   {r2:.4f}')",
        "            ",
        "            # Create scatter plot",
        "            save_path = f'{model_name}_{target_col}_scatter.png' if save_plots else None",
        "            r2_calc = create_scatter_plot(",
        "                y_true_col, y_pred_col,",
        "                title=f'Predictions vs Actual - {target_col}',",
        "                save_path=save_path",
        "            )",
        "            ",
        "            # Create residuals plot",
        "            save_path = f'{model_name}_{target_col}_residuals.png' if save_plots else None",
        "            residuals_stats = create_residuals_plot(",
        "                y_true_col, y_pred_col,",
        "                title=f'Residuals Analysis - {target_col}',",
        "                save_path=save_path",
        "            )",
        "            ",
        "            analysis_results[target_col] = {",
        "                'type': 'regression',",
        "                'mae': mae,",
        "                'mse': mse,",
        "                'rmse': rmse,",
        "                'r2': r2,",
        "                'residuals_stats': residuals_stats",
        "            }",
        "    ",
        "    print('\\nAnalysis completed!')",
        "    return analysis_results",
        "",
        "",
        "# =============================================================================",
        "# MAIN EXECUTION - CONFIGURE YOUR DATASET PATH HERE",
        "# =============================================================================",
        "",
        "# DATASET CONFIGURATION",
        "DATASET_PATH = 'path/to/your/dataset.csv'  # UPDATE THIS TO YOUR DATASET PATH",
        "",
        "if __name__ == '__main__':",
        f"    print('Starting {model_def.name} Model Training')",
        "    print('=' * 50)",
        "",
        "    try:",
        "        # Step 1: Load and preprocess data",
        "        print('Step 1: Loading and preprocessing data...')",
        "        data = load_and_preprocess_data(DATASET_PATH, test_size=0.2)",
        "",
        "        # Step 2: Create model",
        "        print('Step 2: Creating model...')",
        "        model = create_model(data['input_shape'], data['output_shape'])",
        "        model.summary()",
        "",
        "        # Step 3: Train model",
        "        print('Step 3: Training model...')",
        "        history = train_model(",
        "            model, ",
        "            data['X_train'], data['y_train'], ",
        "            data['X_test'], data['y_test']",
        "        )",
        "",
        "        # Step 4: Evaluate model",
        "        print('Step 4: Evaluating model...')",
        "        predictions = evaluate_model(",
        "            model, ",
        "            data['X_test'], data['y_test'], ",
        "            data['target_encoders']",
        "        )",
        "",
        "        # Step 5: Analyze model performance with visualizations",
        "        print('Step 5: Creating analysis visualizations...')",
        "        analysis_results = analyze_model_performance(",
        "            model, data['X_test'], data['y_test'],",
        f"            {target_columns},",
        f"            model_name='{model_def.name.replace(' ', '_')}',",
        "            save_plots=True",
        "        )",
        "",
        "        # Step 6: Save model",
        "        print('Step 6: Saving model...')",
        f"        model.save('{model_def.name.replace(' ', '_')}_model.h5')",
        f"        print('Model saved as: {model_def.name.replace(' ', '_')}_model.h5')",
        "",
        "        print('Training completed successfully!')",
        "        print('=' * 50)",
        "",
        "        # Optional: Plot training history",
        "        try:",
        "            import matplotlib.pyplot as plt",
        "            ",
        "            plt.figure(figsize=(12, 4))",
        "            ",
        "            plt.subplot(1, 2, 1)",
        "            plt.plot(history.history['loss'], label='Training Loss')",
        "            if 'val_loss' in history.history:",
        "                plt.plot(history.history['val_loss'], label='Validation Loss')",
        "            plt.title('Model Loss')",
        "            plt.xlabel('Epoch')",
        "            plt.ylabel('Loss')",
        "            plt.legend()",
        "            ",
        "            plt.subplot(1, 2, 2)",
        "            if 'mae' in history.history:",
        "                plt.plot(history.history['mae'], label='Training MAE')",
        "                if 'val_mae' in history.history:",
        "                    plt.plot(history.history['val_mae'], label='Validation MAE')",
        "            plt.title('Model Metrics')",
        "            plt.xlabel('Epoch')",
        "            plt.ylabel('MAE')",
        "            plt.legend()",
        "            ",
        "            plt.tight_layout()",
        f"            plt.savefig('{model_def.name.replace(' ', '_')}_training_history.png')",
        "            plt.show()",
        "            ",
        f"            print('Training plots saved as: {model_def.name.replace(' ', '_')}_training_history.png')",
        "        except ImportError:",
        "            print('Warning: Matplotlib not available. Skipping plots.')",
        "",
        "    except FileNotFoundError:",
        "        print(f'Error: Dataset file not found at: {DATASET_PATH}')",
        "        print('Please update the DATASET_PATH variable with the correct path to your dataset.')",
        "    except Exception as e:",
        "        print(f'Error during training: {str(e)}')",
        "        import traceback",
        "        traceback.print_exc()",
        "",
        "    # Instructions for usage",
        r"    print('\n' + '=' * 60)",
        "    print('USAGE INSTRUCTIONS:')",
        "    print('=' * 60)",
        "    print('1. Update DATASET_PATH with your CSV file path')",
        "    print('2. Ensure your dataset has these columns:')",
        f"    print('   - Predictors: {predictor_columns}')",
        f"    print('   - Targets: {target_columns}')",
        f"    print('3. Run: python {model_def.name.replace(' ', '_')}_model.py')",
        "    print('=' * 60)"
    ])
    
    # Join all lines with newlines
    return '\n'.join(code_lines)


# Keep existing PyTorch and parsing functions unchanged
def generate_pytorch_code(model_def) -> str:
    """Generate PyTorch code from model definition - keeping existing implementation"""
    
    code = f'''"""
Auto-generated PyTorch model code
Model: {model_def.name}
Generated at: {model_def.updated_at}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    
    # Generate model class
    class_name = f"{model_def.name.replace(' ', '').replace('-', '_')}Model"
    
    if model_def.use_custom_architecture and model_def.custom_architecture:
        code += f"class {class_name}(nn.Module):\n"
        code += f"    def __init__(self, input_size, output_size):\n"
        code += f"        super({class_name}, self).__init__()\n\n"
        
        # Build layers
        layer_names = []
        prev_size = None
        
        for i, layer in enumerate(model_def.custom_architecture):
            layer_type = layer.get('type')
            params = layer.get('params', {})
            layer_name = f"layer{i+1}"
            
            # Normalize layer type to uppercase for consistency
            layer_type_normalized = _normalize_layer_type(layer_type)
            
            if layer_type_normalized == 'DENSE':
                units = params.get('units', 32)
                if prev_size is None:
                    code += f"        self.{layer_name} = nn.Linear(input_size, {units})\n"
                else:
                    code += f"        self.{layer_name} = nn.Linear({prev_size}, {units})\n"
                prev_size = units
                layer_names.append((layer_name, 'linear', params.get('activation', 'relu')))
                
            elif layer_type_normalized == 'LSTM':
                units = params.get('units', 50)
                if prev_size is None:
                    code += f"        self.{layer_name} = nn.LSTM(input_size, {units}, batch_first=True)\n"
                else:
                    code += f"        self.{layer_name} = nn.LSTM({prev_size}, {units}, batch_first=True)\n"
                prev_size = units
                layer_names.append((layer_name, 'lstm', None))
                
            elif layer_type_normalized == 'GRU':
                units = params.get('units', 50)
                if prev_size is None:
                    code += f"        self.{layer_name} = nn.GRU(input_size, {units}, batch_first=True)\n"
                else:
                    code += f"        self.{layer_name} = nn.GRU({prev_size}, {units}, batch_first=True)\n"
                prev_size = units
                layer_names.append((layer_name, 'gru', None))
                
            elif layer_type_normalized == 'DROPOUT':
                code += f"        self.{layer_name} = nn.Dropout({params.get('rate', 0.5)})\n"
                layer_names.append((layer_name, 'dropout', None))
                
            elif layer_type_normalized == 'BATCHNORMALIZATION':
                if prev_size:
                    code += f"        self.{layer_name} = nn.BatchNorm1d({prev_size})\n"
                layer_names.append((layer_name, 'batchnorm', None))
        
        # Output layer
        if prev_size:
            code += f"        self.output = nn.Linear({prev_size}, output_size)\n\n"
        else:
            code += f"        self.output = nn.Linear(input_size, output_size)\n\n"
        
        # Forward method
        code += f"    def forward(self, x):\n"
        for layer_name, layer_type, activation in layer_names:
            if layer_type == 'linear':
                code += f"        x = self.{layer_name}(x)\n"
                if activation == 'relu':
                    code += f"        x = F.relu(x)\n"
                elif activation == 'tanh':
                    code += f"        x = torch.tanh(x)\n"
                elif activation == 'sigmoid':
                    code += f"        x = torch.sigmoid(x)\n"
            elif layer_type in ['lstm', 'gru']:
                code += f"        x, _ = self.{layer_name}(x)\n"
                code += f"        x = x[:, -1, :]  # Get last output\n"
            else:
                code += f"        x = self.{layer_name}(x)\n"
        
        code += f"        x = self.output(x)\n"
        code += f"        return x\n\n"
        
    else:
        # Generate standard model
        model_type = model_def.model_type
        hyperparams = model_def.hyperparameters
        
        if model_type == 'lstm':
            code += f"class {class_name}(nn.Module):\n"
            code += f"    def __init__(self, input_size, hidden_size={hyperparams.get('units', 50)}, "
            code += f"num_layers={hyperparams.get('layers', 2)}, output_size=1, "
            code += f"dropout={hyperparams.get('dropout', 0.2)}):\n"
            code += f"        super({class_name}, self).__init__()\n"
            code += f"        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \n"
            code += f"                           batch_first=True, dropout=dropout if num_layers > 1 else 0)\n"
            code += f"        self.dropout = nn.Dropout(dropout)\n"
            code += f"        self.fc = nn.Linear(hidden_size, output_size)\n\n"
            code += f"    def forward(self, x):\n"
            code += f"        lstm_out, _ = self.lstm(x)\n"
            code += f"        last_output = lstm_out[:, -1, :]\n"
            code += f"        out = self.dropout(last_output)\n"
            code += f"        out = self.fc(out)\n"
            code += f"        return out\n\n"
            
        elif model_type == 'gru':
            code += f"class {class_name}(nn.Module):\n"
            code += f"    def __init__(self, input_size, hidden_size={hyperparams.get('units', 50)}, "
            code += f"num_layers={hyperparams.get('layers', 2)}, output_size=1, "
            code += f"dropout={hyperparams.get('dropout', 0.2)}):\n"
            code += f"        super({class_name}, self).__init__()\n"
            code += f"        self.gru = nn.GRU(input_size, hidden_size, num_layers, \n"
            code += f"                         batch_first=True, dropout=dropout if num_layers > 1 else 0)\n"
            code += f"        self.dropout = nn.Dropout(dropout)\n"
            code += f"        self.fc = nn.Linear(hidden_size, output_size)\n\n"
            code += f"    def forward(self, x):\n"
            code += f"        gru_out, _ = self.gru(x)\n"
            code += f"        last_output = gru_out[:, -1, :]\n"
            code += f"        out = self.dropout(last_output)\n"
            code += f"        out = self.fc(out)\n"
            code += f"        return out\n\n"
            
        elif model_type == 'cnn':
            code += f"class {class_name}(nn.Module):\n"
            code += f"    def __init__(self, input_channels, sequence_length, output_size, "
            code += f"filters={hyperparams.get('filters', 64)}, kernel_size={hyperparams.get('kernel_size', 3)}, "
            code += f"dropout={hyperparams.get('dropout', 0.2)}):\n"
            code += f"        super({class_name}, self).__init__()\n"
            code += f"        self.conv1 = nn.Conv1d(input_channels, filters, kernel_size, padding=1)\n"
            code += f"        self.pool1 = nn.MaxPool1d(2)\n"
            code += f"        self.dropout1 = nn.Dropout(dropout)\n"
            code += f"        self.conv2 = nn.Conv1d(filters, filters*2, kernel_size, padding=1)\n"
            code += f"        self.pool2 = nn.AdaptiveMaxPool1d(1)\n"
            code += f"        self.dropout2 = nn.Dropout(dropout)\n"
            code += f"        self.fc = nn.Linear(filters*2, output_size)\n\n"
            code += f"    def forward(self, x):\n"
            code += f"        x = x.transpose(1, 2)  # (batch, features, sequence)\n"
            code += f"        x = F.relu(self.conv1(x))\n"
            code += f"        x = self.pool1(x)\n"
            code += f"        x = self.dropout1(x)\n"
            code += f"        x = F.relu(self.conv2(x))\n"
            code += f"        x = self.pool2(x)\n"
            code += f"        x = self.dropout2(x)\n"
            code += f"        x = x.squeeze(-1)\n"
            code += f"        x = self.fc(x)\n"
            code += f"        return x\n\n"
    
    # Add example usage
    code += f"\n# Example usage:\n"
    code += f"# model = {class_name}(input_size=5, output_size=2)\n"
    code += f"# x = torch.randn(32, 10, 5)  # batch_size=32, sequence_length=10, features=5\n"
    code += f"# output = model(x)\n"
    code += f"# print(output.shape)  # torch.Size([32, 2])\n"
    
    return code


# Keep all existing parsing functions unchanged
def parse_keras_code(code: str) -> Dict[str, Any]:
    """Parse Keras code and extract architecture"""
    
    architecture = []
    hyperparameters = {}
    
    # Parse the code using AST
    try:
        tree = ast.parse(code)
        
        # Find the create_model function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'create_model':
                # Parse function body
                for stmt in node.body:
                    # Look for model.add() calls
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        if (hasattr(stmt.value.func, 'attr') and 
                            stmt.value.func.attr == 'add' and
                            hasattr(stmt.value.func.value, 'id') and
                            stmt.value.func.value.id == 'model'):
                            
                            # Extract layer info
                            layer_call = stmt.value.args[0]
                            if isinstance(layer_call, ast.Call):
                                layer_info = _extract_layer_info(layer_call)
                                if layer_info:
                                    architecture.append(layer_info)
                    
                    # Look for model.compile() call
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        if (hasattr(stmt.value.func, 'attr') and 
                            stmt.value.func.attr == 'compile'):
                            # Extract compile parameters
                            for keyword in stmt.value.keywords:
                                if keyword.arg == 'optimizer':
                                    # Extract optimizer info
                                    if isinstance(keyword.value, ast.Call):
                                        hyperparameters['optimizer'] = _extract_optimizer_info(keyword.value)
                                elif keyword.arg == 'loss':
                                    if isinstance(keyword.value, ast.Constant):
                                        hyperparameters['loss'] = keyword.value.value
        
    except SyntaxError as e:
        print(f"Error parsing code: {e}")
        return None
    
    return {
        'architecture': architecture,
        'hyperparameters': hyperparameters
    }


def _extract_layer_info(layer_call: ast.Call) -> Dict[str, Any]:
    """Extract layer information from AST Call node"""
    
    layer_type = None
    params = {}
    
    # Get layer type
    if hasattr(layer_call.func, 'attr'):
        layer_type = layer_call.func.attr
    
    # Extract parameters
    for keyword in layer_call.keywords:
        key = keyword.arg
        value = _extract_value(keyword.value)
        if value is not None:
            params[key] = value
    
    if layer_type:
        return {
            'type': layer_type,
            'params': params
        }
    
    return None


def _extract_value(node):
    """Extract value from AST node"""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_extract_value(node.value)}.{node.attr}"
    elif isinstance(node, ast.Call):
        # Handle function calls (like optimizers)
        if hasattr(node.func, 'attr'):
            return node.func.attr
    return None


def _extract_optimizer_info(optimizer_call: ast.Call) -> str:
    """Extract optimizer name from AST Call node"""
    if hasattr(optimizer_call.func, 'attr'):
        return optimizer_call.func.attr.lower()
    return 'adam'


def parse_pytorch_code(code: str) -> Dict[str, Any]:
    """Parse PyTorch code and extract architecture"""
    
    architecture = []
    hyperparameters = {}
    
    # Use regex to extract layer definitions
    layer_pattern = r'self\.(\w+)\s*=\s*nn\.(\w+)\((.*?)\)'
    matches = re.findall(layer_pattern, code, re.MULTILINE)
    
    for layer_name, layer_type, params_str in matches:
        if layer_name == 'output':
            continue
            
        # Parse parameters
        params = {}
        
        if layer_type == 'Linear':
            # Extract units
            units_match = re.search(r'(\d+)\s*,\s*(\d+)', params_str)
            if units_match:
                params['units'] = int(units_match.group(2))
            architecture.append({'type': 'Dense', 'params': params})
            
        elif layer_type == 'LSTM':
            units_match = re.search(r'(\w+)\s*,\s*(\d+)', params_str)
            if units_match:
                params['units'] = int(units_match.group(2))
            params['return_sequences'] = 'batch_first=True' in params_str
            architecture.append({'type': 'LSTM', 'params': params})
            
        elif layer_type == 'GRU':
            units_match = re.search(r'(\w+)\s*,\s*(\d+)', params_str)
            if units_match:
                params['units'] = int(units_match.group(2))
            params['return_sequences'] = 'batch_first=True' in params_str
            architecture.append({'type': 'GRU', 'params': params})
            
        elif layer_type == 'Dropout':
            rate_match = re.search(r'([\d.]+)', params_str)
            if rate_match:
                params['rate'] = float(rate_match.group(1))
            architecture.append({'type': 'Dropout', 'params': params})
            
        elif layer_type == 'Conv1d':
            filters_match = re.search(r'(\w+)\s*,\s*(\d+)', params_str)
            if filters_match:
                params['filters'] = int(filters_match.group(2))
            kernel_match = re.search(r'kernel_size=(\d+)', params_str)
            if kernel_match:
                params['kernel_size'] = int(kernel_match.group(1))
            architecture.append({'type': 'Conv1D', 'params': params})
            
        elif layer_type == 'BatchNorm1d':
            architecture.append({'type': 'BatchNormalization', 'params': params})
    
    return {
        'architecture': architecture,
        'hyperparameters': hyperparameters
    }


def validate_architecture(architecture: List[Dict]) -> bool:
    """Validate that the architecture is valid"""
    
    if not architecture:
        return False
    
    # Check that each layer has required fields
    for layer in architecture:
        if 'type' not in layer:
            return False
        if 'params' not in layer:
            layer['params'] = {}
    
    return True


def _generate_sklearn_header(model_def) -> List[str]:
    """Generate header and imports for sklearn code"""
    model_type = model_def.model_type
    hyperparams = model_def.hyperparameters or {}
    
    # Get Module 1 and Module 2 configurations
    split_method = model_def.default_split_method if hasattr(model_def, 'default_split_method') else 'random'
    split_config = model_def.default_split_config if hasattr(model_def, 'default_split_config') else {}
    execution_method = model_def.default_execution_method if hasattr(model_def, 'default_execution_method') else 'standard'
    execution_config = model_def.default_execution_config if hasattr(model_def, 'default_execution_config') else {}
    
    header_lines = [
        '"""',
        f'Auto-generated scikit-learn {model_type.upper()} model code',
        f'Model: {model_def.name}',
        f'Type: {model_def.model_type.upper()}',
        f'Generated at: {model_def.updated_at}',
        '',
        'Configuration:',
        f'- Target columns: {model_def.target_columns}',
        f'- Predictor columns: {len(model_def.predictor_columns)} features',
        f'- Problem type: {hyperparams.get("problem_type", "auto-detect")}',
        '',
        'Module 1 - Data Split Configuration:',
        f'- Split method: {split_method}',
        f'- Train size: {split_config.get("train_size", 0.7)}',
        f'- Validation size: {split_config.get("val_size", 0.15)}',
        f'- Test size: {split_config.get("test_size", 0.15)}',
        '',
        'Module 2 - Execution Configuration:',
        f'- Execution method: {execution_method}',
        f'- Configuration: {json.dumps(execution_config)}' if execution_config else f'- Configuration: Default',
        '"""',
        '',
        'import numpy as np',
        'import pandas as pd',
        'import joblib',
        'import json',
        'import warnings',
        'from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV',
        'from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit',
        'from sklearn.model_selection import LeaveOneOut, RepeatedKFold, RepeatedStratifiedKFold',
        'from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder',
        'from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score',
        'from sklearn.metrics import accuracy_score, classification_report, confusion_matrix',
        'from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score',
        'import matplotlib.pyplot as plt',
        'import seaborn as sns',
        '',
        'warnings.filterwarnings("ignore")'
    ]
    
    # Import model-specific libraries
    if model_type == 'random_forest':
        problem_type = hyperparams.get('problem_type', 'regression')
        if problem_type == 'classification':
            header_lines.append('from sklearn.ensemble import RandomForestClassifier')
        else:
            header_lines.append('from sklearn.ensemble import RandomForestRegressor')
    elif model_type == 'decision_tree':
        header_lines.append('from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier')
    elif model_type == 'xgboost':
        header_lines.append('import xgboost as xgb')
    
    header_lines.extend(['', '', ''])
    return header_lines


def _generate_decision_tree_params(hyperparams) -> List[str]:
    """Generate Decision Tree parameter configuration"""
    problem_type = hyperparams.get('problem_type', 'regression')
    params_lines = ['    # Model parameters', '    params = {']
    
    # Criterion
    criterion = hyperparams.get('criterion', 'squared_error' if problem_type == 'regression' else 'gini')
    params_lines.append(f'        "criterion": "{criterion}",')
    
    # Splitter
    params_lines.append(f'        "splitter": "{hyperparams.get("splitter", "best")}",')
    
    # Handle max_depth
    if hyperparams.get('max_depth_enabled', True) and hyperparams.get('max_depth'):
        params_lines.append(f'        "max_depth": {hyperparams["max_depth"]},')
    else:
        params_lines.append('        "max_depth": None,')
    
    # Min samples split and leaf
    params_lines.append(f'        "min_samples_split": {hyperparams.get("min_samples_split", 2)},')
    
    # Handle min_samples_leaf (can be int or float)
    min_samples_leaf = hyperparams.get('min_samples_leaf', 1)
    if isinstance(min_samples_leaf, float) and min_samples_leaf < 1:
        params_lines.append(f'        "min_samples_leaf": {min_samples_leaf},')
    else:
        params_lines.append(f'        "min_samples_leaf": {int(min_samples_leaf)},')
    
    # Min weight fraction leaf
    params_lines.append(f'        "min_weight_fraction_leaf": {hyperparams.get("min_weight_fraction_leaf", 0.0)},')
    
    # Max features
    max_features = hyperparams.get('max_features')
    if max_features is None:
        params_lines.append('        "max_features": None,')
    elif isinstance(max_features, (int, float)):
        params_lines.append(f'        "max_features": {max_features},')
    else:
        params_lines.append(f'        "max_features": "{max_features}",')
    
    # Max leaf nodes
    if hyperparams.get('max_leaf_nodes'):
        params_lines.append(f'        "max_leaf_nodes": {hyperparams["max_leaf_nodes"]},')
    else:
        params_lines.append('        "max_leaf_nodes": None,')
    
    # Min impurity decrease
    params_lines.append(f'        "min_impurity_decrease": {hyperparams.get("min_impurity_decrease", 0.0)},')
    
    # CCP alpha
    params_lines.append(f'        "ccp_alpha": {hyperparams.get("ccp_alpha", 0.0)},')
    
    # Classification specific - class weight
    if problem_type == 'classification' and hyperparams.get('class_weight'):
        class_weight = hyperparams['class_weight']
        if class_weight == 'balanced':
            params_lines.append('        "class_weight": "balanced",')
        elif class_weight != 'None':
            params_lines.append(f'        "class_weight": {class_weight},')
    
    # Random state
    if hyperparams.get('random_state') is not None:
        params_lines.append(f'        "random_state": {hyperparams["random_state"]},')
    
    # Remove trailing comma from last parameter
    if params_lines[-1].endswith(','):
        params_lines[-1] = params_lines[-1][:-1]
    
    params_lines.append('    }')
    params_lines.append('')
    
    # Model creation
    if problem_type == 'classification':
        params_lines.append('    model = DecisionTreeClassifier(**params)')
    else:
        params_lines.append('    model = DecisionTreeRegressor(**params)')
    
    params_lines.append('')
    params_lines.append('    # Train model')
    params_lines.append('    model.fit(X_train, y_train)')
    
    # Add validation method code if specified
    validation_method = hyperparams.get('validation_method', 'holdout')
    if validation_method == 'cv':
        params_lines.extend([
            '',
            '    # Cross-validation (opcional)',
            '    from sklearn.model_selection import cross_val_score',
            '    cv_scores = cross_val_score(model, X_train, y_train, cv=5)',
            '    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")'
        ])
    
    return params_lines


def _generate_xgboost_params(hyperparams) -> List[str]:
    """Generate XGBoost parameter configuration"""
    problem_type = hyperparams.get('problem_type', 'regression')
    params_lines = ['    # Model parameters', '    params = {']
    
    # Basic parameters
    params_lines.append(f'        "n_estimators": {hyperparams.get("n_estimators", 500)},')
    params_lines.append(f'        "learning_rate": {hyperparams.get("learning_rate", 0.05)},')
    
    # Handle max_depth (0 means no limit in XGBoost)
    max_depth = hyperparams.get('max_depth', 6)
    if max_depth and max_depth > 0:
        params_lines.append(f'        "max_depth": {max_depth},')
    
    params_lines.append(f'        "subsample": {hyperparams.get("subsample", 0.8)},')
    params_lines.append(f'        "colsample_bytree": {hyperparams.get("colsample_bytree", 0.8)},')
    
    # Objective and eval_metric
    if problem_type == 'classification':
        params_lines.append('        "objective": "binary:logistic",  # Adjust for multiclass')
    else:
        params_lines.append('        "objective": "reg:squarederror",')
    
    eval_metric = hyperparams.get('eval_metric', 'rmse' if problem_type == 'regression' else 'logloss')
    params_lines.append(f'        "eval_metric": "{eval_metric}",')
    
    # Advanced parameters
    params_lines.append(f'        "min_child_weight": {hyperparams.get("min_child_weight", 1)},')
    params_lines.append(f'        "gamma": {hyperparams.get("gamma", 0)},')
    params_lines.append(f'        "reg_lambda": {hyperparams.get("reg_lambda", 1)},')
    params_lines.append(f'        "reg_alpha": {hyperparams.get("reg_alpha", 0)},')
    
    if hyperparams.get('max_delta_step', 0) > 0:
        params_lines.append(f'        "max_delta_step": {hyperparams["max_delta_step"]},')
    
    # Column sampling
    if hyperparams.get('colsample_bylevel', 1.0) != 1.0:
        params_lines.append(f'        "colsample_bylevel": {hyperparams["colsample_bylevel"]},')
    if hyperparams.get('colsample_bynode', 1.0) != 1.0:
        params_lines.append(f'        "colsample_bynode": {hyperparams["colsample_bynode"]},')
    
    # Tree method and device
    tree_method = hyperparams.get('tree_method', 'auto')
    if tree_method != 'auto':
        params_lines.append(f'        "tree_method": "{tree_method}",')
    
    if hyperparams.get('use_gpu', False):
        params_lines.append('        "device": "cuda",')
        if tree_method == 'auto':
            params_lines.append('        "tree_method": "hist",')
    
    # Booster
    booster = hyperparams.get('booster', 'gbtree')
    if booster != 'gbtree':
        params_lines.append(f'        "booster": "{booster}",')
        
    # DART parameters
    if booster == 'dart':
        params_lines.append(f'        "rate_drop": {hyperparams.get("rate_drop", 0.1)},')
        params_lines.append(f'        "skip_drop": {hyperparams.get("skip_drop", 0.5)},')
    
    # Growth policy
    grow_policy = hyperparams.get('grow_policy', 'depthwise')
    if grow_policy != 'depthwise':
        params_lines.append(f'        "grow_policy": "{grow_policy}",')
        if grow_policy == 'lossguide' and hyperparams.get('max_leaves'):
            params_lines.append(f'        "max_leaves": {hyperparams["max_leaves"]},')
    
    # Classification specific
    if problem_type == 'classification' and hyperparams.get('scale_pos_weight', 1) != 1:
        params_lines.append(f'        "scale_pos_weight": {hyperparams["scale_pos_weight"]},')
    
    # Other parameters
    if hyperparams.get('max_bin', 256) != 256:
        params_lines.append(f'        "max_bin": {hyperparams["max_bin"]},')
    
    # Execution parameters
    params_lines.append(f'        "n_jobs": {hyperparams.get("n_jobs", -1)},')
    params_lines.append(f'        "verbosity": {hyperparams.get("verbosity", 1)},')
    
    if hyperparams.get('random_state') is not None:
        params_lines.append(f'        "random_state": {hyperparams["random_state"]},')
    
    # Remove trailing comma from last parameter
    if params_lines[-1].endswith(','):
        params_lines[-1] = params_lines[-1][:-1]
    
    params_lines.append('    }')
    params_lines.append('')
    
    # Model creation
    if problem_type == 'classification':
        params_lines.append('    # Check if multiclass')
        params_lines.append('    n_classes = len(np.unique(y_train))')
        params_lines.append('    if n_classes > 2:')
        params_lines.append('        params["objective"] = "multi:softprob"')
        params_lines.append('        params["num_class"] = n_classes')
        params_lines.append('        if params.get("eval_metric") == "logloss":')
        params_lines.append('            params["eval_metric"] = "mlogloss"')
        params_lines.append('')
        params_lines.append('    model = xgb.XGBClassifier(**params)')
    else:
        params_lines.append('    model = xgb.XGBRegressor(**params)')
    
    # Early stopping code
    if hyperparams.get('early_stopping_enabled', True):
        params_lines.append('')
        params_lines.append('    # Train with early stopping')
        params_lines.append(f'    early_stopping_rounds = {hyperparams.get("early_stopping_rounds", 50)}')
        params_lines.append('    eval_set = [(X_val, y_val)]')
        params_lines.append('    model.fit(X_train, y_train,')
        params_lines.append('              eval_set=eval_set,')
        params_lines.append('              early_stopping_rounds=early_stopping_rounds,')
        params_lines.append('              verbose=True)')
    else:
        params_lines.append('')
        params_lines.append('    # Train model')
        params_lines.append('    model.fit(X_train, y_train)')
    
    return params_lines


def _generate_random_forest_params(hyperparams) -> List[str]:
    """Generate Random Forest parameter configuration with enhanced features"""
    problem_type = hyperparams.get('problem_type', 'regression')
    params_lines = ['    # Model parameters', '    params = {']
    
    # Basic parameters
    if hyperparams.get('n_estimators'):
        params_lines.append(f'        "n_estimators": {hyperparams["n_estimators"]},')
    
    # Handle max_depth
    if hyperparams.get('max_depth_enabled', False) and hyperparams.get('max_depth'):
        params_lines.append(f'        "max_depth": {hyperparams["max_depth"]},')
    else:
        params_lines.append('        "max_depth": None,')
    
    # Handle max_features with auto-detection
    max_features = hyperparams.get('max_features', 'auto')
    if max_features == 'auto':
        if problem_type == 'classification':
            params_lines.append('        "max_features": "sqrt",  # Auto-selected for classification')
        else:
            params_lines.append('        "max_features": 1.0,  # Auto-selected for regression')
    elif isinstance(max_features, str) and max_features in ['sqrt', 'log2']:
        params_lines.append(f'        "max_features": "{max_features}",')
    elif max_features == 'custom' and hyperparams.get('max_features_fraction'):
        params_lines.append(f'        "max_features": {hyperparams["max_features_fraction"]},')
    else:
        try:
            max_features_val = float(max_features)
            params_lines.append(f'        "max_features": {max_features_val},')
        except:
            params_lines.append('        "max_features": "sqrt",  # Default fallback')
    
    # Handle criterion with auto-detection
    criterion = hyperparams.get('criterion', 'auto')
    if criterion == 'auto':
        if problem_type == 'classification':
            params_lines.append('        "criterion": "gini",  # Auto-selected for classification')
        else:
            params_lines.append('        "criterion": "squared_error",  # Auto-selected for regression')
    else:
        params_lines.append(f'        "criterion": "{criterion}",')
    
    # Tree structure parameters
    if hyperparams.get('min_samples_split'):
        params_lines.append(f'        "min_samples_split": {hyperparams["min_samples_split"]},')
    if hyperparams.get('min_samples_leaf'):
        params_lines.append(f'        "min_samples_leaf": {hyperparams["min_samples_leaf"]},')
    if hyperparams.get('min_weight_fraction_leaf'):
        params_lines.append(f'        "min_weight_fraction_leaf": {hyperparams["min_weight_fraction_leaf"]},')
    if hyperparams.get('min_impurity_decrease'):
        params_lines.append(f'        "min_impurity_decrease": {hyperparams["min_impurity_decrease"]},')
    
    # Sampling parameters
    bootstrap = hyperparams.get('bootstrap', True)
    params_lines.append(f'        "bootstrap": {bootstrap},')
    
    # OOB score only if bootstrap is True
    if bootstrap and hyperparams.get('oob_score', False):
        params_lines.append('        "oob_score": True,')
    
    # Performance parameters
    params_lines.append(f'        "n_jobs": {hyperparams.get("n_jobs", -1)},  # Use all CPU cores')
    if hyperparams.get('random_state') is not None:
        params_lines.append(f'        "random_state": {hyperparams["random_state"]},')
    
    # Class weight for classification
    if problem_type == 'classification':
        class_weight = hyperparams.get('class_weight', None)
        if class_weight in ['balanced', 'balanced_subsample']:
            params_lines.append(f'        "class_weight": "{class_weight}",')
    
    # Remove trailing comma from last line
    if params_lines[-1].endswith(','):
        params_lines[-1] = params_lines[-1][:-1]
    
    params_lines.append('    }')
    params_lines.append('')
    
    # Add preset comment
    preset = hyperparams.get('preset', 'balanceado')
    preset_comments = {
        'rapido': '# Preset: Rápido (menos árboles, más rápido)',
        'balanceado': '# Preset: Balanceado (buen compromiso velocidad/precisión)',
        'preciso': '# Preset: Preciso (más árboles, mejor calidad)'
    }
    if preset in preset_comments:
        params_lines.append(f'    {preset_comments[preset]}')
    
    # Create model instance
    if problem_type == 'classification':
        params_lines.append('    model = RandomForestClassifier(**params)')
    else:
        params_lines.append('    model = RandomForestRegressor(**params)')
    
    return params_lines


def _generate_model_creation_code(model_type, hyperparams) -> List[str]:
    """Generate model creation code based on model type"""
    if model_type == 'random_forest':
        return _generate_random_forest_params(hyperparams)
        
    elif model_type == 'decision_tree':
        return _generate_decision_tree_params(hyperparams)
        
    elif model_type == 'xgboost':
        return _generate_xgboost_params(hyperparams)
    
    return []




def _generate_data_loading_function(model_def) -> List[str]:
    """Generate data loading and preprocessing function"""
    predictor_columns = [str(col) for col in model_def.predictor_columns]
    target_columns = [str(col) for col in model_def.target_columns]
    hyperparams = model_def.hyperparameters or {}
    model_type = model_def.model_type
    
    lines = [
        'def load_and_preprocess_data(file_path):',
        '    """',
        '    Load and preprocess the dataset',
        '    """',
        '    print(f"Loading data from: {file_path}")',
        '    df = pd.read_csv(file_path)',
        '    print(f"Dataset shape: {df.shape}")',
        '    print(f"Columns: {list(df.columns)})")',
        '',
        '    # Define columns',
        f'    predictor_columns = {predictor_columns}',
        f'    target_columns = {target_columns}',
        '',
        '    # Validate columns',
        '    missing_predictors = [col for col in predictor_columns if col not in df.columns]',
        '    missing_targets = [col for col in target_columns if col not in df.columns]',
        '    ',
        '    if missing_predictors:',
        '        raise ValueError(f"Missing predictor columns: {missing_predictors}")',
        '    if missing_targets:',
        '        raise ValueError(f"Missing target columns: {missing_targets}")',
        '',
        '    # Extract features and target',
        '    X = df[predictor_columns].copy()',
        '    y = df[target_columns[0]] if len(target_columns) == 1 else df[target_columns]',
        '',
        '    # Handle missing values',
        '    print("\\nHandling missing values...")',
        '    print(f"Missing values in features: {X.isnull().sum().sum()}")',
        '    print(f"Missing values in target: {y.isnull().sum() if hasattr(y, \'isnull\') else 0}")',
        ''
    ]
    
    # Fill missing values
    lines.extend([
        '    # Fill missing values with mean',
        '    X = X.fillna(X.mean())',
        '    if hasattr(y, \'fillna\'):',
        '        y = y.fillna(y.mean() if y.dtype in [np.float64, np.float32] else y.mode()[0])',
        ''
    ])
    
    lines.extend([
        '    return X, y, predictor_columns, target_columns',
        '',
        ''
    ])
    
    return lines


def _generate_data_split_module(model_def) -> List[str]:
    """Generate Module 1: Data Split implementation"""
    split_method = model_def.default_split_method if hasattr(model_def, 'default_split_method') else 'random'
    split_config = model_def.default_split_config if hasattr(model_def, 'default_split_config') else {}
    
    lines = [
        '# =============================================================================',
        '# MODULE 1: DATA SPLIT CONFIGURATION',
        '# =============================================================================',
        '',
        'class DataSplitter:',
        '    """Module 1: Handles data splitting according to configured strategy"""',
        '    ',
        f'    def __init__(self, strategy="{split_method}", config=None):',
        '        self.strategy = strategy',
        '        self.config = config or {}',
        '        ',
        '    def split(self, X, y):',
        '        """Split data according to configured strategy"""',
        '        train_size = self.config.get("train_size", 0.7)',
        '        val_size = self.config.get("val_size", 0.15)',
        '        test_size = self.config.get("test_size", 0.15)',
        '        random_state = self.config.get("random_state", 42)',
        '        ',
        '        if self.strategy == "random":',
        '            # Random split with shuffling',
        '            X_temp, X_test, y_temp, y_test = train_test_split(',
        '                X, y, test_size=test_size, random_state=random_state, shuffle=True',
        '            )',
        '            val_proportion = val_size / (train_size + val_size)',
        '            X_train, X_val, y_train, y_val = train_test_split(',
        '                X_temp, y_temp, test_size=val_proportion, random_state=random_state, shuffle=True',
        '            )',
        '            ',
        '        elif self.strategy == "stratified":',
        '            # Stratified split for classification',
        '            y_stratify = y if len(y.shape) == 1 else y[:, 0]',
        '            X_temp, X_test, y_temp, y_test = train_test_split(',
        '                X, y, test_size=test_size, random_state=random_state, stratify=y_stratify, shuffle=True',
        '            )',
        '            val_proportion = val_size / (train_size + val_size)',
        '            y_temp_stratify = y_temp if len(y_temp.shape) == 1 else y_temp[:, 0]',
        '            X_train, X_val, y_train, y_val = train_test_split(',
        '                X_temp, y_temp, test_size=val_proportion, random_state=random_state, stratify=y_temp_stratify, shuffle=True',
        '            )',
        '            ',
        '        elif self.strategy == "temporal":',
        '            # Temporal split maintaining order',
        '            n_samples = len(X)',
        '            train_end = int(n_samples * train_size)',
        '            val_end = int(n_samples * (train_size + val_size))',
        '            ',
        '            X_train = X[:train_end]',
        '            y_train = y[:train_end]',
        '            X_val = X[train_end:val_end]',
        '            y_val = y[train_end:val_end]',
        '            X_test = X[val_end:]',
        '            y_test = y[val_end:]',
        '            ',
        '        elif self.strategy == "sequential":',
        '            # Sequential split without shuffling',
        '            n_samples = len(X)',
        '            train_end = int(n_samples * train_size)',
        '            val_end = int(n_samples * (train_size + val_size))',
        '            ',
        '            X_train = X.iloc[:train_end] if hasattr(X, "iloc") else X[:train_end]',
        '            y_train = y.iloc[:train_end] if hasattr(y, "iloc") else y[:train_end]',
        '            X_val = X.iloc[train_end:val_end] if hasattr(X, "iloc") else X[train_end:val_end]',
        '            y_val = y.iloc[train_end:val_end] if hasattr(y, "iloc") else y[train_end:val_end]',
        '            X_test = X.iloc[val_end:] if hasattr(X, "iloc") else X[val_end:]',
        '            y_test = y.iloc[val_end:] if hasattr(y, "iloc") else y[val_end:]',
        '            ',
        '        else:',
        '            # Default to random split',
        '            X_temp, X_test, y_temp, y_test = train_test_split(',
        '                X, y, test_size=test_size, random_state=random_state, shuffle=True',
        '            )',
        '            val_proportion = val_size / (train_size + val_size)',
        '            X_train, X_val, y_train, y_val = train_test_split(',
        '                X_temp, y_temp, test_size=val_proportion, random_state=random_state, shuffle=True',
        '            )',
        '            ',
        '        return X_train, X_val, X_test, y_train, y_val, y_test',
        '',
        ''
    ]
    
    return lines


def _generate_execution_module(model_def) -> List[str]:
    """Generate Module 2: Execution Configuration implementation"""
    execution_method = model_def.default_execution_method if hasattr(model_def, 'default_execution_method') else 'standard'
    execution_config = model_def.default_execution_config if hasattr(model_def, 'default_execution_config') else {}
    
    lines = [
        '# =============================================================================',
        '# MODULE 2: EXECUTION CONFIGURATION',
        '# =============================================================================',
        '',
        'class ExecutionStrategy:',
        '    """Module 2: Handles model execution strategy (cross-validation, etc.)"""',
        '    ',
        f'    def __init__(self, method="{execution_method}", config=None):',
        '        self.method = method',
        '        self.config = config or {}',
        '        ',
        '    def execute(self, model, X_train, y_train, X_val, y_val):',
        '        """Execute training according to configured strategy"""',
        '        ',
        '        if self.method == "standard":',
        '            # Standard training without cross-validation',
        '            print("Executing standard training...")',
        '            model.fit(X_train, y_train)',
        '            train_score = model.score(X_train, y_train)',
        '            val_score = model.score(X_val, y_val)',
        '            print(f"Training score: {train_score:.4f}")',
        '            print(f"Validation score: {val_score:.4f}")',
        '            return {"train_score": train_score, "val_score": val_score}',
        '            ',
        '        elif self.method == "kfold":',
        '            # K-Fold Cross Validation',
        '            n_splits = self.config.get("n_splits", 5)',
        '            shuffle = self.config.get("shuffle", True)',
        '            random_state = self.config.get("random_state", 42)',
        '            print(f"\\n🔧 Module 2: Configuration d\'Exécution")',
        '            print(f"   Méthode: {self.method}")',
        '            print(f"   Nombre de folds: {n_splits}")',
        '            print(f"   Mélange: {\'Oui\' if shuffle else \'Non\'}")',
        '            print(f"   Random state: {random_state}")',
        '            print(f"\\nExecuting {n_splits}-fold cross validation...")',
        '            ',
        '            from sklearn.model_selection import KFold',
        '            from sklearn.base import clone',
        '            import time',
        '            ',
        '            # Only set random_state if shuffle is True',
        '            if shuffle:',
        '                kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)',
        '            else:',
        '                kf = KFold(n_splits=n_splits, shuffle=shuffle)',
        '            cv_scores = []',
        '            fold = 0',
        '            ',
        '            # Combine train and validation for CV',
        '            X_full = np.vstack([X_train, X_val])',
        '            y_full = np.concatenate([y_train, y_val])',
        '            ',
        '            for train_idx, val_idx in kf.split(X_full, y_full):',
        '                fold += 1',
        '                fold_start_time = time.time()',
        '                print(f"\\n🔄 Fold {fold}/{n_splits} - Démarrage de l\'entraînement...")',
        '                ',
        '                X_fold_train = X_full[train_idx]',
        '                y_fold_train = y_full[train_idx]',
        '                X_fold_val = X_full[val_idx]',
        '                y_fold_val = y_full[val_idx]',
        '                ',
        '                print(f"   📊 Données d\'entraînement du fold: {X_fold_train.shape[0]} échantillons ({X_fold_train.shape[0]/X_full.shape[0]*100:.1f}% du total)")',
        '                print(f"   📋 Données de validation du fold: {X_fold_val.shape[0]} échantillons ({X_fold_val.shape[0]/X_full.shape[0]*100:.1f}% du total)")',
        '                print(f"   💡 Note: Ces pourcentages sont pour la cross-validation, pas la division train/val/test configurée")',
        '                print(f"   🔧 Entraînement du modèle en cours...")',
        '                ',
        '                # Clone and train model for this fold',
        '                fold_model = clone(model)',
        '                train_start = time.time()',
        '                fold_model.fit(X_fold_train, y_fold_train)',
        '                train_time = time.time() - train_start',
        '                ',
        '                print(f"   ⏱️  Temps d\'entraînement: {train_time:.2f}s")',
        '                ',
        '                # Evaluate on fold',
        '                train_score = fold_model.score(X_fold_train, y_fold_train)',
        '                val_score = fold_model.score(X_fold_val, y_fold_val)',
        '                cv_scores.append(val_score)',
        '                ',
        '                fold_total_time = time.time() - fold_start_time',
        '                print(f"   ✅ Score d\'entraînement: {train_score:.4f}")',
        '                print(f"   📈 Score de validation: {val_score:.4f}")',
        '                print(f"   ⏰ Temps total fold: {fold_total_time:.2f}s")',
        '                print(f"   📊 Score moyen jusqu\'ici: {np.mean(cv_scores):.4f}")',
        '            ',
        '            # Train final model on full training data',
        '            print(f"\\n✅ Cross-validation complétée:")',
        '            print(f"   📊 Meilleur score: {max(cv_scores):.4f}")',
        '            print(f"   📈 Score moyen: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")',
        '            print(f"   📉 Score le plus bas: {min(cv_scores):.4f}")',
        '            print(f"\\nEntraînement du modèle final sur toutes les données d\'entraînement...")',
        '            model.fit(X_train, y_train)',
        '            return {"cv_scores": cv_scores, "cv_mean": np.mean(cv_scores), "cv_std": np.std(cv_scores)}',
        '            ',
        '        elif self.method == "stratified_kfold":',
        '            # Stratified K-Fold for classification',
        '            n_splits = self.config.get("n_splits", 5)',
        '            shuffle = self.config.get("shuffle", True)',
        '            random_state = self.config.get("random_state", 42)',
        '            print(f"\\n🔧 Module 2: Configuration d\'Exécution")',
        '            print(f"   Méthode: {self.method}")',
        '            print(f"   Nombre de folds: {n_splits}")',
        '            print(f"   Mélange: {\'Oui\' if shuffle else \'Non\'}")',
        '            print(f"   Random state: {random_state}")',
        '            print(f"\\nExecuting stratified {n_splits}-fold cross validation...")',
        '            ',
        '            from sklearn.model_selection import StratifiedKFold',
        '            from sklearn.base import clone',
        '            import time',
        '            ',
        '            # Only set random_state if shuffle is True',
        '            if shuffle:',
        '                skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)',
        '            else:',
        '                skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)',
        '            cv_scores = []',
        '            fold = 0',
        '            ',
        '            # Combine train and validation for CV',
        '            X_full = np.vstack([X_train, X_val])',
        '            y_full = np.concatenate([y_train, y_val])',
        '            y_stratify = y_full if len(y_full.shape) == 1 else y_full[:, 0]',
        '            ',
        '            for train_idx, val_idx in skf.split(X_full, y_stratify):',
        '                fold += 1',
        '                fold_start_time = time.time()',
        '                print(f"\\n🔄 Fold {fold}/{n_splits} - Démarrage de l\'entraînement...")',
        '                ',
        '                X_fold_train = X_full[train_idx]',
        '                y_fold_train = y_full[train_idx]',
        '                X_fold_val = X_full[val_idx]',
        '                y_fold_val = y_full[val_idx]',
        '                ',
        '                print(f"   📊 Données d\'entraînement du fold: {X_fold_train.shape[0]} échantillons ({X_fold_train.shape[0]/X_full.shape[0]*100:.1f}% du total)")',
        '                print(f"   📋 Données de validation du fold: {X_fold_val.shape[0]} échantillons ({X_fold_val.shape[0]/X_full.shape[0]*100:.1f}% du total)")',
        '                print(f"   💡 Note: Ces pourcentages sont pour la cross-validation, pas la division train/val/test configurée")',
        '                print(f"   🔧 Entraînement du modèle en cours...")',
        '                ',
        '                # Clone and train model for this fold',
        '                fold_model = clone(model)',
        '                train_start = time.time()',
        '                fold_model.fit(X_fold_train, y_fold_train)',
        '                train_time = time.time() - train_start',
        '                ',
        '                print(f"   ⏱️  Temps d\'entraînement: {train_time:.2f}s")',
        '                ',
        '                # Evaluate on fold',
        '                train_score = fold_model.score(X_fold_train, y_fold_train)',
        '                val_score = fold_model.score(X_fold_val, y_fold_val)',
        '                cv_scores.append(val_score)',
        '                ',
        '                fold_total_time = time.time() - fold_start_time',
        '                print(f"   ✅ Score d\'entraînement: {train_score:.4f}")',
        '                print(f"   📈 Score de validation: {val_score:.4f}")',
        '                print(f"   ⏰ Temps total fold: {fold_total_time:.2f}s")',
        '                print(f"   📊 Score moyen jusqu\'ici: {np.mean(cv_scores):.4f}")',
        '            ',
        '            # Train final model on full training data',
        '            print(f"\\n✅ Cross-validation complétée:")',
        '            print(f"   📊 Meilleur score: {max(cv_scores):.4f}")',
        '            print(f"   📈 Score moyen: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")',
        '            print(f"   📉 Score le plus bas: {min(cv_scores):.4f}")',
        '            print(f"\\nEntraînement du modèle final sur toutes les données d\'entraînement...")',
        '            model.fit(X_train, y_train)',
        '            return {"cv_scores": cv_scores, "cv_mean": np.mean(cv_scores), "cv_std": np.std(cv_scores)}',
        '            ',
        '        elif self.method == "time_series_split":',
        '            # Time Series Split',
        '            n_splits = self.config.get("n_splits", 5)',
        '            print(f"Executing time series split with {n_splits} splits...")',
        '            from sklearn.model_selection import TimeSeriesSplit, cross_val_score',
        '            tss = TimeSeriesSplit(n_splits=n_splits)',
        '            scores = cross_val_score(model, X_train, y_train, cv=tss, scoring="neg_mean_squared_error")',
        '            print(f"CV MSE: {-scores.mean():.4f} (+/- {scores.std() * 2:.4f})")',
        '            model.fit(X_train, y_train)',
        '            return {"cv_scores": scores.tolist(), "cv_mean": -scores.mean(), "cv_std": scores.std()}',
        '            ',
        '        else:',
        '            # Default to standard execution',
        '            print("Executing standard training (default)...")',
        '            model.fit(X_train, y_train)',
        '            return {}',
        '',
        ''
    ]
    
    return lines


def _generate_training_function() -> List[str]:
    """Generate model training function that uses Module 1 and Module 2"""
    return [
        'def train_model(model, X, y, data_splitter=None, execution_strategy=None):',
        '    """',
        '    Train the model using Module 1 (Data Split) and Module 2 (Execution)',
        '    """',
        '    # Use default modules if not provided',
        '    if data_splitter is None:',
        '        data_splitter = DataSplitter()',
        '    if execution_strategy is None:',
        '        execution_strategy = ExecutionStrategy()',
        '    ',
        '    # Module 1: Split the data',
        '    print("\\n" + "="*50)',
        '    print("MODULE 1: DATA SPLITTING")',
        '    print("="*50)',
        '    X_train, X_val, X_test, y_train, y_val, y_test = data_splitter.split(X, y)',
        '    ',
        '    print(f"Training set size: {X_train.shape}")',
        '    print(f"Validation set size: {X_val.shape}")',
        '    print(f"Test set size: {X_test.shape}")',
        '    ',
        '    # Module 2: Execute training strategy',
        '    print("\\n" + "="*50)',
        '    print("MODULE 2: EXECUTION STRATEGY")',
        '    print("="*50)',
        '    execution_results = execution_strategy.execute(model, X_train, y_train, X_val, y_val)',
        '    ',
        '    # Make predictions',
        '    print("\\nMaking predictions...")',
        '    y_pred_train = model.predict(X_train)',
        '    y_pred_test = model.predict(X_test)',
        '    ',
        '    return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test',
        '',
        ''
    ]


def _generate_evaluation_function(model_type, hyperparams) -> List[str]:
    """Generate model evaluation function with multi-output support"""
    problem_type = hyperparams.get('problem_type', 'regression')
    
    lines = [
        'def evaluate_model(model, y_true, y_pred, dataset_name="Dataset", target_names=None):',
        '    """',
        '    Evaluate model performance (supports multi-output)',
        '    """',
        '    print(f"\\n=== {dataset_name} Evaluation ===")',
        '    ',
        '    # Check if multi-output',
        '    if len(y_true.shape) > 1 and y_true.shape[1] > 1:',
        '        print(f"Multi-output evaluation: {y_true.shape[1]} targets")',
        '        return evaluate_multi_output(y_true, y_pred, target_names)',
        '    ',
        ''
    ]
    
    if model_type == 'random_forest' and problem_type == 'classification':
        lines.extend([
            '    # Classification metrics',
            '    accuracy = accuracy_score(y_true, y_pred)',
            '    print(f"Accuracy: {accuracy:.4f}")',
            '    print("\\nClassification Report:")',
            '    print(classification_report(y_true, y_pred))',
            '    print("\\nConfusion Matrix:")',
            '    print(confusion_matrix(y_true, y_pred))',
            '    ',
            '    return {"accuracy": accuracy}'
        ])
    else:
        lines.extend([
            '    # Regression metrics',
            '    mae = mean_absolute_error(y_true, y_pred)',
            '    mse = mean_squared_error(y_true, y_pred)',
            '    rmse = np.sqrt(mse)',
            '    r2 = r2_score(y_true, y_pred)',
            '    ',
            '    print(f"MAE: {mae:.4f}")',
            '    print(f"MSE: {mse:.4f}")',
            '    print(f"RMSE: {rmse:.4f}")',
            '    print(f"R²: {r2:.4f}")',
            '    ',
            '    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}'
        ])
    
    lines.extend([
        '',
        '',
        'def evaluate_multi_output(y_true, y_pred, target_names=None):',
        '    """Evaluate multi-output predictions"""',
        '    results = {}',
        '    n_outputs = y_true.shape[1]',
        '    ',
        '    for i in range(n_outputs):',
        '        target_name = target_names[i] if target_names and i < len(target_names) else f"Target_{i+1}"',
        '        print(f"\\n--- Evaluation for {target_name} ---")',
        '        ',
        '        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])',
        '        mse = mean_squared_error(y_true[:, i], y_pred[:, i])',
        '        rmse = np.sqrt(mse)',
        '        r2 = r2_score(y_true[:, i], y_pred[:, i])',
        '        ',
        '        print(f"MAE: {mae:.4f}")',
        '        print(f"MSE: {mse:.4f}")',
        '        print(f"RMSE: {rmse:.4f}")',
        '        print(f"R²: {r2:.4f}")',
        '        ',
        '        results[target_name] = {',
        '            "mae": mae,',
        '            "mse": mse,',
        '            "rmse": rmse,',
        '            "r2": r2',
        '        }',
        '    ',
        '    return results',
        ''
    ])
    
    lines.extend(['', ''])
    return lines


def _json_to_python_str(obj):
    """Convert JSON object to Python code string"""
    if isinstance(obj, bool):
        return 'True' if obj else 'False'
    elif obj is None:
        return 'None'
    elif isinstance(obj, str):
        return repr(obj)  # This will add quotes properly
    elif isinstance(obj, (int, float)):
        return str(obj)
    elif isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            items.append(f'"{k}": {_json_to_python_str(v)}')
        return '{' + ', '.join(items) + '}'
    elif isinstance(obj, list):
        items = [_json_to_python_str(item) for item in obj]
        return '[' + ', '.join(items) + ']'
    else:
        return repr(obj)


def _generate_main_function(model_def, hyperparams) -> List[str]:
    """Generate main execution function"""
    # Get Module 1 and Module 2 configurations
    split_method = model_def.default_split_method if hasattr(model_def, 'default_split_method') else 'random'
    split_config = model_def.default_split_config if hasattr(model_def, 'default_split_config') else {}
    execution_method = model_def.default_execution_method if hasattr(model_def, 'default_execution_method') else 'standard'
    execution_config = model_def.default_execution_config if hasattr(model_def, 'default_execution_config') else {}
    
    # Convert configs to Python format
    split_config_str = _json_to_python_str(split_config)
    execution_config_str = _json_to_python_str(execution_config)
    
    return [
        '# =============================================================================',
        '# MAIN EXECUTION',
        '# =============================================================================',
        '',
        'if __name__ == "__main__":',
        '    # Configuration',
        '    DATA_FILE = "your_dataset.csv"  # UPDATE THIS PATH',
        '    ',
        '    # Module 1 Configuration',
        f'    split_config = {split_config_str}',
        f'    data_splitter = DataSplitter(strategy="{split_method}", config=split_config)',
        '    ',
        '    # Module 2 Configuration',
        f'    execution_config = {execution_config_str}',
        f'    execution_strategy = ExecutionStrategy(method="{execution_method}", config=execution_config)',
        '    ',
        '    try:',
        '        # Step 1: Load and preprocess data',
        '        print("STEP 1: Loading and preprocessing data...")',
        '        X, y, predictor_cols, target_cols = load_and_preprocess_data(DATA_FILE)',
        '        ',
        '        # Step 2: Create model',
        '        print("\\nSTEP 2: Creating model...")',
        '        model = create_model()',
        '        ',
        '        # Step 3: Train model using Module 1 and Module 2',
        '        print("\\nSTEP 3: Training model with configured modules...")',
        '        X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_model(',
        '            model, X, y, data_splitter=data_splitter, execution_strategy=execution_strategy',
        '        )',
        '        ',
        '        # Step 4: Evaluate model',
        '        print("\\nSTEP 4: Evaluating model...")',
        '        train_metrics = evaluate_model(model, y_train, y_pred_train, "Training Set", target_cols)',
        '        test_metrics = evaluate_model(model, y_test, y_pred_test, "Test Set", target_cols)',
        '        ',
        '        # Step 5: Analyze model performance with visualizations',
        '        print("\\nSTEP 5: Creating analysis visualizations...")',
        '        analysis_results = analyze_model_performance(',
        '            model, X_test, y_test, target_cols,',
        f'            model_name="{model_def.name.replace(" ", "_")}",',
        '            save_plots=True',
        '        )',
        '        ',
        '        # Step 6: Save model',
        '        print("\\nSTEP 6: Saving model...")',
        f'        model_filename = "{model_def.name.replace(" ", "_").lower()}_model.pkl"',
        '        save_model(model, model_filename)',
        '        ',
        '        # Optional: Save model info',
        '        model_info = {',
        f'            "model_name": "{model_def.name}",',
        f'            "model_type": "{model_def.model_type}",',
        '            "predictor_columns": predictor_cols,',
        '            "target_columns": target_cols,',
        '            "hyperparameters": model.get_params(),',
        f'            "generated_at": "{model_def.updated_at}"',
        '        }',
        '        ',
        '        with open("model_info.json", "w") as f:',
        '            json.dump(model_info, f, indent=2)',
        '        print("Model info saved to: model_info.json")',
        '        ',
        '        print("\\nTraining completed successfully!")',
        '        ',
        '    except FileNotFoundError:',
        '        print(f"❌ Error: Could not find data file: {DATA_FILE}")',
        '        print("Please update the DATA_FILE variable with your dataset path.")',
        '    except Exception as e:',
        '        print(f"❌ Error: {str(e)}")',
        '        import traceback',
        '        traceback.print_exc()'
    ]


def generate_sklearn_code(model_def) -> str:
    """Generate scikit-learn code from model definition"""
    
    hyperparams = model_def.hyperparameters or {}
    model_type = model_def.model_type
    
    code_lines = []
    
    # Generate header and imports
    code_lines.extend(_generate_sklearn_header(model_def))
    
    # Add Module 1: Data Split Configuration
    code_lines.extend(_generate_data_split_module(model_def))
    
    # Add Module 2: Execution Configuration
    code_lines.extend(_generate_execution_module(model_def))
    
    # Create model function
    code_lines.extend([
        'def create_model():',
        '    """',
        '    Create and return the configured model',
        '    """',
        f'    print("Creating {model_type.upper()} model...")',
        ''
    ])
    
    # Generate model creation based on type
    code_lines.extend(_generate_model_creation_code(model_type, hyperparams))
    
    code_lines.extend([
        '',
        '    print("Model created successfully!")',
        '    print(f"Model type: {type(model).__name__}")',
        '    print(f"Parameters: {model.get_params()}")',
        '    return model',
        '',
        ''
    ])
    
    # Add data loading function
    code_lines.extend(_generate_data_loading_function(model_def))
    
    # Add training function
    code_lines.extend(_generate_training_function())
    
    # Add evaluation function
    code_lines.extend(_generate_evaluation_function(model_type, hyperparams))
    
    # Save model function
    code_lines.extend([
        'def save_model(model, filename="model.pkl"):',
        '    """',
        '    Save the trained model',
        '    """',
        '    joblib.dump(model, filename)',
        '    print(f"\\nModel saved to: {filename}")',
        '',
        '',
        'def load_model(filename="model.pkl"):',
        '    """',
        '    Load a saved model',
        '    """',
        '    model = joblib.load(filename)',
        '    print(f"Model loaded from: {filename}")',
        '    return model',
        '',
        ''
    ])
    
    # Add analysis visualization functions (same as Keras version)
    code_lines.extend([
        "def create_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', save_path=None):",
        '    """',
        '    Create and display confusion matrix',
        '    """',
        "    from sklearn.metrics import confusion_matrix",
        "    import matplotlib.pyplot as plt",
        "    import seaborn as sns",
        "    import numpy as np",
        "    ",
        "    # Calculate confusion matrix",
        "    cm = confusion_matrix(y_true, y_pred)",
        "    ",
        "    # Create figure",
        "    plt.figure(figsize=(10, 8))",
        "    ",
        "    # Use seaborn heatmap for better visualization",
        "    if labels is None:",
        "        labels = [f'Class_{i}' for i in range(len(cm))]",
        "    ",
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',",
        "                xticklabels=labels, yticklabels=labels,",
        "                cbar_kws={'label': 'Count'})",
        "    ",
        "    plt.title(title, fontsize=16, fontweight='bold')",
        "    plt.xlabel('Predicted Label', fontsize=12)",
        "    plt.ylabel('True Label', fontsize=12)",
        "    ",
        "    # Rotate labels if they are long",
        "    if any(len(str(label)) > 10 for label in labels):",
        "        plt.xticks(rotation=45, ha='right')",
        "        plt.yticks(rotation=0)",
        "    ",
        "    plt.tight_layout()",
        "    ",
        "    if save_path:",
        "        plt.savefig(save_path, dpi=300, bbox_inches='tight')",
        "        print(f'Confusion matrix saved to: {save_path}')",
        "    ",
        "    plt.show()",
        "    return cm",
        "",
        "",
        "def create_scatter_plot(y_true, y_pred, title='Predictions vs Actual', save_path=None):",
        '    """',
        '    Create scatter plot of predictions vs actual values',
        '    """',
        "    import matplotlib.pyplot as plt",
        "    import numpy as np",
        "    from sklearn.metrics import r2_score",
        "    ",
        "    plt.figure(figsize=(10, 8))",
        "    ",
        "    # Create scatter plot",
        "    plt.scatter(y_true, y_pred, alpha=0.6, s=50)",
        "    ",
        "    # Add perfect prediction line",
        "    min_val = min(min(y_true), min(y_pred))",
        "    max_val = max(max(y_true), max(y_pred))",
        "    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')",
        "    ",
        "    # Calculate and display R² score",
        "    r2 = r2_score(y_true, y_pred)",
        "    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,",
        "             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),",
        "             fontsize=12, verticalalignment='top')",
        "    ",
        "    plt.xlabel('Actual Values', fontsize=12)",
        "    plt.ylabel('Predicted Values', fontsize=12)",
        "    plt.title(title, fontsize=16, fontweight='bold')",
        "    plt.legend()",
        "    plt.grid(True, alpha=0.3)",
        "    plt.tight_layout()",
        "    ",
        "    if save_path:",
        "        plt.savefig(save_path, dpi=300, bbox_inches='tight')",
        "        print(f'Scatter plot saved to: {save_path}')",
        "    ",
        "    plt.show()",
        "    return r2",
        "",
        "",
        "def create_residuals_plot(y_true, y_pred, title='Residuals Analysis', save_path=None):",
        '    """',
        '    Create residuals plot for regression analysis',
        '    """',
        "    import matplotlib.pyplot as plt",
        "    import numpy as np",
        "    ",
        "    residuals = y_true - y_pred",
        "    ",
        "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))",
        "    ",
        "    # Residuals vs Predicted",
        "    ax1.scatter(y_pred, residuals, alpha=0.6)",
        "    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)",
        "    ax1.set_xlabel('Predicted Values')",
        "    ax1.set_ylabel('Residuals')",
        "    ax1.set_title('Residuals vs Predicted')",
        "    ax1.grid(True, alpha=0.3)",
        "    ",
        "    # Histogram of residuals",
        "    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')",
        "    ax2.set_xlabel('Residuals')",
        "    ax2.set_ylabel('Frequency')",
        "    ax2.set_title('Distribution of Residuals')",
        "    ax2.grid(True, alpha=0.3)",
        "    ",
        "    # Add statistics",
        "    mean_residuals = np.mean(residuals)",
        "    std_residuals = np.std(residuals)",
        "    ax2.axvline(mean_residuals, color='red', linestyle='--', ",
        "               label=f'Mean: {mean_residuals:.4f}')",
        "    ax2.legend()",
        "    ",
        "    plt.suptitle(title, fontsize=16, fontweight='bold')",
        "    plt.tight_layout()",
        "    ",
        "    if save_path:",
        "        plt.savefig(save_path, dpi=300, bbox_inches='tight')",
        "        print(f'Residuals plot saved to: {save_path}')",
        "    ",
        "    plt.show()",
        "    ",
        "    return {'mean': mean_residuals, 'std': std_residuals}",
        "",
        "",
        "def analyze_model_performance(model, X_test, y_test, target_columns, ",
        "                             model_name='Model', save_plots=True):",
        '    """',
        '    Comprehensive model performance analysis with visualizations',
        '    """',
        "    print('Analyzing model performance...')",
        "    print('=' * 50)",
        "    ",
        "    # Make predictions",
        "    y_pred = model.predict(X_test)",
        "    ",
        "    # Ensure predictions are properly shaped",
        "    # Handle both numpy arrays and pandas Series/DataFrames",
        "    if hasattr(y_pred, 'values'):",
        "        y_pred = y_pred.values",
        "    if hasattr(y_test, 'values'):",
        "        y_test = y_test.values",
        "    ",
        "    if len(y_pred.shape) == 1:",
        "        y_pred = y_pred.reshape(-1, 1)",
        "    if len(y_test.shape) == 1:",
        "        y_test = y_test.reshape(-1, 1)",
        "    ",
        "    analysis_results = {}",
        "    ",
        "    # Analyze each target column",
        "    for i, target_col in enumerate(target_columns):",
        "        print(f'\\nAnalyzing target: {target_col}')",
        "        print('-' * 30)",
        "        ",
        "        y_true_col = y_test[:, i] if y_test.shape[1] > 1 else y_test.flatten()",
        "        y_pred_col = y_pred[:, i] if y_pred.shape[1] > 1 else y_pred.flatten()",
        "        ",
        "        # Determine if classification or regression",
        "        unique_values = len(np.unique(y_true_col))",
        "        is_classification = unique_values <= 20  # Threshold for classification",
        "        ",
        "        if is_classification:",
        "            print(f'Classification Analysis for {target_col}')",
        "            ",
        "            # Classification metrics",
        "            from sklearn.metrics import accuracy_score, classification_report",
        "            ",
        "            # Round predictions for classification",
        "            y_pred_rounded = np.round(y_pred_col).astype(int)",
        "            y_true_int = y_true_col.astype(int)",
        "            ",
        "            accuracy = accuracy_score(y_true_int, y_pred_rounded)",
        "            print(f'Accuracy: {accuracy:.4f}')",
        "            ",
        "            # Create confusion matrix",
        "            save_path = f'{model_name}_{target_col}_confusion_matrix.png' if save_plots else None",
        "            cm = create_confusion_matrix(",
        "                y_true_int, y_pred_rounded,",
        "                title=f'Confusion Matrix - {target_col}',",
        "                save_path=save_path",
        "            )",
        "            ",
        "            analysis_results[target_col] = {",
        "                'type': 'classification',",
        "                'accuracy': accuracy,",
        "                'confusion_matrix': cm.tolist()",
        "            }",
        "        ",
        "        else:",
        "            print(f'Regression Analysis for {target_col}')",
        "            ",
        "            # Regression metrics",
        "            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
        "            ",
        "            mae = mean_absolute_error(y_true_col, y_pred_col)",
        "            mse = mean_squared_error(y_true_col, y_pred_col)",
        "            rmse = np.sqrt(mse)",
        "            r2 = r2_score(y_true_col, y_pred_col)",
        "            ",
        "            print(f'MAE:  {mae:.4f}')",
        "            print(f'MSE:  {mse:.4f}')",
        "            print(f'RMSE: {rmse:.4f}')",
        "            print(f'R²:   {r2:.4f}')",
        "            ",
        "            # Create scatter plot",
        "            save_path = f'{model_name}_{target_col}_scatter.png' if save_plots else None",
        "            r2_calc = create_scatter_plot(",
        "                y_true_col, y_pred_col,",
        "                title=f'Predictions vs Actual - {target_col}',",
        "                save_path=save_path",
        "            )",
        "            ",
        "            # Create residuals plot",
        "            save_path = f'{model_name}_{target_col}_residuals.png' if save_plots else None",
        "            residuals_stats = create_residuals_plot(",
        "                y_true_col, y_pred_col,",
        "                title=f'Residuals Analysis - {target_col}',",
        "                save_path=save_path",
        "            )",
        "            ",
        "            analysis_results[target_col] = {",
        "                'type': 'regression',",
        "                'mae': mae,",
        "                'mse': mse,",
        "                'rmse': rmse,",
        "                'r2': r2,",
        "                'residuals_stats': residuals_stats",
        "            }",
        "    ",
        "    print('\\nAnalysis completed!')",
        "    return analysis_results",
        "",
        "",
    ])
    
    # Add main execution function
    code_lines.extend(_generate_main_function(model_def, hyperparams))
    
    # Add usage instructions
    code_lines.extend([
        '',
        '# =============================================================================',
        '# USAGE INSTRUCTIONS',
        '# =============================================================================',
        '# 1. Update DATA_FILE with your CSV file path',
        '# 2. Ensure your dataset contains these columns:',
        f'#    - Predictors: {model_def.predictor_columns}',
        f'#    - Targets: {model_def.target_columns}',
        '# 3. Run: python this_script.py',
        '# =============================================================================',
    ])
    
    return '\n'.join(code_lines)