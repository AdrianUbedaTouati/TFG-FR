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
        '    print(f"Creating model with input_shape={{input_shape}} and output_shape={{output_shape}}")',
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
    
    # Add main execution section
    code_lines.extend([
        "# =============================================================================",
        "# MAIN EXECUTION - CONFIGURE YOUR DATASET PATH HERE",
        "# =============================================================================",
        "",
        "# DATASET CONFIGURATION",
        "DATASET_PATH = 'path/to/your/dataset.csv'  # ⚠️ CHANGE THIS TO YOUR DATASET PATH",
        "",
        "if __name__ == '__main__':",
        f"    print('🚀 Starting {model_def.name} Model Training')",
        "    print('=' * 50)",
        "",
        "    try:",
        "        # Step 1: Load and preprocess data",
        "        print('📊 Step 1: Loading and preprocessing data...')",
        "        data = load_and_preprocess_data(DATASET_PATH, test_size=0.2)",
        "",
        "        # Step 2: Create model",
        "        print('🧠 Step 2: Creating model...')",
        "        model = create_model(data['input_shape'], data['output_shape'])",
        "        model.summary()",
        "",
        "        # Step 3: Train model",
        "        print('🏋️ Step 3: Training model...')",
        "        history = train_model(",
        "            model, ",
        "            data['X_train'], data['y_train'], ",
        "            data['X_test'], data['y_test']",
        "        )",
        "",
        "        # Step 4: Evaluate model",
        "        print('📈 Step 4: Evaluating model...')",
        "        predictions = evaluate_model(",
        "            model, ",
        "            data['X_test'], data['y_test'], ",
        "            data['target_encoders']",
        "        )",
        "",
        "        # Step 5: Save model",
        "        print('💾 Step 5: Saving model...')",
        f"        model.save('{model_def.name.replace(' ', '_')}_model.h5')",
        f"        print(f'Model saved as: {model_def.name.replace(' ', '_')}_model.h5')",
        "",
        "        print('✅ Training completed successfully!')",
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
        f"            print('📊 Training plots saved as: {model_def.name.replace(' ', '_')}_training_history.png')",
        "        except ImportError:",
        "            print('⚠️ Matplotlib not available. Skipping plots.')",
        "",
        "    except FileNotFoundError:",
        "        print(f'❌ Error: Dataset file not found at: {DATASET_PATH}')",
        "        print('Please update the DATASET_PATH variable with the correct path to your dataset.')",
        "    except Exception as e:",
        "        print(f'❌ Error during training: {str(e)}')",
        "        import traceback",
        "        traceback.print_exc()",
        "",
        "    # Instructions for usage",
        r"    print('\n' + '=' * 60)",
        "    print('📋 USAGE INSTRUCTIONS:')",
        "    print('=' * 60)",
        "    print('1. Update DATASET_PATH with your CSV file path')",
        "    print('2. Ensure your dataset has these columns:')",
        f"    print(f'   - Predictors: {predictor_columns}')",
        f"    print(f'   - Targets: {target_columns}')",
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
        '"""',
        '',
        'import numpy as np',
        'import pandas as pd',
        'import joblib',
        'import json',
        'from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV',
        'from sklearn.preprocessing import StandardScaler, LabelEncoder',
        'from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score',
        'from sklearn.metrics import accuracy_score, classification_report, confusion_matrix',
        ''
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


def _generate_random_forest_params(hyperparams) -> List[str]:
    """Generate Random Forest parameter configuration"""
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
    
    # Handle max_features
    max_features = hyperparams.get('max_features', 'sqrt')
    if isinstance(max_features, str) and max_features != 'custom':
        if max_features == '1.0':
            params_lines.append('        "max_features": 1.0,')
        else:
            params_lines.append(f'        "max_features": "{max_features}",')
    elif hyperparams.get('max_features_fraction'):
        params_lines.append(f'        "max_features": {hyperparams["max_features_fraction"]},')
    
    # Handle criterion
    criterion = hyperparams.get('criterion', 'auto')
    if criterion == 'auto':
        if problem_type == 'classification':
            params_lines.append('        "criterion": "gini",')
        else:
            params_lines.append('        "criterion": "squared_error",')
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
    params_lines.append(f'        "bootstrap": {hyperparams.get("bootstrap", True)},')
    if hyperparams.get('oob_score'):
        params_lines.append(f'        "oob_score": {hyperparams["oob_score"]},')
    
    # Performance parameters
    params_lines.append(f'        "n_jobs": {hyperparams.get("n_jobs", -1)},')
    if hyperparams.get('random_state') is not None:
        params_lines.append(f'        "random_state": {hyperparams["random_state"]},')
    
    # Class weight for classification
    if problem_type == 'classification' and hyperparams.get('class_weight_balanced'):
        params_lines.append('        "class_weight": "balanced",')
    
    # Remove trailing comma from last line
    if params_lines[-1].endswith(','):
        params_lines[-1] = params_lines[-1][:-1]
    
    params_lines.append('    }')
    params_lines.append('')
    
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
        return [
            '    # Model parameters',
            '    params = {',
            f'        "max_depth": {hyperparams.get("max_depth", 10)},',
            f'        "min_samples_split": {hyperparams.get("min_samples_split", 2)},',
            f'        "min_samples_leaf": {hyperparams.get("min_samples_leaf", 1)},',
            f'        "max_features": "{0}"'.format(hyperparams.get("max_features", "auto")),
            '    }',
            '',
            '    model = DecisionTreeRegressor(**params)'
        ]
        
    elif model_type == 'xgboost':
        return [
            '    # Model parameters',
            '    params = {',
            f'        "n_estimators": {hyperparams.get("n_estimators", 100)},',
            f'        "max_depth": {hyperparams.get("max_depth", 6)},',
            f'        "learning_rate": {hyperparams.get("learning_rate", 0.3)},',
            f'        "subsample": {hyperparams.get("subsample", 0.8)},',
            f'        "colsample_bytree": {hyperparams.get("colsample_bytree", 0.8)}',
            '    }',
            '',
            '    model = xgb.XGBRegressor(**params)'
        ]
    
    return []


def _generate_data_loading_function(model_def) -> List[str]:
    """Generate data loading and preprocessing function"""
    predictor_columns = [str(col) for col in model_def.predictor_columns]
    target_columns = [str(col) for col in model_def.target_columns]
    
    return [
        'def load_and_preprocess_data(file_path):',
        '    """',
        '    Load and preprocess the dataset',
        '    """',
        '    print(f"Loading data from: {{file_path}}")',
        '    df = pd.read_csv(file_path)',
        '    print(f"Dataset shape: {{df.shape}}")',
        '    print(f"Columns: {{list(df.columns)}}")',
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
        '    X = df[predictor_columns]',
        '    y = df[target_columns[0]] if len(target_columns) == 1 else df[target_columns]',
        '',
        '    # Handle missing values',
        '    print("\\nHandling missing values...")',
        '    print(f"Missing values in features: {{X.isnull().sum().sum()}}")',
        '    print(f"Missing values in target: {{y.isnull().sum() if hasattr(y, \'isnull\') else 0}}")',
        '',
        '    # Fill missing values',
        '    X = X.fillna(X.mean())',
        '    if hasattr(y, \'fillna\'):',
        '        y = y.fillna(y.mean())',
        '',
        '    return X, y, predictor_columns, target_columns',
        '',
        ''
    ]


def _generate_training_function() -> List[str]:
    """Generate model training function"""
    return [
        'def train_model(model, X, y, test_size=0.2, cv_folds=None):',
        '    """',
        '    Train the model with optional cross-validation',
        '    """',
        '    # Split the data',
        '    X_train, X_test, y_train, y_test = train_test_split(',
        '        X, y, test_size=test_size, random_state=42',
        '    )',
        '    ',
        '    print(f"\\nTraining set size: {{X_train.shape}}")',
        '    print(f"Test set size: {{X_test.shape}}")',
        '',
        '    # Perform cross-validation if requested',
        '    if cv_folds:',
        '        print(f"\\nPerforming {{cv_folds}}-fold cross-validation...")',
        '        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=\'neg_mean_squared_error\')',
        '        print(f"CV MSE: {{-cv_scores.mean():.4f}} (+/- {{cv_scores.std() * 2:.4f}})")',
        '',
        '    # Train the model',
        '    print("\\nTraining model...")',
        '    model.fit(X_train, y_train)',
        '    ',
        '    # Make predictions',
        '    y_pred_train = model.predict(X_train)',
        '    y_pred_test = model.predict(X_test)',
        '    ',
        '    return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test',
        '',
        ''
    ]


def _generate_evaluation_function(model_type, hyperparams) -> List[str]:
    """Generate model evaluation function"""
    problem_type = hyperparams.get('problem_type', 'regression')
    
    lines = [
        'def evaluate_model(model, y_true, y_pred, dataset_name="Dataset"):',
        '    """',
        '    Evaluate model performance',
        '    """',
        '    print(f"\\n=== {{dataset_name}} Evaluation ===")',
        ''
    ]
    
    if model_type == 'random_forest' and problem_type == 'classification':
        lines.extend([
            '    # Classification metrics',
            '    accuracy = accuracy_score(y_true, y_pred)',
            '    print(f"Accuracy: {{accuracy:.4f}}")',
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
            '    print(f"MAE: {{mae:.4f}}")',
            '    print(f"MSE: {{mse:.4f}}")',
            '    print(f"RMSE: {{rmse:.4f}}")',
            '    print(f"R²: {{r2:.4f}}")',
            '    ',
            '    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}'
        ])
    
    lines.extend(['', ''])
    return lines


def _generate_main_function() -> List[str]:
    """Generate main execution function"""
    return [
        'def main():',
        '    """',
        '    Main execution function',
        '    """',
        '    # Configuration',
        '    data_file = "your_dataset.csv"  # Update with your file path',
        '    model_file = "trained_model.pkl"',
        '    test_size = 0.2',
        '    cv_folds = 5  # Set to None to skip cross-validation',
        '    ',
        '    try:',
        '        # Create model',
        '        model = create_model()',
        '        ',
        '        # Load and preprocess data',
        '        X, y, predictor_cols, target_cols = load_and_preprocess_data(data_file)',
        '        ',
        '        # Train model',
        '        X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_model(',
        '            model, X, y, test_size, cv_folds',
        '        )',
        '        ',
        '        # Evaluate model',
        '        print("\\n" + "="*50)',
        '        train_metrics = evaluate_model(model, y_train, y_pred_train, "Training Set")',
        '        test_metrics = evaluate_model(model, y_test, y_pred_test, "Test Set")',
        '        ',
        '        # Feature importance (if available)',
        '        if hasattr(model, \'feature_importances_\'):',
        '            print("\\n=== Feature Importance ===")',
        '            feature_importance = pd.DataFrame({',
        '                \'feature\': predictor_cols,',
        '                \'importance\': model.feature_importances_',
        '            }).sort_values(\'importance\', ascending=False)',
        '            print(feature_importance.head(10))',
        '        ',
        '        # Save model',
        '        print(f"\\nSaving model to: {{model_file}}")',
        '        joblib.dump({',
        '            \'model\': model,',
        '            \'predictor_columns\': predictor_cols,',
        '            \'target_columns\': target_cols,',
        '            \'metrics\': dict(train=train_metrics, test=test_metrics)',
        '        }, model_file)',
        '        print("Model saved successfully!")',
        '        ',
        '    except Exception as e:',
        '        print(f"\\nError: {{str(e)}}")',
        '        import traceback',
        '        traceback.print_exc()',
        '',
        '',
        'if __name__ == "__main__":',
        '    main()'
    ]


def generate_sklearn_code(model_def) -> str:
    """Generate scikit-learn code from model definition"""
    
    hyperparams = model_def.hyperparameters or {}
    model_type = model_def.model_type
    
    code_lines = []
    
    # Header section
    code_lines.extend([
        '"""',
        f'Auto-generated scikit-learn {model_type.upper()} model code',
        f'Model: {model_def.name}',
        f'Type: {model_def.model_type.upper()}',
        f'Generated at: {model_def.updated_at}',
        '',
        'Configuration:',
        f'- Target columns: {model_def.target_columns}',
        f'- Predictor columns: {len(model_def.predictor_columns)} features',
        '"""',
        '',
        'import numpy as np',
        'import pandas as pd',
        'import joblib',
        'import json',
        'from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV',
        'from sklearn.preprocessing import StandardScaler, LabelEncoder',
        'from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score',
        'from sklearn.metrics import accuracy_score, classification_report, confusion_matrix',
        ''
    ])
    
    # Import model-specific libraries
    if model_type == 'random_forest':
        problem_type = hyperparams.get('problem_type', 'regression')
        if problem_type == 'classification':
            code_lines.append('from sklearn.ensemble import RandomForestClassifier')
        else:
            code_lines.append('from sklearn.ensemble import RandomForestRegressor')
    elif model_type == 'decision_tree':
        code_lines.append('from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier')
    elif model_type == 'xgboost':
        code_lines.append('import xgboost as xgb')
    
    code_lines.extend(['', '', ''])
    
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
    if model_type == 'random_forest':
        problem_type = hyperparams.get('problem_type', 'regression')
        
        # Build parameters dict
        params_lines = ['    # Model parameters']
        params_lines.append('    params = {')
        
        # Basic parameters
        if hyperparams.get('n_estimators'):
            params_lines.append(f'        "n_estimators": {hyperparams["n_estimators"]},')
        
        # Handle max_depth
        if hyperparams.get('max_depth_enabled', False) and hyperparams.get('max_depth'):
            params_lines.append(f'        "max_depth": {hyperparams["max_depth"]},')
        else:
            params_lines.append('        "max_depth": None,')
        
        # Handle max_features
        max_features = hyperparams.get('max_features', 'sqrt')
        if isinstance(max_features, str) and max_features != 'custom':
            if max_features == '1.0':
                params_lines.append('        "max_features": 1.0,')
            else:
                params_lines.append(f'        "max_features": "{max_features}",')
        elif hyperparams.get('max_features_fraction'):
            params_lines.append(f'        "max_features": {hyperparams["max_features_fraction"]},')
        
        # Handle criterion
        criterion = hyperparams.get('criterion', 'auto')
        if criterion == 'auto':
            if problem_type == 'classification':
                params_lines.append('        "criterion": "gini",')
            else:
                params_lines.append('        "criterion": "squared_error",')
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
        params_lines.append(f'        "bootstrap": {hyperparams.get("bootstrap", True)},')
        if hyperparams.get('oob_score'):
            params_lines.append(f'        "oob_score": {hyperparams["oob_score"]},')
        
        # Performance parameters
        params_lines.append(f'        "n_jobs": {hyperparams.get("n_jobs", -1)},')
        if hyperparams.get('random_state') is not None:
            params_lines.append(f'        "random_state": {hyperparams["random_state"]},')
        
        # Class weight for classification
        if problem_type == 'classification' and hyperparams.get('class_weight_balanced'):
            params_lines.append('        "class_weight": "balanced",')
        
        # Remove trailing comma from last line
        if params_lines[-1].endswith(','):
            params_lines[-1] = params_lines[-1][:-1]
        
        params_lines.append('    }')
        params_lines.append('')
        
        code_lines.extend(params_lines)
        
        # Create model instance
        if problem_type == 'classification':
            code_lines.append('    model = RandomForestClassifier(**params)')
        else:
            code_lines.append('    model = RandomForestRegressor(**params)')
        
    elif model_type == 'decision_tree':
        code_lines.extend([
            '    # Model parameters',
            '    params = {',
            f'        "max_depth": {hyperparams.get("max_depth", 10)},',
            f'        "min_samples_split": {hyperparams.get("min_samples_split", 2)},',
            f'        "min_samples_leaf": {hyperparams.get("min_samples_leaf", 1)},',
            f'        "max_features": "{0}"'.format(hyperparams.get("max_features", "auto")),
            '    }',
            '',
            '    model = DecisionTreeRegressor(**params)'
        ])
        
    elif model_type == 'xgboost':
        code_lines.extend([
            '    # Model parameters',
            '    params = {',
            f'        "n_estimators": {hyperparams.get("n_estimators", 100)},',
            f'        "max_depth": {hyperparams.get("max_depth", 6)},',
            f'        "learning_rate": {hyperparams.get("learning_rate", 0.3)},',
            f'        "subsample": {hyperparams.get("subsample", 0.8)},',
            f'        "colsample_bytree": {hyperparams.get("colsample_bytree", 0.8)}',
            '    }',
            '',
            '    model = xgb.XGBRegressor(**params)'
        ])
    
    code_lines.extend([
        '',
        '    print("Model created successfully!")',
        '    print(f"Model type: {type(model).__name__}")',
        '    print(f"Parameters: {model.get_params()}")',
        '    return model',
        '',
        ''
    ])
    
    # Data loading function
    predictor_columns = [str(col) for col in model_def.predictor_columns]
    target_columns = [str(col) for col in model_def.target_columns]
    
    code_lines.extend([
        'def load_and_preprocess_data(file_path):',
        '    """',
        '    Load and preprocess the dataset',
        '    """',
        '    print(f"Loading data from: {file_path}")',
        '    df = pd.read_csv(file_path)',
        '    print(f"Dataset shape: {df.shape}")',
        '    print(f"Columns: {list(df.columns)}")',
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
        '    X = df[predictor_columns]',
        '    y = df[target_columns[0]] if len(target_columns) == 1 else df[target_columns]',
        '',
        '    # Handle missing values',
        '    print("\\nHandling missing values...")',
        '    print(f"Missing values in features: {X.isnull().sum().sum()}")',
        '    print(f"Missing values in target: {y.isnull().sum() if hasattr(y, \'isnull\') else 0}")',
        '',
        '    # Fill missing values',
        '    X = X.fillna(X.mean())',
        '    if hasattr(y, \'fillna\'):',
        '        y = y.fillna(y.mean())',
        '',
        '    return X, y, predictor_columns, target_columns',
        '',
        ''
    ])
    
    # Training function
    code_lines.extend([
        'def train_model(model, X, y, test_size=0.2, cv_folds=None):',
        '    """',
        '    Train the model with optional cross-validation',
        '    """',
        '    # Split the data',
        '    X_train, X_test, y_train, y_test = train_test_split(',
        '        X, y, test_size=test_size, random_state=42',
        '    )',
        '    ',
        '    print(f"\\nTraining set size: {X_train.shape}")',
        '    print(f"Test set size: {X_test.shape}")',
        '',
        '    # Perform cross-validation if requested',
        '    if cv_folds:',
        '        print(f"\\nPerforming {cv_folds}-fold cross-validation...")',
        '        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=\'neg_mean_squared_error\')',
        '        print(f"CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")',
        '',
        '    # Train the model',
        '    print("\\nTraining model...")',
        '    model.fit(X_train, y_train)',
        '    ',
        '    # Make predictions',
        '    y_pred_train = model.predict(X_train)',
        '    y_pred_test = model.predict(X_test)',
        '    ',
        '    return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test',
        '',
        ''
    ])
    
    # Evaluation function
    code_lines.extend([
        'def evaluate_model(model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test):',
        '    """',
        '    Evaluate model performance',
        '    """',
        '    print("\\n" + "="*50)',
        '    print("MODEL EVALUATION RESULTS")',
        '    print("="*50)',
        ''
    ])
    
    # Add appropriate metrics based on problem type
    if model_type == 'random_forest' and hyperparams.get('problem_type') == 'classification':
        code_lines.extend([
            '    # Classification metrics',
            '    train_accuracy = accuracy_score(y_train, y_pred_train)',
            '    test_accuracy = accuracy_score(y_test, y_pred_test)',
            '    ',
            '    print(f"\\nAccuracy:")',
            '    print(f"  Training: {train_accuracy:.4f}")',
            '    print(f"  Test:     {test_accuracy:.4f}")',
            '    ',
            '    print("\\nClassification Report (Test Set):")',
            '    print(classification_report(y_test, y_pred_test))',
            '    ',
            '    print("\\nConfusion Matrix (Test Set):")',
            '    print(confusion_matrix(y_test, y_pred_test))'
        ])
    else:
        # Regression metrics
        code_lines.extend([
            '    # Regression metrics',
            '    metrics = {',
            '        "MAE": {',
            '            "train": mean_absolute_error(y_train, y_pred_train),',
            '            "test": mean_absolute_error(y_test, y_pred_test)',
            '        },',
            '        "MSE": {',
            '            "train": mean_squared_error(y_train, y_pred_train),',
            '            "test": mean_squared_error(y_test, y_pred_test)',
            '        },',
            '        "RMSE": {',
            '            "train": np.sqrt(mean_squared_error(y_train, y_pred_train)),',
            '            "test": np.sqrt(mean_squared_error(y_test, y_pred_test))',
            '        },',
            '        "R²": {',
            '            "train": r2_score(y_train, y_pred_train),',
            '            "test": r2_score(y_test, y_pred_test)',
            '        }',
            '    }',
            '    ',
            '    # Print metrics',
            '    for metric_name, values in metrics.items():',
            '        print(f"\\n{metric_name}:")',
            '        print(f"  Training: {values[\'train\']:.4f}")',
            '        print(f"  Test:     {values[\'test\']:.4f}")'
        ])
    
    # Feature importance for tree-based models
    if model_type in ['random_forest', 'decision_tree', 'xgboost']:
        code_lines.extend([
            '',
            '    # Feature importance',
            '    if hasattr(model, \'feature_importances_\'):',
            '        print("\\n" + "="*50)',
            '        print("FEATURE IMPORTANCE")',
            '        print("="*50)',
            '        feature_importance = pd.DataFrame({',
            '            \'feature\': X_train.columns,',
            '            \'importance\': model.feature_importances_',
            '        }).sort_values(\'importance\', ascending=False)',
            '        ',
            '        for idx, row in feature_importance.iterrows():',
            '            print(f"{row[\'feature\']:30s}: {row[\'importance\']:.4f}")'
        ])
    
    code_lines.extend(['', ''])
    
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
    
    # Main execution
    code_lines.extend([
        '# =============================================================================',
        '# MAIN EXECUTION',
        '# =============================================================================',
        '',
        'if __name__ == "__main__":',
        '    # Configuration',
        '    DATA_FILE = "your_dataset.csv"  # ⚠️ UPDATE THIS PATH',
        '    TEST_SIZE = 0.2',
        f'    USE_CROSS_VALIDATION = {"5" if hyperparams.get("validation_method") == "cv" else "None"}',
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
        '        # Step 3: Train model',
        '        print("\\nSTEP 3: Training model...")',
        '        X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_model(',
        '            model, X, y, test_size=TEST_SIZE, cv_folds=USE_CROSS_VALIDATION',
        '        )',
        '        ',
        '        # Step 4: Evaluate model',
        '        print("\\nSTEP 4: Evaluating model...")',
        '        evaluate_model(model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test)',
        '        ',
        '        # Step 5: Save model',
        '        print("\\nSTEP 5: Saving model...")',
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
        '        print("\\n✅ Training completed successfully!")',
        '        ',
        '    except FileNotFoundError:',
        '        print(f"❌ Error: Could not find data file: {DATA_FILE}")',
        '        print("Please update the DATA_FILE variable with your dataset path.")',
        '    except Exception as e:',
        '        print(f"❌ Error: {str(e)}")',
        '        import traceback',
        '        traceback.print_exc()',
        '',
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