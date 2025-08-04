"""
Code generation and parsing for neural network architectures - UPDATED VERSION
"""
import json
import ast
import re
from typing import Dict, List, Any


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
        '    print(f"Creating model with input_shape={input_shape} and output_shape={output_shape}")',
        '    '
    ])
    
    if model_def.use_custom_architecture and model_def.custom_architecture:
        # Generate from custom architecture defined by user
        code += "    print('Building custom architecture...')\\n"
        code += "    model = models.Sequential()\\n\\n"
        
        for i, layer in enumerate(model_def.custom_architecture):
            layer_type = layer.get('type')
            params = layer.get('params', {})
            layer_name = layer.get('name', f'{layer_type} Layer {i+1}')
            
            code += f"    # Layer {i+1}: {layer_name}\\n"
            
            if i == 0:
                # First layer needs input_shape
                if layer_type == 'Dense':
                    code += f"    model.add(layers.Dense(\\n"
                    code += f"        units={params.get('units', 32)},\\n"
                    code += f"        activation='{params.get('activation', 'relu')}',\\n"
                    code += f"        use_bias={params.get('use_bias', True)},\\n"
                    code += f"        kernel_initializer='{params.get('kernel_initializer', 'glorot_uniform')}',\\n"
                    code += f"        input_shape=input_shape,\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'LSTM':
                    code += f"    model.add(layers.LSTM(\\n"
                    code += f"        units={params.get('units', 50)},\\n"
                    code += f"        activation='{params.get('activation', 'tanh')}',\\n"
                    code += f"        recurrent_activation='{params.get('recurrent_activation', 'sigmoid')}',\\n"
                    code += f"        return_sequences={params.get('return_sequences', True)},\\n"
                    code += f"        dropout={params.get('dropout', 0.0)},\\n"
                    code += f"        recurrent_dropout={params.get('recurrent_dropout', 0.0)},\\n"
                    code += f"        input_shape=input_shape,\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'GRU':
                    code += f"    model.add(layers.GRU(\\n"
                    code += f"        units={params.get('units', 50)},\\n"
                    code += f"        activation='{params.get('activation', 'tanh')}',\\n"
                    code += f"        recurrent_activation='{params.get('recurrent_activation', 'sigmoid')}',\\n"
                    code += f"        return_sequences={params.get('return_sequences', True)},\\n"
                    code += f"        dropout={params.get('dropout', 0.0)},\\n"
                    code += f"        recurrent_dropout={params.get('recurrent_dropout', 0.0)},\\n"
                    code += f"        input_shape=input_shape,\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'Conv1D':
                    code += f"    model.add(layers.Conv1D(\\n"
                    code += f"        filters={params.get('filters', 32)},\\n"
                    code += f"        kernel_size={params.get('kernel_size', 3)},\\n"
                    code += f"        strides={params.get('strides', 1)},\\n"
                    code += f"        padding='{params.get('padding', 'valid')}',\\n"
                    code += f"        activation='{params.get('activation', 'relu')}',\\n"
                    code += f"        input_shape=input_shape,\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
            else:
                # Subsequent layers - no input_shape needed
                if layer_type == 'Dense':
                    code += f"    model.add(layers.Dense(\\n"
                    code += f"        units={params.get('units', 32)},\\n"
                    code += f"        activation='{params.get('activation', 'relu')}',\\n"
                    code += f"        use_bias={params.get('use_bias', True)},\\n"
                    code += f"        kernel_initializer='{params.get('kernel_initializer', 'glorot_uniform')}',\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'LSTM':
                    code += f"    model.add(layers.LSTM(\\n"
                    code += f"        units={params.get('units', 50)},\\n"
                    code += f"        activation='{params.get('activation', 'tanh')}',\\n"
                    code += f"        recurrent_activation='{params.get('recurrent_activation', 'sigmoid')}',\\n"
                    code += f"        return_sequences={params.get('return_sequences', False)},\\n"
                    code += f"        dropout={params.get('dropout', 0.0)},\\n"
                    code += f"        recurrent_dropout={params.get('recurrent_dropout', 0.0)},\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'GRU':
                    code += f"    model.add(layers.GRU(\\n"
                    code += f"        units={params.get('units', 50)},\\n"
                    code += f"        activation='{params.get('activation', 'tanh')}',\\n"
                    code += f"        recurrent_activation='{params.get('recurrent_activation', 'sigmoid')}',\\n"
                    code += f"        return_sequences={params.get('return_sequences', False)},\\n"
                    code += f"        dropout={params.get('dropout', 0.0)},\\n"
                    code += f"        recurrent_dropout={params.get('recurrent_dropout', 0.0)},\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'Conv1D':
                    code += f"    model.add(layers.Conv1D(\\n"
                    code += f"        filters={params.get('filters', 32)},\\n"
                    code += f"        kernel_size={params.get('kernel_size', 3)},\\n"
                    code += f"        strides={params.get('strides', 1)},\\n"
                    code += f"        padding='{params.get('padding', 'valid')}',\\n"
                    code += f"        activation='{params.get('activation', 'relu')}',\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'Dropout':
                    code += f"    model.add(layers.Dropout(\\n"
                    code += f"        rate={params.get('rate', 0.5)},\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'BatchNormalization':
                    code += f"    model.add(layers.BatchNormalization(\\n"
                    code += f"        momentum={params.get('momentum', 0.99)},\\n"
                    code += f"        epsilon={params.get('epsilon', 0.001)},\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'Flatten':
                    code += f"    model.add(layers.Flatten(\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'MaxPooling1D':
                    code += f"    model.add(layers.MaxPooling1D(\\n"
                    code += f"        pool_size={params.get('pool_size', 2)},\\n"
                    code += f"        strides={params.get('strides', None)},\\n"
                    code += f"        padding='{params.get('padding', 'valid')}',\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'GlobalMaxPooling1D':
                    code += f"    model.add(layers.GlobalMaxPooling1D(\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
                    
                elif layer_type == 'GlobalAveragePooling1D':
                    code += f"    model.add(layers.GlobalAveragePooling1D(\\n"
                    code += f"        name='{layer_name.replace(' ', '_').lower()}'\\n"
                    code += f"    ))\\n\\n"
        
        # Add final output layer
        output_activation = hyperparams.get('output_activation', 'linear')
        output_units = hyperparams.get('output_units', len(model_def.target_columns))
        code += f"    # Final Output Layer\\n"
        code += f"    model.add(layers.Dense(\\n"
        code += f"        units={output_units},\\n"
        code += f"        activation='{output_activation}',\\n"
        code += f"        name='output_layer'\\n"
        code += f"    ))\\n\\n"
        
    else:
        # Generate standard model based on model_type
        model_type = model_def.model_type
        
        code += f"    print('Building {model_type.upper()} model...')\\n"
        code += f"    model = models.Sequential()\\n\\n"
        
        if model_type == 'lstm':
            num_layers = hyperparams.get('layers', 2)
            units = hyperparams.get('units', 50)
            dropout = hyperparams.get('dropout', 0.2)
            
            for i in range(num_layers):
                return_sequences = i < num_layers - 1  # All except last layer return sequences
                code += f"    # LSTM Layer {i+1}\\n"
                code += f"    model.add(layers.LSTM(\\n"
                code += f"        units={units},\\n"
                code += f"        activation='{hyperparams.get('activation', 'tanh')}',\\n"
                code += f"        recurrent_activation='{hyperparams.get('recurrent_activation', 'sigmoid')}',\\n"
                code += f"        return_sequences={return_sequences},\\n"
                code += f"        dropout={dropout},\\n"
                code += f"        recurrent_dropout={hyperparams.get('recurrent_dropout', 0.0)},\\n"
                if i == 0:
                    code += f"        input_shape=input_shape,\\n"
                code += f"        name='lstm_layer_{i+1}'\\n"
                code += f"    ))\\n\\n"
                
                if dropout > 0 and i < num_layers - 1:
                    code += f"    # Dropout after LSTM Layer {i+1}\\n"
                    code += f"    model.add(layers.Dropout({dropout}, name='dropout_{i+1}'))\\n\\n"
            
            # Output layer
            output_units = hyperparams.get('output_units', len(model_def.target_columns))
            output_activation = hyperparams.get('output_activation', 'linear')
            code += f"    # Output Layer\\n"
            code += f"    model.add(layers.Dense({output_units}, activation='{output_activation}', name='output'))\\n\\n"
            
        elif model_type == 'gru':
            num_layers = hyperparams.get('layers', 2)
            units = hyperparams.get('units', 50)
            dropout = hyperparams.get('dropout', 0.2)
            
            for i in range(num_layers):
                return_sequences = i < num_layers - 1
                code += f"    # GRU Layer {i+1}\\n"
                code += f"    model.add(layers.GRU(\\n"
                code += f"        units={units},\\n"
                code += f"        activation='{hyperparams.get('activation', 'tanh')}',\\n"
                code += f"        recurrent_activation='{hyperparams.get('recurrent_activation', 'sigmoid')}',\\n"
                code += f"        return_sequences={return_sequences},\\n"
                code += f"        dropout={dropout},\\n"
                code += f"        recurrent_dropout={hyperparams.get('recurrent_dropout', 0.0)},\\n"
                if i == 0:
                    code += f"        input_shape=input_shape,\\n"
                code += f"        name='gru_layer_{i+1}'\\n"
                code += f"    ))\\n\\n"
                
                if dropout > 0 and i < num_layers - 1:
                    code += f"    # Dropout after GRU Layer {i+1}\\n"
                    code += f"    model.add(layers.Dropout({dropout}, name='dropout_{i+1}'))\\n\\n"
            
            # Output layer
            output_units = hyperparams.get('output_units', len(model_def.target_columns))
            output_activation = hyperparams.get('output_activation', 'linear')
            code += f"    # Output Layer\\n"
            code += f"    model.add(layers.Dense({output_units}, activation='{output_activation}', name='output'))\\n\\n"
            
        elif model_type == 'cnn':
            filters = hyperparams.get('filters', 64)
            kernel_size = hyperparams.get('kernel_size', 3)
            pool_size = hyperparams.get('pool_size', 2)
            dropout = hyperparams.get('dropout', 0.2)
            dense_units = hyperparams.get('dense_units', 100)
            
            # First Conv1D layer
            code += f"    # First Convolutional Layer\\n"
            code += f"    model.add(layers.Conv1D(\\n"
            code += f"        filters={filters},\\n"
            code += f"        kernel_size={kernel_size},\\n"
            code += f"        activation='{hyperparams.get('activation', 'relu')}',\\n"
            code += f"        input_shape=input_shape,\\n"
            code += f"        name='conv1d_1'\\n"
            code += f"    ))\\n\\n"
            
            # Max pooling
            code += f"    # Max Pooling Layer\\n"
            code += f"    model.add(layers.MaxPooling1D(pool_size={pool_size}, name='maxpool_1'))\\n\\n"
            
            if dropout > 0:
                code += f"    # Dropout Layer\\n"
                code += f"    model.add(layers.Dropout({dropout}, name='dropout_1'))\\n\\n"
            
            # Second Conv1D layer
            code += f"    # Second Convolutional Layer\\n"
            code += f"    model.add(layers.Conv1D(\\n"
            code += f"        filters={filters * 2},\\n"
            code += f"        kernel_size={kernel_size},\\n"
            code += f"        activation='{hyperparams.get('activation', 'relu')}',\\n"
            code += f"        name='conv1d_2'\\n"
            code += f"    ))\\n\\n"
            
            # Global max pooling
            code += f"    # Global Max Pooling Layer\\n"
            code += f"    model.add(layers.GlobalMaxPooling1D(name='global_maxpool'))\\n\\n"
            
            # Dense layer
            code += f"    # Dense Layer\\n"
            code += f"    model.add(layers.Dense(\\n"
            code += f"        units={dense_units},\\n"
            code += f"        activation='{hyperparams.get('activation', 'relu')}',\\n"
            code += f"        name='dense_1'\\n"
            code += f"    ))\\n\\n"
            
            if dropout > 0:
                code += f"    # Final Dropout Layer\\n"
                code += f"    model.add(layers.Dropout({dropout}, name='dropout_final'))\\n\\n"
            
            # Output layer
            output_units = hyperparams.get('output_units', len(model_def.target_columns))
            output_activation = hyperparams.get('output_activation', 'linear')
            code += f"    # Output Layer\\n"
            code += f"    model.add(layers.Dense({output_units}, activation='{output_activation}', name='output'))\\n\\n"
            
        elif model_type == 'transformer':
            # Basic transformer architecture
            d_model = hyperparams.get('d_model', 64)
            num_heads = hyperparams.get('num_heads', 8)
            ff_dim = hyperparams.get('ff_dim', 256)
            dropout = hyperparams.get('dropout', 0.1)
            
            code += f"    # Multi-Head Self-Attention\\n"
            code += f"    attention_layer = layers.MultiHeadAttention(\\n"
            code += f"        num_heads={num_heads},\\n"
            code += f"        key_dim={d_model // num_heads},\\n"
            code += f"        dropout={dropout},\\n"
            code += f"        name='multi_head_attention'\\n"
            code += f"    )\\n\\n"
            
            code += f"    # Input projection\\n"
            code += f"    inputs = layers.Input(shape=input_shape)\\n"
            code += f"    x = layers.Dense({d_model}, name='input_projection')(inputs)\\n\\n"
            
            code += f"    # Self-attention\\n"
            code += f"    attention_output = attention_layer(x, x)\\n"
            code += f"    x = layers.Add(name='attention_add')([x, attention_output])\\n"
            code += f"    x = layers.LayerNormalization(name='attention_norm')(x)\\n\\n"
            
            code += f"    # Feed Forward Network\\n"
            code += f"    ffn = layers.Dense({ff_dim}, activation='relu', name='ffn_1')(x)\\n"
            code += f"    ffn = layers.Dropout({dropout}, name='ffn_dropout')(ffn)\\n"
            code += f"    ffn = layers.Dense({d_model}, name='ffn_2')(ffn)\\n"
            code += f"    x = layers.Add(name='ffn_add')([x, ffn])\\n"
            code += f"    x = layers.LayerNormalization(name='ffn_norm')(x)\\n\\n"
            
            code += f"    # Global average pooling and output\\n"
            code += f"    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)\\n"
            
            # Output layer
            output_units = hyperparams.get('output_units', len(model_def.target_columns))
            output_activation = hyperparams.get('output_activation', 'linear')
            code += f"    outputs = layers.Dense({output_units}, activation='{output_activation}', name='output')(x)\\n\\n"
            
            code += f"    # Create model\\n"
            code += f"    model = models.Model(inputs=inputs, outputs=outputs)\\n\\n"
    
    # Add compilation with all configured parameters
    optimizer_name = hyperparams.get('optimizer', 'Adam').lower()
    learning_rate = hyperparams.get('learning_rate', 0.001)
    loss_function = hyperparams.get('loss_function', 'mse')
    metrics = hyperparams.get('metrics', ['mae', 'mse'])
    
    code += f"    # Compile model with configured parameters\\n"
    code += f"    model.compile(\\n"
    
    # Handle different optimizers
    if optimizer_name == 'adam':
        code += f"        optimizer=optimizers.Adam(learning_rate={learning_rate}),\\n"
    elif optimizer_name == 'sgd':
        code += f"        optimizer=optimizers.SGD(learning_rate={learning_rate}),\\n"
    elif optimizer_name == 'rmsprop':
        code += f"        optimizer=optimizers.RMSprop(learning_rate={learning_rate}),\\n"
    else:
        code += f"        optimizer=optimizers.Adam(learning_rate={learning_rate}),\\n"
    
    code += f"        loss='{loss_function}',\\n"
    code += f"        metrics={metrics}\\n"
    code += f"    )\\n\\n"
    
    code += f"    print('Model compiled successfully!')\\n"
    code += f"    return model\\n\\n"
    
    # Add training function with callbacks
    code += f"\\ndef train_model(model, X_train, y_train, X_val=None, y_val=None):\\n"
    code += f"    \\\"\\\"\\\"\\n"
    code += f"    Train the model with configured parameters\\n"
    code += f"    \\\"\\\"\\\"\\n"
    code += f"    batch_size = {hyperparams.get('batch_size', 32)}\\n"
    code += f"    epochs = {hyperparams.get('epochs', 50)}\\n\\n"
    
    # Add callbacks if configured
    callbacks_config = hyperparams.get('callbacks', {})
    if callbacks_config:
        code += f"    # Configure callbacks\\n"
        code += f"    callback_list = []\\n\\n"
        
        if callbacks_config.get('early_stopping', False):
            code += f"    # Early stopping\\n"
            code += f"    callback_list.append(callbacks.EarlyStopping(\\n"
            code += f"        monitor='val_loss',\\n"
            code += f"        patience=10,\\n"
            code += f"        restore_best_weights=True\\n"
            code += f"    ))\\n\\n"
        
        if callbacks_config.get('reduce_lr', False):
            code += f"    # Reduce learning rate on plateau\\n"
            code += f"    callback_list.append(callbacks.ReduceLROnPlateau(\\n"
            code += f"        monitor='val_loss',\\n"
            code += f"        factor=0.5,\\n"
            code += f"        patience=5,\\n"
            code += f"        min_lr=1e-7\\n"
            code += f"    ))\\n\\n"
        
        if callbacks_config.get('model_checkpoint', False):
            code += f"    # Model checkpoint\\n"
            code += f"    callback_list.append(callbacks.ModelCheckpoint(\\n"
            code += f"        'best_model.h5',\\n"
            code += f"        monitor='val_loss',\\n"
            code += f"        save_best_only=True\\n"
            code += f"    ))\\n\\n"
        
        code += f"    # Train the model\\n"
        code += f"    history = model.fit(\\n"
        code += f"        X_train, y_train,\\n"
        code += f"        batch_size=batch_size,\\n"
        code += f"        epochs=epochs,\\n"
        code += f"        validation_data=(X_val, y_val) if X_val is not None else None,\\n"
        code += f"        callbacks=callback_list,\\n"
        code += f"        verbose=1\\n"
        code += f"    )\\n\\n"
    else:
        code += f"    # Train the model\\n"
        code += f"    history = model.fit(\\n"
        code += f"        X_train, y_train,\\n"
        code += f"        batch_size=batch_size,\\n"
        code += f"        epochs=epochs,\\n"
        code += f"        validation_data=(X_val, y_val) if X_val is not None else None,\\n"
        code += f"        verbose=1\\n"
        code += f"    )\\n\\n"
    
    code += f"    return history\\n\\n"
    
    # Add data loading and preprocessing functions
    code += f"\\ndef load_and_preprocess_data(dataset_path, test_size=0.2):\\n"
    code += f"    \\\"\\\"\\\"\\n"
    code += f"    Load dataset and preprocess for training\\n"
    code += f"    \\\"\\\"\\\"\\n"
    code += f"    import pandas as pd\\n"
    code += f"    from sklearn.model_selection import train_test_split\\n"
    code += f"    from sklearn.preprocessing import StandardScaler, LabelEncoder\\n"
    code += f"    from sklearn.impute import SimpleImputer\\n\\n"
    
    code += f"    print(f'Loading dataset from: {{dataset_path}}')\\n"
    code += f"    df = pd.read_csv(dataset_path)\\n"
    code += f"    print(f'Dataset shape: {{df.shape}}')\\n\\n"
    
    # Add columns info
    predictor_columns = [str(col) for col in model_def.predictor_columns]
    target_columns = [str(col) for col in model_def.target_columns]
    
    code += f"    # Define columns based on model configuration\\n"
    code += f"    predictor_columns = {predictor_columns}\\n"
    code += f"    target_columns = {target_columns}\\n\\n"
    
    code += f"    # Validate columns exist in dataset\\n"
    code += f"    missing_predictors = [col for col in predictor_columns if col not in df.columns]\\n"
    code += f"    missing_targets = [col for col in target_columns if col not in df.columns]\\n"
    code += f"    \\n"
    code += f"    if missing_predictors:\\n"
    code += f"        raise ValueError(f'Missing predictor columns: {{missing_predictors}}')\\n"
    code += f"    if missing_targets:\\n"
    code += f"        raise ValueError(f'Missing target columns: {{missing_targets}}')\\n\\n"
    
    code += f"    # Extract features and targets\\n"
    code += f"    X = df[predictor_columns].copy()\\n"
    code += f"    y = df[target_columns].copy()\\n\\n"
    
    code += f"    # Handle missing values\\n"
    code += f"    print('Handling missing values...')\\n"
    code += f"    # For numeric columns, use mean imputation\\n"
    code += f"    numeric_cols = X.select_dtypes(include=[np.number]).columns\\n"
    code += f"    if len(numeric_cols) > 0:\\n"
    code += f"        imputer_num = SimpleImputer(strategy='mean')\\n"
    code += f"        X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])\\n\\n"
    
    code += f"    # For categorical columns, use most frequent imputation and label encoding\\n"
    code += f"    categorical_cols = X.select_dtypes(include=['object']).columns\\n"
    code += f"    label_encoders = {{}}\\n"
    code += f"    if len(categorical_cols) > 0:\\n"
    code += f"        imputer_cat = SimpleImputer(strategy='most_frequent')\\n"
    code += f"        X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])\\n"
    code += f"        \\n"
    code += f"        # Label encode categorical variables\\n"
    code += f"        for col in categorical_cols:\\n"
    code += f"            le = LabelEncoder()\\n"
    code += f"            X[col] = le.fit_transform(X[col].astype(str))\\n"
    code += f"            label_encoders[col] = le\\n\\n"
    
    code += f"    # Handle target variables\\n"
    code += f"    target_encoders = {{}}\\n"
    code += f"    for col in target_columns:\\n"
    code += f"        if y[col].dtype == 'object':\\n"
    code += f"            # Categorical target - encode\\n"
    code += f"            le = LabelEncoder()\\n"
    code += f"            y[col] = le.fit_transform(y[col].astype(str))\\n"
    code += f"            target_encoders[col] = le\\n"
    code += f"        else:\\n"
    code += f"            # Numeric target - handle missing values\\n"
    code += f"            y[col] = y[col].fillna(y[col].mean())\\n\\n"
    
    code += f"    # Normalize features\\n"
    code += f"    print('Normalizing features...')\\n"
    code += f"    scaler = StandardScaler()\\n"
    code += f"    X_scaled = scaler.fit_transform(X)\\n\\n"
    
    # Add sequence creation for RNN models
    if model_def.model_type in ['lstm', 'gru']:
        code += f"    # Create sequences for RNN models\\n"
        code += f"    def create_sequences(data, target, timesteps=10):\\n"
        code += f"        X_seq, y_seq = [], []\\n"
        code += f"        for i in range(timesteps, len(data)):\\n"
        code += f"            X_seq.append(data[i-timesteps:i])\\n"
        code += f"            y_seq.append(target[i])\\n"
        code += f"        return np.array(X_seq), np.array(y_seq)\\n\\n"
        
        code += f"    print('Creating sequences for RNN...')\\n"
        code += f"    timesteps = 10\\n"
        code += f"    X_sequences, y_sequences = create_sequences(X_scaled, y.values, timesteps)\\n"
        code += f"    print(f'Sequence shape: X={{X_sequences.shape}}, y={{y_sequences.shape}}')\\n\\n"
        
        code += f"    # Split data\\n"
        code += f"    X_train, X_test, y_train, y_test = train_test_split(\\n"
        code += f"        X_sequences, y_sequences, test_size=test_size, random_state=42\\n"
        code += f"    )\\n\\n"
        
        code += f"    input_shape = (timesteps, X_scaled.shape[1])\\n"
    else:
        code += f"    # Split data\\n"
        code += f"    X_train, X_test, y_train, y_test = train_test_split(\\n"
        code += f"        X_scaled, y.values, test_size=test_size, random_state=42\\n"
        code += f"    )\\n\\n"
        
        code += f"    input_shape = (X_scaled.shape[1],)\\n"
    
    code += f"    output_shape = (len(target_columns),)\\n\\n"
    
    code += f"    print(f'Training set: X={{X_train.shape}}, y={{y_train.shape}}')\\n"
    code += f"    print(f'Test set: X={{X_test.shape}}, y={{y_test.shape}}')\\n\\n"
    
    code += f"    return {{\\n"
    code += f"        'X_train': X_train, 'X_test': X_test,\\n"
    code += f"        'y_train': y_train, 'y_test': y_test,\\n"
    code += f"        'input_shape': input_shape, 'output_shape': output_shape,\\n"
    code += f"        'scaler': scaler, 'label_encoders': label_encoders,\\n"
    code += f"        'target_encoders': target_encoders\\n"
    code += f"    }}\\n\\n"
    
    # Add model evaluation function
    code += f"\\ndef evaluate_model(model, X_test, y_test, target_encoders=None):\\n"
    code += f"    \\\"\\\"\\\"\\n"
    code += f"    Evaluate the trained model\\n"
    code += f"    \\\"\\\"\\\"\\n"
    code += f"    print('Evaluating model...')\\n"
    code += f"    \\n"
    code += f"    # Make predictions\\n"
    code += f"    y_pred = model.predict(X_test)\\n"
    code += f"    \\n"
    code += f"    # Calculate metrics\\n"
    code += f"    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\\n"
    code += f"    \\n"
    code += f"    print('\\\\n=== Model Evaluation Results ===')\\n"
    code += f"    for i, target_col in enumerate({target_columns}):\\n"
    code += f"        y_true_col = y_test[:, i] if y_test.ndim > 1 else y_test\\n"
    code += f"        y_pred_col = y_pred[:, i] if y_pred.ndim > 1 else y_pred\\n"
    code += f"        \\n"
    code += f"        mae = mean_absolute_error(y_true_col, y_pred_col)\\n"
    code += f"        mse = mean_squared_error(y_true_col, y_pred_col)\\n"
    code += f"        rmse = np.sqrt(mse)\\n"
    code += f"        r2 = r2_score(y_true_col, y_pred_col)\\n"
    code += f"        \\n"
    code += f"        print(f'Target: {{target_col}}')\\n"
    code += f"        print(f'  MAE:  {{mae:.4f}}')\\n"
    code += f"        print(f'  MSE:  {{mse:.4f}}')\\n"
    code += f"        print(f'  RMSE: {{rmse:.4f}}')\\n"
    code += f"        print(f'  R²:   {{r2:.4f}}')\\n"
    code += f"        print()\\n\\n"
    
    code += f"    return y_pred\\n\\n"
    
    # Add main execution section
    code += f"\\n# =============================================================================\\n"
    code += f"# MAIN EXECUTION - CONFIGURE YOUR DATASET PATH HERE\\n"
    code += f"# =============================================================================\\n\\n"
    
    code += f"# DATASET CONFIGURATION\\n"
    code += f"DATASET_PATH = 'path/to/your/dataset.csv'  # ⚠️ CHANGE THIS TO YOUR DATASET PATH\\n\\n"
    
    code += f"if __name__ == '__main__':\\n"
    code += f"    print('🚀 Starting {model_def.name} Model Training')\\n"
    code += f"    print('=' * 50)\\n\\n"
    
    code += f"    try:\\n"
    code += f"        # Step 1: Load and preprocess data\\n"
    code += f"        print('📊 Step 1: Loading and preprocessing data...')\\n"
    code += f"        data = load_and_preprocess_data(DATASET_PATH, test_size=0.2)\\n\\n"
    
    code += f"        # Step 2: Create model\\n"
    code += f"        print('🧠 Step 2: Creating model...')\\n"
    code += f"        model = create_model(data['input_shape'], data['output_shape'])\\n"
    code += f"        model.summary()\\n\\n"
    
    code += f"        # Step 3: Train model\\n"
    code += f"        print('🏋️ Step 3: Training model...')\\n"
    code += f"        history = train_model(\\n"
    code += f"            model, \\n"
    code += f"            data['X_train'], data['y_train'], \\n"
    code += f"            data['X_test'], data['y_test']\\n"
    code += f"        )\\n\\n"
    
    code += f"        # Step 4: Evaluate model\\n"
    code += f"        print('📈 Step 4: Evaluating model...')\\n"
    code += f"        predictions = evaluate_model(\\n"
    code += f"            model, \\n"
    code += f"            data['X_test'], data['y_test'], \\n"
    code += f"            data['target_encoders']\\n"
    code += f"        )\\n\\n"
    
    code += f"        # Step 5: Save model\\n"
    code += f"        print('💾 Step 5: Saving model...')\\n"
    code += f"        model.save('{model_def.name.replace(' ', '_')}_model.h5')\\n"
    code += f"        print(f'Model saved as: {model_def.name.replace(' ', '_')}_model.h5')\\n\\n"
    
    code += f"        print('✅ Training completed successfully!')\\n"
    code += f"        print('=' * 50)\\n\\n"
    
    code += f"        # Optional: Plot training history\\n"
    code += f"        try:\\n"
    code += f"            import matplotlib.pyplot as plt\\n"
    code += f"            \\n"
    code += f"            plt.figure(figsize=(12, 4))\\n"
    code += f"            \\n"
    code += f"            plt.subplot(1, 2, 1)\\n"
    code += f"            plt.plot(history.history['loss'], label='Training Loss')\\n"
    code += f"            if 'val_loss' in history.history:\\n"
    code += f"                plt.plot(history.history['val_loss'], label='Validation Loss')\\n"
    code += f"            plt.title('Model Loss')\\n"
    code += f"            plt.xlabel('Epoch')\\n"
    code += f"            plt.ylabel('Loss')\\n"
    code += f"            plt.legend()\\n"
    code += f"            \\n"
    code += f"            plt.subplot(1, 2, 2)\\n"
    code += f"            if 'mae' in history.history:\\n"
    code += f"                plt.plot(history.history['mae'], label='Training MAE')\\n"
    code += f"                if 'val_mae' in history.history:\\n"
    code += f"                    plt.plot(history.history['val_mae'], label='Validation MAE')\\n"
    code += f"            plt.title('Model Metrics')\\n"
    code += f"            plt.xlabel('Epoch')\\n"
    code += f"            plt.ylabel('MAE')\\n"
    code += f"            plt.legend()\\n"
    code += f"            \\n"
    code += f"            plt.tight_layout()\\n"
    code += f"            plt.savefig('{model_def.name.replace(' ', '_')}_training_history.png')\\n"
    code += f"            plt.show()\\n"
    code += f"            \\n"
    code += f"            print('📊 Training plots saved as: {model_def.name.replace(' ', '_')}_training_history.png')\\n"
    code += f"        except ImportError:\\n"
    code += f"            print('⚠️ Matplotlib not available. Skipping plots.')\\n\\n"
    
    code += f"    except FileNotFoundError:\\n"
    code += f"        print(f'❌ Error: Dataset file not found at: {{DATASET_PATH}}')\\n"
    code += f"        print('Please update the DATASET_PATH variable with the correct path to your dataset.')\\n"
    code += f"    except Exception as e:\\n"
    code += f"        print(f'❌ Error during training: {{str(e)}}')\\n"
    code += f"        import traceback\\n"
    code += f"        traceback.print_exc()\\n\\n"
    
    code += f"    # Instructions for usage\\n"
    code += f"    print('\\\\n' + '=' * 60)\\n"
    code += f"    print('📋 USAGE INSTRUCTIONS:')\\n"
    code += f"    print('=' * 60)\\n"
    code += f"    print('1. Update DATASET_PATH with your CSV file path')\\n"
    code += f"    print('2. Ensure your dataset has these columns:')\\n"
    code += f"    print(f'   - Predictors: {predictor_columns}')\\n"
    code += f"    print(f'   - Targets: {target_columns}')\\n"
    code += f"    print('3. Run: python {model_def.name.replace(' ', '_')}_model.py')\\n"
    code += f"    print('=' * 60)\\n"
    
    return code


# Rest of the file remains the same - just copy the existing PyTorch and parsing functions
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
        code += f"class {class_name}(nn.Module):\\n"
        code += f"    def __init__(self, input_size, output_size):\\n"
        code += f"        super({class_name}, self).__init__()\\n\\n"
        
        # Build layers
        layer_names = []
        prev_size = None
        
        for i, layer in enumerate(model_def.custom_architecture):
            layer_type = layer.get('type')
            params = layer.get('params', {})
            layer_name = f"layer{i+1}"
            
            if layer_type == 'Dense':
                units = params.get('units', 32)
                if prev_size is None:
                    code += f"        self.{layer_name} = nn.Linear(input_size, {units})\\n"
                else:
                    code += f"        self.{layer_name} = nn.Linear({prev_size}, {units})\\n"
                prev_size = units
                layer_names.append((layer_name, 'linear', params.get('activation', 'relu')))
                
            elif layer_type == 'LSTM':
                units = params.get('units', 50)
                if prev_size is None:
                    code += f"        self.{layer_name} = nn.LSTM(input_size, {units}, batch_first=True)\\n"
                else:
                    code += f"        self.{layer_name} = nn.LSTM({prev_size}, {units}, batch_first=True)\\n"
                prev_size = units
                layer_names.append((layer_name, 'lstm', None))
                
            elif layer_type == 'GRU':
                units = params.get('units', 50)
                if prev_size is None:
                    code += f"        self.{layer_name} = nn.GRU(input_size, {units}, batch_first=True)\\n"
                else:
                    code += f"        self.{layer_name} = nn.GRU({prev_size}, {units}, batch_first=True)\\n"
                prev_size = units
                layer_names.append((layer_name, 'gru', None))
                
            elif layer_type == 'Dropout':
                code += f"        self.{layer_name} = nn.Dropout({params.get('rate', 0.5)})\\n"
                layer_names.append((layer_name, 'dropout', None))
                
            elif layer_type == 'BatchNormalization':
                if prev_size:
                    code += f"        self.{layer_name} = nn.BatchNorm1d({prev_size})\\n"
                layer_names.append((layer_name, 'batchnorm', None))
        
        # Output layer
        if prev_size:
            code += f"        self.output = nn.Linear({prev_size}, output_size)\\n\\n"
        else:
            code += f"        self.output = nn.Linear(input_size, output_size)\\n\\n"
        
        # Forward method
        code += f"    def forward(self, x):\\n"
        for layer_name, layer_type, activation in layer_names:
            if layer_type == 'linear':
                code += f"        x = self.{layer_name}(x)\\n"
                if activation == 'relu':
                    code += f"        x = F.relu(x)\\n"
                elif activation == 'tanh':
                    code += f"        x = torch.tanh(x)\\n"
                elif activation == 'sigmoid':
                    code += f"        x = torch.sigmoid(x)\\n"
            elif layer_type in ['lstm', 'gru']:
                code += f"        x, _ = self.{layer_name}(x)\\n"
                code += f"        x = x[:, -1, :]  # Get last output\\n"
            else:
                code += f"        x = self.{layer_name}(x)\\n"
        
        code += f"        x = self.output(x)\\n"
        code += f"        return x\\n\\n"
        
    else:
        # Generate standard model
        model_type = model_def.model_type
        hyperparams = model_def.hyperparameters
        
        if model_type == 'lstm':
            code += f"class {class_name}(nn.Module):\\n"
            code += f"    def __init__(self, input_size, hidden_size={hyperparams.get('units', 50)}, "
            code += f"num_layers={hyperparams.get('layers', 2)}, output_size=1, "
            code += f"dropout={hyperparams.get('dropout', 0.2)}):\\n"
            code += f"        super({class_name}, self).__init__()\\n"
            code += f"        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \\n"
            code += f"                           batch_first=True, dropout=dropout if num_layers > 1 else 0)\\n"
            code += f"        self.dropout = nn.Dropout(dropout)\\n"
            code += f"        self.fc = nn.Linear(hidden_size, output_size)\\n\\n"
            code += f"    def forward(self, x):\\n"
            code += f"        lstm_out, _ = self.lstm(x)\\n"
            code += f"        last_output = lstm_out[:, -1, :]\\n"
            code += f"        out = self.dropout(last_output)\\n"
            code += f"        out = self.fc(out)\\n"
            code += f"        return out\\n\\n"
            
        elif model_type == 'gru':
            code += f"class {class_name}(nn.Module):\\n"
            code += f"    def __init__(self, input_size, hidden_size={hyperparams.get('units', 50)}, "
            code += f"num_layers={hyperparams.get('layers', 2)}, output_size=1, "
            code += f"dropout={hyperparams.get('dropout', 0.2)}):\\n"
            code += f"        super({class_name}, self).__init__()\\n"
            code += f"        self.gru = nn.GRU(input_size, hidden_size, num_layers, \\n"
            code += f"                         batch_first=True, dropout=dropout if num_layers > 1 else 0)\\n"
            code += f"        self.dropout = nn.Dropout(dropout)\\n"
            code += f"        self.fc = nn.Linear(hidden_size, output_size)\\n\\n"
            code += f"    def forward(self, x):\\n"
            code += f"        gru_out, _ = self.gru(x)\\n"
            code += f"        last_output = gru_out[:, -1, :]\\n"
            code += f"        out = self.dropout(last_output)\\n"
            code += f"        out = self.fc(out)\\n"
            code += f"        return out\\n\\n"
            
        elif model_type == 'cnn':
            code += f"class {class_name}(nn.Module):\\n"
            code += f"    def __init__(self, input_channels, sequence_length, output_size, "
            code += f"filters={hyperparams.get('filters', 64)}, kernel_size={hyperparams.get('kernel_size', 3)}, "
            code += f"dropout={hyperparams.get('dropout', 0.2)}):\\n"
            code += f"        super({class_name}, self).__init__()\\n"
            code += f"        self.conv1 = nn.Conv1d(input_channels, filters, kernel_size, padding=1)\\n"
            code += f"        self.pool1 = nn.MaxPool1d(2)\\n"
            code += f"        self.dropout1 = nn.Dropout(dropout)\\n"
            code += f"        self.conv2 = nn.Conv1d(filters, filters*2, kernel_size, padding=1)\\n"
            code += f"        self.pool2 = nn.AdaptiveMaxPool1d(1)\\n"
            code += f"        self.dropout2 = nn.Dropout(dropout)\\n"
            code += f"        self.fc = nn.Linear(filters*2, output_size)\\n\\n"
            code += f"    def forward(self, x):\\n"
            code += f"        x = x.transpose(1, 2)  # (batch, features, sequence)\\n"
            code += f"        x = F.relu(self.conv1(x))\\n"
            code += f"        x = self.pool1(x)\\n"
            code += f"        x = self.dropout1(x)\\n"
            code += f"        x = F.relu(self.conv2(x))\\n"
            code += f"        x = self.pool2(x)\\n"
            code += f"        x = self.dropout2(x)\\n"
            code += f"        x = x.squeeze(-1)\\n"
            code += f"        x = self.fc(x)\\n"
            code += f"        return x\\n\\n"
    
    # Add example usage
    code += f"\\n# Example usage:\\n"
    code += f"# model = {class_name}(input_size=5, output_size=2)\\n"
    code += f"# x = torch.randn(32, 10, 5)  # batch_size=32, sequence_length=10, features=5\\n"
    code += f"# output = model(x)\\n"
    code += f"# print(output.shape)  # torch.Size([32, 2])\\n"
    
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