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
    
    code = f'''"""
Auto-generated Keras model code
Model: {model_def.name}
Type: {model_def.model_type.upper()}
Generated at: {model_def.updated_at}

Configuration:
- Target columns: {model_def.target_columns}
- Predictor columns: {len(model_def.predictor_columns)} features
- Loss function: {hyperparams.get('loss_function', 'mse')}
- Optimizer: {hyperparams.get('optimizer', 'Adam')}
- Learning rate: {hyperparams.get('learning_rate', 0.001)}
- Batch size: {hyperparams.get('batch_size', 32)}
- Epochs: {hyperparams.get('epochs', 50)}
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np

def create_model(input_shape, output_shape):
    """
    Create and return the configured model
    
    Args:
        input_shape: Tuple defining input shape (e.g., (timesteps, features) for sequences)
        output_shape: Tuple defining output shape (e.g., (num_targets,) for outputs)
    
    Returns:
        Compiled Keras model ready for training
    """
    print(f"Creating model with input_shape={{input_shape}} and output_shape={{output_shape}}")
    
'''
    
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
    
    # Add example usage section
    code += f"\\n# Example usage:\\n"
    code += f"if __name__ == '__main__':\\n"
    code += f"    # Define your data shapes\\n"
    if model_def.model_type in ['lstm', 'gru']:
        code += f"    timesteps = 10  # Number of time steps\\n"
        code += f"    features = {len(model_def.predictor_columns)}  # Number of features\\n"
        code += f"    input_shape = (timesteps, features)\\n"
    else:
        code += f"    features = {len(model_def.predictor_columns)}  # Number of features\\n"
        code += f"    input_shape = (features,)\\n"
    
    code += f"    output_shape = ({len(model_def.target_columns)},)  # Number of target variables\\n\\n"
    
    code += f"    # Create and display model\\n"
    code += f"    model = create_model(input_shape, output_shape)\\n"
    code += f"    model.summary()\\n\\n"
    
    code += f"    # Example data (replace with your actual data)\\n"
    code += f"    # X_train = np.random.randn(1000, *input_shape)\\n"
    code += f"    # y_train = np.random.randn(1000, *output_shape)\\n"
    code += f"    # X_val = np.random.randn(200, *input_shape)\\n"
    code += f"    # y_val = np.random.randn(200, *output_shape)\\n\\n"
    
    code += f"    # Train the model\\n"
    code += f"    # history = train_model(model, X_train, y_train, X_val, y_val)\\n"
    
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