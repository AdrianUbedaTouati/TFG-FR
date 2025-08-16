"""
Normalization method compatibility definitions
"""

# Define input/output types for each normalization method
NORMALIZATION_IO_TYPES = {
    # Numeric methods
    'MIN_MAX': {
        'input': 'numeric',
        'output': 'numeric',
        'output_dtype': 'float64',
        'output_columns': 'single',  # outputs single column
        'description': 'Escala valores numéricos al rango [0, 1]'
    },
    'Z_SCORE': {
        'input': 'numeric',
        'output': 'numeric',
        'output_dtype': 'float64',
        'output_columns': 'single',
        'description': 'Estandariza valores numéricos a media 0 y desviación 1'
    },
    'LSTM_TCN': {
        'input': 'numeric',
        'output': 'numeric',
        'output_dtype': 'float64',
        'output_columns': 'single',
        'description': 'Escala valores numéricos al rango [-1, 1]'
    },
    'CNN': {
        'input': 'numeric',
        'output': 'numeric',
        'output_dtype': 'float64',
        'output_columns': 'single',
        'description': 'Normalización Z-Score para CNN'
    },
    'TRANSFORMER': {
        'input': 'numeric',
        'output': 'numeric',
        'output_dtype': 'float64',
        'output_columns': 'single',
        'description': 'RobustScaler para valores numéricos'
    },
    'TREE': {
        'input': 'numeric',
        'output': 'numeric',
        'output_dtype': 'original',  # Maintains original type
        'output_columns': 'single',
        'description': 'Sin transformación (para modelos de árbol)'
    },
    
    # Text methods
    'LOWER': {
        'input': 'text',
        'output': 'text',
        'output_dtype': 'object',
        'output_columns': 'single',
        'description': 'Convierte texto a minúsculas'
    },
    'STRIP': {
        'input': 'text',
        'output': 'text',
        'output_dtype': 'object',
        'output_columns': 'single',
        'description': 'Elimina espacios al inicio y final del texto'
    },
    'ONE_HOT': {
        'input': 'text',
        'output': 'numeric',
        'output_dtype': 'Int64',  # Nullable integer
        'output_columns': 'single',  # Current implementation outputs single column with numeric codes
        'description': 'Convierte categorías de texto a códigos numéricos'
    },
    # Note: If we want true one-hot encoding with multiple columns, we would need:
    # 'ONE_HOT_MULTI': {
    #     'input': 'text',
    #     'output': 'numeric',
    #     'output_columns': 'multiple',
    #     'description': 'One-hot encoding con múltiples columnas binarias'
    # },
}

def get_compatible_methods(previous_method=None, column_type='numeric'):
    """
    Get methods compatible with the output of the previous method
    
    Args:
        previous_method: The previous normalization method applied
        column_type: The original column type ('numeric' or 'text')
    
    Returns:
        List of compatible method names
    """
    if previous_method is None:
        # First layer - return methods compatible with column type
        return [
            method for method, info in NORMALIZATION_IO_TYPES.items()
            if info['input'] == column_type
        ]
    
    # Get output type of previous method
    if previous_method in NORMALIZATION_IO_TYPES:
        output_type = NORMALIZATION_IO_TYPES[previous_method]['output']
    elif previous_method.startswith('CUSTOM_'):
        # For custom functions, we need to check their type
        # This will be handled separately
        return list(NORMALIZATION_IO_TYPES.keys())  # Return all for now
    else:
        # Unknown method, return all
        return list(NORMALIZATION_IO_TYPES.keys())
    
    # Return methods that can accept the output type
    compatible = [
        method for method, info in NORMALIZATION_IO_TYPES.items()
        if info['input'] == output_type
    ]
    
    return compatible

def can_chain_methods(method1, method2):
    """
    Check if method2 can be chained after method1
    
    Args:
        method1: First normalization method
        method2: Second normalization method
    
    Returns:
        Boolean indicating if chaining is valid
    """
    if method1 not in NORMALIZATION_IO_TYPES:
        return True  # Allow if we don't know the method
    
    if method2 not in NORMALIZATION_IO_TYPES:
        return True  # Allow if we don't know the method
    
    output_type = NORMALIZATION_IO_TYPES[method1]['output']
    input_type = NORMALIZATION_IO_TYPES[method2]['input']
    
    return output_type == input_type

def get_method_io_type(method, custom_function=None):
    """Get input/output type information for a method
    
    Args:
        method: The method name
        custom_function: Optional CustomNormalizationFunction instance for custom methods
    """
    if method in NORMALIZATION_IO_TYPES:
        return NORMALIZATION_IO_TYPES[method]
    
    # For custom functions, try to get more specific info
    if method.startswith('CUSTOM_'):
        if custom_function:
            # Determine output columns based on custom function configuration
            output_columns = 'multiple' if custom_function.new_columns and len(custom_function.new_columns) > 1 else 'single'
            
            return {
                'input': custom_function.function_type,  # 'numeric' or 'text'
                'output': custom_function.function_type,  # Custom functions maintain the same type
                'output_columns': output_columns,
                'description': custom_function.description or 'Función personalizada'
            }
        else:
            # Default for custom functions without info
            return {
                'input': 'any',
                'output': 'any',
                'output_columns': 'single',
                'description': 'Función personalizada'
            }
    
    return {
        'input': 'unknown',
        'output': 'unknown',
        'output_columns': 'single',
        'description': 'Método desconocido'
    }

def validate_method_chain(method_chain, initial_column_type='numeric', custom_functions_info=None):
    """
    Validate if a chain of normalization methods is compatible
    
    Args:
        method_chain: List of normalization methods to apply in sequence
        initial_column_type: The type of the original column ('numeric' or 'text')
        custom_functions_info: Optional dict mapping custom function IDs to their info
    
    Returns:
        tuple: (is_valid, error_message, final_output_type)
    """
    if not method_chain:
        return True, None, initial_column_type
    
    current_type = initial_column_type
    current_columns = 'single'
    
    for i, method in enumerate(method_chain):
        # Get custom function info if available
        custom_func = None
        if custom_functions_info and method.startswith('CUSTOM_'):
            try:
                func_id = int(method.replace('CUSTOM_', ''))
                custom_func = custom_functions_info.get(func_id)
            except:
                pass
        
        method_info = get_method_io_type(method, custom_func)
        
        # Check if method can accept current type - now just a warning, not a blocker
        if method_info['input'] not in ['any', 'unknown', current_type]:
            # Instead of blocking, just log a warning and continue
            print(f"Warning: Capa {i+1}: {method} expects input type '{method_info['input']}' but receives type '{current_type}'")
        
        # Check column compatibility
        if current_columns == 'multiple':
            return False, f"Capa {i+1}: Las funciones que generan múltiples columnas no pueden ser seguidas por otras transformaciones", None
        
        # Update current type and columns for next iteration
        if method_info['output'] not in ['any', 'unknown']:
            current_type = method_info['output']
        current_columns = method_info.get('output_columns', 'single')
    
    return True, None, current_type

def detect_column_data_type(series):
    """
    Detect the actual data type of a pandas Series
    
    Args:
        series: pandas Series to analyze
    
    Returns:
        str: 'numeric' or 'text'
    """
    import pandas as pd
    
    # Check if it's numeric
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    
    # Check if it can be converted to numeric
    try:
        pd.to_numeric(series, errors='coerce')
        # If more than 50% of non-null values can be converted, consider it numeric
        numeric_count = pd.to_numeric(series, errors='coerce').notna().sum()
        total_count = series.notna().sum()
        if total_count > 0 and numeric_count / total_count > 0.5:
            return 'numeric'
    except:
        pass
    
    # Otherwise, it's text
    return 'text'