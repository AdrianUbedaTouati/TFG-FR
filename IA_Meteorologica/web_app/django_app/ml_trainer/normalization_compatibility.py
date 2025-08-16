"""
Normalization method compatibility definitions
"""

# Define input/output types for each normalization method
NORMALIZATION_IO_TYPES = {
    # Numeric methods
    'MIN_MAX': {
        'input': 'numeric',
        'output': 'numeric',
        'description': 'Escala valores numéricos al rango [0, 1]'
    },
    'Z_SCORE': {
        'input': 'numeric',
        'output': 'numeric',
        'description': 'Estandariza valores numéricos a media 0 y desviación 1'
    },
    'LSTM_TCN': {
        'input': 'numeric',
        'output': 'numeric',
        'description': 'Escala valores numéricos al rango [-1, 1]'
    },
    'CNN': {
        'input': 'numeric',
        'output': 'numeric',
        'description': 'Normalización Z-Score para CNN'
    },
    'TRANSFORMER': {
        'input': 'numeric',
        'output': 'numeric',
        'description': 'RobustScaler para valores numéricos'
    },
    'TREE': {
        'input': 'numeric',
        'output': 'numeric',
        'description': 'Sin transformación (para modelos de árbol)'
    },
    
    # Text methods
    'LOWER': {
        'input': 'text',
        'output': 'text',
        'description': 'Convierte texto a minúsculas'
    },
    'STRIP': {
        'input': 'text',
        'output': 'text',
        'description': 'Elimina espacios al inicio y final del texto'
    },
    'ONE_HOT': {
        'input': 'text',
        'output': 'numeric',
        'description': 'Convierte categorías de texto a códigos numéricos'
    },
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

def get_method_io_type(method):
    """Get input/output type information for a method"""
    if method in NORMALIZATION_IO_TYPES:
        return NORMALIZATION_IO_TYPES[method]
    
    # For custom functions, return generic info
    if method.startswith('CUSTOM_'):
        return {
            'input': 'any',
            'output': 'any',
            'description': 'Función personalizada'
        }
    
    return {
        'input': 'unknown',
        'output': 'unknown',
        'description': 'Método desconocido'
    }