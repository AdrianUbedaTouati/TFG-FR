"""
Type conversions for normalization pipeline - Data type conversions only
"""

# Define available type conversions
TYPE_CONVERSIONS = {
    'numeric_to_numeric': [
        {
            'value': 'TO_FLOAT64',
            'label': 'Convertir a Float64',
            'description': 'Convierte a números decimales de doble precisión',
            'input': 'numeric',
            'output': 'numeric'
        },
        {
            'value': 'TO_FLOAT32',
            'label': 'Convertir a Float32',
            'description': 'Convierte a números decimales de precisión simple',
            'input': 'numeric',
            'output': 'numeric'
        },
        {
            'value': 'TO_INT64',
            'label': 'Convertir a Int64',
            'description': 'Convierte a enteros de 64 bits (pierde decimales)',
            'input': 'numeric',
            'output': 'numeric'
        },
        {
            'value': 'TO_INT32',
            'label': 'Convertir a Int32',
            'description': 'Convierte a enteros de 32 bits (pierde decimales)',
            'input': 'numeric',
            'output': 'numeric'
        },
        {
            'value': 'TO_INT16',
            'label': 'Convertir a Int16',
            'description': 'Convierte a enteros de 16 bits (pierde decimales)',
            'input': 'numeric',
            'output': 'numeric'
        }
    ],
    'numeric_to_text': [
        {
            'value': 'TO_STRING',
            'label': 'Convertir a Texto',
            'description': 'Convierte números a su representación en texto',
            'input': 'numeric',
            'output': 'text'
        }
    ],
    'text_to_numeric': [
        {
            'value': 'TO_NUMERIC',
            'label': 'Convertir a Numérico',
            'description': 'Intenta convertir texto a números (valores no numéricos serán NaN)',
            'input': 'text',
            'output': 'numeric'
        }
    ],
    'text_to_text': [
        {
            'value': 'TO_STRING',
            'label': 'Asegurar tipo Texto',
            'description': 'Convierte cualquier valor a texto',
            'input': 'text',
            'output': 'text'
        }
    ]
}

def get_conversion_options(from_type, to_type):
    """
    Get available conversion options based on input and output types
    
    Args:
        from_type: 'numeric' or 'text'
        to_type: 'numeric' or 'text'
    
    Returns:
        List of conversion options
    """
    key = f"{from_type}_to_{to_type}"
    return TYPE_CONVERSIONS.get(key, [])