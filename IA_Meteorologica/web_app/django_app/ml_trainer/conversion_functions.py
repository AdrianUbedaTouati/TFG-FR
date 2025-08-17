"""
Implementation of data type conversion functions
"""
import pandas as pd
import numpy as np

# Type conversion functions
def to_float64_conversion(series: pd.Series) -> pd.Series:
    """Convert to float64"""
    return series.astype('float64')

def to_float32_conversion(series: pd.Series) -> pd.Series:
    """Convert to float32"""
    return series.astype('float32')

def to_int64_conversion(series: pd.Series) -> pd.Series:
    """Convert to int64 (truncates decimals)"""
    # Handle NaN values by converting to nullable Int64
    if series.isna().any():
        return series.round().astype('Int64')  # Nullable integer type
    else:
        return series.round().astype('int64')

def to_int32_conversion(series: pd.Series) -> pd.Series:
    """Convert to int32 (truncates decimals)"""
    if series.isna().any():
        # Convert to float first, round, then to nullable int
        return series.round().astype('float32').astype('Int32')
    else:
        return series.round().astype('int32')

def to_int16_conversion(series: pd.Series) -> pd.Series:
    """Convert to int16 (truncates decimals)"""
    if series.isna().any():
        # Convert to float first, round, then to nullable int
        return series.round().astype('float32').astype('Int16')
    else:
        return series.round().astype('int16')

def to_string_conversion(series: pd.Series) -> pd.Series:
    """Convert to string"""
    # Replace 'nan' with empty string for better display
    return series.astype(str).replace('nan', '')

def to_numeric_conversion(series: pd.Series) -> pd.Series:
    """Convert to numeric (non-numeric values become NaN)"""
    return pd.to_numeric(series, errors='coerce')

def truncate_conversion(series: pd.Series, decimals: int = 2) -> pd.Series:
    """Truncate numeric values to N decimal places (without rounding)"""
    if not pd.api.types.is_numeric_dtype(series):
        # Provide more helpful error message
        dtype_str = str(series.dtype)
        sample_values = series.dropna().head(3).tolist()
        sample_str = ', '.join(str(v) for v in sample_values) if sample_values else 'no values'
        # Try to get the series name for better error message
        column_name = series.name if hasattr(series, 'name') and series.name else 'columna desconocida'
        raise ValueError(
            f"La conversión de truncamiento requiere datos numéricos. "
            f"La columna '{column_name}' tiene tipo '{dtype_str}' con valores como: {sample_str}"
        )
    
    # Handle negative decimals (truncate before decimal point)
    if decimals < 0:
        factor = 10 ** (-decimals)
        return (series // factor) * factor
    else:
        factor = 10 ** decimals
        return np.trunc(series * factor) / factor

# Dispatch table
CONVERSION_FUNCTIONS = {
    'TO_FLOAT64': to_float64_conversion,
    'TO_FLOAT32': to_float32_conversion,
    'TO_INT64': to_int64_conversion,
    'TO_INT32': to_int32_conversion,
    'TO_INT16': to_int16_conversion,
    'TO_STRING': to_string_conversion,
    'TO_NUMERIC': to_numeric_conversion,
    'TRUNCATE': truncate_conversion,
}

def apply_conversion(series: pd.Series, conversion_method: str, params: dict = None) -> pd.Series:
    """
    Apply a type conversion to a pandas Series
    
    Args:
        series: Input series
        conversion_method: Name of the conversion method
        params: Optional parameters for the conversion (e.g., decimals for TRUNCATE)
    
    Returns:
        Converted series
    """
    if conversion_method not in CONVERSION_FUNCTIONS:
        raise ValueError(f"Unknown conversion method: {conversion_method}")
    
    func = CONVERSION_FUNCTIONS[conversion_method]
    
    # Handle parameterized conversions
    if conversion_method == 'TRUNCATE' and params:
        decimals = params.get('decimals', 2)
        return func(series, decimals=decimals)
    else:
        return func(series)

def get_conversion_warnings(from_dtype, to_conversion):
    """
    Get warnings for type conversions
    
    Returns:
        str or None: Warning message if conversion may lose data
    """
    warnings = {
        ('float', 'TO_INT'): "Se perderán los decimales al convertir a entero",
        ('int64', 'TO_INT32'): "Valores fuera del rango -2,147,483,648 a 2,147,483,647 causarán overflow",
        ('int64', 'TO_INT16'): "Valores fuera del rango -32,768 a 32,767 causarán overflow",
        ('float64', 'TO_FLOAT32'): "Posible pérdida de precisión en números muy grandes o muy pequeños",
        ('text', 'TO_NUMERIC'): "Valores no numéricos se convertirán a NaN",
    }
    
    # Check for matching warning patterns
    from_str = str(from_dtype).lower()
    
    # Check for float to int conversions
    if 'float' in from_str and to_conversion.startswith('TO_INT'):
        return warnings[('float', 'TO_INT')]
    
    # Check for int64 to smaller int
    if 'int64' in from_str:
        if to_conversion == 'TO_INT32':
            return warnings[('int64', 'TO_INT32')]
        elif to_conversion == 'TO_INT16':
            return warnings[('int64', 'TO_INT16')]
    
    # Check for float64 to float32
    if 'float64' in from_str and to_conversion == 'TO_FLOAT32':
        return warnings[('float64', 'TO_FLOAT32')]
    
    # Check for text to numeric
    if from_str in ['object', 'string'] and to_conversion == 'TO_NUMERIC':
        return warnings[('text', 'TO_NUMERIC')]
    
    return None