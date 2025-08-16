"""
Type conversion utilities for normalization pipeline
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class TypeConversionWarning:
    """Container for type conversion warnings"""
    def __init__(self, from_type: str, to_type: str, warning_msg: str):
        self.from_type = from_type
        self.to_type = to_type
        self.warning_msg = warning_msg
    
    def __str__(self):
        return f"Conversión de {self.from_type} a {self.to_type}: {self.warning_msg}"


def can_convert_safely(from_dtype, to_dtype) -> Tuple[bool, Optional[str]]:
    """
    Check if conversion between data types is safe
    
    Returns:
        tuple: (can_convert, warning_message)
    """
    # String representation of dtypes for easier comparison
    from_str = str(from_dtype).lower()
    to_str = str(to_dtype).lower()
    
    # Safe conversions (no data loss)
    safe_conversions = {
        ('int8', 'int16'), ('int8', 'int32'), ('int8', 'int64'),
        ('int16', 'int32'), ('int16', 'int64'),
        ('int32', 'int64'),
        ('uint8', 'uint16'), ('uint8', 'uint32'), ('uint8', 'uint64'),
        ('uint16', 'uint32'), ('uint16', 'uint64'),
        ('uint32', 'uint64'),
        ('float16', 'float32'), ('float16', 'float64'),
        ('float32', 'float64'),
        # Int to float is safe (no precision loss for reasonable values)
        ('int8', 'float32'), ('int8', 'float64'),
        ('int16', 'float32'), ('int16', 'float64'),
        ('int32', 'float64'),
        ('uint8', 'float32'), ('uint8', 'float64'),
        ('uint16', 'float32'), ('uint16', 'float64'),
        ('uint32', 'float64'),
        # Special pandas nullable int to float
        ('int64', 'float64'), ('int32', 'float32'),
    }
    
    # Check if it's a safe conversion
    for safe_from, safe_to in safe_conversions:
        if safe_from in from_str and safe_to in to_str:
            return True, None
    
    # Conversions with potential data loss
    warning_conversions = {
        ('float', 'int'): "Se perderán los decimales al convertir de float a int",
        ('int64', 'int32'): "Posible pérdida de precisión si los valores son muy grandes",
        ('float64', 'float32'): "Posible pérdida de precisión en valores muy pequeños o grandes",
        ('object', 'float'): "Se intentará convertir texto a números, valores no numéricos serán NaN",
        ('object', 'int'): "Se intentará convertir texto a enteros, valores no numéricos serán NaN",
    }
    
    for (warn_from, warn_to), warning_msg in warning_conversions.items():
        if warn_from in from_str and warn_to in to_str:
            return True, warning_msg
    
    # Same type conversions
    if from_dtype == to_dtype:
        return True, None
    
    # Default: allow conversion but warn about potential issues
    return True, f"Conversión de {from_dtype} a {to_dtype} puede causar cambios en los datos"


def convert_series_dtype(series: pd.Series, target_dtype: str, 
                        force: bool = False) -> Tuple[pd.Series, Optional[TypeConversionWarning]]:
    """
    Convert a pandas Series to target dtype with safety checks
    
    Args:
        series: Input series
        target_dtype: Target data type ('numeric', 'text', or specific dtype)
        force: If True, force conversion even if unsafe
    
    Returns:
        tuple: (converted_series, warning)
    """
    current_dtype = series.dtype
    warning = None
    
    # Handle high-level type specifications
    if target_dtype == 'numeric':
        # For numeric target, prefer float64 to handle all numeric types
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                # Convert integers to float for normalization
                target_dtype = 'float64'
            else:
                # Already float, keep as is
                return series, None
        else:
            # Try to convert to numeric
            target_dtype = 'float64'
    elif target_dtype == 'text':
        target_dtype = 'object'
    
    # Check if conversion is needed
    if str(current_dtype) == str(target_dtype):
        return series, None
    
    # Check conversion safety
    can_convert, warning_msg = can_convert_safely(current_dtype, target_dtype)
    
    if not can_convert and not force:
        raise ValueError(f"Conversión insegura de {current_dtype} a {target_dtype}")
    
    if warning_msg:
        warning = TypeConversionWarning(str(current_dtype), str(target_dtype), warning_msg)
    
    # Perform conversion
    try:
        if target_dtype in ['float64', 'float32', 'float16']:
            # Use pd.to_numeric for better handling of mixed types
            converted = pd.to_numeric(series, errors='coerce')
            if target_dtype != 'float64':
                converted = converted.astype(target_dtype)
        else:
            converted = series.astype(target_dtype)
        
        return converted, warning
    except Exception as e:
        if force:
            # If forced, try harder
            if 'float' in str(target_dtype):
                return pd.to_numeric(series, errors='coerce'), warning
            elif 'int' in str(target_dtype):
                return pd.to_numeric(series, errors='coerce').fillna(0).astype(target_dtype), warning
            else:
                return series.astype(target_dtype), warning
        else:
            raise ValueError(f"Error al convertir de {current_dtype} a {target_dtype}: {str(e)}")


def ensure_numeric_compatibility(series: pd.Series) -> Tuple[pd.Series, Optional[TypeConversionWarning]]:
    """
    Ensure a series is compatible with numeric operations
    Converts integers to float to avoid sklearn issues
    """
    if pd.api.types.is_integer_dtype(series):
        return convert_series_dtype(series, 'float64')
    elif pd.api.types.is_numeric_dtype(series):
        return series, None
    else:
        # Try to convert to numeric
        return convert_series_dtype(series, 'float64', force=True)