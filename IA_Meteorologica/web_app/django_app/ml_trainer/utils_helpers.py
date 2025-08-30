"""
Helper functions for safe JSON serialization
"""
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Union


def safe_to_list(values: Any) -> List[Any]:
    """
    Safely convert values to list, replacing NaN with None for JSON serialization
    
    Args:
        values: Values to convert (Series, array, list, etc.)
        
    Returns:
        List with NaN values replaced by None
    """
    if hasattr(values, 'tolist'):
        # For numpy arrays or pandas Series
        result = values.tolist()
    elif isinstance(values, list):
        result = values
    else:
        # Try to convert to list
        try:
            result = list(values)
        except:
            return []
    
    # Replace NaN values with None
    return [None if pd.isna(val) else val for val in result]


def safe_float(value: Any) -> Union[float, None]:
    """
    Safely convert value to float, returning None if NaN
    
    Args:
        value: Value to convert
        
    Returns:
        Float value or None if NaN
    """
    if pd.isna(value):
        return None
    try:
        return float(value)
    except:
        return None


def safe_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively clean dictionary values, replacing NaN with None
    
    Args:
        data: Dictionary to clean
        
    Returns:
        Cleaned dictionary safe for JSON serialization
    """
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned[key] = safe_dict_values(value)
        elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            cleaned[key] = safe_to_list(value)
        elif isinstance(value, (float, np.floating)):
            cleaned[key] = safe_float(value)
        elif pd.isna(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned