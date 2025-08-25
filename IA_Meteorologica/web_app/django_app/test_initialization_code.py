#!/usr/bin/env python
"""
Test script to verify the initialization code functionality for custom normalization functions
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_initialization_code_simulation():
    """
    Simulate the initialization code execution like it would work in the custom function system
    """
    print("\n=== Testing Initialization Code Functionality ===")
    
    # Create sample data (simulating column_data)
    column_data = pd.Series([
        'apple', 'banana', 'apple', 'cherry', 'banana', 'apple', 'date', 'cherry'
    ], name='fruits')
    
    print(f"Sample data (column_data):\n{column_data}")
    
    # Test 1: Category mapping initialization code
    print("\n--- Test 1: Category Mapping ---")
    
    # Simulate initialization code
    initialization_code = """
# Crear mapeo de categorías únicas
unique_categories = sorted(column_data.unique())
category_map = {cat: i for i, cat in enumerate(unique_categories)}
reverse_map = {i: cat for cat, i in category_map.items()}

print(f"Unique categories: {unique_categories}")
print(f"Category mapping: {category_map}")
"""
    
    # Simulate function code
    function_code = """
def normalize(value):
    # Usar las variables globales creadas en la inicialización
    if value in category_map:
        return category_map[value]
    else:
        return -1  # Unknown category
"""
    
    # Create safe execution environment
    safe_globals = {
        '__builtins__': {
            'len': len,
            'str': str,
            'int': int,
            'enumerate': enumerate,
            'sorted': sorted,
            'print': print,
            'dict': dict,
            'list': list,
        },
        'column_data': column_data
    }
    
    # Execute initialization code
    print("Executing initialization code...")
    exec(initialization_code, safe_globals)
    
    # Execute function code
    print("\nExecuting function code...")
    exec(function_code, safe_globals)
    
    # Test the normalize function
    normalize_func = safe_globals['normalize']
    
    print("\nTesting normalize function with sample values:")
    test_values = ['apple', 'banana', 'cherry', 'unknown']
    
    for value in test_values:
        result = normalize_func(value)
        print(f"  normalize('{value}') = {result}")
    
    print("\n✓ Test 1 passed!")

def test_statistical_initialization():
    """
    Test initialization code that calculates statistics
    """
    print("\n--- Test 2: Statistical Variables ---")
    
    # Create numerical sample data
    column_data = pd.Series([1.2, 2.8, 3.5, 4.1, 5.7, 2.3, 3.9, 4.6], name='values')
    
    print(f"Sample numerical data:\n{column_data}")
    
    # Initialization code for statistics
    initialization_code = """
# Calcular estadísticas globales
global_mean = column_data.mean()
global_std = column_data.std()
global_min = column_data.min()
global_max = column_data.max()

print(f"Global statistics:")
print(f"  Mean: {global_mean:.3f}")
print(f"  Std:  {global_std:.3f}")
print(f"  Min:  {global_min:.3f}")
print(f"  Max:  {global_max:.3f}")
"""
    
    # Function code using the statistics
    function_code = """
def normalize(value):
    # Z-score normalization using global variables
    return (value - global_mean) / global_std
"""
    
    # Create safe execution environment
    safe_globals = {
        '__builtins__': {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'print': print,
        },
        'column_data': column_data
    }
    
    # Execute initialization code
    print("Executing statistical initialization code...")
    exec(initialization_code, safe_globals)
    
    # Execute function code
    print("\nExecuting function code...")
    exec(function_code, safe_globals)
    
    # Test the normalize function
    normalize_func = safe_globals['normalize']
    
    print("\nTesting normalize function with sample values:")
    test_values = [1.2, 3.5, 5.7]
    
    for value in test_values:
        result = normalize_func(value)
        print(f"  normalize({value}) = {result:.3f}")
    
    # Verify the results manually
    mean = column_data.mean()
    std = column_data.std()
    expected = (3.5 - mean) / std
    actual = normalize_func(3.5)
    
    assert abs(expected - actual) < 1e-10, f"Expected {expected}, got {actual}"
    
    print("\n✓ Test 2 passed!")

def test_multi_output_with_initialization():
    """
    Test initialization code with a multi-output function
    """
    print("\n--- Test 3: Multi-output with Initialization ---")
    
    # Create datetime-like sample data
    column_data = pd.Series([
        '2023-01-15', '2023-06-20', '2023-12-31', '2023-03-10'
    ], name='dates')
    
    print(f"Sample date data:\n{column_data}")
    
    # Initialization code that prepares date processing
    initialization_code = """
# Convertir a datetime y extraer información global
import pandas as pd
from datetime import datetime

date_series = pd.to_datetime(column_data)
year_range = (date_series.dt.year.min(), date_series.dt.year.max())
month_counts = date_series.dt.month.value_counts().to_dict()

print(f"Year range: {year_range}")
print(f"Month distribution: {month_counts}")
"""
    
    # Function code that uses the global date information
    function_code = """
def normalize(value):
    import pandas as pd
    
    # Convert single value to datetime
    dt = pd.to_datetime(value)
    
    # Extract components using global information
    year_normalized = (dt.year - year_range[0]) / max(1, year_range[1] - year_range[0])
    
    return {
        'year_normalized': year_normalized,
        'month': dt.month,
        'day': dt.day,
        'is_common_month': dt.month in month_counts and month_counts[dt.month] > 1
    }
"""
    
    # Create safe execution environment
    safe_globals = {
        '__builtins__': {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'max': max,
            'print': print,
            'dict': dict,
            '__import__': __import__,
        },
        'column_data': column_data
    }
    
    # Add pandas
    import pandas as pd
    from datetime import datetime
    safe_globals['pd'] = pd
    safe_globals['datetime'] = datetime
    
    # Execute initialization code
    print("Executing date initialization code...")
    exec(initialization_code, safe_globals)
    
    # Execute function code
    print("\nExecuting multi-output function code...")
    exec(function_code, safe_globals)
    
    # Test the normalize function
    normalize_func = safe_globals['normalize']
    
    print("\nTesting normalize function with sample date:")
    test_value = '2023-06-20'
    result = normalize_func(test_value)
    
    print(f"  normalize('{test_value}') = {result}")
    
    # Verify the result is a dictionary with expected keys
    expected_keys = {'year_normalized', 'month', 'day', 'is_common_month'}
    assert set(result.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(result.keys())}"
    assert isinstance(result['year_normalized'], float), "year_normalized should be float"
    assert isinstance(result['month'], int), "month should be int"
    assert isinstance(result['day'], int), "day should be int"
    assert isinstance(result['is_common_month'], bool), "is_common_month should be bool"
    
    print("\n✓ Test 3 passed!")

if __name__ == '__main__':
    print("Testing Initialization Code Functionality")
    print("=" * 50)
    
    try:
        test_initialization_code_simulation()
        test_statistical_initialization()
        test_multi_output_with_initialization()
        
        print("\n\n" + "=" * 50)
        print("✅ All initialization code tests passed successfully!")
        print("=" * 50)
        print("\nKey features tested:")
        print("- Global variable creation in initialization code")
        print("- Access to column_data variable")
        print("- Variable persistence between initialization and main function")
        print("- Statistical calculations")
        print("- Category mapping")
        print("- Multi-output functions with initialization")
        
    except AssertionError as e:
        print(f"\n\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)