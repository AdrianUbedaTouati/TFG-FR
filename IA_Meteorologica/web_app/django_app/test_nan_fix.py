#!/usr/bin/env python
"""
Test script to verify NaN values are properly handled in JSON responses
"""
import os
import sys
import django
import json
import numpy as np
import pandas as pd

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.utils import detect_column_type
from ml_trainer.utils_helpers import safe_to_list, safe_float, safe_dict_values


def test_safe_to_list():
    """Test safe_to_list function with various inputs including NaN"""
    print("Testing safe_to_list...")
    
    # Test with numpy array containing NaN
    arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    result = safe_to_list(arr)
    print(f"  Numpy array with NaN: {arr} -> {result}")
    assert result == [1.0, 2.0, None, 4.0, None], f"Expected [1.0, 2.0, None, 4.0, None], got {result}"
    
    # Test with pandas Series containing NaN
    series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    result = safe_to_list(series)
    print(f"  Pandas Series with NaN: {series.tolist()} -> {result}")
    assert result == [1.0, None, 3.0, None, 5.0], f"Expected [1.0, None, 3.0, None, 5.0], got {result}"
    
    # Test with list containing NaN
    lst = [1.0, float('nan'), 3.0]
    result = safe_to_list(lst)
    print(f"  List with NaN: {lst} -> {result}")
    assert result[0] == 1.0 and result[1] is None and result[2] == 3.0, f"Expected [1.0, None, 3.0], got {result}"
    
    print("  ✓ All tests passed!")


def test_safe_float():
    """Test safe_float function with various inputs"""
    print("\nTesting safe_float...")
    
    # Test with normal float
    assert safe_float(3.14) == 3.14, "Normal float failed"
    print("  ✓ Normal float: 3.14 -> 3.14")
    
    # Test with NaN
    assert safe_float(np.nan) is None, "NaN should return None"
    print("  ✓ NaN -> None")
    
    # Test with float('nan')
    assert safe_float(float('nan')) is None, "float('nan') should return None"
    print("  ✓ float('nan') -> None")
    
    # Test with pd.NA
    assert safe_float(pd.NA) is None, "pd.NA should return None"
    print("  ✓ pd.NA -> None")
    
    print("  ✓ All tests passed!")


def test_detect_column_type():
    """Test detect_column_type with series containing NaN"""
    print("\nTesting detect_column_type with NaN values...")
    
    # Create a numeric series with NaN
    series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0])
    result = detect_column_type(series)
    
    print(f"  Series: {series.tolist()}")
    print(f"  Result sample_values: {result['sample_values']}")
    
    # Verify no NaN in sample_values
    assert all(not pd.isna(v) and v is not None for v in result['sample_values'] if v is not None), \
        "sample_values should not contain NaN"
    
    # Verify statistics are properly handled
    assert result['mean'] is not None, "Mean should be calculated"
    assert result['std'] is not None, "Std should be calculated"
    
    # Try to serialize to JSON
    try:
        json_str = json.dumps(result)
        print("  ✓ Result can be serialized to JSON")
    except Exception as e:
        print(f"  ✗ JSON serialization failed: {e}")
        raise
    
    print("  ✓ All tests passed!")


def test_json_serialization():
    """Test that our data structures can be properly serialized to JSON"""
    print("\nTesting JSON serialization...")
    
    # Create a complex data structure with NaN values
    data = {
        'values': [1.0, np.nan, 3.0],
        'stats': {
            'mean': np.nan,
            'std': 2.5,
            'min': 1.0,
            'max': np.nan
        },
        'arrays': {
            'numpy': np.array([1.0, np.nan, 3.0]),
            'pandas': pd.Series([np.nan, 2.0, 3.0])
        }
    }
    
    # Clean the data
    cleaned = safe_dict_values(data)
    
    # Try to serialize
    try:
        json_str = json.dumps(cleaned)
        print("  ✓ Complex data structure serialized successfully")
        
        # Verify the content
        parsed = json.loads(json_str)
        assert parsed['values'] == [1.0, None, 3.0], "Values not properly cleaned"
        assert parsed['stats']['mean'] is None, "NaN mean not converted to None"
        assert parsed['stats']['std'] == 2.5, "Valid float changed"
        print("  ✓ Deserialized data matches expected values")
    except Exception as e:
        print(f"  ✗ JSON serialization failed: {e}")
        raise


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing NaN handling in JSON serialization")
    print("=" * 60)
    
    try:
        test_safe_to_list()
        test_safe_float()
        test_detect_column_type()
        test_json_serialization()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED - NaN values are properly handled!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()