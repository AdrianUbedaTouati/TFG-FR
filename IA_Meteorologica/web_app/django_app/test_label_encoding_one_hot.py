#!/usr/bin/env python
"""
Test script to verify Label Encoding and One-Hot Encoding functions
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_trainer.normalization_methods import label_encoding_text, one_hot_text

def test_label_encoding():
    """Test the label encoding function"""
    print("\n=== Testing Label Encoding ===")
    
    # Create test data
    data = pd.Series(['cat', 'dog', 'cat', 'bird', 'dog', 'cat', np.nan, 'bird'], name='animals')
    print(f"Original data:\n{data}")
    
    # Apply label encoding
    encoded = label_encoding_text(data)
    print(f"\nLabel encoded:\n{encoded}")
    print(f"Type: {type(encoded)}")
    print(f"Dtype: {encoded.dtype}")
    
    # Verify it's a Series with numeric codes
    assert isinstance(encoded, pd.Series), "Label encoding should return a Series"
    assert pd.api.types.is_integer_dtype(encoded) or encoded.dtype == 'Int64', "Label encoding should return integer codes"
    
    # Check unique values
    unique_values = encoded.dropna().unique()
    print(f"\nUnique encoded values: {sorted(unique_values)}")
    assert len(unique_values) == 3, "Should have 3 unique codes for 3 categories"
    assert set(unique_values) == {0, 1, 2}, "Codes should be 0, 1, 2"
    
    print("\n✓ Label Encoding test passed!")

def test_one_hot_encoding():
    """Test the one-hot encoding function"""
    print("\n=== Testing One-Hot Encoding ===")
    
    # Create test data
    data = pd.Series(['cat', 'dog', 'cat', 'bird', 'dog', 'cat', np.nan, 'bird'], name='animals')
    print(f"Original data:\n{data}")
    
    # Apply one-hot encoding
    encoded = one_hot_text(data)
    print(f"\nOne-hot encoded:\n{encoded}")
    print(f"Type: {type(encoded)}")
    print(f"Shape: {encoded.shape}")
    print(f"Columns: {list(encoded.columns)}")
    
    # Verify it's a DataFrame with multiple columns
    assert isinstance(encoded, pd.DataFrame), "One-hot encoding should return a DataFrame"
    assert encoded.shape[1] == 3, "Should have 3 columns for 3 categories"
    
    # Check column names
    expected_cols = ['animals_bird', 'animals_cat', 'animals_dog']
    assert list(encoded.columns) == expected_cols, f"Expected columns {expected_cols}, got {list(encoded.columns)}"
    
    # Check values are binary
    for col in encoded.columns:
        unique_vals = encoded[col].unique()
        assert set(unique_vals).issubset({0, 1}), f"Column {col} should only contain 0 and 1"
    
    # Check each row has exactly one 1 (except NaN rows)
    row_sums = encoded.sum(axis=1)
    non_nan_mask = data.notna()
    assert (row_sums[non_nan_mask] == 1).all(), "Each non-NaN row should have exactly one 1"
    
    print("\n✓ One-Hot Encoding test passed!")

def test_integration():
    """Test integration with a sample dataset"""
    print("\n=== Testing Integration ===")
    
    # Create a sample dataset
    df = pd.DataFrame({
        'city': ['New York', 'London', 'Paris', 'New York', 'London', 'Paris', 'Berlin'],
        'weather': ['sunny', 'rainy', 'cloudy', 'rainy', 'sunny', 'cloudy', 'sunny'],
        'temperature': [25.5, 15.2, 18.7, 22.1, 16.8, 19.3, 20.5]
    })
    
    print("Original DataFrame:")
    print(df)
    
    # Test Label Encoding on 'city' column
    print("\n\nApplying Label Encoding to 'city' column:")
    df['city_label'] = label_encoding_text(df['city'])
    print(df[['city', 'city_label']])
    
    # Test One-Hot Encoding on 'weather' column
    print("\n\nApplying One-Hot Encoding to 'weather' column:")
    weather_one_hot = one_hot_text(df['weather'])
    print("One-hot encoded columns:")
    print(weather_one_hot)
    
    # Combine with original dataframe
    df_with_one_hot = pd.concat([df, weather_one_hot], axis=1)
    print("\n\nFinal DataFrame with both encodings:")
    print(df_with_one_hot)
    
    print("\n✓ Integration test passed!")

if __name__ == '__main__':
    print("Testing Label Encoding and One-Hot Encoding functions")
    print("=" * 50)
    
    try:
        test_label_encoding()
        test_one_hot_encoding()
        test_integration()
        
        print("\n\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)