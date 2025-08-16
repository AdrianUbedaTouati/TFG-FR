#!/usr/bin/env python3
"""
Simple test for normalization methods
"""
import pandas as pd
import sys
import os

# Add the django app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_trainer.normalization_methods import one_hot_text, min_max_numeric

def test_one_hot_to_minmax():
    """Test applying MIN_MAX after ONE_HOT encoding"""
    
    # Create sample data with text column
    data = {
        'Summary': ['Clear', 'Cloudy', 'Rain', 'Clear', 'Cloudy', 'Snow', 'Rain', 'Clear'],
    }
    df = pd.DataFrame(data)
    
    print("Original Series:")
    print(df['Summary'])
    print(f"dtype: {df['Summary'].dtype}")
    print(f"Unique values: {df['Summary'].unique()}\n")
    
    # Step 1: Apply ONE_HOT
    print("Step 1: Applying ONE_HOT encoding...")
    encoded = one_hot_text(df['Summary'])
    print(f"After ONE_HOT:")
    print(encoded)
    print(f"dtype: {encoded.dtype}")
    print(f"Unique values: {encoded.unique()}\n")
    
    # Step 2: Apply MIN_MAX
    print("Step 2: Applying MIN_MAX normalization...")
    normalized = min_max_numeric(encoded)
    print(f"After MIN_MAX:")
    print(normalized)
    print(f"dtype: {normalized.dtype}")
    print(f"Value range: [{normalized.min()}, {normalized.max()}]\n")
    
    # Show mapping
    print("Final mapping:")
    mapping = pd.DataFrame({
        'Original': df['Summary'],
        'After ONE_HOT': encoded,
        'After MIN_MAX': normalized
    }).drop_duplicates().sort_values('After ONE_HOT')
    print(mapping)

if __name__ == "__main__":
    test_one_hot_to_minmax()