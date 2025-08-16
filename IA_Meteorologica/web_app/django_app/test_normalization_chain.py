#!/usr/bin/env python3
"""
Test script for chained normalization: ONE_HOT -> MIN_MAX
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

import pandas as pd
from ml_trainer.views.normalization_views import DatasetNormalizationView

def test_one_hot_to_minmax():
    """Test applying MIN_MAX after ONE_HOT encoding"""
    
    # Create sample data with text column
    data = {
        'Summary': ['Clear', 'Cloudy', 'Rain', 'Clear', 'Cloudy', 'Snow', 'Rain', 'Clear'],
        'Temperature': [20, 15, 10, 22, 16, -5, 12, 21]
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print(f"\nSummary dtype: {df['Summary'].dtype}")
    print(f"Unique values: {df['Summary'].unique()}")
    
    # Create normalization config for chained normalization
    config = {
        'Summary': [
            {'method': 'ONE_HOT', 'keep_original': False},  # Convert text to numeric
            {'method': 'MIN_MAX', 'keep_original': False}   # Then scale to [0, 1]
        ]
    }
    
    print("\nNormalization config:")
    print(config)
    
    # Apply normalization
    view = DatasetNormalizationView()
    
    try:
        normalized_df = view._apply_normalization(df, config)
        
        print("\nNormalized DataFrame:")
        print(normalized_df)
        print(f"\nSummary dtype after normalization: {normalized_df['Summary'].dtype}")
        print(f"Summary values after normalization:")
        print(normalized_df['Summary'].value_counts().sort_index())
        
        # Verify the transformation
        print("\nVerification:")
        print("- ONE_HOT should convert: Clear->0, Cloudy->1, Rain->2, Snow->3")
        print("- MIN_MAX should then scale [0,3] to [0,1]")
        print("- Expected: Clear->0.0, Cloudy->0.333, Rain->0.667, Snow->1.0")
        
        # Check if values are in expected range
        min_val = normalized_df['Summary'].min()
        max_val = normalized_df['Summary'].max()
        print(f"\nActual range: [{min_val}, {max_val}]")
        
        if min_val >= 0 and max_val <= 1:
            print("✓ Success: Values are in [0, 1] range as expected")
        else:
            print("✗ Error: Values are not in expected range")
            
    except Exception as e:
        print(f"\n✗ Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_one_hot_to_minmax()