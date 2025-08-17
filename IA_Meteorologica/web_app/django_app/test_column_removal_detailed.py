#!/usr/bin/env python
"""
Test script to verify column removal behavior during normalization
"""
import os
import sys
import django
import pandas as pd
import json

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.views.normalization_views import DatasetNormalizationView
from ml_trainer.models import Dataset, CustomNormalizationFunction
from django.contrib.auth.models import User
from django.test import RequestFactory

def test_column_removal():
    print("=== Testing Column Removal During Normalization ===\n")
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'numeric_col': [1.0, 2.0, 3.0, 4.0, 5.0],
        'text_col': ['A', 'B', 'C', 'A', 'B']
    })
    
    print("Original DataFrame:")
    print(test_df)
    print(f"Columns: {list(test_df.columns)}\n")
    
    # Create normalization view instance
    view = DatasetNormalizationView()
    
    # Test 1: Single-layer normalization with keep_original=False
    print("\n=== Test 1: Single-layer normalization (keep_original=False) ===")
    config1 = {
        'numeric_col': {
            'method': 'MIN_MAX',
            'keep_original': False
        }
    }
    result1 = view._apply_normalization(test_df.copy(), config1)
    print(f"Result columns: {list(result1.columns)}")
    print(f"Expected: numeric_col should be transformed in-place")
    print(f"Actual result:\n{result1}\n")
    
    # Test 2: Single-layer normalization with keep_original=True
    print("\n=== Test 2: Single-layer normalization (keep_original=True) ===")
    config2 = {
        'numeric_col': {
            'method': 'MIN_MAX',
            'keep_original': True
        }
    }
    result2 = view._apply_normalization(test_df.copy(), config2)
    print(f"Result columns: {list(result2.columns)}")
    print(f"Expected: Should have both numeric_col and numeric_col_normalized")
    print(f"Actual result:\n{result2}\n")
    
    # Test 3: Multi-layer normalization with all keep_original=False
    print("\n=== Test 3: Multi-layer normalization (all keep_original=False) ===")
    config3 = {
        'numeric_col': [
            {'method': 'MIN_MAX', 'keep_original': False},
            {'method': 'Z_SCORE', 'keep_original': False}
        ]
    }
    result3 = view._apply_normalization(test_df.copy(), config3)
    print(f"Result columns: {list(result3.columns)}")
    print(f"Expected: Original numeric_col should be removed, only transformed column remains")
    print(f"Actual result:\n{result3}\n")
    
    # Test 4: Multi-layer normalization with one keep_original=True
    print("\n=== Test 4: Multi-layer normalization (one keep_original=True) ===")
    config4 = {
        'numeric_col': [
            {'method': 'MIN_MAX', 'keep_original': True},
            {'method': 'Z_SCORE', 'keep_original': False}
        ]
    }
    result4 = view._apply_normalization(test_df.copy(), config4)
    print(f"Result columns: {list(result4.columns)}")
    print(f"Expected: Original numeric_col should be kept plus transformed columns")
    print(f"Actual result:\n{result4}\n")
    
    # Test 5: Text column with ONE_HOT (transforms in-place)
    print("\n=== Test 5: Text column with ONE_HOT (transforms in-place) ===")
    config5 = {
        'text_col': {
            'method': 'ONE_HOT',
            'keep_original': False
        }
    }
    result5 = view._apply_normalization(test_df.copy(), config5)
    print(f"Result columns: {list(result5.columns)}")
    print(f"Expected: text_col should be transformed to numeric codes in-place")
    print(f"Actual result:\n{result5}\n")
    
    # Test 6: Multi-layer with conversion
    print("\n=== Test 6: Multi-layer with conversion ===")
    config6 = {
        'numeric_col': [
            {'method': 'MIN_MAX', 'keep_original': False, 'conversion': 'TO_INT'},
            {'method': 'Z_SCORE', 'keep_original': False}
        ]
    }
    result6 = view._apply_normalization(test_df.copy(), config6)
    print(f"Result columns: {list(result6.columns)}")
    print(f"Expected: Original column removed, only final transformed column")
    print(f"Actual result:\n{result6}\n")
    
    # Debug: Check what happens with column detection
    print("\n=== Debug: Column Detection Logic ===")
    test_df_debug = test_df.copy()
    config_debug = {
        'numeric_col': [
            {'method': 'MIN_MAX', 'keep_original': False},
            {'method': 'Z_SCORE', 'keep_original': False}
        ]
    }
    
    # Manually trace through the logic
    print(f"Initial columns: {list(test_df_debug.columns)}")
    
    # After step 1
    suffix1 = "_step1"
    new_col1 = f"numeric_col{suffix1}"
    print(f"After step 1: Should create {new_col1}")
    
    # After step 2
    suffix2 = "_step2"
    new_col2 = f"numeric_col{suffix2}"
    print(f"After step 2: Should create {new_col2}")
    
    # Check for new columns
    print(f"Pattern for new columns: numeric_col_*")
    print(f"Should find new columns and remove original")
    
    result_debug = view._apply_normalization(test_df_debug, config_debug)
    print(f"\nFinal columns: {list(result_debug.columns)}")
    print(f"Original 'numeric_col' present: {'numeric_col' in result_debug.columns}")

if __name__ == "__main__":
    test_column_removal()