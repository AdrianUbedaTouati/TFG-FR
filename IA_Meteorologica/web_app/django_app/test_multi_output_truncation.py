#!/usr/bin/env python
"""Test script to verify multi-output truncation functionality"""

import os
import sys
import django
import json

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.models import Dataset
from django.contrib.auth.models import User
from ml_trainer.views.normalization_views import DatasetNormalizationView

def test_multi_output_truncation():
    """Test that multi-output truncation works correctly"""
    
    # Get a test dataset
    try:
        dataset = Dataset.objects.filter(name__icontains="weather").first()
        if not dataset:
            dataset = Dataset.objects.first()
        
        if not dataset:
            print("No datasets found in the database!")
            return
        
        print(f"Using dataset: {dataset.name} (ID: {dataset.id})")
        
        # Create a test normalization config with multi-output conversion
        normalization_config = {
            "Formatted Date": {
                "layers": [
                    {
                        "method": "date_extraction",
                        "keep_original": False
                    }
                ],
                "output_conversions": [
                    {
                        "column": "Formatted Date_year",
                        "conversion": "TRUNCATE",
                        "params": {"decimals": 0}
                    },
                    {
                        "column": "Formatted Date_month", 
                        "conversion": "TRUNCATE",
                        "params": {"decimals": -1}
                    }
                ]
            }
        }
        
        print("\nNormalization config:")
        print(json.dumps(normalization_config, indent=2))
        
        # Create view instance
        view = DatasetNormalizationView()
        
        # Load dataset
        import pandas as pd
        df = pd.read_csv(dataset.file.path)
        print(f"\nOriginal dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Apply normalization
        print("\nApplying normalization...")
        normalized_df = view._apply_normalization(df, normalization_config)
        
        print(f"\nNormalized dataset shape: {normalized_df.shape}")
        print(f"Columns: {list(normalized_df.columns)}")
        
        # Check if date extraction worked
        if "Formatted Date_year" in normalized_df.columns:
            print(f"\nYear column sample:")
            print(normalized_df["Formatted Date_year"].head(10))
        
        if "Formatted Date_month" in normalized_df.columns:
            print(f"\nMonth column sample:")
            print(normalized_df["Formatted Date_month"].head(10))
        
        # Check for warnings
        if hasattr(view, 'conversion_warnings'):
            print(f"\nConversion warnings: {view.conversion_warnings}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_output_truncation()