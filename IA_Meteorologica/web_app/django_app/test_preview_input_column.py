#!/usr/bin/env python
"""
Test script to verify the preview fix for multi-column transformations
where layer 2 uses an input from layer 1's multi-column output.
"""

import os
import sys
import django
import json

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from django.contrib.auth import get_user_model
from ml_trainer.models import Dataset
from ml_trainer.views.normalization_views import DatasetNormalizationView
import pandas as pd

User = get_user_model()

def test_preview_with_input_column():
    """Test preview when layer 2 uses an input from multi-column transformation"""
    
    # Get or create test user
    user, _ = User.objects.get_or_create(
        username='test_user',
        defaults={'email': 'test@example.com'}
    )
    
    # Create a test dataset with a date column
    test_data = pd.DataFrame({
        'Formatted Date': ['2023-01-15', '2023-02-20', '2023-03-25', '2023-04-30'],
        'Temperature': [20.5, 22.3, 19.8, 21.0]
    })
    
    # Save as CSV
    csv_path = '/tmp/test_preview_input.csv'
    test_data.to_csv(csv_path, index=False)
    
    # Create dataset
    dataset = Dataset.objects.create(
        name='Test Preview Input Column',
        file_path=csv_path,
        user=user
    )
    
    print("Test Dataset Created")
    print("=" * 50)
    
    # Test normalization config with:
    # Layer 1: Date extraction (creates multiple columns including _day)
    # Layer 2: One-hot encoding on the _day column
    normalization_config = {
        'Formatted Date': [
            {
                'method': 'date_extraction',
                'keep_original': True
            },
            {
                'method': 'one_hot',
                'keep_original': False,
                'input_column': 'Formatted Date_day'  # This uses output from layer 1
            }
        ]
    }
    
    print("\nNormalization Config:")
    print(json.dumps(normalization_config, indent=2))
    print("\n" + "=" * 50)
    
    # Create view instance
    view = DatasetNormalizationView()
    
    # Simulate the preview request
    class MockRequest:
        def __init__(self, data):
            self.data = data
            self.user = user
    
    request_data = {
        'dataset_id': dataset.id,
        'normalization_config': normalization_config,
        'show_steps': True  # This triggers the multi-layer preview logic
    }
    
    request = MockRequest(request_data)
    
    try:
        # Call the preview method
        print("\nCalling preview_normalization...")
        response = view.preview_normalization(request)
        
        if response.status_code == 200:
            print("\nPreview successful!")
            data = response.data
            
            # Check the preview data
            if 'preview' in data and 'Formatted Date' in data['preview']:
                preview = data['preview']['Formatted Date']
                
                print("\nUnique Mapping (first 5 entries):")
                if 'unique_mapping' in preview:
                    for i, mapping in enumerate(preview['unique_mapping'][:5]):
                        print(f"  {mapping['original']} -> {mapping['normalized']}")
                    
                    # Check if we have null values
                    null_count = sum(1 for m in preview['unique_mapping'] if m['normalized'] is None or str(m['normalized']) == 'None')
                    total_count = len(preview['unique_mapping'])
                    
                    print(f"\nNull values in mapping: {null_count}/{total_count}")
                    
                    if null_count == 0:
                        print("✓ SUCCESS: No null values in the preview mapping!")
                    else:
                        print("✗ FAILED: Still have null values in the preview mapping")
                else:
                    print("✗ No unique_mapping found in preview")
                
                # Show transformation steps
                if 'transformation_steps' in data:
                    print("\nTransformation Steps:")
                    for col, steps in data['transformation_steps'].items():
                        print(f"\n  Column: {col}")
                        for i, step in enumerate(steps):
                            print(f"    Step {i+1}: {step.get('method', 'Unknown')}")
                            if 'new_columns' in step:
                                print(f"      New columns: {step['new_columns']}")
            else:
                print("✗ No preview data found in response")
                
        else:
            print(f"\n✗ Preview failed with status {response.status_code}")
            print(f"Error: {response.data}")
            
    except Exception as e:
        print(f"\n✗ Exception during preview: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        dataset.delete()
        if os.path.exists(csv_path):
            os.remove(csv_path)
        print("\nTest cleanup completed")


if __name__ == '__main__':
    test_preview_with_input_column()