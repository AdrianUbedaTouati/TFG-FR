#!/usr/bin/env python
"""
Test script to verify the layer preview fix for normalization chains.
This script simulates the exact scenario where layer 2 preview should show
the output of layer 1 as its input values.
"""

import os
import sys
import django
import json
from decimal import Decimal

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from ml_trainer.models import Dataset
import pandas as pd

def test_layer_preview():
    # Create test client
    client = Client()
    
    # Login as admin
    try:
        user = User.objects.get(username='admin')
    except User.DoesNotExist:
        user = User.objects.create_superuser('admin', 'admin@test.com', 'admin')
    
    client.force_login(user)
    
    # Get a dataset with text columns
    dataset = Dataset.objects.filter(user=user).first()
    if not dataset:
        print("No dataset found for testing")
        return
    
    print(f"Testing with dataset: {dataset.name} (ID: {dataset.id})")
    
    # Read the dataset
    df = pd.read_csv(dataset.file.path)
    
    # Find a text column
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    if not text_columns:
        print("No text columns found in dataset")
        return
    
    test_column = text_columns[0]
    print(f"\nTesting with column: {test_column}")
    print(f"Sample original values: {df[test_column].unique()[:5].tolist()}")
    
    # Test configuration: Layer 1 = LOWER, Layer 2 = ONE_HOT
    normalization_config = {
        test_column: [
            {
                "method": "LOWER",
                "keep_original": False
            },
            {
                "method": "ONE_HOT", 
                "keep_original": False
            }
        ]
    }
    
    # Make preview request
    response = client.post(
        f'/api/datasets/{dataset.id}/normalization/preview/',
        data=json.dumps({
            'normalization': normalization_config,
            'sample_size': 100,
            'show_steps': True
        }),
        content_type='application/json'
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.json())
        return
    
    data = response.json()
    
    # Check transformation steps
    if 'transformation_steps' in data and test_column in data['transformation_steps']:
        steps = data['transformation_steps'][test_column]
        print(f"\nTransformation steps found: {len(steps)}")
        
        for i, step in enumerate(steps):
            print(f"\nStep {i+1} ({step['method']}):")
            print(f"  Before values (first 5): {step['before'][:5]}")
            print(f"  After values (first 5): {step['after'][:5]}")
    
    # Check unique mapping
    if 'preview' in data and test_column in data['preview']:
        preview = data['preview'][test_column]
        if 'unique_mapping' in preview:
            mapping = preview['unique_mapping']
            print(f"\nUnique mapping found: {len(mapping)} values")
            print("First 5 mappings:")
            for item in mapping[:5]:
                print(f"  {item['original']} -> {item['normalized']}")
            
            # Verify that the "original" values in the mapping are lowercase
            # (output of layer 1) not the original uppercase values
            original_values = [item['original'] for item in mapping[:5]]
            if all(val.islower() for val in original_values if isinstance(val, str)):
                print("\n✓ SUCCESS: Layer 2 preview shows lowercase values (output of Layer 1)")
            else:
                print("\n✗ FAILURE: Layer 2 preview still shows original values")
                print(f"  Expected lowercase values, got: {original_values}")

if __name__ == "__main__":
    test_layer_preview()