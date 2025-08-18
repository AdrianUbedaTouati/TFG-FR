#!/usr/bin/env python
"""
Test script to verify NaN serialization fixes in DatasetColumnsView
"""

import os
import sys
import django
import json
import pandas as pd
import numpy as np
from io import StringIO

# Add the project to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from ml_trainer.models import Dataset
from django.core.files.base import ContentFile

User = get_user_model()

def test_nan_serialization():
    """Test that NaN values are properly handled in dataset columns endpoint"""
    print("Testing NaN serialization in DatasetColumnsView...")
    
    # Create test user
    user = User.objects.filter(username='test_user').first()
    if not user:
        user = User.objects.create_user(username='test_user', password='test_pass')
    
    # Create test dataset with NaN values
    csv_content = """col1,col2,col3,col4
1.5,A,2020-01-01,10
2.5,,2020-01-02,20
,C,,30
4.5,D,2020-01-04,
5.5,E,2020-01-05,50
,F,,60
7.5,G,2020-01-07,70
8.5,H,2020-01-08,80
9.5,I,2020-01-09,90
10.5,J,2020-01-10,100"""
    
    dataset = None
    try:
        # Create dataset
        dataset = Dataset.objects.create(
            name='test_nan_dataset',
            user=user,
            description='Test dataset with NaN values'
        )
        dataset.file.save('test_nan.csv', ContentFile(csv_content.encode('utf-8')))
        
        # Test API endpoint
        client = Client()
        client.force_login(user)
        
        response = client.get(f'/api/datasets/{dataset.id}/columns/')
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("✓ Response is valid JSON")
                
                # Check preview data
                preview = data.get('data', {}).get('preview', {})
                print(f"\nPreview data:")
                for col, values in preview.items():
                    print(f"  {col}: {values}")
                    # Verify no NaN values, only None
                    for val in values:
                        if isinstance(val, float) and np.isnan(val):
                            print(f"✗ Found NaN in {col}: {val}")
                            return False
                
                print("\n✓ All NaN values properly converted to None")
                return True
                
            except json.JSONDecodeError as e:
                print(f"✗ JSON decode error: {e}")
                print(f"Response content: {response.content}")
                return False
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"Response: {response.content}")
            return False
        
    finally:
        # Cleanup
        if dataset:
            dataset.delete()

if __name__ == '__main__':
    try:
        success = test_nan_serialization()
        if success:
            print("\n✓ NaN serialization test passed!")
        else:
            print("\n✗ NaN serialization test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)