#!/usr/bin/env python
"""
Test script for partial replace functionality in DatasetReplaceValuesView
"""

import os
import sys
import django
import pandas as pd
from io import StringIO

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from django.contrib.auth import get_user_model
from ml_trainer.models import Dataset
from django.core.files.base import ContentFile


def create_test_dataset():
    """Create a test dataset for testing partial replace functionality"""
    # Create test data
    data = {
        'id': [1, 2, 3, 4, 5],
        'code': ['ABC123', 'DEF456', 'GHI789', 'JKL012', 'MNO345'],
        'value': [100.5, 200.3, 300.7, 400.2, 500.9],
        'description': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # Get or create test user
    User = get_user_model()
    user, _ = User.objects.get_or_create(
        username='test_user',
        defaults={'email': 'test@example.com'}
    )
    
    # Create dataset
    dataset = Dataset.objects.create(
        name='test_partial_replace',
        user=user
    )
    
    dataset.file.save(
        'test_partial_replace.csv',
        ContentFile(csv_content.encode('utf-8'))
    )
    
    return dataset


def test_partial_replace():
    """Test the partial replace functionality"""
    # Create test dataset
    dataset = create_test_dataset()
    print(f"Created test dataset: {dataset.name} (ID: {dataset.id})")
    
    # Load the dataset to show initial state
    df_initial = pd.read_csv(dataset.file.path)
    print("\nInitial dataset:")
    print(df_initial)
    
    # Test case 1: Character by character replacement
    print("\n" + "="*50)
    print("Test Case 1: Character by Character Replacement")
    print("Replace characters at positions [0, 2] with 'XY' in rows 0 and 2")
    
    from ml_trainer.views.dataset_views import DatasetReplaceValuesView
    from rest_framework.test import APIRequestFactory
    from django.contrib.auth import get_user_model
    
    User = get_user_model()
    user = User.objects.get(username='test_user')
    
    factory = APIRequestFactory()
    view = DatasetReplaceValuesView()
    
    # Create request for char by char replacement
    request_data = {
        'column_name': 'code',
        'indices': [0, 2],  # Rows to modify
        'new_value': 'XY',
        'partial_replace': True,
        'partial_pattern': [0, 2],  # Character positions to replace
        'partial_type': 'charByChar'
    }
    
    request = factory.post(f'/api/datasets/{dataset.id}/replace-values/', request_data, format='json')
    request.user = user
    
    response = view.post(request, dataset.id)
    print(f"\nResponse status: {response.status_code}")
    print(f"Response data: {response.data}")
    
    # Show modified dataset
    df_modified1 = pd.read_csv(dataset.file.path)
    print("\nDataset after char-by-char replacement:")
    print(df_modified1)
    
    # Test case 2: Complete replacement at positions
    print("\n" + "="*50)
    print("Test Case 2: Complete Replacement at Positions")
    print("Replace positions [1, 2, 3] with 'NEW' in row 1")
    
    request_data = {
        'column_name': 'code',
        'indices': [1],  # Row to modify
        'new_value': 'NEW',
        'partial_replace': True,
        'partial_pattern': [1, 2, 3],  # Character positions to remove and replace
        'partial_type': 'complete'
    }
    
    request = factory.post(f'/api/datasets/{dataset.id}/replace-values/', request_data, format='json')
    request.user = user
    
    response = view.post(request, dataset.id)
    print(f"\nResponse status: {response.status_code}")
    print(f"Response data: {response.data}")
    
    # Show final dataset
    df_final = pd.read_csv(dataset.file.path)
    print("\nFinal dataset:")
    print(df_final)
    
    # Test case 3: Numeric column partial replace
    print("\n" + "="*50)
    print("Test Case 3: Partial Replace on Numeric Column")
    print("Replace first digit (position 0) with '9' in value column for row 3")
    
    request_data = {
        'column_name': 'value',
        'indices': [3],  # Row to modify
        'new_value': '9',
        'partial_replace': True,
        'partial_pattern': [0],  # Replace first character
        'partial_type': 'charByChar'
    }
    
    request = factory.post(f'/api/datasets/{dataset.id}/replace-values/', request_data, format='json')
    request.user = user
    
    response = view.post(request, dataset.id)
    print(f"\nResponse status: {response.status_code}")
    print(f"Response data: {response.data}")
    
    # Show final dataset
    df_final2 = pd.read_csv(dataset.file.path)
    print("\nFinal dataset after numeric partial replace:")
    print(df_final2)
    
    # Clean up
    dataset.delete()
    print("\n" + "="*50)
    print("Test completed and dataset cleaned up.")


if __name__ == '__main__':
    try:
        test_partial_replace()
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()