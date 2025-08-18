#!/usr/bin/env python
"""
Test script to verify the DatasetReplaceValuesView implementation
Tests numeric validation, char replacement, and NaN prevention
"""

import os
import sys
import django
import pandas as pd
import numpy as np
from io import StringIO

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from django.test import RequestFactory
from django.contrib.auth import get_user_model
from ml_trainer.models import Dataset
from ml_trainer.views.dataset_views import DatasetReplaceValuesView
from django.core.files.base import ContentFile

User = get_user_model()

def create_test_dataset():
    """Create a test dataset with mixed data types"""
    data = {
        'numeric_col': [1.5, 2.0, 3.5, 4.0, 5.5],
        'int_col': [10, 20, 30, 40, 50],
        'string_col': ['apple', 'banana', 'cherry', 'date', 'elderberry'],
        'mixed_col': ['100.5', '200.0', 'text300', '400.5', 'value500']
    }
    
    df = pd.DataFrame(data)
    
    # Create CSV content
    csv_content = df.to_csv(index=False)
    
    # Get or create test user
    user, _ = User.objects.get_or_create(
        username='test_user',
        defaults={'email': 'test@example.com', 'is_staff': True}
    )
    
    # Create dataset
    dataset = Dataset.objects.create(
        name='test_replace_values',
        user=user,
        description='Test dataset for replace values functionality'
    )
    
    # Save the file
    dataset.file.save(
        'test_replace_values.csv',
        ContentFile(csv_content.encode('utf-8'))
    )
    
    return dataset, user

def test_numeric_column_replacement():
    """Test replacement in numeric columns"""
    print("\n=== Testing Numeric Column Replacement ===")
    
    dataset, user = create_test_dataset()
    factory = RequestFactory()
    view = DatasetReplaceValuesView()
    
    # Test 1: Valid numeric replacement
    request = factory.post(f'/api/datasets/{dataset.id}/replace-values/', {
        'column_name': 'numeric_col',
        'indices': [0, 2],
        'new_value': '99.9'
    })
    request.user = user
    
    response = view.post(request, dataset.id)
    print(f"Test 1 - Valid numeric: {response.data}")
    
    # Verify the changes
    df = pd.read_csv(dataset.file.path)
    assert df.loc[0, 'numeric_col'] == 99.9
    assert df.loc[2, 'numeric_col'] == 99.9
    
    # Test 2: Invalid numeric value
    request = factory.post(f'/api/datasets/{dataset.id}/replace-values/', {
        'column_name': 'numeric_col',
        'indices': [1],
        'new_value': 'invalid_number'
    })
    request.user = user
    
    response = view.post(request, dataset.id)
    print(f"Test 2 - Invalid numeric: {response.data}")
    assert response.status_code != 200  # Should fail
    
    # Cleanup
    dataset.delete()

def test_char_replacement():
    """Test character/substring replacement"""
    print("\n=== Testing Character Replacement ===")
    
    dataset, user = create_test_dataset()
    factory = RequestFactory()
    view = DatasetReplaceValuesView()
    
    # Test 1: Replace decimal point in mixed column
    request = factory.post(f'/api/datasets/{dataset.id}/replace-values/', {
        'column_name': 'mixed_col',
        'indices': [0, 1, 3],  # '100.5', '200.0', '400.5'
        'char_replace': True,
        'char_to_find': '.',
        'char_to_replace': ','
    })
    request.user = user
    
    response = view.post(request, dataset.id)
    print(f"Test 1 - Char replace: {response.data}")
    
    # Verify the changes
    df = pd.read_csv(dataset.file.path)
    assert df.loc[0, 'mixed_col'] == '100,5'
    assert df.loc[1, 'mixed_col'] == '200,0'
    assert df.loc[3, 'mixed_col'] == '400,5'
    
    # Test 2: Remove characters (empty replacement)
    dataset2, _ = create_test_dataset()
    request = factory.post(f'/api/datasets/{dataset2.id}/replace-values/', {
        'column_name': 'string_col',
        'indices': [0, 1],  # 'apple', 'banana'
        'char_replace': True,
        'char_to_find': 'a',
        'char_to_replace': ''
    })
    request.user = user
    
    response = view.post(request, dataset2.id)
    print(f"Test 2 - Char remove: {response.data}")
    
    # Verify the changes
    df = pd.read_csv(dataset2.file.path)
    assert df.loc[0, 'string_col'] == 'pple'  # 'apple' -> 'pple'
    assert df.loc[1, 'string_col'] == 'bnn'   # 'banana' -> 'bnn'
    
    # Cleanup
    dataset.delete()
    dataset2.delete()

def test_integer_column_preservation():
    """Test that integer columns maintain their type"""
    print("\n=== Testing Integer Column Type Preservation ===")
    
    dataset, user = create_test_dataset()
    factory = RequestFactory()
    view = DatasetReplaceValuesView()
    
    # Replace with integer value
    request = factory.post(f'/api/datasets/{dataset.id}/replace-values/', {
        'column_name': 'int_col',
        'indices': [0, 2],
        'new_value': '999'
    })
    request.user = user
    
    response = view.post(request, dataset.id)
    print(f"Test - Integer replacement: {response.data}")
    
    # Verify the changes and type
    df = pd.read_csv(dataset.file.path)
    assert df.loc[0, 'int_col'] == 999
    assert df.loc[2, 'int_col'] == 999
    
    # Check that the column is still integer-like after reload
    assert all(isinstance(x, (int, np.integer)) or pd.isna(x) for x in df['int_col'])
    
    # Cleanup
    dataset.delete()

def test_nan_prevention():
    """Test that NaN values are handled properly"""
    print("\n=== Testing NaN Prevention ===")
    
    dataset, user = create_test_dataset()
    factory = RequestFactory()
    view = DatasetReplaceValuesView()
    
    # Test empty string replacement in numeric column
    request = factory.post(f'/api/datasets/{dataset.id}/replace-values/', {
        'column_name': 'numeric_col',
        'indices': [1, 3],
        'new_value': ''  # This should create NaN
    })
    request.user = user
    
    response = view.post(request, dataset.id)
    print(f"Test - Empty value replacement: {response.data}")
    
    # Verify NaN was created intentionally
    df = pd.read_csv(dataset.file.path)
    assert pd.isna(df.loc[1, 'numeric_col'])
    assert pd.isna(df.loc[3, 'numeric_col'])
    
    # Cleanup
    dataset.delete()

def test_char_replace_on_numeric():
    """Test character replacement on numeric columns"""
    print("\n=== Testing Char Replace on Numeric Columns ===")
    
    # Create a dataset with numeric values that have decimal points
    data = {
        'prices': [10.50, 20.75, 30.00, 40.25, 50.99]
    }
    df = pd.DataFrame(data)
    csv_content = df.to_csv(index=False)
    
    user = User.objects.get(username='test_user')
    dataset = Dataset.objects.create(
        name='test_numeric_char_replace',
        user=user,
        description='Test char replacement on numeric'
    )
    dataset.file.save('test_numeric_char.csv', ContentFile(csv_content.encode('utf-8')))
    
    factory = RequestFactory()
    view = DatasetReplaceValuesView()
    
    # Replace decimal point with comma (should convert to string)
    request = factory.post(f'/api/datasets/{dataset.id}/replace-values/', {
        'column_name': 'prices',
        'indices': [0, 1, 2],
        'char_replace': True,
        'char_to_find': '.',
        'char_to_replace': ','
    })
    request.user = user
    
    response = view.post(request, dataset.id)
    print(f"Test - Numeric char replace: {response.data}")
    
    # Verify the changes
    df = pd.read_csv(dataset.file.path)
    # After char replacement that breaks numeric format, values become strings
    assert df.loc[0, 'prices'] == '10,5'
    assert df.loc[1, 'prices'] == '20,75'
    assert df.loc[2, 'prices'] == '30,0'
    
    # Cleanup
    dataset.delete()

def main():
    """Run all tests"""
    print("Starting DatasetReplaceValuesView Tests...")
    
    try:
        test_numeric_column_replacement()
        test_char_replacement()
        test_integer_column_preservation()
        test_nan_prevention()
        test_char_replace_on_numeric()
        
        print("\n✅ All tests passed successfully!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()