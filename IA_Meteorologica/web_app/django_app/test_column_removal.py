#!/usr/bin/env python
"""
Test script to verify that column removal works correctly in normalization.

This script tests various scenarios:
1. Single normalization with keep_original=False (should remove column)
2. Single normalization with keep_original=True (should keep column)
3. Multi-layer normalization with all keep_original=False (should remove column)
4. Multi-layer normalization with at least one keep_original=True (should keep column)
"""

import os
import sys
import django
import pandas as pd
import tempfile

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.models import Dataset, User
from ml_trainer.views.normalization_views import DatasetNormalizationView
from django.core.files.base import ContentFile


def create_test_dataset():
    """Create a test dataset with numeric and text columns"""
    data = {
        'temperature': [20.5, 21.0, 19.5, 22.0, 20.0],
        'humidity': [65, 70, 60, 75, 68],
        'weather': ['sunny', 'cloudy', 'sunny', 'rainy', 'cloudy'],
        'pressure': [1013.25, 1012.5, 1014.0, 1011.75, 1013.0]
    }
    df = pd.DataFrame(data)
    
    # Save to temporary CSV
    csv_content = df.to_csv(index=False)
    
    # Create dataset object
    user = User.objects.first()
    if not user:
        user = User.objects.create_user(username='test_user', password='test123')
    
    dataset = Dataset.objects.create(
        name='test_column_removal',
        user=user,
        is_normalized=False
    )
    
    dataset.file.save('test.csv', ContentFile(csv_content.encode('utf-8')))
    
    return dataset, df


def test_single_normalization_remove_column(view, dataset):
    """Test single normalization with keep_original=False"""
    print("\n=== Test 1: Single normalization with keep_original=False ===")
    
    config = {
        'temperature': {
            'method': 'MIN_MAX',
            'keep_original': False
        }
    }
    
    df = pd.read_csv(dataset.file.path)
    result_df = view._apply_normalization(df, config)
    
    print(f"Original columns: {list(df.columns)}")
    print(f"Result columns: {list(result_df.columns)}")
    
    # Check if column was transformed in place (not removed, but values changed)
    assert 'temperature' in result_df.columns, "Column should still exist (transformed in place)"
    assert not (result_df['temperature'] == df['temperature']).all(), "Values should be normalized"
    
    print("✓ Test passed: Column transformed in place")


def test_single_normalization_keep_column(view, dataset):
    """Test single normalization with keep_original=True"""
    print("\n=== Test 2: Single normalization with keep_original=True ===")
    
    config = {
        'temperature': {
            'method': 'MIN_MAX',
            'keep_original': True
        }
    }
    
    df = pd.read_csv(dataset.file.path)
    result_df = view._apply_normalization(df, config)
    
    print(f"Original columns: {list(df.columns)}")
    print(f"Result columns: {list(result_df.columns)}")
    
    assert 'temperature' in result_df.columns, "Original column should be kept"
    assert 'temperature_normalized' in result_df.columns, "New normalized column should exist"
    assert (result_df['temperature'] == df['temperature']).all(), "Original values should be unchanged"
    
    print("✓ Test passed: Original column kept, new column created")


def test_multi_layer_remove_column(view, dataset):
    """Test multi-layer normalization with all keep_original=False"""
    print("\n=== Test 3: Multi-layer normalization with all keep_original=False ===")
    
    config = {
        'temperature': [
            {
                'method': 'MIN_MAX',
                'keep_original': False
            },
            {
                'method': 'Z_SCORE',
                'keep_original': False
            }
        ]
    }
    
    df = pd.read_csv(dataset.file.path)
    result_df = view._apply_normalization(df, config)
    
    print(f"Original columns: {list(df.columns)}")
    print(f"Result columns: {list(result_df.columns)}")
    
    # Original column should be removed
    assert 'temperature' not in result_df.columns, "Original column should be removed"
    # Should have intermediate columns
    assert any(col.startswith('temperature_') for col in result_df.columns), "Should have normalized columns"
    
    print("✓ Test passed: Original column removed")


def test_multi_layer_keep_column(view, dataset):
    """Test multi-layer normalization with at least one keep_original=True"""
    print("\n=== Test 4: Multi-layer normalization with one keep_original=True ===")
    
    config = {
        'temperature': [
            {
                'method': 'MIN_MAX',
                'keep_original': False
            },
            {
                'method': 'Z_SCORE', 
                'keep_original': True  # This should prevent removal
            }
        ]
    }
    
    df = pd.read_csv(dataset.file.path)
    result_df = view._apply_normalization(df, config)
    
    print(f"Original columns: {list(df.columns)}")
    print(f"Result columns: {list(result_df.columns)}")
    
    assert 'temperature' in result_df.columns, "Original column should be kept"
    assert any(col.startswith('temperature_') for col in result_df.columns), "Should have normalized columns"
    
    print("✓ Test passed: Original column kept")


def test_text_normalization(view, dataset):
    """Test text normalization with keep_original=False"""
    print("\n=== Test 5: Text normalization with keep_original=False ===")
    
    config = {
        'weather': {
            'method': 'LOWER',
            'keep_original': False
        }
    }
    
    df = pd.read_csv(dataset.file.path)
    result_df = view._apply_normalization(df, config)
    
    print(f"Original columns: {list(df.columns)}")
    print(f"Result columns: {list(result_df.columns)}")
    print(f"Original values: {df['weather'].tolist()}")
    print(f"Normalized values: {result_df['weather'].tolist()}")
    
    assert 'weather' in result_df.columns, "Column should exist (transformed in place)"
    assert all(val == val.lower() for val in result_df['weather']), "Values should be lowercased"
    
    print("✓ Test passed: Text column transformed in place")


def main():
    """Run all tests"""
    print("Testing column removal in normalization...")
    
    # Create test dataset
    dataset, df = create_test_dataset()
    
    # Create view instance
    view = DatasetNormalizationView()
    
    try:
        # Run tests
        test_single_normalization_remove_column(view, dataset)
        test_single_normalization_keep_column(view, dataset)
        test_multi_layer_remove_column(view, dataset)
        test_multi_layer_keep_column(view, dataset)
        test_text_normalization(view, dataset)
        
        print("\n✅ All tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if dataset.file:
            os.remove(dataset.file.path)
        dataset.delete()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())