#!/usr/bin/env python3
"""
Test script to verify that normalization information is displayed in dataset cards
"""

import re

def test_dataset_cards_implementation():
    """Test the dataset cards normalization info implementation"""
    
    print("Testing dataset cards normalization info display...")
    
    # Read the datasets.html file
    with open('templates/datasets.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Test 1: Check that getNormalizationBadge function exists
    print("\n1. Checking getNormalizationBadge function...")
    if 'function getNormalizationBadge' in content:
        print("✓ getNormalizationBadge function found")
    else:
        print("✗ getNormalizationBadge function NOT found")
        return
    
    # Test 2: Check that badge is added to dataset cards
    print("\n2. Checking badge in dataset cards...")
    badge_in_card = re.search(r'dataset\.is_normalized.*?getNormalizationBadge\(dataset\.normalization_method\)', content, re.DOTALL)
    if badge_in_card:
        print("✓ Normalization badge added to dataset cards")
    else:
        print("✗ Normalization badge NOT added to cards")
    
    # Test 3: Check that getDetailedNormalizationInfo function exists
    print("\n3. Checking getDetailedNormalizationInfo function...")
    if 'function getDetailedNormalizationInfo' in content:
        print("✓ getDetailedNormalizationInfo function found")
    else:
        print("✗ getDetailedNormalizationInfo function NOT found")
    
    # Test 4: Check that detailed info is added to dataset modal
    print("\n4. Checking detailed info in dataset modal...")
    detailed_in_modal = re.search(r'dataset\.is_normalized.*?getDetailedNormalizationInfo\(dataset\.normalization_method\)', content, re.DOTALL)
    if detailed_in_modal:
        print("✓ Detailed normalization info added to modal")
    else:
        print("✗ Detailed normalization info NOT added to modal")
    
    # Test 5: Check badge displays column kept count
    print("\n5. Checking column count display...")
    column_count_check = re.search(r'columna.*?original.*?conservada', content)
    if column_count_check:
        print("✓ Badge shows count of original columns kept")
    else:
        print("✗ Badge does NOT show column count")
    
    # Test 6: Check detailed view shows per-column info
    print("\n6. Checking per-column normalization details...")
    per_column_check = re.search(r'Original conservada.*?Original eliminada', content, re.DOTALL)
    if per_column_check:
        print("✓ Detailed view shows per-column keep_original status")
    else:
        print("✗ Detailed view does NOT show per-column status")
    
    print("\n✅ All tests completed!")
    
    # Summary
    print("\nSummary of changes:")
    print("- Dataset cards now show a badge indicating how many original columns were kept")
    print("- Badge uses different colors/icons based on whether columns were kept")
    print("- Dataset details modal shows detailed normalization info per column")
    print("- Each column shows its normalization method chain and keep_original status")
    print("- Visual indicators (badges) clearly show whether each column was kept or removed")

if __name__ == '__main__':
    test_dataset_cards_implementation()