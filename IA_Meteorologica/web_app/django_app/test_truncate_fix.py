#!/usr/bin/env python
"""
Test script to verify TRUNCATE conversion is only available for numeric columns
"""
import os
import django
import json

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.type_conversions import TYPE_CONVERSIONS, get_conversion_options

print("=== Testing Type Conversion Options ===")
print()

# Test numeric to numeric conversions
print("1. Numeric to Numeric conversions:")
numeric_convs = get_conversion_options('numeric', 'numeric')
for conv in numeric_convs:
    print(f"   - {conv['value']}: {conv['label']}")
    if conv['value'] == 'TRUNCATE':
        print(f"     ✓ TRUNCATE is available for numeric inputs")

print()

# Test text to numeric conversions
print("2. Text to Numeric conversions:")
text_to_numeric = get_conversion_options('text', 'numeric')
for conv in text_to_numeric:
    print(f"   - {conv['value']}: {conv['label']}")
if not any(conv['value'] == 'TRUNCATE' for conv in text_to_numeric):
    print("     ✓ TRUNCATE is NOT available for text inputs")

print()

# Test text to text conversions
print("3. Text to Text conversions:")
text_to_text = get_conversion_options('text', 'text')
for conv in text_to_text:
    print(f"   - {conv['value']}: {conv['label']}")
if not any(conv['value'] == 'TRUNCATE' for conv in text_to_text):
    print("     ✓ TRUNCATE is NOT available for text to text conversions")

print()

# Test numeric to text conversions
print("4. Numeric to Text conversions:")
numeric_to_text = get_conversion_options('numeric', 'text')
for conv in numeric_to_text:
    print(f"   - {conv['value']}: {conv['label']}")
if not any(conv['value'] == 'TRUNCATE' for conv in numeric_to_text):
    print("     ✓ TRUNCATE is NOT available for numeric to text conversions")

print()
print("=== Frontend Type Detection Test Cases ===")
print()

# Simulate frontend column type detection
test_columns = [
    ("Formatted Date", "text"),
    ("Formatted Date_year", "numeric"),
    ("Formatted Date_month", "numeric"), 
    ("Formatted Date_day", "numeric"),
    ("Formatted Date_day_name", "text"),
    ("Formatted Date_month_name", "text"),
    ("Temperature_normalized", "numeric"),
    ("Summary_encoded", "text"),
]

print("Column name patterns and expected types:")
for col_name, expected_type in test_columns:
    # Simulate the frontend logic
    detected_type = "unknown"
    
    if (col_name.endswith('_year') or 
        col_name.endswith('_month') or 
        col_name.endswith('_day') or
        col_name.endswith('_hour') or
        col_name.endswith('_minute') or
        col_name.endswith('_second')):
        detected_type = 'numeric'
    elif (col_name.endswith('_day_name') or 
          col_name.endswith('_month_name') or
          col_name == 'Formatted Date' or
          col_name.startswith('Formatted Date_')):
        # Special handling for Formatted Date
        if not any(col_name.endswith(suffix) for suffix in ['_year', '_month', '_day', '_hour', '_minute', '_second']):
            detected_type = 'text'
    
    status = "✓" if detected_type == expected_type else "✗"
    print(f"   {status} {col_name}: detected as {detected_type} (expected: {expected_type})")

print()
print("=== Summary ===")
print("TRUNCATE should only be available when:")
print("1. Input type is 'numeric'")
print("2. Output type is 'numeric'")
print("3. It's in the numeric_to_numeric conversion category")