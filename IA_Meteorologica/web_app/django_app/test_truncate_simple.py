#!/usr/bin/env python
"""
Simple test to verify TRUNCATE logic
"""

# Simulate the frontend logic for column type detection
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
print("Frontend will show TRUNCATE option only when detected_type == 'numeric'")