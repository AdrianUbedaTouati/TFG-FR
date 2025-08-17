#!/usr/bin/env python3
"""Test script to verify column name escaping issues"""

# Column names that might cause issues
test_columns = [
    "Temperature (C)",
    "Apparent Temperature (C)",
    "Pressure (millibars)",
    "Wind Speed (km/h)",
    "Humidity",
    "Visibility (km)",
    "Daily Summary",
    "Column-with-dash",
    "Column.with.dots",
    "Column[with]brackets"
]

# JavaScript escaping pattern from the template
def js_escape(column):
    # Current pattern: column.replace(/['"\\]/g, '\\$&').replace(/\s/g, '_')
    import re
    # Replace quotes, backslashes
    escaped = re.sub(r"['\"]", lambda m: '\\' + m.group(0), column)
    # Replace spaces with underscores
    escaped = escaped.replace(' ', '_')
    return escaped

# Better escaping pattern that handles all special characters
def safe_escape(column):
    # Replace all non-alphanumeric characters (except underscore) with underscore
    import re
    return re.sub(r'[^a-zA-Z0-9_]', '_', column)

print("Testing column name escaping:\n")
print("Column Name                    | Current Escape              | Safe Escape")
print("-" * 80)

for col in test_columns:
    current = js_escape(col)
    safe = safe_escape(col)
    print(f"{col:<30} | {current:<27} | {safe}")
    
print("\nProblem: Parentheses, dots, brackets, and dashes are not escaped in the current pattern!")
print("This causes getElementById to fail when looking for elements with these characters.")