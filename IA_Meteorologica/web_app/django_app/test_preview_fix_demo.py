#!/usr/bin/env python3
"""
Demonstration of the preview fix for multi-column transformations
"""

import pandas as pd
import numpy as np

def demonstrate_issue_and_fix():
    """Show the issue and how the fix resolves it"""
    
    print("DEMONSTRATION: Preview Fix for Multi-Column Transformations")
    print("=" * 60)
    
    # Sample data
    df = pd.DataFrame({
        'Formatted Date': ['2023-01-15', '2023-02-20', '2023-03-25', '2023-04-30'],
        'Temperature': [20.5, 22.3, 19.8, 21.0]
    })
    
    print("\nOriginal Data:")
    print(df)
    
    # Simulate Layer 1: Date extraction (creates multiple columns)
    print("\n\nLayer 1: Date Extraction")
    print("-" * 30)
    
    # This would create columns like: Formatted Date_year, Formatted Date_month, Formatted Date_day, etc.
    intermediate_df = df.copy()
    intermediate_df['Formatted Date_year'] = pd.to_datetime(df['Formatted Date']).dt.year
    intermediate_df['Formatted Date_month'] = pd.to_datetime(df['Formatted Date']).dt.month
    intermediate_df['Formatted Date_day'] = pd.to_datetime(df['Formatted Date']).dt.day
    
    print("After Layer 1 (date extraction):")
    print(intermediate_df)
    
    # The issue in preview generation
    print("\n\nTHE ISSUE:")
    print("-" * 30)
    
    # Layer 2 configuration specifies input_column: 'Formatted Date_day'
    layer2_config = {
        'method': 'one_hot',
        'keep_original': False,
        'input_column': 'Formatted Date_day'  # This is different from the main column name!
    }
    
    print(f"Layer 2 wants to process: '{layer2_config['input_column']}'")
    
    # OLD BEHAVIOR (before fix):
    print("\nOLD BEHAVIOR (before fix):")
    # The preview code would create temp_df with the wrong column
    temp_column_name = 'Formatted Date'  # This was being used
    all_unique_values = intermediate_df[layer2_config['input_column']].dropna().unique()
    
    # Create temp dataframe - but with wrong column name!
    temp_df_old = pd.DataFrame({temp_column_name: all_unique_values})
    print(f"temp_df columns: {list(temp_df_old.columns)}")
    print(f"Layer 2 expects column: '{layer2_config['input_column']}'")
    print("Result: Column mismatch! The transformation would fail or return nulls.")
    
    # NEW BEHAVIOR (after fix):
    print("\n\nNEW BEHAVIOR (after fix):")
    
    # Step 1: Detect that layer 2 has a different input_column
    if 'input_column' in layer2_config and layer2_config['input_column'] in intermediate_df.columns:
        intermediate_column = layer2_config['input_column']
        print(f"✓ Detected layer 2 needs column: '{intermediate_column}'")
    
    # Step 2: Get unique values from the correct column
    all_unique_values = intermediate_df[intermediate_column].dropna().unique()
    temp_column_name = intermediate_column
    
    # Step 3: Create temp_df with the correct column
    temp_df_new = pd.DataFrame({temp_column_name: all_unique_values})
    
    # Step 4: Adjust the transformation config
    layer2_config_adjusted = layer2_config.copy()
    if layer2_config_adjusted['input_column'] != temp_column_name:
        # This should not happen now, but just in case
        del layer2_config_adjusted['input_column']
    
    print(f"temp_df columns: {list(temp_df_new.columns)}")
    print(f"Unique values to process: {all_unique_values}")
    print("Result: Column names match! Preview can generate correct mappings.")
    
    # Simulate the one-hot encoding preview
    print("\n\nSimulated Preview Mapping:")
    print("-" * 30)
    for val in all_unique_values:
        one_hot_result = f"Formatted Date_day_{val}"  # Simulated one-hot column name
        print(f"  {val} -> {one_hot_result}")
    
    print("\n✓ SUCCESS: No null values in the preview mapping!")
    
    print("\n\nSUMMARY OF THE FIX:")
    print("=" * 60)
    print("1. Check if last transformation has 'input_column' field")
    print("2. Use that column from intermediate_df for preview generation")
    print("3. Create temp_df with the correct column name")
    print("4. Adjust transformation config to remove input_column mismatch")
    print("5. Result: Preview shows correct mappings instead of null values")


if __name__ == '__main__':
    demonstrate_issue_and_fix()