# Column Removal Analysis for Normalization Process

## Overview
This document analyzes why columns are not being removed during normalization in the Django ML Trainer application.

## Key Findings

### 1. Column Removal Logic Location
The main column removal logic is in `_apply_normalization()` method at lines 407-420:

```python
# After all layers, check if we need to remove the original column
# Only remove if NONE of the layers wanted to keep the original
if isinstance(method_config, list) and len(method_config) > 0:
    # Check if any layer has keep_original = True
    any_keep_original = any(step.get('keep_original', False) for step in method_config)
    
    if not any_keep_original and column in normalized_df.columns:
        # Check if there are new columns created (with suffix)
        new_cols = [col for col in normalized_df.columns if col.startswith(f"{column}_") and col != column]
        if new_cols:
            # There are new columns, so we can safely remove the original
            print(f"Removing original column '{column}' as no layer requested to keep it")
            normalized_df = normalized_df.drop(columns=[column])
        # If no new columns, the original was transformed in place, so don't drop it
```

### 2. The Problem: Column Naming Pattern

The issue is in the column detection pattern. The code looks for columns that start with `{column}_` (underscore), but the actual suffix pattern used is:
- For multi-layer: `_step{step_index + 1}` (line 454)
- For single-layer: `_normalized` (line 454)

This creates column names like:
- `numeric_col_step1`
- `numeric_col_step2`
- `numeric_col_normalized`

**BUT** the detection pattern at line 415 looks for `{column}_`, which means it's looking for columns like `numeric_col_*`.

### 3. The Root Cause

The issue is that when creating new columns in `_apply_single_normalization()`:

For multi-layer normalizations (lines 1002-1004):
```python
if total_steps > 1:
    new_column_name = f"{column}{suffix}"  # This creates "numeric_col_step1" (NO underscore!)
    normalized_df[new_column_name] = func(normalized_df[column])
```

But the detection logic looks for (line 415):
```python
new_cols = [col for col in normalized_df.columns if col.startswith(f"{column}_") and col != column]
```

The pattern `f"{column}_"` expects an underscore after the column name, but the actual column names don't have that underscore!

### 4. Why Columns Are Not Being Removed

1. **Multi-layer normalizations**: 
   - Creates columns like `numeric_col_step1`, `numeric_col_step2`
   - Detection looks for `numeric_col_*` (with underscore)
   - No match found, so `new_cols` is empty
   - Original column is not removed

2. **Single-layer normalizations with keep_original=False**:
   - Transforms the column in-place (lines 1010-1011, 1028-1029)
   - No new columns are created
   - Original column contains the transformed data

3. **Custom functions**:
   - Have their own removal logic (line 877)
   - Only remove if `total_steps == 1 and not keep_original and custom_func.remove_original_column`

### 5. The Fix

The detection pattern needs to be fixed to match the actual column naming:

```python
# Current (incorrect):
new_cols = [col for col in normalized_df.columns if col.startswith(f"{column}_") and col != column]

# Fixed:
new_cols = [col for col in normalized_df.columns if col.startswith(column) and col != column]
```

Or alternatively, change the suffix generation to include an underscore:

```python
# Current:
new_column_name = f"{column}{suffix}"

# Alternative fix:
new_column_name = f"{column}_{suffix}"  # Add underscore
```

### 6. Additional Issues

1. **Suffix Generation**: The suffix at line 454 doesn't include an underscore:
   ```python
   suffix = f"_step{step_index + 1}" if total_steps > 1 else "_normalized"
   ```
   This creates `column_step1` instead of `column_step1`.

2. **Column Tracking**: After each step, the code tries to update `current_column` (lines 398-404), but this logic also depends on the same flawed pattern.

3. **Input Column Logic**: The code supports specifying `input_column` for steps (lines 347-356), which adds complexity to tracking which columns should be removed.

## Recommendations

1. **Immediate Fix**: Change line 415 to remove the underscore requirement:
   ```python
   new_cols = [col for col in normalized_df.columns if col.startswith(column) and col != column and len(col) > len(column)]
   ```

2. **Better Fix**: Standardize column naming to always include a separator:
   ```python
   new_column_name = f"{column}_{suffix.lstrip('_')}"  # Ensures single underscore
   ```

3. **Track Created Columns**: Maintain a list of created columns during normalization to ensure accurate removal:
   ```python
   created_columns = []
   # When creating a new column:
   created_columns.append(new_column_name)
   # When checking for removal:
   if created_columns and not any_keep_original:
       normalized_df = normalized_df.drop(columns=[column])
   ```

## Summary

The column removal is failing because of a mismatch between:
- How new columns are named: `columnSUFFIX` (no separator)
- How they are detected for removal: `column_*` (expects underscore)

This causes the removal logic to never find the new columns, so it never removes the original column even when all layers have `keep_original=False`.