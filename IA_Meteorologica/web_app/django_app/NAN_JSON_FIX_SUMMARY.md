# Fix for NaN Values in JSON Serialization

## Problem
The application was throwing `ValueError: Out of range float values are not JSON compliant` when trying to serialize data containing NaN (Not a Number) values to JSON. This occurred in various places including:
- Dataset column preview data
- Sample values in column statistics
- Normalization preview data
- Statistical calculations (mean, std, min, max, etc.)

## Root Cause
NaN values from NumPy and Pandas are not JSON serializable. When using `.tolist()` on arrays/series containing NaN, or when converting float NaN values directly, JSON serialization fails.

## Solution
Created helper functions to safely convert values for JSON serialization:

### 1. Created `ml_trainer/utils_helpers.py`
This file contains three helper functions:

- `safe_to_list(values)`: Converts arrays/series/lists to lists, replacing NaN with None
- `safe_float(value)`: Converts values to float, returning None if NaN
- `safe_dict_values(data)`: Recursively cleans dictionary values for JSON serialization

### 2. Updated `ml_trainer/utils.py`
- Added import for helper functions
- Replaced all `.tolist()` calls with `safe_to_list()`
- Updated `detect_column_type()` to use safe conversions

### 3. Updated `ml_trainer/views/dataset_views.py`
- Added import for helper functions
- Replaced direct `.tolist()` calls with `safe_to_list()`
- Replaced manual NaN checks with `safe_float()` for statistics
- Updated preview data generation to use `safe_to_list()`

### 4. Updated `ml_trainer/views/normalization_views.py`
- Added import for helper functions
- Replaced all `.tolist()` calls with `safe_to_list()`
- Ensured sample values are properly converted

## Changes Made

### Files Modified:
1. **Created `ml_trainer/utils_helpers.py`** - New helper functions
2. **Modified `ml_trainer/utils.py`** - Updated to use safe conversions
3. **Modified `ml_trainer/views/dataset_views.py`** - Fixed all float/list conversions
4. **Modified `ml_trainer/views/normalization_views.py`** - Fixed sample value conversions

### Key Replacements:
- `series.tolist()` → `safe_to_list(series)`
- `array.tolist()` → `safe_to_list(array)`
- `float(value) if pd.notna(value) else None` → `safe_float(value)`
- `[None if pd.isna(val) else val for val in data]` → `safe_to_list(data)`

## Testing
The fix ensures that:
1. All NaN values are converted to None before JSON serialization
2. Arrays and Series are safely converted to lists
3. Float statistics handle NaN properly
4. The application no longer throws JSON serialization errors

## Usage
The helper functions are now used throughout the codebase wherever data might contain NaN values and needs to be serialized to JSON. This includes:
- API responses
- Preview data
- Statistical calculations
- Sample values
- Normalization results