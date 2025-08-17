# Fix for TRUNCATE Conversion Error with Multi-Column Outputs

## Problem Description

When using multi-column transformations (like date extraction), the TRUNCATE conversion option was being shown for text columns, causing an error:
```
Error aplicando TRUNCATE a columna Formatted Date: Truncate conversion requires numeric data
```

## Root Cause

The conversion options were determined based on the general output type of the transformation method, not the specific type of the selected output column. For example:
- Date extraction has `output_type: 'numeric'` because it produces numeric columns (_year, _month, _day)
- But if the user selects "Formatted Date" (the original text column), TRUNCATE was still shown

## Solution

### 1. Modified Frontend Type Detection

Added `updateConversionOptionsForLayer` function in `templates/normalize.html` that:
- Checks the specific input column selected for the current layer
- Determines type based on column name patterns:
  - Columns ending with `_year`, `_month`, `_day`, etc. → numeric
  - Columns ending with `_day_name`, `_month_name` → text
  - Column exactly named `Formatted Date` or starting with `Formatted Date_` → text (unless it's a numeric suffix)

### 2. Dynamic Conversion Options Update

- Conversion options are now populated dynamically when:
  - A new layer is added
  - The input column selection changes
  - The normalization method changes
- The conversion dropdown only shows appropriate options based on the actual column type

### 3. Integration Points

- `addLayer()`: Calls `updateConversionOptionsForLayer` after creating a new layer
- `onInputColumnChange()`: Updates conversion options when input column changes
- `onNormalizationChange()`: Updates conversion options for the next layer when method changes

## Files Modified

1. `/mnt/c/Users/andri/Desktop/TFG_FR/IA_Meteorologica/web_app/django_app/templates/normalize.html`
   - Added `updateConversionOptionsForLayer()` function (lines 1829-1917)
   - Modified `addLayer()` to call the update function (line 3175)
   - Updated `onInputColumnChange()` to refresh conversion options (lines 2130-2133)
   - Updated `onNormalizationChange()` to update next layer's conversions (lines 2814-2817)

## Impact

- TRUNCATE conversion is now only available for numeric columns
- No changes to backend logic needed - the frontend correctly filters options
- Better user experience with appropriate conversion options
- Prevents runtime errors from invalid conversions

## Testing

Users can verify the fix by:
1. Creating a date extraction transformation (layer 1)
2. Adding a second layer and selecting "Formatted Date" as input
3. The conversion dropdown should NOT show TRUNCATE option
4. Selecting a numeric output like "Formatted Date_year" should show TRUNCATE