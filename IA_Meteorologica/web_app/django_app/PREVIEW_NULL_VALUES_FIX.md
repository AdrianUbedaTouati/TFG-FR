# Fix for Preview Null Values with Multi-Column Transformations

## Problem Description

When a normalization chain has:
1. **Layer 1**: Creates multiple output columns (e.g., date extraction creates _year, _month, _day columns)
2. **Layer 2**: Uses one of those outputs via `input_column` parameter (e.g., processes the _day column)

The preview was showing all values as `null` in the unique mapping, even though the actual transformation worked correctly (as shown in "Muestras de datos").

## Root Cause

The preview generation code had a mismatch between:
- The column name used to create the temporary DataFrame for preview
- The `input_column` specified in the layer 2 transformation config

### Example Scenario:
```json
{
  "Formatted Date": [
    {
      "method": "date_extraction",
      "keep_original": true
    },
    {
      "method": "one_hot",
      "input_column": "Formatted Date_day"
    }
  ]
}
```

**What was happening:**
1. Layer 1 creates: `Formatted Date_year`, `Formatted Date_month`, `Formatted Date_day`
2. Preview code creates temp_df with column `Formatted Date`
3. Layer 2 expects `Formatted Date_day` but can't find it in temp_df
4. Result: All mappings show as null

## Solution

Modified the preview generation in `ml_trainer/views/normalization_views.py` (lines 1745-1794):

### Key Changes:

1. **Detect input_column requirement** (lines 1745-1749):
   ```python
   # Check if the last transformation expects a different input column
   if 'input_column' in last_transformation and last_transformation['input_column'] in intermediate_df.columns:
       intermediate_column = last_transformation['input_column']
       print(f"Layer preview: Last transformation specifies input_column: '{intermediate_column}'")
   ```

2. **Use the correct column for temp_df** (lines 1751-1764):
   - Extract unique values from the actual input column needed by layer 2
   - Create temp_df with the correct column name

3. **Adjust transformation config** (lines 1766-1774):
   ```python
   # Prepare the transformation config for the last step
   last_transformation_adjusted = last_transformation.copy()
   if 'input_column' in last_transformation_adjusted:
       if last_transformation_adjusted['input_column'] != temp_column_name:
           # Remove the input_column since we're applying to temp_column_name
           del last_transformation_adjusted['input_column']
   ```

4. **Apply same fix to fallback path** (lines 1787-1794):
   - When the main preview path fails, the fallback also handles input_column correctly

## Impact

- Preview now correctly shows the mapping values instead of null
- No impact on actual normalization process (which was already working)
- Better user experience when configuring multi-layer transformations

## Files Modified

- `/mnt/c/Users/andri/Desktop/TFG_FR/IA_Meteorologica/web_app/django_app/ml_trainer/views/normalization_views.py`

## Testing

To verify the fix:
1. Create a normalization with date extraction (layer 1) 
2. Add a second layer that uses one of the date components (e.g., one-hot on _day)
3. Check the preview - it should show actual values in the mapping, not nulls