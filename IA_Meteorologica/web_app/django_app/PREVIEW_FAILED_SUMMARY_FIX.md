# Preview Failed 'Summary' Fix

## Issue
When previewing the second layer of a multi-layer normalization, the system would fail with "Preview failed: 'Summary'" error. This happened when:
1. The first layer transformed a column name (e.g., "Summary" → "summary" via lowercase)
2. The second layer tried to preview normalization on "Summary" which no longer existed in the original dataset

## Root Cause
The preview code was trying to access `df[column]` where `column` was "Summary", but this column only existed in the normalized output from the first layer, not in the original dataset.

## Solution
Modified the preview logic to handle columns that come from previous normalizations:

1. **Check if column exists** (line ~1522):
   - Added `original_column_exists` flag to track if the column exists in the original dataset
   - If not, check if it exists in the normalized data

2. **Handle missing columns gracefully** (lines ~1713-1722):
   - If column doesn't exist in original dataset, use the normalized data instead
   - For getting sample values, use `normalized_sample[column]` instead of `sample_df[column]`

3. **Fix statistics calculation** (multiple locations):
   - When calculating stats, use `df[column] if original_column_exists else normalized_full[column]`
   - This ensures we get statistics from the appropriate source

4. **Fix source data for mappings** (line ~1748):
   - Set `source_df_for_mapping` to use normalized data when column doesn't exist in original

5. **Fix normalization application** (line ~1758):
   - When applying previous transformations, use the appropriate base dataset

## Result
Now the system can properly handle multi-layer normalizations where column names change between layers. The preview will correctly show the transformation even when the column only exists as output from a previous layer.