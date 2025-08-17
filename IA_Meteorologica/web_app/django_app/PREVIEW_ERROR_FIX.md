# Preview Error Fix Summary

## Issue
The user reported an error "No se recibieron datos de previsualización" (No preview data received) when trying to preview a transformation that creates multiple columns (like date_extraction).

The server logs showed:
```
Preview: Column 'Formatted Date' not found in normalized sample
```

## Root Cause
When using `date_extraction` with `keep_original=False`, the original column is removed and new columns are created (e.g., Formatted Date_year, Formatted Date_month, etc.). The preview code was looking for the original column in the normalized data, which no longer exists.

## Changes Made

### 1. Backend Preview Handling
**File**: `ml_trainer/views/normalization_views.py`
**Class**: `DatasetNormalizationPreviewView`

Added special handling for methods that create multiple columns:
```python
# Check if this method creates multiple columns (like date_extraction)
creates_multiple_columns = False
if isinstance(actual_config, list):
    for step in actual_config:
        if isinstance(step, dict) and step.get('method') in ['date_extraction'] and not step.get('keep_original', False):
            creates_multiple_columns = True
            break
        elif isinstance(step, dict) and step.get('method', '').startswith('CUSTOM_') and not step.get('keep_original', False):
            creates_multiple_columns = True
            break

if creates_multiple_columns:
    # Handle multi-column output case
    comparison[column] = {
        'multi_column_output': True,
        'new_columns': new_columns,
        'original_removed': True,
        # ... preview data for each new column
    }
```

### 2. Frontend Preview Display
**File**: `templates/normalize.html`
**Function**: `showPreview`

Updated to handle both the old format (`preview.new_columns`) and new format (`preview.multi_column_output`):
```javascript
const hasMultipleOutputs = preview && (
    (preview.new_columns && preview.new_columns.length > 0) || 
    preview.multi_column_output === true
);
```

### 3. Multi-Output Truncation Parameter Fix
**File**: `templates/normalize.html`
**Function**: `applyMultiOutputTruncateParams`

Fixed the parameters sent for text truncation to be compatible with backend:
```javascript
// For text columns, include both chars and decimals
conv.params = { chars: chars, decimals: chars };
```

### 4. Added Debug Logging
Added extensive logging in both frontend and backend to track:
- Configuration being sent to preview
- Multi-column output conversions
- Data transformation at each step

## How It Works Now

1. **Multi-Column Detection**: The preview correctly identifies when a normalization creates multiple columns.

2. **Preview Structure**: For multi-column outputs, the preview response includes:
   ```json
   {
     "preview": {
       "Formatted Date": {
         "multi_column_output": true,
         "new_columns": ["Formatted Date_year", "Formatted Date_month", ...],
         "original_removed": true,
         "Formatted Date_year": { "sample": [...], "stats": {...} },
         "Formatted Date_month": { "sample": [...], "stats": {...} }
       }
     }
   }
   ```

3. **Display**: The frontend correctly displays:
   - Information about new columns being created
   - Preview data for each new column
   - Warning that original column will be removed

## Testing
To test:
1. Select a date column (e.g., "Formatted Date")
2. Choose "date_extraction" method
3. Set keep_original to False
4. Click preview - should show all generated columns
5. Add output conversions (truncation) to specific columns
6. Preview again - should show truncated values