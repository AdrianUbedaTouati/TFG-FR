# Numeric Column Type Detection Fix

## Issue
When a custom function creates numeric columns from a text column (like "Formatted Date"), the system was incorrectly detecting the generated columns as text type instead of numeric. This caused the truncation dialog to show text truncation options (character limit) instead of numeric truncation options (decimal places).

## Root Cause
The `getOutputColumnType` function was falling back to the original column type when it couldn't determine the type of the generated columns. Since "Formatted Date" is a text column, all its outputs were being treated as text, even though the custom function generates numeric columns (trend, h_sin, h_cos, etc.).

## Changes Made

### 1. Enhanced Type Detection in `getOutputColumnType`
**File**: `templates/normalize.html`

Added pattern matching to infer column types from common naming patterns:
```javascript
// Common numeric patterns
if (outputColumnName.match(/_(sin|cos|trend|normalized|count|sum|mean|std|min|max|year|month|day|hour|minute|second)$/i) ||
    outputColumnName.match(/^(h|dow|doy)_(sin|cos)$/i) ||
    outputColumnName.match(/^trend$|_hours?$|_days?$|_years?$/i)) {
    console.log(`Inferred numeric type for ${outputColumnName} based on pattern`);
    return 'numeric';
}
```

Added special handling for custom functions:
```javascript
// For custom functions that normalize dates, the outputs are typically numeric
if (layer && layer.method && layer.method.startsWith('CUSTOM_')) {
    console.log(`Custom function detected, defaulting to numeric for ${outputColumnName}`);
    return 'numeric';
}
```

### 2. Improved Column Type Fetching
**File**: `templates/normalize.html`

Updated `fetchColumnTypesForOutput` to properly prepare the configuration:
```javascript
// Prepare config with output_conversions to ensure we get all columns
const config = Array.isArray(normalizationConfig[column]) ? {
    layers: normalizationConfig[column],
    output_conversions: normalizationConfig[column].output_conversions || []
} : normalizationConfig[column];
```

### 3. Added Debug Logging
Added console logging throughout to track:
- Column type detection
- Type inference from patterns
- Stored column types from preview API

### 4. Proactive Type Fetching
Updated `addOutputConversion` to fetch column types if not already available:
```javascript
// Fetch column types if not already available
if (outputColumns.length > 0 && (!outputIndicator || !outputIndicator.dataset.columnTypes)) {
    console.log('Column types not available, fetching...');
    fetchColumnTypesForOutput(column, outputColumns);
}
```

## How It Works Now

1. **Initial Detection**: When output conversions are displayed, the system first checks for stored type information from the preview API.

2. **Pattern Matching**: If type info isn't available, it tries to infer the type from the column name:
   - Names ending in _sin, _cos, _trend, etc. → numeric
   - Names like h_sin, dow_cos, doy_sin → numeric
   - Names ending in _text, _str, _label → text

3. **Custom Function Default**: For custom functions (CUSTOM_*), the default is now numeric since most date normalization functions output numeric values.

4. **Fallback**: Only as a last resort does it use the original column type.

## Testing
To test:
1. Select "Formatted Date" column (text type)
2. Apply custom function "Normalize Date Sin/Cos" 
3. Add output conversion and select a generated column (e.g., "h_sin")
4. Choose TRUNCATE → Should show decimal truncation options, not character limit

The console will show:
- "Inferred numeric type for h_sin based on pattern"
- Or "Custom function detected, defaulting to numeric for h_sin"