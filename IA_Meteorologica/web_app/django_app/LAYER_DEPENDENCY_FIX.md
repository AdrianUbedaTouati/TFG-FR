# Layer Dependency Fix

## Issue
When fetching column types for output conversions, the system was getting an error:
```
ERROR: Input column 'trend' not found in dataframe for step 1
```

This happened because:
1. Layer 0: Custom function creates columns like 'trend', 'h_sin', etc.
2. Layer 1: MIN_MAX tries to normalize 'trend'
3. When fetching column types, only Layer 1 configuration was sent
4. Backend couldn't find 'trend' because Layer 0 wasn't included

## Root Cause
The `fetchColumnTypesForOutput` function was only sending the last layer configuration:
```javascript
// OLD CODE - Only sent last layer
currentConfig[column] = [normalizationConfig[column][lastLayerIndex]];
```

This failed when the last layer depended on columns created by previous layers.

## Solution
Updated `fetchColumnTypesForOutput` to send all layers up to and including the last layer:

```javascript
// NEW CODE - Sends all layers to ensure dependencies are satisfied
if (Array.isArray(normalizationConfig[column])) {
    // Include all layers up to the last layer to ensure columns like 'trend' exist
    currentConfig[column] = normalizationConfig[column].slice(0, lastLayerIndex + 1);
} else if (normalizationConfig[column]) {
    currentConfig[column] = [normalizationConfig[column]];
}
```

## How It Works Now

1. **Layer Dependencies**: When Layer 1 uses `input_column: 'trend'`, it needs Layer 0 to create that column first.

2. **Complete Chain**: The preview API now receives the complete transformation chain, so it can:
   - Execute Layer 0 (creates 'trend', 'h_sin', etc.)
   - Execute Layer 1 (applies MIN_MAX to 'trend')
   - Return accurate type information for all columns

3. **Type Detection**: With the complete chain, the system can accurately detect that:
   - 'trend' is numeric (float64)
   - 'h_sin' is numeric (float64)
   - etc.

## Result
- Column types are correctly detected for multi-layer transformations
- Output conversions show appropriate options (decimal truncation for numeric columns)
- No more "column not found" errors when fetching types

## Testing
1. Apply a custom function that creates numeric columns
2. Add a second layer that uses one of those columns as input
3. Add output conversions - types should be correctly detected
4. Console will show successful type detection without errors