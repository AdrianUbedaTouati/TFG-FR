# Type Detection and Configuration Fixes

## Issues Fixed

### 1. Inconsistent Type Detection for Truncation Dialogs
**Problem**: Sometimes the system detected a column as numeric but showed text truncation options, then when clicked, showed numeric truncation.

**Solution**:
- Made all truncation dialog functions (`showTruncateParamsDialog`, `showMultiOutputTruncateDialog`, `showOutputTruncateDialog`) async
- Added proper type fetching using the preview API before showing dialogs
- For single-column outputs: Uses `fetchSingleColumnOutputType` for accurate detection
- For multi-column outputs: Fetches types if not cached, ensuring correct dialog shows
- Added extensive console logging for debugging type detection

**Key Changes**:
```javascript
// showOutputTruncateDialog now fetches actual type
async function showOutputTruncateDialog(column) {
    let outputType = await fetchSingleColumnOutputType(column);
    console.log(`Single column output type for ${column}: ${outputType}`);
    // ... shows correct dialog based on actual type
}

// Multi-output dialog fetches types if needed
async function showMultiOutputTruncateDialog(column, convIndex) {
    if (!outputIndicator || !outputIndicator.dataset.columnTypesInfo) {
        await fetchColumnTypesForOutput(column, outputColumns);
    }
    // ... shows correct dialog based on fetched type
}
```

### 2. Configuration Not Cleared When Function Deselected
**Problem**: When a normalization function was deselected (set to empty), it still appeared in the final normalization.

**Solution**:
- Enhanced the `else` block in `onNormalizationChange` to properly remove configurations
- Handles both single-layer and multi-layer configurations
- Filters out null entries and reindexes layers
- Clears all cached type information when removing a function
- Completely removes column config when no layers remain

**Key Changes**:
```javascript
} else {
    // Remove this layer
    if (Array.isArray(normalizationConfig[column])) {
        if (layerIndex === 0 && normalizationConfig[column].length === 1) {
            // If this is the only layer, remove entire config
            delete normalizationConfig[column];
        } else {
            // Remove specific layer and clean up
            normalizationConfig[column][layerIndex] = null;
            normalizationConfig[column] = normalizationConfig[column].filter(layer => layer !== null);
        }
    }
    // Clear all cached type information
    const allIndicators = document.querySelectorAll(`[id^="output-type-${escapedColumn}-"]`);
    allIndicators.forEach(indicator => {
        delete indicator.dataset.columnTypes;
        delete indicator.dataset.columnTypesInfo;
        // ... clear all cached data
    });
}
```

## Testing

1. **Type Detection Test**:
   - Apply One-Hot Encoding to a text column
   - Add output conversion → TRUNCATE
   - Should show decimal truncation (numeric), not character truncation

2. **Configuration Removal Test**:
   - Select a normalization function
   - Preview to confirm it's applied
   - Change dropdown back to empty/none
   - Preview again - the function should not be applied

Both issues are now resolved with proper async type detection and complete configuration cleanup.