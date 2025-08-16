# Individual Keep Original Checkbox Implementation

## Summary of Changes

I have successfully implemented individual "keep original" checkboxes for each normalization layer as requested. Each layer now has its own checkbox to control whether the original column should be preserved.

## Key Changes Made

### 1. HTML Structure
- Each normalization layer (both initial and dynamically added) now includes its own checkbox with unique ID: `keep-original-checkbox-${escapedColumn}-${layerIndex}`
- The checkbox is contained within a `keep-original-container` div for proper styling and visibility control

### 2. JavaScript Functions Updated

#### `onKeepOriginalChangeForLayer(column, layerIndex)`
- New function that handles checkbox changes for individual layers
- Updates only the specific layer's `keep_original` value in the configuration

#### `onNormalizationChange(column, layerIndex)`
- Modified to preserve the checkbox state when methods are changed
- Reads the current checkbox value if it exists
- Initializes the checkbox based on the configuration value
- Shows/hides the checkbox based on whether it's a default function or custom function

#### `removeNormalizationLayer(column, layerIndex)`
- Updated to properly reindex checkbox IDs and event handlers when layers are removed
- Ensures the onchange attribute calls `onKeepOriginalChangeForLayer` with the correct parameters

### 3. Behavior
- Each layer can independently control whether to keep the original column
- Checkbox state is preserved when:
  - Changing normalization methods
  - Adding new layers
  - Removing layers (remaining layers are properly reindexed)
- Custom functions always keep the original column (checkbox hidden)
- Default functions show the checkbox allowing user choice

## Testing
A test script was created to verify all functionality is working correctly. All tests pass:
- ✓ Function exists and is properly implemented
- ✓ Initial layer has per-layer checkbox
- ✓ New layers get per-layer checkboxes
- ✓ Checkbox state is preserved
- ✓ Layer reindexing works correctly
- ✓ Checkbox initialization from config works

## Usage
Users can now:
1. Select different "keep original" settings for each normalization layer
2. Chain normalizations with fine-grained control over which intermediate columns to preserve
3. See immediate feedback as checkboxes appear/disappear based on the selected method type

The implementation is complete and ready for use!