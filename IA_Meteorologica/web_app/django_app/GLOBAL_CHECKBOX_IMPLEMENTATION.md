# Global Column-Level Keep Original Checkbox Implementation

## Summary of Changes

I have implemented a global column-level "Mantener columna original" checkbox that appears when any normalization method is selected for a column. This checkbox controls the `keep_original` setting for all layers at once.

## Key Features

### 1. Global Checkbox Visibility
- The global checkbox appears automatically when any normalization method is selected
- Located below the layer controls with prominent styling
- Hidden when no normalization is applied

### 2. Checkbox Behavior
- **Global Control**: When checked/unchecked, it updates ALL layers for that column
- **Bidirectional Sync**: 
  - Global checkbox → Updates all per-layer checkboxes
  - Per-layer checkboxes → When all have same value, updates global checkbox
- **Clear Label**: "Mantener columna original (aplicar a todas las capas)"

### 3. Visual Design
- Gradient background with cyan/purple tones
- Prominent border and shadow effects
- Hover effects for better interactivity
- Fade-in animation when appearing
- Larger checkbox size (18x18px) for easier interaction

### 4. Implementation Details

#### Modified Functions:
1. **`onNormalizationChange`**: Shows global checkbox when normalization is selected
2. **`onKeepOriginalChange`**: Updates all layers when global checkbox changes
3. **`onKeepOriginalChangeForLayer`**: Syncs global checkbox when all layers match

#### HTML Structure:
```html
<div class="keep-original-container" id="keep-original-${column}">
    <label>
        <input type="checkbox" onchange="onKeepOriginalChange('${column}')">
        <span>Mantener columna original (aplicar a todas las capas)</span>
    </label>
</div>
```

## User Workflow

1. User selects a normalization method for a column
2. Global checkbox appears below the layer controls
3. User can:
   - Check the global checkbox to keep original for ALL layers
   - Use individual per-layer checkboxes for fine control
   - Global checkbox auto-syncs when all layers have same value

## Benefits

- **Convenience**: One checkbox controls all layers
- **Flexibility**: Still allows per-layer control when needed
- **Visibility**: Prominent styling makes it easy to see and use
- **Consistency**: Syncs with per-layer checkboxes automatically

The implementation provides both convenience (global control) and flexibility (per-layer control) for managing column preservation during normalization.