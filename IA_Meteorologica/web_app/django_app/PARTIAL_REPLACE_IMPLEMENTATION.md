# Partial Replace Implementation

## Overview

The `DatasetReplaceValuesView` endpoint now supports partial replacement functionality, allowing users to replace specific characters at given positions within cell values.

## Parameters

The endpoint accepts the following parameters for partial replacement:

- **partial_replace**: `boolean` - Enables partial replacement mode
- **partial_pattern**: `array of integers` - Indices of characters to replace (0-based)
- **partial_type**: `string` - Type of partial replacement:
  - `'charByChar'` - Replace each character individually
  - `'complete'` - Remove characters at specified indices and insert replacement string

## Implementation Details

### Character by Character Mode (`charByChar`)

In this mode, each character at the specified indices is replaced with the corresponding character from the `new_value` string.

**Algorithm:**
1. Convert the cell value to a string
2. Convert to a list of characters
3. For each index in `partial_pattern`:
   - Replace the character at that index with the corresponding character from `new_value`
   - If `new_value` has fewer characters than `partial_pattern` indices, remaining positions are replaced with empty string

**Example:**
```python
original = "ABC123"
partial_pattern = [0, 2, 4]
new_value = "XYZ"
# Result: "XBY1Z3"
```

### Complete Mode (`complete`)

In this mode, characters at the specified indices are removed, and the complete `new_value` string is inserted at the position of the first removed character.

**Algorithm:**
1. Convert the cell value to a string
2. Convert to a list of characters
3. Remove characters at all specified indices (in reverse order to maintain indices)
4. Insert the complete `new_value` string at the position of the first removed character

**Example:**
```python
original = "ABC123"
partial_pattern = [1, 2, 3]
new_value = "XYZ"
# Result: "AXYZ23"
```

## Numeric Column Handling

When applying partial replacement to numeric columns:

1. The value is converted to string for manipulation
2. After replacement, the system attempts to convert back to numeric
3. For integer columns, if the result has no decimal part, it's converted to integer
4. If conversion fails, a warning is logged but the operation continues

## API Request Example

```json
{
  "column_name": "product_code",
  "indices": [0, 5, 10],  // Rows to modify
  "new_value": "XY",
  "partial_replace": true,
  "partial_pattern": [2, 4],  // Character positions to replace
  "partial_type": "charByChar"
}
```

## Response Format

The response includes information about the partial replacement:

```json
{
  "status": "success",
  "data": {
    "message": "Successfully replaced characters at positions [2, 4] in 3 values",
    "replaced_count": 3,
    "mode": "partial_charByChar",
    "partial_info": {
      "type": "charByChar",
      "pattern": [2, 4]
    }
  }
}
```

## Error Handling

- Invalid indices are silently skipped
- Out-of-bounds character indices are ignored
- Null values in cells are skipped
- Numeric conversion failures generate warnings but don't stop the operation

## Use Cases

1. **Fixing date formats**: Replace specific characters in date strings
2. **Code modifications**: Update specific positions in product codes
3. **Data cleaning**: Remove or replace characters at known positions
4. **Format standardization**: Ensure consistent formatting by replacing characters at specific positions