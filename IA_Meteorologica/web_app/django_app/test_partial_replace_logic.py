#!/usr/bin/env python3
"""
Test the partial replace logic implementation
"""

def partial_replace_char_by_char(original_value, partial_pattern, new_value):
    """
    Character by character replacement
    partial_pattern contains indices of characters to replace
    new_value contains the replacement string (char by char)
    """
    str_value = str(original_value)
    char_list = list(str_value)
    replacement_chars = list(new_value) if new_value else []
    
    # Replace characters at specified indices
    for i, char_idx in enumerate(partial_pattern):
        if isinstance(char_idx, int) and 0 <= char_idx < len(char_list):
            # Use corresponding replacement character if available
            if i < len(replacement_chars):
                char_list[char_idx] = replacement_chars[i]
            else:
                # If not enough replacement chars, use empty string
                char_list[char_idx] = ''
    
    return ''.join(char_list)


def partial_replace_complete(original_value, partial_pattern, new_value):
    """
    Complete replacement at specific indices
    partial_pattern contains indices of characters to replace
    new_value is the complete replacement string
    """
    str_value = str(original_value)
    char_list = list(str_value)
    
    # Remove characters at specified indices (in reverse order to maintain indices)
    for char_idx in sorted(partial_pattern, reverse=True):
        if isinstance(char_idx, int) and 0 <= char_idx < len(char_list):
            char_list.pop(char_idx)
    
    # Insert the new value at the first index position
    if partial_pattern and new_value:
        insert_pos = min(partial_pattern)
        # Insert each character of new_value starting at insert_pos
        for i, char in enumerate(new_value):
            if insert_pos + i <= len(char_list):
                char_list.insert(insert_pos + i, char)
    
    return ''.join(char_list)


def test_implementations():
    """Test both implementation types"""
    print("Testing Partial Replace Implementations")
    print("=" * 50)
    
    # Test Case 1: Character by Character
    print("\nTest Case 1: Character by Character Replacement")
    original = "ABC123"
    pattern = [0, 2, 4]  # Replace positions 0, 2, 4
    new_value = "XYZ"
    
    result = partial_replace_char_by_char(original, pattern, new_value)
    print(f"Original: {original}")
    print(f"Pattern: {pattern}")
    print(f"New value: {new_value}")
    print(f"Result: {result}")
    print(f"Expected: XBY1Z3")
    
    # Test Case 2: Not enough replacement characters
    print("\nTest Case 2: Not enough replacement characters")
    original = "ABC123"
    pattern = [0, 2, 4, 5]  # Replace 4 positions
    new_value = "XY"  # Only 2 replacement chars
    
    result = partial_replace_char_by_char(original, pattern, new_value)
    print(f"Original: {original}")
    print(f"Pattern: {pattern}")
    print(f"New value: {new_value}")
    print(f"Result: {result}")
    print(f"Expected: XBY1 (positions 4,5 removed)")
    
    # Test Case 3: Complete replacement
    print("\nTest Case 3: Complete Replacement")
    original = "ABC123"
    pattern = [1, 2, 3]  # Remove positions 1, 2, 3
    new_value = "XYZ"
    
    result = partial_replace_complete(original, pattern, new_value)
    print(f"Original: {original}")
    print(f"Pattern: {pattern}")
    print(f"New value: {new_value}")
    print(f"Result: {result}")
    print(f"Expected: AXYZ23")
    
    # Test Case 4: Numeric values
    print("\nTest Case 4: Numeric value handling")
    original = 123.45
    pattern = [0, 1]  # Replace first two digits
    new_value = "99"
    
    result = partial_replace_char_by_char(original, pattern, new_value)
    print(f"Original: {original}")
    print(f"Pattern: {pattern}")
    print(f"New value: {new_value}")
    print(f"Result: {result}")
    print(f"Can convert to float: {result} -> {float(result)}")
    
    # Test Case 5: Edge cases
    print("\nTest Case 5: Edge cases")
    original = "A"
    pattern = [0]
    new_value = "XYZ"
    
    result1 = partial_replace_char_by_char(original, pattern, new_value)
    result2 = partial_replace_complete(original, pattern, new_value)
    
    print(f"Original: {original}")
    print(f"Pattern: {pattern}")
    print(f"New value: {new_value}")
    print(f"Char by char result: {result1}")
    print(f"Complete result: {result2}")


if __name__ == '__main__':
    test_implementations()