#!/usr/bin/env python
"""
Fix for column removal issue in normalization process
"""

# The issue is in ml_trainer/views/normalization_views.py at line 415
# Current code:
# new_cols = [col for col in normalized_df.columns if col.startswith(f"{column}_") and col != column]

# The problem: column names are created without underscore separator
# Example: "numeric_col_step1" is created as "numeric_colstep1"

# FIX OPTION 1: Change the detection pattern (line 415)
# From:
#   new_cols = [col for col in normalized_df.columns if col.startswith(f"{column}_") and col != column]
# To:
#   new_cols = [col for col in normalized_df.columns if col.startswith(column) and col != column and len(col) > len(column)]

# FIX OPTION 2: Change column naming to include underscore
# Lines 1003-1004, 1008-1009, 1021-1022, 1026-1027
# From:
#   new_column_name = f"{column}{suffix}"
# To:
#   new_column_name = f"{column}{suffix}" if suffix.startswith("_") else f"{column}_{suffix}"

# RECOMMENDED FIX: Option 1 is simpler and less invasive

print("""
Column Removal Fix Instructions:

1. Open ml_trainer/views/normalization_views.py

2. Find line 415:
   new_cols = [col for col in normalized_df.columns if col.startswith(f"{column}_") and col != column]

3. Replace with:
   new_cols = [col for col in normalized_df.columns if col.startswith(column) and col != column and len(col) > len(column)]

4. Also fix line 402 (for column tracking):
   From:
   new_cols = [col for col in normalized_df.columns if col.startswith(f"{current_column}_")]
   To:
   new_cols = [col for col in normalized_df.columns if col.startswith(current_column) and col != current_column and len(col) > len(current_column)]

5. Also fix lines 1231-1232 in the preview function:
   From:
   new_cols = [col for col in current_df.columns if col.startswith(f"{current_column}_step{step_index + 1}")]
   To:
   new_cols = [col for col in current_df.columns if col.startswith(f"{current_column}_step{step_index + 1}") or col == f"{current_column}_step{step_index + 1}"]

This will properly detect columns created during normalization and remove the original when appropriate.
""")