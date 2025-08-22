# Code Generation Fixes

## Overview
This document summarizes the fixes applied to the code generation module to address three issues:
1. Remove unnecessary rounding in classification predictions
2. Remove unused cv_train_size/cv_val_size parameters from generated code
3. Use real class names in confusion matrix instead of generic labels

## Changes Made

### 1. Classification Prediction Handling
**Issue**: The generated code was unnecessarily rounding predictions for classification tasks, which is incorrect for Random Forest classifiers that already output class labels.

**Fix**: Updated the classification code to:
- Check if the model supports probability predictions
- Work directly with the predictions without rounding
- Only convert to int for models that need it

**Before**:
```python
# Round predictions for classification
y_pred_rounded = np.round(y_pred_col).astype(int)
y_true_int = y_true_col.astype(int)
accuracy = accuracy_score(y_true_int, y_pred_rounded)
```

**After**:
```python
# For Random Forest classification, predictions are already class labels
# No need to round - work with predictions directly
if hasattr(model, 'predict_proba'):
    # If model supports probability predictions, use regular predict for labels
    y_pred_labels = y_pred_col
else:
    # For other models, ensure integer labels
    y_pred_labels = y_pred_col.astype(int)

y_true_labels = y_true_col.astype(int)
accuracy = accuracy_score(y_true_labels, y_pred_labels)
```

### 2. CV Parameters
**Issue**: The code was supposed to have unused cv_train_size/cv_val_size parameters documented but not used.

**Analysis**: After searching the code_generator.py file, no cv_train_size or cv_val_size parameters were found. These parameters are handled at the training level in ml_utils.py, not in the generated code. The generated code receives the already-split data, so no changes were needed.

### 3. Confusion Matrix Labels
**Issue**: The confusion matrix was using generic labels like "Class_0", "Class_1" instead of actual class values.

**Fix**: Updated the confusion matrix creation to:
- Extract unique labels from the actual data
- Pass real class names to the confusion matrix function
- Update the default label generation to use actual values

**Before**:
```python
# Create confusion matrix
cm = create_confusion_matrix(
    y_true_int, y_pred_rounded,
    title=f'Confusion Matrix - {target_col}',
    save_path=save_path
)

# In create_confusion_matrix function:
if labels is None:
    labels = [f'Class_{i}' for i in range(len(cm))]
```

**After**:
```python
# Get unique class labels from the data
unique_labels = np.unique(np.concatenate([y_true_labels, y_pred_labels]))
class_names = [f'{int(label)}' for label in unique_labels]

# Create confusion matrix with actual class labels
cm = create_confusion_matrix(
    y_true_labels, y_pred_labels,
    labels=class_names,
    title=f'Confusion Matrix - {target_col}',
    save_path=save_path
)

# In create_confusion_matrix function:
if labels is None:
    # Extract unique labels from the data
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = [str(label) for label in unique_labels]
```

## Impact
These changes ensure that:
1. Classification models work correctly without unnecessary rounding
2. Confusion matrices display actual class values instead of generic labels
3. The generated code is more accurate and professional

## Files Modified
- `/mnt/c/Users/andri/Desktop/TFG_FR/IA_Meteorologica/web_app/django_app/ml_trainer/code_generator.py`
  - Updated classification prediction handling (2 occurrences)
  - Updated confusion matrix label generation (2 occurrences)