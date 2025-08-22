# Cross-Validation Train/Validation Split Fix Summary

## Overview
This document summarizes the fixes implemented to ensure the CV train/validation split configuration is properly saved, loaded, and used during model training.

## Issues Fixed

### 1. CV Split Not Being Saved in Model Updates
- **Issue**: When updating a model, the CV train/validation split percentages were not being included in the execution configuration.
- **Fix**: Modified the `updateModel` function in `models.html` to include `cv_train_size` and `cv_val_size` when saving k-fold based execution methods.

### 2. CV Split Not Being Used During Training
- **Issue**: The CV train/validation split was configured in the UI but not actually applied during cross-validation training.
- **Fix**: Implemented the split logic in the training process to divide each fold's training data according to the configured percentages.

## Implementation Details

### 1. Frontend Changes (models.html)

#### A. Model Update Function
Added CV split configuration to the execution config in `updateModel`:
```javascript
if (executionMethod === 'kfold' || executionMethod === 'stratified_kfold') {
    executionConfig.cv_train_size = parseInt(document.getElementById('cvTrainSize').value) / 100;
    executionConfig.cv_val_size = parseInt(document.getElementById('cvValSize').value) / 100;
}
```

### 2. Backend Changes

#### A. Execution Strategies (execution_config.py)
Added `cv_train_size` and `cv_val_size` attributes to all k-fold based execution strategies:
- `KFoldExecution`
- `StratifiedKFoldExecution`
- `RepeatedKFoldExecution`
- `RepeatedStratifiedKFoldExecution`

Example:
```python
self.cv_train_size = config.get('cv_train_size', 0.8)
self.cv_val_size = config.get('cv_val_size', 0.2)
```

#### B. Training Logic (ml_utils.py)
Implemented the actual split within each fold during cross-validation:

1. **Extract CV split configuration** from execution strategy
2. **Split fold training data** if CV validation is configured:
   ```python
   if cv_val_size > 0 and cv_train_size < 1.0:
       X_fold_train_split, X_fold_val_split, y_fold_train_split, y_fold_val_split = train_test_split(
           X_fold_train, y_fold_train, 
           test_size=cv_val_size,
           random_state=execution_config.get('random_state', None)
       )
   ```
3. **Use split data for training** instead of full fold data
4. **Log detailed information** about the splits for transparency

### 3. Training Process Flow

1. **Standard Execution**: Uses Module 1 data splits (train/val/test)
2. **Cross-Validation Execution**:
   - Combines train+validation data from Module 1
   - For each fold:
     - Splits data into fold_train and fold_test
     - If CV split is configured (e.g., 80/20):
       - Further splits fold_train into actual_train (80%) and actual_val (20%)
       - Uses actual_train for model training
       - Uses actual_val for validation during training (e.g., early stopping)
       - Uses fold_test for final evaluation
     - If no CV split configured:
       - Uses entire fold_train for training
       - Uses fold_test for evaluation

### 4. User Experience

- CV train/validation split configuration is now properly saved with the model
- When loading a model for editing, the CV split values are restored
- During training, users see detailed logs showing:
  - Total fold data sizes
  - CV train/validation split within each fold
  - Actual sample counts for each subset
  - Clear explanations of what each dataset is used for

## Benefits

1. **Flexibility**: Users can configure validation data within CV folds for early stopping and hyperparameter tuning
2. **Persistence**: Configuration is saved with the model and restored when editing
3. **Transparency**: Clear logging shows exactly how data is being split and used
4. **Robustness**: Fallback logic ensures training continues even if splits fail

## Testing Recommendations

1. Create a model with k-fold CV and custom train/val split (e.g., 70/30)
2. Save the model and reload it - verify splits are preserved
3. Train the model and verify logs show correct data splits
4. Check that early stopping (if configured) uses the validation split within folds
5. Verify final evaluation uses the held-out test portion of each fold