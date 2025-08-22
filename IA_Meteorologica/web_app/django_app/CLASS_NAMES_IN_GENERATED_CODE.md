# Class Names in Generated Code

## Overview
This document summarizes the changes made to ensure that the generated code saves and uses actual class names in confusion matrices instead of showing numeric labels (0, 1, 2, etc.).

## Changes Made

### 1. Data Loading Function
Updated `load_and_preprocess_data()` to handle categorical target encoding and save the encoders:
- Detects categorical target columns
- Uses LabelEncoder to encode them
- Saves the encoders in a `target_encoders` dictionary
- Returns the encoders along with the processed data

### 2. Model Saving Function
Updated `save_model()` to save target encoders with the model:
- Accepts `target_encoders` parameter
- Saves both model and encoders in a dictionary
- Prints the class names when saving for reference
- Maintains backward compatibility with old format

### 3. Model Loading Function
Updated `load_model()` to load target encoders:
- Loads the model data dictionary
- Extracts both model and target encoders
- Handles legacy format (just model) for backward compatibility
- Prints the class names when loading

### 4. Analysis Function
Updated `analyze_model_performance()` to use real class names:
- Accepts `target_encoders` parameter
- When creating confusion matrix for classification:
  - Checks if target encoder exists for the column
  - Maps numeric labels back to original class names
  - Falls back to numeric labels if no encoder available

### 5. Main Execution
Updated the main execution block to:
- Receive target encoders from data loading
- Pass target encoders to analysis function
- Save target encoders with the model

## Example Output
When the generated code runs, it will now:

1. During data loading:
```
Encoding categorical target column: weather_condition
Classes in weather_condition: ['Clear', 'Cloudy', 'Rainy', 'Snowy']
```

2. During model saving:
```
Model saved to: weather_model.pkl
Target column 'weather_condition' classes: ['Clear', 'Cloudy', 'Rainy', 'Snowy']
```

3. In confusion matrix:
Instead of showing labels like "0", "1", "2", "3", it will show:
"Clear", "Cloudy", "Rainy", "Snowy"

## Benefits
1. **Better interpretability**: Users can see actual class names in confusion matrices
2. **Consistency**: Generated code matches the behavior of the web interface
3. **Portability**: Class mappings are saved with the model for future use
4. **Backward compatibility**: Still works with models that don't have encoders

## Files Modified
- `/mnt/c/Users/andri/Desktop/TFG_FR/IA_Meteorologica/web_app/django_app/ml_trainer/code_generator.py`
  - Updated `_generate_data_loading_function()`
  - Updated save/load model functions in generated code
  - Updated `analyze_model_performance()` in generated code
  - Updated main execution block to pass target encoders