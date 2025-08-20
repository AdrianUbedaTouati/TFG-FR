# Confusion Matrix Label Debug Summary

## Issue
The confusion matrix in the training results page shows generic labels (Class_0, Class_1) instead of the actual class names from the dataset.

## Debug Logging Added

### Client-side (JavaScript)
Added console.log statements in `templates/training_results.html`:

1. **Line 1159-1165**: Log the full predictions analysis data and confusion matrix details
   - `console.log('Predictions analysis data:', analysisData.predictions_analysis);`
   - `console.log('First target data:', firstTarget);`
   - `console.log('Confusion matrix labels:', firstTarget.confusion_matrix.labels);`
   - `console.log('Confusion matrix matrix:', firstTarget.confusion_matrix.matrix);`

2. **Line 1427-1428**: Log the labels being used in the confusion matrix chart
   - `console.log('Using labels for confusion matrix:', labels);`
   - `console.log('Label source:', confusionMatrix.labels ? 'From server' : 'Generated default');`

### Server-side (Python)
Added debug prints in `ml_trainer/views/training_views.py`:

1. **Line 323, 329, 332**: Enhanced target encoder loading
   - Print loaded encoder columns
   - Print encoder classes for each column
   - Print if no encoders found in model file

2. **Line 403-405**: Log encoder usage when analyzing predictions
   - Print encoder object for each target column
   - Print encoder classes if encoder exists

3. **Line 481-518**: Enhanced _calculate_confusion_matrix method
   - Print if label_encoder is provided
   - Print unique labels found in data
   - Print label encoder classes
   - Print final labels being used
   - Print complete result being returned

## How to Debug

1. Open the browser developer console (F12)
2. Navigate to the training results page: http://127.0.0.1:8001/training-sessions/50/results/
3. Look for the console logs starting with:
   - "Predictions analysis data:"
   - "Creating confusion matrix:"
   - "Using labels for confusion matrix:"

4. Check the Django server console for prints starting with:
   - "[_calculate_confusion_matrix]"
   - "Found saved target encoders"
   - "Analyzing target column"

## Expected Flow

1. Model is saved with `target_encoders` dictionary containing LabelEncoder objects
2. When loading analysis, the encoders are retrieved from the saved model
3. The encoder is passed to `_calculate_confusion_matrix`
4. The method uses `encoder.classes_` to get real label names
5. These labels are sent to the client in the confusion matrix data

## Potential Issues to Check

1. **Encoder not saved**: Check if `target_encoders` is in the saved model file
2. **Encoder not loaded**: Check if the encoder is successfully loaded from the model
3. **Encoder not passed**: Check if the encoder is passed to _analyze_predictions
4. **Wrong encoder**: Check if the correct encoder is used for the target column
5. **Client-side issue**: Check if labels are received but not displayed correctly

## Next Steps

After examining the debug output:
1. Identify where in the flow the real labels are lost
2. Fix the specific issue (e.g., encoder not saved, not loaded, or not used correctly)
3. Remove the debug logging once the issue is fixed