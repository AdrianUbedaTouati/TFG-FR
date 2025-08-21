"""
Auto-generated scikit-learn RANDOM_FOREST model code
Model: Random forest
Type: RANDOM_FOREST
Generated at: 2025-08-20 11:22:58.129399+00:00

Configuration:
- Target columns: ['Summary']
- Predictor columns: 15 features
- Problem type: classification
"""

import numpy as np
import pandas as pd
import joblib
import json
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier



def create_model():
    """
    Create and return the configured model
    """
    print("Creating RANDOM_FOREST model...")

    # Model parameters
    params = {
        "n_estimators": 1000,
        "max_depth": None,
        "max_features": "sqrt",
        "criterion": "gini",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "oob_score": True,
        "n_jobs": -1,  # Use all CPU cores
        "random_state": 44,
        "class_weight": "balanced"
    }

    # Preset: Balanceado (buen compromiso velocidad/precisión)
    model = RandomForestClassifier(**params)

    print("Model created successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Parameters: {model.get_params()}")
    return model


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset
    """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)})")

    # Define columns
    predictor_columns = ['Precip Type', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)', 'trend', 'h_sin', 'h_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos', 'tz_offset_hours']
    target_columns = ['Summary']

    # Validate columns
    missing_predictors = [col for col in predictor_columns if col not in df.columns]
    missing_targets = [col for col in target_columns if col not in df.columns]
    
    if missing_predictors:
        raise ValueError(f"Missing predictor columns: {missing_predictors}")
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    # Extract features and target
    X = df[predictor_columns].copy()
    y = df[target_columns[0]] if len(target_columns) == 1 else df[target_columns]

    # Handle missing values
    print("\nHandling missing values...")
    print(f"Missing values in features: {X.isnull().sum().sum()}")
    print(f"Missing values in target: {y.isnull().sum() if hasattr(y, 'isnull') else 0}")

    # Fill missing values with mean
    X = X.fillna(X.mean())
    if hasattr(y, 'fillna'):
        y = y.fillna(y.mean() if y.dtype in [np.float64, np.float32] else y.mode()[0])

    return X, y, predictor_columns, target_columns


def train_model(model, X, y, test_size=0.2, cv_folds=None):
    """
    Train the model with optional cross-validation
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Perform cross-validation if requested
    if cv_folds:
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
        print(f"CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train the model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test


def evaluate_model(model, y_true, y_pred, dataset_name="Dataset", target_names=None):
    """
    Evaluate model performance (supports multi-output)
    """
    print(f"\n=== {dataset_name} Evaluation ===")
    
    # Check if multi-output
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        print(f"Multi-output evaluation: {y_true.shape[1]} targets")
        return evaluate_multi_output(y_true, y_pred, target_names)
    

    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return {"accuracy": accuracy}


def evaluate_multi_output(y_true, y_pred, target_names=None):
    """Evaluate multi-output predictions"""
    results = {}
    n_outputs = y_true.shape[1]
    
    for i in range(n_outputs):
        target_name = target_names[i] if target_names and i < len(target_names) else f"Target_{i+1}"
        print(f"\n--- Evaluation for {target_name} ---")
        
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        results[target_name] = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
    
    return results



def save_model(model, filename="model.pkl"):
    """
    Save the trained model
    """
    joblib.dump(model, filename)
    print(f"\nModel saved to: {filename}")


def load_model(filename="model.pkl"):
    """
    Load a saved model
    """
    model = joblib.load(filename)
    print(f"Model loaded from: {filename}")
    return model


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_FILE = "db.csv"  # ⚠️ UPDATE THIS PATH
    TEST_SIZE = 0.2
    USE_CROSS_VALIDATION = None
    
    try:
        # Step 1: Load and preprocess data
        print("STEP 1: Loading and preprocessing data...")
        X, y, predictor_cols, target_cols = load_and_preprocess_data(DATA_FILE)
        
        # Step 2: Create model
        print("\nSTEP 2: Creating model...")
        model = create_model()
        
        # Step 3: Train model
        print("\nSTEP 3: Training model...")
        X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_model(
            model, X, y, test_size=TEST_SIZE, cv_folds=USE_CROSS_VALIDATION
        )
        
        # Step 4: Evaluate model
        print("\nSTEP 4: Evaluating model...")
        train_metrics = evaluate_model(model, y_train, y_pred_train, "Training Set", target_cols)
        test_metrics = evaluate_model(model, y_test, y_pred_test, "Test Set", target_cols)
        
        # Step 5: Save model
        print("\nSTEP 5: Saving model...")
        model_filename = "random_forest_model.pkl"
        save_model(model, model_filename)
        
        # Optional: Save model info
        model_info = {
            "model_name": "Random forest",
            "model_type": "random_forest",
            "predictor_columns": predictor_cols,
            "target_columns": target_cols,
            "hyperparameters": model.get_params(),
            "generated_at": "2025-08-20 11:22:58.129399+00:00"
        }
        
        with open("model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        print("Model info saved to: model_info.json")
        
        print("\n✅ Training completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file: {DATA_FILE}")
        print("Please update the DATA_FILE variable with your dataset path.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================
# 1. Update DATA_FILE with your CSV file path
# 2. Ensure your dataset contains these columns:
#    - Predictors: ['Precip Type', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)', 'trend', 'h_sin', 'h_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos', 'tz_offset_hours']
#    - Targets: ['Summary']
# 3. Run: python this_script.py
# =============================================================================