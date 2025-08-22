"""
Auto-generated scikit-learn RANDOM_FOREST model code
Model: pepe
Type: RANDOM_FOREST
Generated at: 2025-08-22 02:12:55.212115+00:00

Configuration:
- Target columns: ['Summary']
- Predictor columns: 13 features
- Problem type: classification

Module 1 - Data Split Configuration:
- Split method: random
- Train size: 0.8
- Validation size: 0.1
- Test size: 0.1

Module 2 - Execution Configuration:
- Execution method: kfold
- Configuration: {"n_splits": 5, "shuffle": false}
"""

import numpy as np
import pandas as pd
import joblib
import json
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier



# =============================================================================
# MODULE 1: DATA SPLIT CONFIGURATION
# =============================================================================

class DataSplitter:
    """Module 1: Handles data splitting according to configured strategy"""
    
    def __init__(self, strategy="random", config=None):
        self.strategy = strategy
        self.config = config or {}
        
    def split(self, X, y):
        """Split data according to configured strategy"""
        train_size = self.config.get("train_size", 0.7)
        val_size = self.config.get("val_size", 0.15)
        test_size = self.config.get("test_size", 0.15)
        random_state = self.config.get("random_state", 42)
        
        if self.strategy == "random":
            # Random split with shuffling
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=True
            )
            val_proportion = val_size / (train_size + val_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_proportion, random_state=random_state, shuffle=True
            )
            
        elif self.strategy == "stratified":
            # Stratified split for classification
            y_stratify = y if len(y.shape) == 1 else y[:, 0]
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y_stratify, shuffle=True
            )
            val_proportion = val_size / (train_size + val_size)
            y_temp_stratify = y_temp if len(y_temp.shape) == 1 else y_temp[:, 0]
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_proportion, random_state=random_state, stratify=y_temp_stratify, shuffle=True
            )
            
        elif self.strategy == "temporal":
            # Temporal split maintaining order
            n_samples = len(X)
            train_end = int(n_samples * train_size)
            val_end = int(n_samples * (train_size + val_size))
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]
            X_test = X[val_end:]
            y_test = y[val_end:]
            
        elif self.strategy == "sequential":
            # Sequential split without shuffling
            n_samples = len(X)
            train_end = int(n_samples * train_size)
            val_end = int(n_samples * (train_size + val_size))
            
            X_train = X.iloc[:train_end] if hasattr(X, "iloc") else X[:train_end]
            y_train = y.iloc[:train_end] if hasattr(y, "iloc") else y[:train_end]
            X_val = X.iloc[train_end:val_end] if hasattr(X, "iloc") else X[train_end:val_end]
            y_val = y.iloc[train_end:val_end] if hasattr(y, "iloc") else y[train_end:val_end]
            X_test = X.iloc[val_end:] if hasattr(X, "iloc") else X[val_end:]
            y_test = y.iloc[val_end:] if hasattr(y, "iloc") else y[val_end:]
            
        else:
            # Default to random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=True
            )
            val_proportion = val_size / (train_size + val_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_proportion, random_state=random_state, shuffle=True
            )
            
        return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# MODULE 2: EXECUTION CONFIGURATION
# =============================================================================

class ExecutionStrategy:
    """Module 2: Handles model execution strategy (cross-validation, etc.)"""
    
    def __init__(self, method="kfold", config=None):
        self.method = method
        self.config = config or {}
        
    def execute(self, model, X_train, y_train, X_val, y_val):
        """Execute training according to configured strategy"""
        
        if self.method == "standard":
            # Standard training without cross-validation
            print("Executing standard training...")
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            print(f"Training score: {train_score:.4f}")
            print(f"Validation score: {val_score:.4f}")
            return {"train_score": train_score, "val_score": val_score}
            
        elif self.method == "kfold":
            # K-Fold Cross Validation
            n_splits = self.config.get("n_splits", 5)
            print(f"Executing {n_splits}-fold cross validation...")
            from sklearn.model_selection import KFold, cross_val_score
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error")
            print(f"CV MSE: {-scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            # Train final model on full training data
            model.fit(X_train, y_train)
            return {"cv_scores": scores.tolist(), "cv_mean": -scores.mean(), "cv_std": scores.std()}
            
        elif self.method == "stratified_kfold":
            # Stratified K-Fold for classification
            n_splits = self.config.get("n_splits", 5)
            print(f"Executing stratified {n_splits}-fold cross validation...")
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            y_stratify = y_train if len(y_train.shape) == 1 else y_train[:, 0]
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
            print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            model.fit(X_train, y_train)
            return {"cv_scores": scores.tolist(), "cv_mean": scores.mean(), "cv_std": scores.std()}
            
        elif self.method == "time_series_split":
            # Time Series Split
            n_splits = self.config.get("n_splits", 5)
            print(f"Executing time series split with {n_splits} splits...")
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            tss = TimeSeriesSplit(n_splits=n_splits)
            scores = cross_val_score(model, X_train, y_train, cv=tss, scoring="neg_mean_squared_error")
            print(f"CV MSE: {-scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            model.fit(X_train, y_train)
            return {"cv_scores": scores.tolist(), "cv_mean": -scores.mean(), "cv_std": scores.std()}
            
        else:
            # Default to standard execution
            print("Executing standard training (default)...")
            model.fit(X_train, y_train)
            return {}


def create_model():
    """
    Create and return the configured model
    """
    print("Creating RANDOM_FOREST model...")

    # Model parameters
    params = {
        "n_estimators": 300,
        "max_depth": None,
        "max_features": "sqrt",
        "criterion": "gini",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "oob_score": True,
        "n_jobs": -1,  # Use all CPU cores
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
    predictor_columns = ['Precip Type', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)', 'h_sin', 'h_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
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


def train_model(model, X, y, data_splitter=None, execution_strategy=None):
    """
    Train the model using Module 1 (Data Split) and Module 2 (Execution)
    """
    # Use default modules if not provided
    if data_splitter is None:
        data_splitter = DataSplitter()
    if execution_strategy is None:
        execution_strategy = ExecutionStrategy()
    
    # Module 1: Split the data
    print("\n" + "="*50)
    print("MODULE 1: DATA SPLITTING")
    print("="*50)
    X_train, X_val, X_test, y_train, y_val, y_test = data_splitter.split(X, y)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Module 2: Execute training strategy
    print("\n" + "="*50)
    print("MODULE 2: EXECUTION STRATEGY")
    print("="*50)
    execution_results = execution_strategy.execute(model, X_train, y_train, X_val, y_val)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
    
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


def create_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', save_path=None):
    """
    Create and display confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Use seaborn heatmap for better visualization
    if labels is None:
        labels = [f'Class_{i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Rotate labels if they are long
    if any(len(str(label)) > 10 for label in labels):
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Confusion matrix saved to: {save_path}')
    
    plt.show()
    return cm


def create_scatter_plot(y_true, y_pred, title='Predictions vs Actual', save_path=None):
    """
    Create scatter plot of predictions vs actual values
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate and display R² score
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=12, verticalalignment='top')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Scatter plot saved to: {save_path}')
    
    plt.show()
    return r2


def create_residuals_plot(y_true, y_pred, title='Residuals Analysis', save_path=None):
    """
    Create residuals plot for regression analysis
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)
    ax2.axvline(mean_residuals, color='red', linestyle='--', 
               label=f'Mean: {mean_residuals:.4f}')
    ax2.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Residuals plot saved to: {save_path}')
    
    plt.show()
    
    return {'mean': mean_residuals, 'std': std_residuals}


def analyze_model_performance(model, X_test, y_test, target_columns, 
                             model_name='Model', save_plots=True):
    """
    Comprehensive model performance analysis with visualizations
    """
    print('Analyzing model performance...')
    print('=' * 50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Ensure predictions are properly shaped
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
    
    analysis_results = {}
    
    # Analyze each target column
    for i, target_col in enumerate(target_columns):
        print(f'\nAnalyzing target: {target_col}')
        print('-' * 30)
        
        y_true_col = y_test[:, i] if y_test.shape[1] > 1 else y_test.flatten()
        y_pred_col = y_pred[:, i] if y_pred.shape[1] > 1 else y_pred.flatten()
        
        # Determine if classification or regression
        unique_values = len(np.unique(y_true_col))
        is_classification = unique_values <= 20  # Threshold for classification
        
        if is_classification:
            print(f'Classification Analysis for {target_col}')
            
            # Classification metrics
            from sklearn.metrics import accuracy_score, classification_report
            
            # Round predictions for classification
            y_pred_rounded = np.round(y_pred_col).astype(int)
            y_true_int = y_true_col.astype(int)
            
            accuracy = accuracy_score(y_true_int, y_pred_rounded)
            print(f'Accuracy: {accuracy:.4f}')
            
            # Create confusion matrix
            save_path = f'{model_name}_{target_col}_confusion_matrix.png' if save_plots else None
            cm = create_confusion_matrix(
                y_true_int, y_pred_rounded,
                title=f'Confusion Matrix - {target_col}',
                save_path=save_path
            )
            
            analysis_results[target_col] = {
                'type': 'classification',
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist()
            }
        
        else:
            print(f'Regression Analysis for {target_col}')
            
            # Regression metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(y_true_col, y_pred_col)
            mse = mean_squared_error(y_true_col, y_pred_col)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_col, y_pred_col)
            
            print(f'MAE:  {mae:.4f}')
            print(f'MSE:  {mse:.4f}')
            print(f'RMSE: {rmse:.4f}')
            print(f'R²:   {r2:.4f}')
            
            # Create scatter plot
            save_path = f'{model_name}_{target_col}_scatter.png' if save_plots else None
            r2_calc = create_scatter_plot(
                y_true_col, y_pred_col,
                title=f'Predictions vs Actual - {target_col}',
                save_path=save_path
            )
            
            # Create residuals plot
            save_path = f'{model_name}_{target_col}_residuals.png' if save_plots else None
            residuals_stats = create_residuals_plot(
                y_true_col, y_pred_col,
                title=f'Residuals Analysis - {target_col}',
                save_path=save_path
            )
            
            analysis_results[target_col] = {
                'type': 'regression',
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'residuals_stats': residuals_stats
            }
    
    print('\nAnalysis completed!')
    return analysis_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_FILE = "your_dataset.csv"  # UPDATE THIS PATH
    
    # Module 1 Configuration
    split_config = {
        "train_size": 0.8,
        "val_size": 0.1,
        "test_size": 0.1,
        "random_state": 42
}
    data_splitter = DataSplitter(strategy="random", config=split_config)
    
    # Module 2 Configuration
    execution_config = {
        "n_splits": 5,
        "shuffle": false
}
    execution_strategy = ExecutionStrategy(method="kfold", config=execution_config)
    
    try:
        # Step 1: Load and preprocess data
        print("STEP 1: Loading and preprocessing data...")
        X, y, predictor_cols, target_cols = load_and_preprocess_data(DATA_FILE)
        
        # Step 2: Create model
        print("\nSTEP 2: Creating model...")
        model = create_model()
        
        # Step 3: Train model using Module 1 and Module 2
        print("\nSTEP 3: Training model with configured modules...")
        X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_model(
            model, X, y, data_splitter=data_splitter, execution_strategy=execution_strategy
        )
        
        # Step 4: Evaluate model
        print("\nSTEP 4: Evaluating model...")
        train_metrics = evaluate_model(model, y_train, y_pred_train, "Training Set", target_cols)
        test_metrics = evaluate_model(model, y_test, y_pred_test, "Test Set", target_cols)
        
        # Step 5: Analyze model performance with visualizations
        print("\nSTEP 5: Creating analysis visualizations...")
        analysis_results = analyze_model_performance(
            model, X_test, y_test, target_cols,
            model_name="pepe",
            save_plots=True
        )
        
        # Step 6: Save model
        print("\nSTEP 6: Saving model...")
        model_filename = "pepe_model.pkl"
        save_model(model, model_filename)
        
        # Optional: Save model info
        model_info = {
            "model_name": "pepe",
            "model_type": "random_forest",
            "predictor_columns": predictor_cols,
            "target_columns": target_cols,
            "hyperparameters": model.get_params(),
            "generated_at": "2025-08-22 02:12:55.212115+00:00"
        }
        
        with open("model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        print("Model info saved to: model_info.json")
        
        print("\nTraining completed successfully!")
        
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
#    - Predictors: ['Precip Type', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)', 'h_sin', 'h_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
#    - Targets: ['Summary']
# 3. Run: python this_script.py
# =============================================================================