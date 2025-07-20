import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import json
import joblib
import os
from datetime import datetime, timedelta


def get_model_config(model_type):
    """Get default hyperparameters for each model type"""
    configs = {
        'lstm': {
            'units': 50,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'layers': 2
        },
        'cnn': {
            'filters': 64,
            'kernel_size': 3,
            'pool_size': 2,
            'dense_units': 100,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'decision_tree': {
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto'
        },
        'transformer': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto'
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    }
    return configs.get(model_type, {})


def get_normalization_methods(model_type):
    """Get available normalization methods for each model type"""
    neural_methods = ['min_max', 'standard', 'robust']
    tree_methods = ['none', 'min_max', 'standard']
    
    if model_type in ['lstm', 'cnn', 'transformer']:
        return neural_methods
    elif model_type in ['decision_tree', 'random_forest', 'xgboost']:
        return tree_methods
    return ['none']


def get_metrics(model_type):
    """Get available metrics for each model type"""
    regression_metrics = ['mae', 'mse', 'rmse', 'r2']
    classification_metrics = ['accuracy', 'roc_auc', 'f1']
    
    # For weather prediction, we'll mainly use regression metrics
    return regression_metrics


def get_scaler(method):
    """Get the appropriate scaler based on method"""
    if method == 'min_max':
        return MinMaxScaler()
    elif method == 'standard':
        return StandardScaler()
    elif method == 'robust':
        return RobustScaler()
    return None


def prepare_data(session):
    """Prepare data for training"""
    df = pd.read_csv(session.dataset.file.path)
    
    # Select columns
    X = df[session.predictor_columns]
    y = df[session.target_columns]
    
    # Split data
    n = len(df)
    train_end = int(n * session.train_split)
    val_end = int(n * (session.train_split + session.val_split))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    # Normalize if needed
    scaler_X = get_scaler(session.normalization_method)
    scaler_y = get_scaler(session.normalization_method)
    
    if scaler_X:
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)
        
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        y_test = scaler_y.transform(y_test)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test), (scaler_X, scaler_y)


def train_sklearn_model(model_type, hyperparams, X_train, y_train, X_val, y_val):
    """Train scikit-learn models"""
    if model_type == 'decision_tree':
        model = DecisionTreeRegressor(**hyperparams)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(**hyperparams)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(**hyperparams)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # Calculate validation metrics
    val_pred = model.predict(X_val)
    history = {
        'train_loss': [mean_squared_error(y_train, model.predict(X_train))],
        'val_loss': [mean_squared_error(y_val, val_pred)]
    }
    
    return model, history


def train_model(session):
    """Main training function"""
    try:
        session.status = 'training'
        session.save()
        
        # Prepare data
        data, scalers = prepare_data(session)
        X_train, y_train, X_val, y_val, X_test, y_test = data
        scaler_X, scaler_y = scalers
        
        # Train model based on type
        if session.model_type in ['decision_tree', 'random_forest', 'xgboost']:
            model, history = train_sklearn_model(
                session.model_type, 
                session.hyperparameters,
                X_train, y_train, 
                X_val, y_val
            )
        else:
            # For neural networks, we'll need TensorFlow/PyTorch
            raise NotImplementedError(f"Neural network models not yet implemented")
        
        # Test evaluation
        y_pred = model.predict(X_test)
        
        if scaler_y:
            y_test = scaler_y.inverse_transform(y_test)
            y_pred = scaler_y.inverse_transform(y_pred)
        
        # Calculate test metrics
        test_results = {}
        for metric in session.selected_metrics:
            if metric == 'mae':
                test_results[metric] = float(mean_absolute_error(y_test, y_pred))
            elif metric == 'mse':
                test_results[metric] = float(mean_squared_error(y_test, y_pred))
            elif metric == 'rmse':
                test_results[metric] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            elif metric == 'r2':
                test_results[metric] = float(r2_score(y_test, y_pred))
        
        # Save model
        model_dir = 'media/models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/model_{session.id}.pkl"
        
        # Save model and scalers
        joblib.dump({
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'predictor_columns': session.predictor_columns,
            'target_columns': session.target_columns
        }, model_path)
        
        # Update session
        session.training_history = history
        session.test_results = test_results
        session.model_file = model_path
        session.status = 'completed'
        session.save()
        
    except Exception as e:
        session.status = 'failed'
        session.error_message = str(e)
        session.save()
        raise


def make_predictions(session, input_file):
    """Make predictions using trained model"""
    # Load model
    model_data = joblib.load(session.model_file.path)
    model = model_data['model']
    scaler_X = model_data['scaler_X']
    scaler_y = model_data['scaler_y']
    
    # Load input data
    df = pd.read_csv(input_file)
    X = df[session.predictor_columns]
    
    # Scale input if needed
    if scaler_X:
        X = scaler_X.transform(X)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Inverse transform if needed
    if scaler_y:
        predictions = scaler_y.inverse_transform(predictions)
    
    # Format results
    results = []
    for i, row in df.iterrows():
        result = {
            'input': row.to_dict(),
            'predictions': {}
        }
        for j, col in enumerate(session.target_columns):
            result['predictions'][col] = float(predictions[i, j])
        results.append(result)
    
    return results


def generate_weather_map_data(session, date):
    """Generate predictions for weather map visualization"""
    # Define grid points for Spain and France
    spain_france_grid = [
        # Spain major cities
        {'region': 'Madrid', 'latitude': 40.4168, 'longitude': -3.7038},
        {'region': 'Barcelona', 'latitude': 41.3851, 'longitude': 2.1734},
        {'region': 'Valencia', 'latitude': 39.4699, 'longitude': -0.3763},
        {'region': 'Sevilla', 'latitude': 37.3891, 'longitude': -5.9845},
        {'region': 'Bilbao', 'latitude': 43.2630, 'longitude': -2.9350},
        # France major cities
        {'region': 'Paris', 'latitude': 48.8566, 'longitude': 2.3522},
        {'region': 'Lyon', 'latitude': 45.7640, 'longitude': 4.8357},
        {'region': 'Marseille', 'latitude': 43.2965, 'longitude': 5.3698},
        {'region': 'Toulouse', 'latitude': 43.6047, 'longitude': 1.4442},
        {'region': 'Bordeaux', 'latitude': 44.8378, 'longitude': -0.5792},
    ]
    
    # Generate synthetic input data for prediction
    # In real implementation, this would come from weather APIs or historical patterns
    predictions = []
    for point in spain_france_grid:
        # Create prediction record
        pred = WeatherPrediction(
            training_session=session,
            prediction_date=date,
            region=point['region'],
            latitude=point['latitude'],
            longitude=point['longitude'],
            predictions={
                'temperature': np.random.uniform(10, 30),
                'humidity': np.random.uniform(40, 80),
                'pressure': np.random.uniform(1000, 1030),
                'wind_speed': np.random.uniform(0, 20)
            }
        )
        predictions.append(pred)
    
    return predictions