#!/usr/bin/env python
"""
Test script to verify Random Forest training fixes
"""
import os
import sys
import django
import numpy as np
import pandas as pd
from datetime import datetime

# Setup Django
sys.path.append('/mnt/c/Users/andri/Desktop/TFG_FR/IA_Meteorologica/web_app/django_app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.models import TrainingSession, Dataset
from ml_trainer.ml_utils import train_model


def test_random_forest_training():
    """Test Random Forest training with feature engineering"""
    print("=" * 50)
    print("Testing Random Forest Training with Preprocessing")
    print("=" * 50)
    
    # Create a test dataset with mixed features
    np.random.seed(42)
    n_samples = 1000
    
    # Numeric features
    temp = np.random.normal(20, 5, n_samples)
    humidity = np.random.normal(60, 15, n_samples)
    pressure = np.random.normal(1013, 10, n_samples)
    
    # Cyclic feature (hour of day)
    hour = np.random.randint(0, 24, n_samples)
    
    # Categorical feature (weather type)
    weather_types = ['Clear', 'Cloudy', 'Rainy', 'Foggy']
    weather = np.random.choice(weather_types, n_samples)
    
    # Target variable (correlated with features)
    target = (
        0.5 * temp + 
        0.3 * humidity + 
        0.1 * pressure + 
        2 * np.sin(2 * np.pi * hour / 24) +  # Cyclic pattern
        np.where(weather == 'Rainy', 5, 0) +  # Categorical effect
        np.random.normal(0, 1, n_samples)  # Noise
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'Temperature': temp,
        'Humidity': humidity,
        'Pressure': pressure,
        'Hour': hour,
        'WeatherType': weather,
        'Target': target
    })
    
    # Save test dataset
    test_file = '/tmp/test_rf_dataset.csv'
    df.to_csv(test_file, index=False)
    print(f"Created test dataset with {n_samples} samples")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Shape: {df.shape}")
    print()
    
    # Create a training session
    session = TrainingSession.objects.create(
        name="Test Random Forest with Preprocessing",
        dataset_id=1,  # Assuming a dataset exists
        model_type='random_forest',
        predictor_columns=['Temperature', 'Humidity', 'Pressure', 'Hour', 'WeatherType'],
        target_columns=['Target'],
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        hyperparameters={
            'problem_type': 'regression',
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'encode_categorical': True,
            'add_cyclic_features': True,
            'encoding_method': 'onehot'
        },
        status='pending'
    )
    
    print(f"Created training session ID: {session.id}")
    print("Hyperparameters:")
    for key, value in session.hyperparameters.items():
        print(f"  {key}: {value}")
    print()
    
    # Mock the dataset file path
    session.dataset.file.name = test_file
    
    print("Starting training...")
    print("-" * 50)
    
    try:
        # Train the model
        train_model(session)
        
        print("-" * 50)
        print("Training completed successfully!")
        print()
        
        # Check results
        session.refresh_from_db()
        print(f"Final status: {session.status}")
        print(f"Preprocessing info: {session.preprocessing_info}")
        
        if session.preprocessing_info:
            print("\nPreprocessing details:")
            print(f"  Original features: {len(session.predictor_columns)}")
            print(f"  Features after preprocessing: {session.preprocessing_info.get('n_features_after_preprocessing')}")
            print(f"  Categorical columns: {session.preprocessing_info.get('categorical_columns')}")
            print(f"  Cyclic columns: {session.preprocessing_info.get('cyclic_columns')}")
        
        if session.training_logs:
            print(f"\nTraining logs ({len(session.training_logs)} entries):")
            for log in session.training_logs[-10:]:  # Show last 10 logs
                print(f"  [{log.get('timestamp', '')}] {log.get('message', '')}")
        
        if session.test_results:
            print("\nTest results:")
            for metric, value in session.test_results.items():
                print(f"  {metric}: {value}")
        
        # Test prediction to ensure preprocessing works
        print("\nTesting prediction...")
        test_input = pd.DataFrame([{
            'Temperature': 25,
            'Humidity': 70,
            'Pressure': 1015,
            'Hour': 14,
            'WeatherType': 'Cloudy'
        }])
        test_input_file = '/tmp/test_rf_input.csv'
        test_input.to_csv(test_input_file, index=False)
        
        from ml_trainer.ml_utils import make_predictions
        predictions = make_predictions(session, test_input_file)
        print(f"Prediction successful! Result: {predictions[0]['predictions']}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Check session logs for more details
        session.refresh_from_db()
        if session.error_message:
            print(f"\nSession error message: {session.error_message}")
        
        if session.training_logs:
            print("\nLast training logs:")
            for log in session.training_logs[-5:]:
                print(f"  {log.get('message', '')}")
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists(test_input_file):
            os.remove(test_input_file)


if __name__ == "__main__":
    test_random_forest_training()