#!/usr/bin/env python
"""
Test script to verify neural network implementation
Run with: python test_neural_networks.py
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

import numpy as np
import pandas as pd
from ml_trainer.models import TrainingSession, Dataset, ModelDefinition
from ml_trainer.ml_utils import train_model
import json

def test_neural_networks():
    """Test neural network training functionality"""
    
    print("=" * 50)
    print("Testing Neural Network Implementation")
    print("=" * 50)
    
    # Create a sample dataset for testing
    print("\n1. Creating sample dataset...")
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='H')
    
    # Create synthetic weather data
    data = pd.DataFrame({
        'Date': dates,
        'Temperature': np.sin(np.arange(1000) * 0.1) * 10 + 20 + np.random.randn(1000) * 2,
        'Humidity': np.cos(np.arange(1000) * 0.05) * 20 + 60 + np.random.randn(1000) * 5,
        'Pressure': np.sin(np.arange(1000) * 0.03) * 5 + 1013 + np.random.randn(1000) * 2,
        'WindSpeed': np.abs(np.sin(np.arange(1000) * 0.08) * 10 + 5 + np.random.randn(1000) * 2),
        'Precipitation': np.maximum(0, np.random.randn(1000) * 2 + 1)
    })
    
    # Save dataset
    test_data_path = 'media/datasets/test_neural_network_data.csv'
    os.makedirs('media/datasets', exist_ok=True)
    data.to_csv(test_data_path, index=False)
    print(f"Sample data saved to {test_data_path}")
    
    # Create dataset in database
    dataset = Dataset.objects.create(
        name="Test Neural Network Dataset",
        file=test_data_path,
        short_description="Test dataset for neural networks",
        is_normalized=False
    )
    print(f"Dataset created with ID: {dataset.id}")
    
    # Test each neural network type
    neural_models = ['lstm', 'gru', 'cnn']
    
    for model_type in neural_models:
        print(f"\n2. Testing {model_type.upper()} model...")
        
        # Create model definition
        model_def = ModelDefinition.objects.create(
            name=f"Test {model_type.upper()} Model",
            description=f"Testing {model_type} implementation",
            model_type=model_type,
            dataset=dataset,
            predictor_columns=['Temperature', 'Humidity', 'Pressure'],
            target_columns=['WindSpeed', 'Precipitation'],
            hyperparameters={
                'epochs': 5,  # Small number for testing
                'batch_size': 32,
                'learning_rate': 0.001,
                'units': 32,
                'dropout': 0.2,
                'sequence_length': 10,
                'optimizer': 'adam',
                'loss': 'mse'
            }
        )
        
        # Create training session
        session = TrainingSession.objects.create(
            model_definition=model_def,
            name=f"Test {model_type.upper()} Training",
            dataset=dataset,
            model_type=model_type,
            predictor_columns=['Temperature', 'Humidity', 'Pressure'],
            target_columns=['WindSpeed', 'Precipitation'],
            normalization_method='min_max',
            hyperparameters=model_def.hyperparameters,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            selected_metrics=['mae', 'mse', 'rmse', 'r2']
        )
        
        print(f"Created training session ID: {session.id}")
        
        try:
            # Train the model
            print(f"Training {model_type.upper()} model...")
            train_model(session)
            
            # Refresh session from database
            session.refresh_from_db()
            
            if session.status == 'completed':
                print(f"✓ {model_type.upper()} training completed successfully!")
                print(f"  Test Results: {json.dumps(session.test_results, indent=2)}")
            else:
                print(f"✗ {model_type.upper()} training failed: {session.error_message}")
                
        except Exception as e:
            print(f"✗ Error training {model_type.upper()}: {str(e)}")
    
    # Test custom architecture
    print("\n3. Testing Custom Architecture...")
    custom_architecture = [
        {'type': 'LSTM', 'params': {'units': 64, 'activation': 'tanh', 'return_sequences': True}},
        {'type': 'Dropout', 'params': {'rate': 0.3}},
        {'type': 'LSTM', 'params': {'units': 32, 'activation': 'tanh', 'return_sequences': False}},
        {'type': 'Dense', 'params': {'units': 16, 'activation': 'relu'}},
        {'type': 'Dense', 'params': {'units': 2, 'activation': 'linear'}}
    ]
    
    # Create custom model
    custom_model_def = ModelDefinition.objects.create(
        name="Test Custom Architecture",
        description="Testing custom neural network architecture",
        model_type='lstm',
        dataset=dataset,
        predictor_columns=['Temperature', 'Humidity', 'Pressure'],
        target_columns=['WindSpeed', 'Precipitation'],
        custom_architecture=custom_architecture,
        use_custom_architecture=True,
        hyperparameters={
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001,
            'sequence_length': 10,
            'optimizer': 'adam',
            'loss': 'mse'
        }
    )
    
    custom_session = TrainingSession.objects.create(
        model_definition=custom_model_def,
        name="Test Custom Architecture Training",
        dataset=dataset,
        model_type='lstm',
        predictor_columns=['Temperature', 'Humidity', 'Pressure'],
        target_columns=['WindSpeed', 'Precipitation'],
        normalization_method='standard',
        hyperparameters=custom_model_def.hyperparameters,
        custom_architecture=custom_architecture,
        use_custom_architecture=True,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        selected_metrics=['mae', 'mse', 'rmse']
    )
    
    try:
        print("Training custom architecture...")
        train_model(custom_session)
        custom_session.refresh_from_db()
        
        if custom_session.status == 'completed':
            print("✓ Custom architecture training completed successfully!")
            print(f"  Test Results: {json.dumps(custom_session.test_results, indent=2)}")
        else:
            print(f"✗ Custom architecture training failed: {custom_session.error_message}")
    except Exception as e:
        print(f"✗ Error training custom architecture: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("=" * 50)
    
    # Cleanup
    print("\nCleaning up test data...")
    TrainingSession.objects.filter(dataset=dataset).delete()
    ModelDefinition.objects.filter(dataset=dataset).delete()
    dataset.delete()
    if os.path.exists(test_data_path):
        os.remove(test_data_path)
    print("Cleanup complete.")

if __name__ == "__main__":
    test_neural_networks()