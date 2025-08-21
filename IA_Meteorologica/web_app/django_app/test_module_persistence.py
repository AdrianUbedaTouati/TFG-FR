#!/usr/bin/env python3
"""
Test script to verify Module 1 and Module 2 persistence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.models import ModelDefinition
import json

def test_module_persistence():
    """Test that Module 1 and Module 2 configurations persist correctly"""
    
    print("Testing Module 1 and Module 2 Persistence")
    print("=" * 60)
    
    # Get all models
    models = ModelDefinition.objects.all()
    
    if not models:
        print("No models found in database.")
        return
    
    print(f"Found {models.count()} models\n")
    
    for model in models[:3]:  # Test first 3 models
        print(f"Model: {model.name}")
        print(f"  ID: {model.id}")
        print(f"  Type: {model.model_type}")
        
        # Check Module 1
        print(f"\n  Module 1 - Data Split:")
        print(f"    Method: {model.default_split_method}")
        print(f"    Config: {json.dumps(model.default_split_config, indent=6) if model.default_split_config else 'None'}")
        
        # Check Module 2
        print(f"\n  Module 2 - Execution:")
        print(f"    Method: {model.default_execution_method}")
        print(f"    Config: {json.dumps(model.default_execution_config, indent=6) if model.default_execution_config else 'None'}")
        
        print("\n" + "-" * 60 + "\n")
    
    # Test updating a model
    if models.count() > 0:
        test_model = models[0]
        print(f"Testing update on model: {test_model.name}")
        
        # Create test configurations
        test_split_config = {
            'train_size': 0.6,
            'val_size': 0.2,
            'test_size': 0.2,
            'random_state': 123
        }
        
        test_exec_config = {
            'n_splits': 10,
            'shuffle': True,
            'random_state': 456
        }
        
        # Update model
        test_model.default_split_method = 'stratified'
        test_model.default_split_config = test_split_config
        test_model.default_execution_method = 'kfold'
        test_model.default_execution_config = test_exec_config
        test_model.save()
        
        print("Model updated with test configurations")
        
        # Reload from database
        test_model.refresh_from_db()
        
        print("\nAfter reload:")
        print(f"  Split method: {test_model.default_split_method}")
        print(f"  Split config: {test_model.default_split_config}")
        print(f"  Execution method: {test_model.default_execution_method}")
        print(f"  Execution config: {test_model.default_execution_config}")
        
        # Verify values match
        if (test_model.default_split_method == 'stratified' and
            test_model.default_split_config == test_split_config and
            test_model.default_execution_method == 'kfold' and
            test_model.default_execution_config == test_exec_config):
            print("\n✅ Module configurations persist correctly in database!")
        else:
            print("\n❌ Module configurations did NOT persist correctly!")

if __name__ == "__main__":
    test_module_persistence()