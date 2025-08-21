#!/usr/bin/env python3
"""
Simple script to check if models have Module 2 configuration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.models import ModelDefinition
import json

def check_models():
    print("Checking Module 2 Configuration in Models")
    print("=" * 50)
    
    models = ModelDefinition.objects.all()
    
    if not models:
        print("No models found.")
        return
    
    for i, model in enumerate(models[:5], 1):  # Check first 5 models
        print(f"\n{i}. Model: {model.name} (ID: {model.id})")
        print(f"   Type: {model.model_type}")
        print(f"   Module 2 Method: {model.default_execution_method}")
        print(f"   Module 2 Config: {model.default_execution_config}")
        
        if model.default_execution_method and model.default_execution_method != 'standard':
            print(f"   ✅ HAS Module 2 configuration!")
        else:
            print(f"   ❌ NO Module 2 configuration (using standard)")

if __name__ == "__main__":
    check_models()