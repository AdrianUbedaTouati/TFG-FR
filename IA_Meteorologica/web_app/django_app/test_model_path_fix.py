"""
Test script to verify model path fix and confusion matrix with real data
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.models import TrainingSession
from django.conf import settings

print("=== Testing Model Path Fix ===")
print(f"MEDIA_ROOT: {settings.MEDIA_ROOT}")

# Get the latest completed training session
session = TrainingSession.objects.filter(status='completed').order_by('-id').first()

if session:
    print(f"\nSession ID: {session.id}")
    print(f"Model type: {session.model_type}")
    print(f"Model file field: {session.model_file}")
    
    if session.model_file:
        print(f"Model file name: {session.model_file.name}")
        print(f"Model file path: {session.model_file.path}")
        
        # Check if file exists
        if os.path.exists(session.model_file.path):
            print(f"✓ Model file exists at: {session.model_file.path}")
        else:
            print(f"✗ Model file NOT found at: {session.model_file.path}")
            
            # Try alternative paths
            alt_path = os.path.join(settings.MEDIA_ROOT, session.model_file.name)
            print(f"\nChecking alternative path: {alt_path}")
            if os.path.exists(alt_path):
                print(f"✓ Model file found at alternative path!")
            else:
                print(f"✗ Model file NOT found at alternative path")
    else:
        print("No model file associated with session")
        
    # Show all model files in media/models directory
    models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
    if os.path.exists(models_dir):
        print(f"\nFiles in {models_dir}:")
        for f in os.listdir(models_dir):
            print(f"  - {f}")
else:
    print("No completed training sessions found")

print("\n=== Creating Test Training Session ===")
# Let's create a test session to see how the path is saved
from ml_trainer.models import Dataset, TrainingSession

# Get a dataset
dataset = Dataset.objects.first()
if dataset:
    print(f"Using dataset: {dataset.name}")
    
    # Create a dummy training session
    test_session = TrainingSession.objects.create(
        name="Test Model Path Fix",
        model_type="random_forest",
        dataset=dataset,
        predictor_columns=["Temperature (C)", "Humidity"],
        target_columns=["Summary"],
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        status='pending',
        user=dataset.user if hasattr(dataset, 'user') else None
    )
    
    print(f"Created test session ID: {test_session.id}")
    print("This session can be used to test the training with the fixed path")
else:
    print("No datasets found to create test session")