"""
Script to assign a default user to existing records
"""
import os
import sys
import django

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from django.contrib.auth.models import User
from ml_trainer.models import Dataset, ModelDefinition, TrainingSession, CustomNormalizationFunction


def assign_default_user():
    # Check if there's a superuser
    superusers = User.objects.filter(is_superuser=True)
    
    if not superusers.exists():
        print("No superuser found. Creating a default admin user...")
        default_user = User.objects.create_superuser(
            username='admin',
            email='admin@weatherai.com',
            password='admin123'
        )
        print(f"Created default admin user: {default_user.username}")
    else:
        default_user = superusers.first()
        print(f"Using existing superuser: {default_user.username}")
    
    # Assign user to datasets without user
    datasets_without_user = Dataset.objects.filter(user__isnull=True)
    count = datasets_without_user.count()
    if count > 0:
        datasets_without_user.update(user=default_user)
        print(f"Assigned {count} datasets to {default_user.username}")
    
    # Assign user to model definitions without user
    models_without_user = ModelDefinition.objects.filter(user__isnull=True)
    count = models_without_user.count()
    if count > 0:
        models_without_user.update(user=default_user)
        print(f"Assigned {count} model definitions to {default_user.username}")
    
    # Assign user to training sessions without user
    sessions_without_user = TrainingSession.objects.filter(user__isnull=True)
    count = sessions_without_user.count()
    if count > 0:
        sessions_without_user.update(user=default_user)
        print(f"Assigned {count} training sessions to {default_user.username}")
    
    # Custom normalization functions can remain without user (public functions)
    functions_without_user = CustomNormalizationFunction.objects.filter(user__isnull=True)
    print(f"Found {functions_without_user.count()} public custom normalization functions (will remain public)")
    
    print("\nDone! All records have been assigned to a user.")
    return default_user


if __name__ == '__main__':
    user = assign_default_user()
    print(f"\nYou can now login with:")
    print(f"Username: {user.username}")
    if user.username == 'admin':
        print("Password: admin123")
        print("\nIMPORTANT: Please change this password after first login!")