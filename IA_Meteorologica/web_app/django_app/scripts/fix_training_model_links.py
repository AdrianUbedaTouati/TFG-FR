#!/usr/bin/env python
"""
Script to check and fix training sessions that are not linked to model definitions
"""
import os
import sys
import django

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.models import TrainingSession, ModelDefinition


def fix_training_model_links():
    """Check and fix training sessions without model_definition"""
    
    # Get all training sessions without model_definition
    unlinked_sessions = TrainingSession.objects.filter(model_definition__isnull=True)
    
    print(f"Found {unlinked_sessions.count()} training sessions without model_definition")
    
    fixed_count = 0
    
    for session in unlinked_sessions:
        print(f"\nProcessing session: {session.id} - {session.name}")
        
        # Try to find matching model definition by matching fields
        potential_models = ModelDefinition.objects.filter(
            model_type=session.model_type,
            dataset=session.dataset,
            predictor_columns=session.predictor_columns,
            target_columns=session.target_columns
        )
        
        if potential_models.count() == 1:
            # Exact match found
            model_def = potential_models.first()
            session.model_definition = model_def
            session.save()
            fixed_count += 1
            print(f"  ✓ Linked to model definition: {model_def.id} - {model_def.name}")
        elif potential_models.count() > 1:
            print(f"  ! Multiple potential model definitions found ({potential_models.count()})")
            # Try to match by hyperparameters or other fields
            for model_def in potential_models:
                if model_def.hyperparameters == session.hyperparameters:
                    session.model_definition = model_def
                    session.save()
                    fixed_count += 1
                    print(f"  ✓ Linked to model definition by hyperparameters: {model_def.id} - {model_def.name}")
                    break
        else:
            print(f"  ✗ No matching model definition found")
            # Could create a new model definition if needed
    
    print(f"\n✅ Fixed {fixed_count} training sessions")
    
    # Show summary
    total_sessions = TrainingSession.objects.count()
    linked_sessions = TrainingSession.objects.filter(model_definition__isnull=False).count()
    
    print(f"\nSummary:")
    print(f"Total training sessions: {total_sessions}")
    print(f"Linked to model definitions: {linked_sessions}")
    print(f"Unlinked: {total_sessions - linked_sessions}")


if __name__ == "__main__":
    fix_training_model_links()