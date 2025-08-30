# Generated manually
# Migration to fix root_dataset_id for existing normalized datasets

from django.db import migrations

def fix_root_dataset_ids(apps, schema_editor):
    """
    Fix root_dataset_id for normalized datasets that might not have it set correctly
    """
    Dataset = apps.get_model('ml_trainer', 'Dataset')
    
    # Get all normalized datasets
    normalized_datasets = Dataset.objects.filter(is_normalized=True)
    
    for dataset in normalized_datasets:
        # If dataset doesn't have root_dataset_id but has parent
        if not dataset.root_dataset_id and dataset.parent_dataset_id:
            # Find the root by traversing up the parent chain
            current = dataset
            visited = set()
            
            while current.parent_dataset_id and current.parent_dataset_id not in visited:
                visited.add(current.id)
                try:
                    parent = Dataset.objects.get(id=current.parent_dataset_id)
                    if not parent.is_normalized:
                        # Found the root
                        dataset.root_dataset_id = parent.id
                        dataset.save()
                        break
                    elif parent.root_dataset_id:
                        # Parent has root_dataset_id, use it
                        dataset.root_dataset_id = parent.root_dataset_id
                        dataset.save()
                        break
                    else:
                        # Continue up the chain
                        current = parent
                except Dataset.DoesNotExist:
                    # Parent doesn't exist, can't determine root
                    break
        
        # If dataset doesn't have root_dataset_id and no parent (orphan)
        elif not dataset.root_dataset_id and not dataset.parent_dataset_id:
            # Use its own ID as root (best we can do)
            dataset.root_dataset_id = dataset.id
            dataset.save()

def reverse_fix(apps, schema_editor):
    """
    This migration is not reversible as we cannot determine 
    what the previous incorrect values were
    """
    pass

class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0014_add_user_to_weatherprediction'),
    ]

    operations = [
        migrations.RunPython(fix_root_dataset_ids, reverse_fix),
    ]