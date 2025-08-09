# Generated manually to fix existing custom functions

from django.db import migrations


def fix_existing_functions(apps, schema_editor):
    """Set default values for existing custom functions without new fields"""
    CustomNormalizationFunction = apps.get_model('ml_trainer', 'CustomNormalizationFunction')
    
    for func in CustomNormalizationFunction.objects.all():
        # If new_columns is None or empty, set it to empty list
        if func.new_columns is None:
            func.new_columns = []
        
        # If remove_original_column is None, set it based on whether new_columns exist
        if not hasattr(func, 'remove_original_column') or func.remove_original_column is None:
            # If function creates new columns, default to removing original
            # Otherwise, keep the original column
            func.remove_original_column = len(func.new_columns) > 0
        
        func.save()


def reverse_fix(apps, schema_editor):
    """Reverse operation - does nothing"""
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0017_add_column_control_fields'),
    ]

    operations = [
        migrations.RunPython(fix_existing_functions, reverse_fix),
    ]