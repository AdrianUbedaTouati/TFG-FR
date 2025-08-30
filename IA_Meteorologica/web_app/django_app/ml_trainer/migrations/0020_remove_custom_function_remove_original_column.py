# Generated migration to remove remove_original_column field

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0019_add_normalization_chains'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='customnormalizationfunction',
            name='remove_original_column',
        ),
    ]