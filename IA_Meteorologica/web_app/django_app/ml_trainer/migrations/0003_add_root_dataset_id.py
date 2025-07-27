# Generated manually
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0002_add_parent_dataset_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='root_dataset_id',
            field=models.IntegerField(blank=True, help_text='ID del dataset raíz original (para mantener agrupación)', null=True),
        ),
    ]