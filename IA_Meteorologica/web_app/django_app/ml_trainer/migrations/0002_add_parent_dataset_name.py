# Generated manually
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='parent_dataset_name',
            field=models.CharField(blank=True, help_text='Nombre del dataset padre (se mantiene aunque se borre el original)', max_length=255, null=True),
        ),
    ]