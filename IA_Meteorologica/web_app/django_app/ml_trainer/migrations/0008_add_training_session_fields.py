# Generated manually
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0007_merge_0004_add_project_model_0006_merge_20250727_1817'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingsession',
            name='name',
            field=models.CharField(max_length=200, default='Unnamed Model'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='target_column',
            field=models.CharField(max_length=100, default=''),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='test_size',
            field=models.FloatField(default=0.2),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='config',
            field=models.JSONField(default=dict),
        ),
    ]