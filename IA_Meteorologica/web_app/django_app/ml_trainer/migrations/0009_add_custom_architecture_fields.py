# Generated manually
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0008_add_training_session_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingsession',
            name='custom_architecture',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='use_custom_architecture',
            field=models.BooleanField(default=False),
        ),
    ]