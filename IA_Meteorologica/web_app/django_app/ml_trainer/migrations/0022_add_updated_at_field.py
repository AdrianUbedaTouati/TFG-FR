# Generated manually
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0021_add_progress_tracking_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingsession',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
        ),
    ]