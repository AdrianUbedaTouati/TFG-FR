# Generated automatically

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0022_add_updated_at_field'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingsession',
            name='preprocessing_info',
            field=models.JSONField(blank=True, null=True, help_text='Information about preprocessing steps applied'),
        ),
    ]