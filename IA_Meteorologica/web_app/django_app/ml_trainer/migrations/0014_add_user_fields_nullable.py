# Generated migration - Step 1: Keep fields nullable

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('ml_trainer', '0013_add_custom_normalization_function'),
    ]

    operations = [
        # Just update the related_names, keep fields nullable for now
        migrations.AlterField(
            model_name='dataset',
            name='user',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='datasets', to=settings.AUTH_USER_MODEL),
        ),
        
        migrations.AlterField(
            model_name='modeldefinition',
            name='user',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='model_definitions', to=settings.AUTH_USER_MODEL),
        ),
        
        migrations.AlterField(
            model_name='trainingsession',
            name='user',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='training_sessions', to=settings.AUTH_USER_MODEL),
        ),
        
        migrations.AlterField(
            model_name='customnormalizationfunction',
            name='user',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='custom_functions', to=settings.AUTH_USER_MODEL),
        ),
    ]