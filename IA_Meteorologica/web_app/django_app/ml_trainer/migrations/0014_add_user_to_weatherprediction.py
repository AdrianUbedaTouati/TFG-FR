# Generated manually

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('ml_trainer', '0013_add_custom_normalization_function'),
    ]

    operations = [
        migrations.AddField(
            model_name='weatherprediction',
            name='user',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='weatherpredictions', to=settings.AUTH_USER_MODEL),
        ),
    ]