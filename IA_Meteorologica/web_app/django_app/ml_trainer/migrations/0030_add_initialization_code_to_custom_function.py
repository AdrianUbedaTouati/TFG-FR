# Generated manually for adding initialization_code field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0029_add_cv_metrics'),
    ]

    operations = [
        migrations.AddField(
            model_name='customnormalizationfunction',
            name='initialization_code',
            field=models.TextField(blank=True, null=True, help_text='Código Python de inicialización que se ejecuta una sola vez para crear variables globales'),
        ),
    ]