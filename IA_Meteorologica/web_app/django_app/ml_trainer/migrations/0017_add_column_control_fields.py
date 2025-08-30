# Generated manually for adding column control fields to CustomNormalizationFunction

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0016_merge_20250808_2005'),
    ]

    operations = [
        migrations.AddField(
            model_name='customnormalizationfunction',
            name='remove_original_column',
            field=models.BooleanField(default=True, help_text='Si eliminar la columna original después de normalizar'),
        ),
        migrations.AddField(
            model_name='customnormalizationfunction',
            name='new_columns',
            field=models.JSONField(default=list, help_text='Lista de nombres de columnas nuevas que creará la función'),
        ),
    ]