# Generated manually

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0027_add_execution_config'),
    ]

    operations = [
        migrations.AddField(
            model_name='modeldefinition',
            name='default_execution_method',
            field=models.CharField(
                choices=[
                    ('standard', 'Ejecución Estándar'),
                    ('kfold', 'K-Fold Cross Validation'),
                    ('stratified_kfold', 'Stratified K-Fold CV'),
                    ('time_series_split', 'Time Series Split'),
                    ('leave_one_out', 'Leave-One-Out CV'),
                    ('repeated_kfold', 'Repeated K-Fold CV'),
                    ('repeated_stratified_kfold', 'Repeated Stratified K-Fold CV'),
                ],
                default='standard',
                help_text='Método de ejecución por defecto',
                max_length=30
            ),
        ),
        migrations.AddField(
            model_name='modeldefinition',
            name='default_execution_config',
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text='Configuración de ejecución por defecto'
            ),
        ),
    ]