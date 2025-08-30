# Generated manually

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0026_add_sequential_split'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingsession',
            name='execution_method',
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
                help_text='Método de ejecución del entrenamiento',
                max_length=30
            ),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='execution_config',
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text='Configuración específica del método de ejecución'
            ),
        ),
    ]