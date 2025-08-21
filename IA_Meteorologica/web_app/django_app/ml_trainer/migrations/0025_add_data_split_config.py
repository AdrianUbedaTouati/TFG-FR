# Generated migration for data split configuration

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0024_alter_normalizationchainstep_unique_together_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingsession',
            name='split_method',
            field=models.CharField(
                max_length=20,
                default='random',
                choices=[
                    ('random', 'División Aleatoria'),
                    ('stratified', 'División Estratificada'),
                    ('group', 'División por Grupos'),
                    ('temporal', 'División Temporal'),
                ],
                help_text='Método de división de datos'
            ),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='split_config',
            field=models.JSONField(
                default=dict,
                blank=True,
                help_text='Configuración específica del método de división'
            ),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='random_state',
            field=models.IntegerField(
                null=True,
                blank=True,
                help_text='Semilla global para reproducibilidad'
            ),
        ),
        migrations.AddField(
            model_name='modeldefinition',
            name='default_split_method',
            field=models.CharField(
                max_length=20,
                default='random',
                choices=[
                    ('random', 'División Aleatoria'),
                    ('stratified', 'División Estratificada'),
                    ('group', 'División por Grupos'),
                    ('temporal', 'División Temporal'),
                ],
                help_text='Método de división de datos por defecto'
            ),
        ),
        migrations.AddField(
            model_name='modeldefinition',
            name='default_split_config',
            field=models.JSONField(
                default=dict,
                blank=True,
                help_text='Configuración de división por defecto'
            ),
        ),
    ]