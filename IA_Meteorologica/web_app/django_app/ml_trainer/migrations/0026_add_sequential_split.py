# Generated migration to add sequential split method

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0025_add_data_split_config'),
    ]

    operations = [
        migrations.AlterField(
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
                    ('sequential', 'División Secuencial'),
                ],
                help_text='Método de división de datos'
            ),
        ),
        migrations.AlterField(
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
                    ('sequential', 'División Secuencial'),
                ],
                help_text='Método de división de datos por defecto'
            ),
        ),
    ]