# Generated manually
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0009_add_custom_architecture_fields'),
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelDefinition',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('description', models.TextField(blank=True)),
                ('model_type', models.CharField(choices=[('random_forest', 'Random Forest'), ('lstm', 'LSTM'), ('gru', 'GRU'), ('xgboost', 'XGBoost'), ('gradient_boosting', 'Gradient Boosting')], max_length=50)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('predictor_columns', models.JSONField(default=list)),
                ('target_columns', models.JSONField(default=list)),
                ('default_config', models.JSONField(default=dict)),
                ('hyperparameters', models.JSONField(default=dict)),
                ('custom_architecture', models.JSONField(blank=True, null=True)),
                ('use_custom_architecture', models.BooleanField(default=False)),
                ('training_count', models.IntegerField(default=0)),
                ('best_score', models.FloatField(blank=True, null=True)),
                ('last_trained', models.DateTimeField(blank=True, null=True)),
                ('is_active', models.BooleanField(default=True)),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ml_trainer.dataset')),
                ('user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='auth.user')),
            ],
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='model_definition',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='ml_trainer.modeldefinition'),
        ),
    ]