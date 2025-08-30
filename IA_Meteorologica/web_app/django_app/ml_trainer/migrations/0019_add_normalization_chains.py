# Generated manually for normalization chains feature

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('ml_trainer', '0018_fix_existing_custom_functions'),
    ]

    operations = [
        migrations.CreateModel(
            name='NormalizationChain',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('description', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='normalization_chains', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
                'unique_together': {('name', 'user')},
            },
        ),
        migrations.CreateModel(
            name='NormalizationChainStep',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('order', models.PositiveIntegerField()),
                ('method', models.CharField(help_text='Método de normalización o ID de función personalizada', max_length=100)),
                ('keep_original', models.BooleanField(default=False, help_text='Mantener la columna original')),
                ('chain', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='steps', to='ml_trainer.normalizationchain')),
            ],
            options={
                'ordering': ['order'],
                'unique_together': {('chain', 'order')},
            },
        ),
        migrations.AddField(
            model_name='dataset',
            name='normalization_chain',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='datasets', to='ml_trainer.normalizationchain'),
        ),
    ]