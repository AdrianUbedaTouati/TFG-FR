# Generated manually
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0020_remove_custom_function_remove_original_column'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingsession',
            name='current_epoch',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='total_epochs',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='current_batch',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='total_batches',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='train_loss',
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='val_loss',
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='train_accuracy',
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='val_accuracy',
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='progress',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='training_logs',
            field=models.JSONField(default=list),
        ),
    ]