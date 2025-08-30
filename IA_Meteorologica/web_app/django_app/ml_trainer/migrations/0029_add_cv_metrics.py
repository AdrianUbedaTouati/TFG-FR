# Generated manually by Claude Code

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_trainer', '0028_add_execution_config_to_modeldefinition'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingsession',
            name='cv_scores',
            field=models.JSONField(blank=True, help_text='All fold scores from cross-validation', null=True),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='cv_best_score',
            field=models.FloatField(blank=True, help_text='Best score from cross-validation', null=True),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='cv_mean_score',
            field=models.FloatField(blank=True, help_text='Average score from cross-validation', null=True),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='cv_worst_score',
            field=models.FloatField(blank=True, help_text='Lowest score from cross-validation', null=True),
        ),
        migrations.AddField(
            model_name='trainingsession',
            name='cv_std_score',
            field=models.FloatField(blank=True, help_text='Standard deviation of CV scores', null=True),
        ),
    ]