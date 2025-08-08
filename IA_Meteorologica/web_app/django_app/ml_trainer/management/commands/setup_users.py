from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from ml_trainer.models import Dataset, ModelDefinition, TrainingSession, CustomNormalizationFunction


class Command(BaseCommand):
    help = 'Setup users for existing data'

    def handle(self, *args, **options):
        # Check if there's a superuser
        superusers = User.objects.filter(is_superuser=True)
        
        if not superusers.exists():
            self.stdout.write(self.style.WARNING("No superuser found. Creating a default admin user..."))
            default_user = User.objects.create_superuser(
                username='admin',
                email='admin@weatherai.com',
                password='admin123'
            )
            self.stdout.write(self.style.SUCCESS(f"Created default admin user: {default_user.username}"))
            self.stdout.write(self.style.WARNING("Default password is: admin123"))
            self.stdout.write(self.style.WARNING("PLEASE CHANGE THIS PASSWORD AFTER FIRST LOGIN!"))
        else:
            default_user = superusers.first()
            self.stdout.write(self.style.SUCCESS(f"Using existing superuser: {default_user.username}"))
        
        # Assign user to datasets without user
        datasets_without_user = Dataset.objects.filter(user__isnull=True)
        count = datasets_without_user.count()
        if count > 0:
            datasets_without_user.update(user=default_user)
            self.stdout.write(self.style.SUCCESS(f"Assigned {count} datasets to {default_user.username}"))
        
        # Assign user to model definitions without user
        models_without_user = ModelDefinition.objects.filter(user__isnull=True)
        count = models_without_user.count()
        if count > 0:
            models_without_user.update(user=default_user)
            self.stdout.write(self.style.SUCCESS(f"Assigned {count} model definitions to {default_user.username}"))
        
        # Assign user to training sessions without user
        sessions_without_user = TrainingSession.objects.filter(user__isnull=True)
        count = sessions_without_user.count()
        if count > 0:
            sessions_without_user.update(user=default_user)
            self.stdout.write(self.style.SUCCESS(f"Assigned {count} training sessions to {default_user.username}"))
        
        # Custom normalization functions can remain without user (public functions)
        functions_without_user = CustomNormalizationFunction.objects.filter(user__isnull=True)
        self.stdout.write(self.style.SUCCESS(f"Found {functions_without_user.count()} public custom normalization functions (will remain public)"))
        
        self.stdout.write(self.style.SUCCESS("\nDone! All records have been assigned to a user."))