import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from django.contrib.auth.models import User

# Check if admin user exists
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin')
    print("Admin user created successfully!")
else:
    print("Admin user already exists.")