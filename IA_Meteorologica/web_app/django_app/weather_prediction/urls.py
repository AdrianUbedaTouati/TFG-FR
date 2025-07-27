from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('predictions/', views.predictions, name='predictions'),
    path('models/', views.models, name='models'),
    path('datasets/', views.datasets, name='datasets'),
    path('datasets/<int:dataset_id>/normalize/', views.dataset_normalize, name='dataset-normalize'),
    path('training-progress/<int:session_id>/', views.training_progress, name='training-progress'),
    path('admin/', admin.site.urls),
    path('api/', include('ml_trainer.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)