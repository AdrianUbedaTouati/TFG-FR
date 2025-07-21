from django.urls import path
from . import views

urlpatterns = [
    # Dataset endpoints
    path('datasets/', views.DatasetListCreateView.as_view(), name='dataset-list-create'),
    path('datasets/<int:pk>/', views.DatasetDetailView.as_view(), name='dataset-detail'),
    path('datasets/<int:pk>/columns/', views.DatasetColumnsView.as_view(), name='dataset-columns'),
    path('datasets/<int:pk>/columns/<str:column_name>/', views.DatasetColumnDetailsView.as_view(), name='dataset-column-details'),
    path('datasets/<int:pk>/download/', views.DatasetDownloadView.as_view(), name='dataset-download'),
    path('datasets/<int:pk>/report/', views.DatasetReportView.as_view(), name='dataset-report'),
    
    # Training session endpoints
    path('training-sessions/', views.TrainingSessionListCreateView.as_view(), name='training-session-list-create'),
    path('training-sessions/<int:pk>/', views.TrainingSessionDetailView.as_view(), name='training-session-detail'),
    path('training-sessions/<int:pk>/train/', views.TrainModelView.as_view(), name='train-model'),
    
    # Model configuration endpoints
    path('model-configs/', views.ModelConfigView.as_view(), name='model-configs'),
    path('normalization-methods/<str:model_type>/', views.NormalizationMethodsView.as_view(), name='normalization-methods'),
    path('metrics/<str:model_type>/', views.MetricsView.as_view(), name='metrics'),
    
    # Prediction endpoints
    path('predictions/', views.PredictionListCreateView.as_view(), name='prediction-list-create'),
    path('predictions/map/', views.PredictionMapView.as_view(), name='prediction-map'),
    path('predict/', views.PredictView.as_view(), name='predict'),
]