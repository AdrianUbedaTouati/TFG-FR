from django.urls import path
from . import views
from . import analysis_views

urlpatterns = [
    # Dataset endpoints
    path('datasets/', views.DatasetListCreateView.as_view(), name='dataset-list-create'),
    path('datasets/<int:pk>/', views.DatasetDetailView.as_view(), name='dataset-detail'),
    path('datasets/<int:pk>/columns/', views.DatasetColumnsView.as_view(), name='dataset-columns'),
    path('datasets/<int:pk>/columns/<path:column_name>/', views.DatasetColumnDetailsView.as_view(), name='dataset-column-details'),
    path('datasets/<int:pk>/columns/<path:column_name>/analysis/', views.DatasetVariableAnalysisView.as_view(), name='dataset-variable-analysis'),
    path('datasets/<int:pk>/analysis/', analysis_views.dataset_analysis, name='dataset-general-analysis'),
    path('datasets/<int:pk>/download/', views.DatasetDownloadView.as_view(), name='dataset-download'),
    path('datasets/<int:pk>/report/', views.DatasetReportView.as_view(), name='dataset-report'),
    path('datasets/<int:pk>/normalization/', views.DatasetNormalizationView.as_view(), name='dataset-normalization'),
    path('datasets/<int:pk>/normalization/preview/', views.DatasetNormalizationPreviewView.as_view(), name='dataset-normalization-preview'),
    
    # Custom normalization function endpoints
    path('custom-normalization-functions/', views.CustomNormalizationFunctionView.as_view(), name='custom-normalization-functions'),
    path('custom-normalization-functions/<int:pk>/', views.CustomNormalizationFunctionView.as_view(), name='custom-normalization-function-detail'),
    path('custom-normalization-functions/test/', views.CustomNormalizationFunctionTestView.as_view(), name='custom-normalization-function-test'),
    
    # Model definition endpoints
    path('models/', views.ModelDefinitionListCreateView.as_view(), name='model-list-create'),
    path('models/<int:pk>/', views.ModelDefinitionDetailView.as_view(), name='model-detail'),
    path('models/<int:pk>/trainings/', views.ModelDefinitionTrainingsView.as_view(), name='model-trainings'),
    path('models/<int:pk>/clone/', views.CloneModelDefinitionView.as_view(), name='model-clone'),
    path('models/<int:pk>/export-code/', views.ExportModelCodeView.as_view(), name='model-export-code'),
    path('models/<int:pk>/import-code/', views.ImportModelCodeView.as_view(), name='model-import-code'),
    
    # Training session endpoints
    path('training-sessions/', views.TrainingSessionListCreateView.as_view(), name='training-session-list-create'),
    path('training-sessions/<int:pk>/', views.TrainingSessionDetailView.as_view(), name='training-session-detail'),
    path('training-sessions/<int:pk>/train/', views.TrainModelView.as_view(), name='train-model'),
    
    # Model configuration endpoints
    path('model-configs/', views.ModelConfigView.as_view(), name='model-configs'),
    path('implemented-models/', views.ImplementedModelsView.as_view(), name='implemented-models'),
    path('normalization-methods/<str:model_type>/', views.NormalizationMethodsView.as_view(), name='normalization-methods'),
    path('metrics/<str:model_type>/', views.MetricsView.as_view(), name='metrics'),
    path('frameworks/<str:model_type>/', views.ModelFrameworkView.as_view(), name='model-frameworks'),
    
    # Prediction endpoints
    path('predictions/', views.PredictionListCreateView.as_view(), name='prediction-list-create'),
    path('predictions/map/', views.PredictionMapView.as_view(), name='prediction-map'),
    path('predict/', views.PredictView.as_view(), name='predict'),
]