from django.shortcuts import render, get_object_or_404
from ml_trainer.models import TrainingSession

def home(request):
    return render(request, 'home.html')

def dashboard(request):
    return render(request, 'dashboard.html')

def predictions(request):
    return render(request, 'predictions.html')

def models(request):
    return render(request, 'models.html')

def datasets(request):
    return render(request, 'datasets.html')

def dataset_normalize(request, dataset_id):
    return render(request, 'normalize.html', {'dataset_id': dataset_id})

def training_progress(request, session_id):
    session = get_object_or_404(TrainingSession, id=session_id)
    return render(request, 'training_progress.html', {
        'session': session,
        'session_id': session_id
    })