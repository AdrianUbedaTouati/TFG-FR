from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from ml_trainer.models import TrainingSession, Dataset

def home(request):
    return render(request, 'home.html')

@login_required
def dashboard(request):
    context = {
        'recent_datasets': Dataset.objects.filter(user=request.user).order_by('-uploaded_at')[:5],
        'recent_trainings': TrainingSession.objects.filter(user=request.user).order_by('-created_at')[:5],
    }
    return render(request, 'dashboard.html', context)

@login_required
def predictions(request):
    return render(request, 'predictions.html')

@login_required
def models(request):
    return render(request, 'models.html')

@login_required
def datasets(request):
    return render(request, 'datasets.html')

@login_required
def dataset_normalize(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Verificar que el usuario tenga acceso al dataset
    if dataset.user != request.user and not request.user.is_staff:
        from django.http import HttpResponseForbidden
        return HttpResponseForbidden("No tienes permiso para acceder a este dataset.")
    
    return render(request, 'normalize.html', {'dataset_id': dataset_id})

@login_required
def training_progress(request, session_id):
    session = get_object_or_404(TrainingSession, id=session_id)
    
    # Verificar que el usuario tenga acceso a la sesión
    if session.user != request.user and not request.user.is_staff:
        from django.http import HttpResponseForbidden
        return HttpResponseForbidden("No tienes permiso para acceder a esta sesión de entrenamiento.")
    
    return render(request, 'training_progress.html', {
        'session': session,
        'session_id': session_id
    })

@login_required
def dataset_analyze(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Verificar que el usuario tenga acceso al dataset
    if dataset.user != request.user and not request.user.is_staff:
        from django.http import HttpResponseForbidden
        return HttpResponseForbidden("No tienes permiso para acceder a este dataset.")
    
    # Calcular el tamaño del archivo en MB
    try:
        file_size = dataset.file.size / (1024 * 1024)  # Convert to MB
    except:
        file_size = 0
    
    context = {
        'dataset': dataset,
        'dataset_id': dataset_id,
        'dataset_name': dataset.name,
        'file_size': file_size,
        'is_normalized': dataset.is_normalized if hasattr(dataset, 'is_normalized') else False
    }
    
    return render(request, 'dataset_analyze.html', context)


@login_required
def training_results(request, pk):
    """Training results page"""
    session = get_object_or_404(TrainingSession, id=pk)
    
    # Verificar que el usuario tenga acceso a la sesión
    if session.user != request.user and not request.user.is_staff:
        from django.http import HttpResponseForbidden
        return HttpResponseForbidden("No tienes permiso para acceder a estos resultados.")
    
    return render(request, 'training_results.html', {
        'session_id': pk,
        'session': session
    })