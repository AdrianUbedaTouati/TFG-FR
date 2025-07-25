from django.shortcuts import render

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