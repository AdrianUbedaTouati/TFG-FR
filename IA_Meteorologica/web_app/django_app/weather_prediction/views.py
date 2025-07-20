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