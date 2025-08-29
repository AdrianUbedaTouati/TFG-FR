from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth.models import User
from django import forms
from django.contrib.auth.decorators import login_required
from .translations import get_translation


class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class': 'form-control'}))
    
    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs['class'] = 'form-control'
        self.fields['password1'].widget.attrs['class'] = 'form-control'
        self.fields['password1'].widget.attrs['id'] = 'id_password1'
        self.fields['password2'].widget.attrs['class'] = 'form-control'
        self.fields['password2'].widget.attrs['id'] = 'id_password2'
        
        # Hacer la validación de contraseña más flexible
        # Language will be set dynamically in the view
        self.fields['password1'].help_text = ''
        self.fields['password2'].help_text = ''
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user


def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            next_url = request.GET.get('next', 'home')
            return redirect(next_url)
        else:
            # Get language from session or cookie
            lang = request.session.get('language', request.COOKIES.get('language', 'fr'))
            messages.error(request, get_translation('login_error', lang))
    
    return render(request, 'login.html')


def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            # Get language from session or cookie
            lang = request.session.get('language', request.COOKIES.get('language', 'fr'))
            messages.success(request, get_translation('welcome_message', lang, username=username))
            return redirect('home')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'register.html', {'form': form})


@login_required
def logout_view(request):
    logout(request)
    # Get language from session or cookie
    lang = request.session.get('language', request.COOKIES.get('language', 'fr'))
    messages.success(request, get_translation('logout_success', lang))
    return redirect('login')


@login_required
def profile_view(request):
    """Vista del perfil del usuario"""
    from ml_trainer.models import Dataset, TrainingSession, ModelDefinition, CustomNormalizationFunction
    
    context = {
        'user': request.user,
        'datasets_count': Dataset.objects.filter(user=request.user).count(),
        'models_count': ModelDefinition.objects.filter(user=request.user).count(),
        'trainings_count': TrainingSession.objects.filter(user=request.user).count(),
        'functions_count': CustomNormalizationFunction.objects.filter(user=request.user).count(),
    }
    
    return render(request, 'profile.html', context)