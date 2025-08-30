# Translation system for the weather prediction app

TRANSLATIONS = {
    'fr': {
        'password_help_text': 'Votre mot de passe doit contenir au moins 8 caractères.',
        'password_confirm_help': 'Entrez le même mot de passe pour confirmer.',
        'login_error': 'Utilisateur ou mot de passe incorrect.',
        'welcome_message': 'Bienvenue {username} ! Votre compte a été créé avec succès.',
        'logout_success': 'Vous avez été déconnecté avec succès.',
    },
    'es': {
        'password_help_text': 'Tu contraseña debe tener al menos 8 caracteres.',
        'password_confirm_help': 'Ingresa la misma contraseña para confirmar.',
        'login_error': 'Usuario o contraseña incorrectos.',
        'welcome_message': '¡Bienvenido {username}! Tu cuenta ha sido creada exitosamente.',
        'logout_success': 'Has cerrado sesión exitosamente.',
    },
    'en': {
        'password_help_text': 'Your password must contain at least 8 characters.',
        'password_confirm_help': 'Enter the same password to confirm.',
        'login_error': 'Incorrect username or password.',
        'welcome_message': 'Welcome {username}! Your account has been created successfully.',
        'logout_success': 'You have successfully logged out.',
    }
}

def get_translation(key, lang='fr', **kwargs):
    """Get translation for a given key in the specified language"""
    if lang not in TRANSLATIONS:
        lang = 'fr'  # Default to French
    
    text = TRANSLATIONS.get(lang, {}).get(key, TRANSLATIONS['fr'].get(key, key))
    
    # Format the text with any provided kwargs
    if kwargs:
        text = text.format(**kwargs)
    
    return text