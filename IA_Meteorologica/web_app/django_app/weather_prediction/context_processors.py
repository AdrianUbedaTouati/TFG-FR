def language_context(request):
    """
    Add current language to the template context
    """
    return {
        'current_language': request.session.get('language', 'fr')
    }