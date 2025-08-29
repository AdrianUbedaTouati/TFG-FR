from django.utils.deprecation import MiddlewareMixin

class LanguageMiddleware(MiddlewareMixin):
    """
    Middleware to handle language preferences from JavaScript localStorage
    """
    def process_request(self, request):
        # Try to get language from POST data (for AJAX requests)
        if request.method == 'POST':
            lang = request.POST.get('language')
            if lang in ['fr', 'es', 'en']:
                request.session['language'] = lang
        
        # Try to get language from GET parameter
        lang_param = request.GET.get('lang')
        if lang_param in ['fr', 'es', 'en']:
            request.session['language'] = lang_param
        
        # Try to get language from cookie (set by JavaScript)
        if 'language' not in request.session:
            lang_cookie = request.COOKIES.get('language', 'fr')
            if lang_cookie in ['fr', 'es', 'en']:
                request.session['language'] = lang_cookie
        
        # Default to French
        if 'language' not in request.session:
            request.session['language'] = 'fr'
    
    def process_response(self, request, response):
        # Set language cookie if session has language
        if hasattr(request, 'session') and 'language' in request.session:
            response.set_cookie('language', request.session['language'], max_age=31536000)  # 1 year
        return response