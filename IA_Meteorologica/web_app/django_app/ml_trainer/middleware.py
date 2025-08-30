import json
from django.http import JsonResponse

class JSONErrorMiddleware:
    """
    Middleware para asegurar que los errores en las APIs devuelvan JSON
    en lugar de páginas HTML
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        # Solo procesar respuestas de error para rutas API
        if request.path.startswith('/api/') and response.status_code >= 400:
            # Si la respuesta no es JSON, convertirla
            if response.get('Content-Type', '').startswith('text/html'):
                error_message = 'Internal Server Error'
                
                # Intentar extraer información del error si es posible
                if response.status_code == 404:
                    error_message = 'Resource not found'
                elif response.status_code == 405:
                    error_message = 'Method not allowed'
                elif response.status_code == 500:
                    error_message = 'Internal server error occurred'
                
                return JsonResponse({
                    'error': error_message,
                    'status_code': response.status_code
                }, status=response.status_code)
        
        return response