"""
Code export and import views
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.http import HttpResponse

from ..models import ModelDefinition
from ..code_generator import (
    generate_keras_code, generate_pytorch_code,
    parse_keras_code, parse_pytorch_code,
    validate_architecture
)
from ..utils import error_response, success_response
from ..constants import (
    SUCCESS_CODE_EXPORTED, SUCCESS_CODE_IMPORTED,
    ERROR_INVALID_FRAMEWORK
)


class ExportModelCodeView(APIView):
    """Export model architecture as Python code"""
    
    def get(self, request, pk):
        """Download model code as file"""
        model_def = get_object_or_404(ModelDefinition, pk=pk)
        framework = request.query_params.get('framework', model_def.framework)
        
        # Generate code based on framework
        code = self._generate_code(model_def, framework)
        if code is None:
            return error_response(ERROR_INVALID_FRAMEWORK.format(framework))
        
        # Update model with exported code
        self._update_exported_code(model_def, code)
        
        # Return as downloadable file
        response = HttpResponse(code, content_type='text/plain')
        filename = f"{model_def.name.replace(' ', '_')}_model.py"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
    
    def post(self, request, pk):
        """Get code as JSON response"""
        model_def = get_object_or_404(ModelDefinition, pk=pk)
        framework = request.data.get('framework', model_def.framework)
        
        # Generate code based on framework
        code = self._generate_code(model_def, framework)
        if code is None:
            return error_response(ERROR_INVALID_FRAMEWORK.format(framework))
        
        # Update model with exported code
        self._update_exported_code(model_def, code)
        
        return success_response({
            'code': code,
            'framework': framework,
            'model_name': model_def.name,
            'version': model_def.code_version
        }, message=SUCCESS_CODE_EXPORTED)
    
    def _generate_code(self, model_def, framework):
        """Generate code based on framework"""
        if framework == 'pytorch':
            return generate_pytorch_code(model_def)
        elif framework == 'keras':
            return generate_keras_code(model_def)
        return None
    
    def _update_exported_code(self, model_def, code):
        """Update model with exported code"""
        model_def.exported_code = code
        model_def.code_version += 1
        model_def.save()


class ImportModelCodeView(APIView):
    """Import model architecture from Python code"""
    
    def post(self, request, pk):
        model_def = get_object_or_404(ModelDefinition, pk=pk)
        
        # Get code from request
        code = self._get_code_from_request(request)
        if not code:
            return error_response("No code provided")
        
        framework = request.data.get('framework', 'keras')
        
        # Parse and validate code
        result = self._parse_and_validate_code(code, framework)
        if result['error']:
            return error_response(result['error'])
        
        # Update model definition
        self._update_model_from_parsed_code(
            model_def, 
            result['architecture'],
            result['hyperparameters'],
            framework,
            code
        )
        
        return success_response({
            'architecture': result['architecture'],
            'framework': framework,
            'version': model_def.code_version
        }, message=SUCCESS_CODE_IMPORTED)
    
    def _get_code_from_request(self, request):
        """Extract code from request"""
        code = request.data.get('code', '')
        
        if not code and 'file' in request.FILES:
            # Try to get from file upload
            uploaded_file = request.FILES['file']
            code = uploaded_file.read().decode('utf-8')
        
        return code
    
    def _parse_and_validate_code(self, code, framework):
        """Parse code and validate architecture"""
        result = {'error': None, 'architecture': [], 'hyperparameters': {}}
        
        try:
            # Parse based on framework
            if framework == 'pytorch':
                parsed = parse_pytorch_code(code)
            else:
                parsed = parse_keras_code(code)
            
            if not parsed:
                result['error'] = "Failed to parse code"
                return result
            
            # Validate architecture
            architecture = parsed.get('architecture', [])
            if not validate_architecture(architecture):
                result['error'] = "Invalid architecture detected"
                return result
            
            result['architecture'] = architecture
            result['hyperparameters'] = parsed.get('hyperparameters', {})
            
        except Exception as e:
            result['error'] = f"Error parsing code: {str(e)}"
        
        return result
    
    def _update_model_from_parsed_code(self, model_def, architecture, 
                                       hyperparameters, framework, code):
        """Update model definition with parsed code"""
        model_def.custom_architecture = architecture
        model_def.use_custom_architecture = True
        model_def.framework = framework
        
        # Update hyperparameters if found
        if hyperparameters:
            model_def.hyperparameters.update(hyperparameters)
        
        # Store the imported code
        model_def.exported_code = code
        model_def.code_version += 1
        
        model_def.save()