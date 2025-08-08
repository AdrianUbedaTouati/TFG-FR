"""
Dataset normalization views
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.core.files.base import ContentFile
from django.conf import settings
from django.db import models
import pandas as pd
import numpy as np
from datetime import datetime
import os

from ..models import Dataset, CustomNormalizationFunction
from ..normalization_methods import DISPATCH_NUM, DISPATCH_TEXT
from ..normalization_mappings import get_numeric_enum, get_text_enum
from ..utils import (
    load_dataset, error_response, success_response,
    validate_dataframe, detect_column_type
)
from ..constants import (
    ERROR_PARSING_FAILED, ERROR_NORMALIZATION_FAILED,
    SUCCESS_NORMALIZATION_COMPLETE
)
from ..serializers import CustomNormalizationFunctionSerializer
import traceback
import sys
from io import StringIO


class DatasetNormalizationView(APIView):
    """Apply normalization to a dataset"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        """Obtiene información de normalización para un dataset"""
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        try:
            df = pd.read_csv(dataset.file.path)
            
            # Obtener funciones personalizadas del usuario
            custom_numeric_functions = []
            custom_text_functions = []
            
            # Obtener funciones personalizadas del usuario
            if request.user.is_staff:
                # Admin ve todas las funciones
                custom_functions = CustomNormalizationFunction.objects.all().order_by('name')
            else:
                # Usuario normal ve solo sus funciones
                custom_functions = CustomNormalizationFunction.objects.filter(user=request.user).order_by('name')
            print(f"DatasetNormalizationView - User: {request.user}")
            
            print(f"Found {custom_functions.count()} custom functions")
            
            # Separar funciones personalizadas por tipo
            for func in custom_functions:
                print(f"Processing function: {func.name} (type: {func.function_type}, user: {func.user})")
                func_data = {
                    'value': f'CUSTOM_{func.id}',
                    'label': f'{func.name} 🔧',
                    'description': func.description or f'Función personalizada: {func.name}',
                    'is_custom': True
                }
                
                if func.function_type == 'numeric':
                    custom_numeric_functions.append(func_data)
                else:  # text
                    custom_text_functions.append(func_data)
            
            print(f"Custom numeric functions: {len(custom_numeric_functions)}")
            print(f"Custom text functions: {len(custom_text_functions)}")
            
            # Analizar cada columna para determinar métodos de normalización disponibles
            normalization_info = {}
            
            for col in df.columns:
                col_type = str(df[col].dtype)
                
                if df[col].dtype in ['int64', 'float64']:
                    # Columna numérica
                    normalization_info[col] = {
                        'type': 'numeric',
                        'primary_methods': [
                            {'value': 'MIN_MAX', 'label': 'Min-Max [0, 1]', 'description': 'Escala valores al rango [0, 1]'},
                            {'value': 'Z_SCORE', 'label': 'Z-Score', 'description': 'Estandariza a media 0 y desviación 1'},
                            {'value': 'LSTM_TCN', 'label': 'LSTM/TCN [-1, 1]', 'description': 'Escala al rango [-1, 1] para RNN/TCN'},
                            {'value': 'CNN', 'label': 'CNN (Z-Score)', 'description': 'Normalización Z-Score para CNN'},
                            {'value': 'TRANSFORMER', 'label': 'Transformer (Robust)', 'description': 'RobustScaler resistente a outliers'},
                            {'value': 'TREE', 'label': 'Tree (Sin cambios)', 'description': 'Sin transformación (modelos de árbol)'}
                        ] + custom_numeric_functions,  # Agregar funciones personalizadas numéricas
                        'secondary_methods': [
                            {'value': 'LOWER', 'label': 'Minúsculas', 'description': 'Convierte a minúsculas (si son texto)'},
                            {'value': 'STRIP', 'label': 'Eliminar espacios', 'description': 'Elimina espacios al inicio/final'},
                            {'value': 'ONE_HOT', 'label': 'One-Hot Encoding', 'description': 'Convierte categorías a códigos numéricos (0, 1, 2...)'}
                        ] + custom_text_functions,  # Agregar funciones personalizadas de texto
                        'stats': {
                            'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                            'std': float(df[col].std()) if not df[col].isna().all() else None,
                            'min': float(df[col].min()) if not df[col].isna().all() else None,
                            'max': float(df[col].max()) if not df[col].isna().all() else None
                        }
                    }
                elif df[col].dtype == 'object':
                    # Columna de texto o objeto
                    unique_values = df[col].nunique()
                    sample_values = df[col].dropna().head(20).tolist()
                    
                    normalization_info[col] = {
                        'type': 'text',
                        'primary_methods': [
                            {'value': 'LOWER', 'label': 'Minúsculas', 'description': 'Convierte todo a minúsculas'},
                            {'value': 'STRIP', 'label': 'Eliminar espacios', 'description': 'Elimina espacios al inicio/final'},
                            {'value': 'ONE_HOT', 'label': 'One-Hot Encoding', 'description': f'Convierte {unique_values} categorías a códigos numéricos (0, 1, 2...)'}
                        ] + custom_text_functions,  # Agregar funciones personalizadas de texto
                        'secondary_methods': [
                            {'value': 'MIN_MAX', 'label': 'Min-Max [0, 1]', 'description': 'Escala valores (si son números como texto)'},
                            {'value': 'Z_SCORE', 'label': 'Z-Score', 'description': 'Estandariza (si son números como texto)'},
                            {'value': 'LSTM_TCN', 'label': 'LSTM/TCN [-1, 1]', 'description': 'Escala [-1, 1] (si son números)'},
                            {'value': 'CNN', 'label': 'CNN (Z-Score)', 'description': 'Z-Score (si son números)'},
                            {'value': 'TRANSFORMER', 'label': 'Transformer (Robust)', 'description': 'RobustScaler (si son números)'},
                            {'value': 'TREE', 'label': 'Tree (Sin cambios)', 'description': 'Sin transformación'}
                        ] + custom_numeric_functions,  # Agregar funciones personalizadas numéricas
                        'stats': {
                            'unique_count': unique_values,
                            'sample_values': sample_values,
                            'most_common': df[col].value_counts().head(3).to_dict() if unique_values < 100 else None
                        }
                    }
                else:
                    # Tipo desconocido o personalizado
                    normalization_info[col] = {
                        'type': 'unknown',
                        'primary_methods': [
                            {'value': 'MIN_MAX', 'label': 'Min-Max [0, 1]', 'description': 'Intenta escalar al rango [0, 1]'},
                            {'value': 'Z_SCORE', 'label': 'Z-Score', 'description': 'Intenta estandarizar'},
                            {'value': 'LSTM_TCN', 'label': 'LSTM/TCN [-1, 1]', 'description': 'Intenta escalar a [-1, 1]'},
                            {'value': 'CNN', 'label': 'CNN (Z-Score)', 'description': 'Intenta Z-Score'},
                            {'value': 'TRANSFORMER', 'label': 'Transformer (Robust)', 'description': 'Intenta RobustScaler'},
                            {'value': 'TREE', 'label': 'Tree (Sin cambios)', 'description': 'Sin transformación'},
                            {'value': 'LOWER', 'label': 'Minúsculas', 'description': 'Intenta convertir a minúsculas'},
                            {'value': 'STRIP', 'label': 'Eliminar espacios', 'description': 'Intenta eliminar espacios'},
                            {'value': 'ONE_HOT', 'label': 'One-Hot Encoding', 'description': 'Convierte categorías a códigos numéricos'}
                        ] + custom_numeric_functions + custom_text_functions,  # Agregar todas las funciones personalizadas
                        'secondary_methods': [],
                        'stats': {
                            'dtype': col_type,
                            'sample_values': df[col].dropna().head(20).astype(str).tolist()
                        }
                    }
            
            # Obtener copias normalizadas existentes
            normalized_copies = Dataset.objects.filter(
                parent_dataset=dataset
            ).order_by('-uploaded_at')
            
            return Response({
                'dataset': {
                    'id': dataset.id,
                    'name': dataset.name,
                    'shape': list(df.shape),  # Convertir tupla a lista
                    'columns': list(df.columns)
                },
                'normalization_info': normalization_info,
                'normalized_copies': [
                    {
                        'id': copy.id,
                        'name': copy.name,
                        'created_at': copy.uploaded_at,
                        'description': f"Dataset normalizado" if '_normalized_' in copy.name else "Dataset"
                    }
                    for copy in normalized_copies
                ]
            })
            
        except Exception as e:
            import traceback
            print(f"Error in DatasetNormalizationView: {str(e)}")
            print(traceback.format_exc())
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get normalization parameters
        normalization_config = request.data.get('normalization_config', {})
        create_copy = request.data.get('create_copy', True)
        copy_name = request.data.get('copy_name', '')
        copy_short_description = request.data.get('copy_short_description', '')
        copy_long_description = request.data.get('copy_long_description', '')
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Validate dataset
        is_valid, error_msg = validate_dataframe(df)
        if not is_valid:
            return error_response(error_msg)
        
        try:
            # Apply normalization
            normalized_df = self._apply_normalization(df, normalization_config)
            
            print(f"Normalized DataFrame shape: {normalized_df.shape}")  # Debug
            
            # Save normalized dataset
            if create_copy:
                new_dataset = self._save_normalized_copy(
                    dataset, normalized_df, copy_name, normalization_config,
                    copy_short_description, copy_long_description
                )
                
                return success_response({
                    'dataset_id': new_dataset.id,
                    'dataset_name': new_dataset.name,
                    'normalization_applied': normalization_config,
                    'new_shape': list(normalized_df.shape)
                }, message=SUCCESS_NORMALIZATION_COMPLETE)
            else:
                # Overwrite original
                csv_content = normalized_df.to_csv(index=False)
                dataset.file.save(
                    os.path.basename(dataset.file.name),
                    ContentFile(csv_content.encode('utf-8')),
                    save=False
                )
                dataset.is_normalized = True
                dataset.normalization_method = str(normalization_config)
                dataset.save()
                
                return success_response({
                    'dataset_id': dataset.id,
                    'normalization_applied': normalization_config,
                    'new_shape': list(normalized_df.shape)
                }, message=SUCCESS_NORMALIZATION_COMPLETE)
                
        except ValueError as e:
            # Error específico de función personalizada con formato detallado
            return Response({
                'error': 'custom_function_error',
                'error_type': 'CustomFunctionError',
                'message': str(e),
                'details': str(e)  # Ya incluye el formato tipo terminal
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return error_response(ERROR_NORMALIZATION_FAILED.format(str(e)))
    
    def _apply_normalization(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply normalization based on configuration"""
        normalized_df = df.copy()
        
        for column, method in config.items():
            if column not in df.columns:
                continue
            
            # Check if it's a custom function
            if method.startswith('CUSTOM_'):
                try:
                    function_id = int(method.replace('CUSTOM_', ''))
                    # Forzar lectura fresca desde la base de datos
                    custom_func = CustomNormalizationFunction.objects.get(id=function_id)
                    print(f"Applying custom function {custom_func.name} (ID: {function_id}) to column {column}")
                    
                    # Create safe execution environment
                    safe_globals = {
                        '__builtins__': {
                            'len': len,
                            'str': str,
                            'int': int,
                            'float': float,
                            'isinstance': isinstance,
                            'min': min,
                            'max': max,
                            'abs': abs,
                            'round': round,
                            'sum': sum,
                        }
                    }
                    
                    # Add pandas for numeric functions
                    if custom_func.function_type == 'numeric':
                        safe_globals['pd'] = pd
                    
                    # Execute the custom function code
                    exec(custom_func.code, safe_globals)
                    
                    if 'normalize' in safe_globals:
                        normalize_func = safe_globals['normalize']
                        
                        # Apply function to each value
                        if custom_func.function_type == 'numeric':
                            # For numeric functions, provide the series as well
                            normalized_df[column] = df[column].apply(
                                lambda x: normalize_func(x, series=df[column])
                            )
                        else:
                            # For text functions, just provide the value
                            normalized_df[column] = df[column].apply(normalize_func)
                        continue
                    else:
                        print(f"Custom function {custom_func.name} does not define 'normalize' function")
                except CustomNormalizationFunction.DoesNotExist:
                    print(f"Custom normalization function with ID {function_id} not found")
                except Exception as e:
                    import traceback
                    error_msg = f"Error applying custom normalization {method} to column {column}: {str(e)}"
                    print(error_msg)
                    
                    # Crear un mensaje de error detallado tipo terminal
                    error_details = {
                        'column': column,
                        'method': method,
                        'function_name': custom_func.name if 'custom_func' in locals() else 'Unknown',
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'traceback': traceback.format_exc()
                    }
                    
                    raise ValueError(f"Custom function error in column '{column}':\n\n"
                                   f"Function: {error_details['function_name']}\n"
                                   f"Error Type: {error_details['error_type']}\n"
                                   f"Error Message: {error_details['error_message']}\n\n"
                                   f"Traceback:\n{error_details['traceback']}")
            
            # Try numeric normalization first
            norm_enum = get_numeric_enum(method)
            if norm_enum in DISPATCH_NUM:
                try:
                    # Attempt to convert to numeric if it's not already
                    if df[column].dtype == 'object':
                        # Try to convert text to numeric
                        numeric_series = pd.to_numeric(df[column], errors='coerce')
                        if not numeric_series.isna().all():  # If at least some values converted
                            func = DISPATCH_NUM[norm_enum]
                            normalized_df[column] = func(numeric_series)
                            continue
                    else:
                        # Already numeric
                        func = DISPATCH_NUM[norm_enum]
                        normalized_df[column] = func(df[column])
                        continue
                except Exception:
                    pass
            
            # Try text normalization
            text_enum = get_text_enum(method)
            if text_enum in DISPATCH_TEXT:
                try:
                    func = DISPATCH_TEXT[text_enum]
                    normalized_df[column] = func(df[column])
                except Exception as e:
                    print(f"Error applying text normalization {method} to column {column}: {str(e)}")
        
        return normalized_df
    
    def _save_normalized_copy(self, original_dataset, normalized_df, 
                              copy_name, config, short_description='', long_description=''):
        """Save normalized dataset as a new copy"""
        if not copy_name:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            copy_name = f"{original_dataset.name}_normalized_{timestamp}"
        
        # Generate CSV content
        csv_content = normalized_df.to_csv(index=False)
        
        # Create new dataset record
        new_dataset = Dataset.objects.create(
            name=copy_name,
            user=original_dataset.user,
            is_normalized=True,
            parent_dataset=original_dataset,
            parent_dataset_name=original_dataset.name,
            root_dataset_id=original_dataset.root_dataset_id or original_dataset.id,
            normalization_method=str(config),
            short_description=short_description or f"Normalized from {original_dataset.name}",
            long_description=long_description or f"Normalization applied: {config}"
        )
        
        # Save file using Django's file storage
        new_dataset.file.save(
            f"{copy_name}.csv",
            ContentFile(csv_content.encode('utf-8'))
        )
        
        return new_dataset


class DatasetNormalizationPreviewView(APIView):
    """Preview normalization results without saving"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get normalization parameters
        normalization_config = request.data.get('normalization', {})
        sample_size = min(request.data.get('sample_size', 100), 1000)
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Take sample
        sample_df = df.head(sample_size)
        
        try:
            # Apply normalization to sample
            view = DatasetNormalizationView()
            normalized_sample = view._apply_normalization(sample_df, normalization_config)
            
            # Prepare comparison
            comparison = {}
            for column in normalization_config.keys():
                if column in df.columns:
                    # Get unique values for better comparison
                    original_values = sample_df[column].dropna()
                    normalized_values = normalized_sample[column].dropna()
                    
                    # For categorical/text columns, show unique value mapping
                    if df[column].dtype == 'object' or df[column].nunique() < 50:
                        unique_mapping = []
                        for val in original_values.unique()[:50]:  # Limit to 50 unique values
                            original_idx = original_values[original_values == val].index[0]
                            if original_idx in normalized_values.index:
                                unique_mapping.append({
                                    'original': val,
                                    'normalized': normalized_values.loc[original_idx]
                                })
                        
                        comparison[column] = {
                            'original': {
                                'sample': original_values.head(50).tolist(),
                                'stats': detect_column_type(sample_df[column])
                            },
                            'normalized': {
                                'sample': normalized_values.head(50).tolist(),
                                'stats': detect_column_type(normalized_sample[column])
                            },
                            'unique_mapping': unique_mapping,
                            'is_categorical': True
                        }
                    else:
                        # For numeric columns, show more samples
                        comparison[column] = {
                            'original': {
                                'sample': original_values.head(50).tolist(),
                                'stats': detect_column_type(sample_df[column])
                            },
                            'normalized': {
                                'sample': normalized_values.head(50).tolist(),
                                'stats': detect_column_type(normalized_sample[column])
                            },
                            'is_categorical': False
                        }
            
            return success_response({
                'preview': comparison,
                'sample_size': len(sample_df)
            })
            
        except ValueError as e:
            # Error específico de función personalizada con formato detallado
            return Response({
                'error': 'custom_function_error',
                'error_type': 'CustomFunctionError',
                'message': str(e),
                'details': str(e)  # Ya incluye el formato tipo terminal
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return error_response(f"Preview failed: {str(e)}")


class CustomNormalizationFunctionView(APIView):
    """CRUD operations for custom normalization functions"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """List all custom normalization functions"""
        print(f"GET request - User: {request.user}")
        
        # Filtrar funciones por usuario
        if request.user.is_staff:
            functions = CustomNormalizationFunction.objects.all()
        else:
            functions = CustomNormalizationFunction.objects.filter(user=request.user)
        print(f"Found {functions.count()} functions for user")
            
        serializer = CustomNormalizationFunctionSerializer(functions, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        """Create a new custom normalization function"""
        print(f"POST data: {request.data}")
        print(f"User authenticated: {request.user.is_authenticated}")
        
        serializer = CustomNormalizationFunctionSerializer(data=request.data)
        if serializer.is_valid():
            # Validate the code syntax
            code = serializer.validated_data['code']
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                return error_response(f"Syntax error in function code: {str(e)}")
            
            # Guardar la función asociada al usuario actual
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        print(f"Serializer errors: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def put(self, request, pk):
        """Update an existing custom normalization function"""
        print(f"PUT request for function ID {pk}")
        print(f"Request data: {request.data}")
        
        try:
            # Verificar que la función pertenezca al usuario
            if request.user.is_staff:
                function = CustomNormalizationFunction.objects.get(pk=pk)
            else:
                function = CustomNormalizationFunction.objects.get(pk=pk, user=request.user)
            print(f"Function found: {function.name}")
            
            # Mostrar el código actual antes de actualizar
            print(f"Current code before update (first 200 chars): {function.code[:200]}...")
            print(f"Current code length: {len(function.code)}")
            
            serializer = CustomNormalizationFunctionSerializer(function, data=request.data)
            if serializer.is_valid():
                # Validate the code syntax
                code = serializer.validated_data['code']
                print(f"New code to save (first 200 chars): {code[:200]}...")
                print(f"New code length: {len(code)}")
                
                try:
                    compile(code, '<string>', 'exec')
                except SyntaxError as e:
                    return error_response(f"Syntax error in function code: {str(e)}")
                
                # Guardar y forzar refresh desde la DB
                updated_function = serializer.save()
                updated_function.refresh_from_db()
                
                print(f"Function updated successfully:")
                print(f"  Name: {updated_function.name}")
                print(f"  Type: {updated_function.function_type}")
                print(f"  Code length: {len(updated_function.code)}")
                print(f"  Code after save (first 200 chars): {updated_function.code[:200]}...")
                
                # Verificar directamente en la DB
                db_function = CustomNormalizationFunction.objects.get(pk=pk)
                print(f"  Code in DB (first 200 chars): {db_function.code[:200]}...")
                
                # Serializar de nuevo para asegurar que devolvemos los datos actualizados
                response_serializer = CustomNormalizationFunctionSerializer(updated_function)
                return Response(response_serializer.data)
            
            print(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        except CustomNormalizationFunction.DoesNotExist:
            return Response(
                {'error': 'Función no encontrada'}, 
                status=status.HTTP_404_NOT_FOUND
            )
    
    def delete(self, request, pk):
        """Delete a custom normalization function"""
        try:
            # Verificar que la función pertenezca al usuario
            if request.user.is_staff:
                function = CustomNormalizationFunction.objects.get(pk=pk)
            else:
                function = CustomNormalizationFunction.objects.get(pk=pk, user=request.user)
            
            function.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
            
        except CustomNormalizationFunction.DoesNotExist:
            return Response(
                {'error': 'Función no encontrada'}, 
                status=status.HTTP_404_NOT_FOUND
            )


class CustomNormalizationFunctionTestView(APIView):
    """Test a custom normalization function"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Test a custom normalization function with a sample value"""
        code = request.data.get('code', '')
        test_value = request.data.get('test_value', '')
        function_type = request.data.get('function_type', 'numeric')
        
        if not code or test_value is None:
            return error_response("Code and test_value are required")
        
        # Create a safe execution environment
        safe_globals = {
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'isinstance': isinstance,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sum': sum,
            }
        }
        
        # Add pandas series methods for numeric functions
        if function_type == 'numeric':
            import pandas as pd
            safe_globals['pd'] = pd
            
        try:
            # Compile and execute the code
            compiled_code = compile(code, '<string>', 'exec')
            exec(compiled_code, safe_globals)
            
            # Find the normalize function
            if 'normalize' not in safe_globals:
                return error_response("Function 'normalize' not found in code")
            
            normalize_func = safe_globals['normalize']
            
            # Convert test value to appropriate type
            if function_type == 'numeric':
                try:
                    test_value = float(test_value)
                except ValueError:
                    return error_response("Invalid numeric value")
            
            # Execute the function
            result = normalize_func(test_value)
            
            return success_response({
                'result': str(result),
                'input': str(test_value),
                'function_type': function_type
            })
            
        except Exception as e:
            # Capture the full traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            
            return error_response(f"Error executing function: {str(e)}\n\nTraceback:\n{tb_str}")