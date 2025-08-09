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
import signal
from contextlib import contextmanager

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


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time"""
    def signal_handler(signum, frame):
        raise TimeoutException("Function execution timed out")
    
    # Set the signal handler and alarm
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # Disable the alarm
    else:
        # On Windows, just yield without timeout
        yield


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
            print(f"Starting normalization process for dataset {dataset.name}")
            print(f"Normalization config: {normalization_config}")
            
            # Apply normalization
            normalized_df = self._apply_normalization(df, normalization_config)
            
            print(f"Normalization complete. DataFrame shape: {normalized_df.shape}")  # Debug
            print(f"DataFrame columns: {list(normalized_df.columns)}")
            
            # Save normalized dataset
            if create_copy:
                print(f"Saving normalized copy...")
                new_dataset = self._save_normalized_copy(
                    dataset, normalized_df, copy_name, normalization_config,
                    copy_short_description, copy_long_description
                )
                print(f"Dataset saved successfully: {new_dataset.name}")
                
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
                    import math
                    from datetime import datetime, timezone, timedelta
                    import re
                    import json
                    import unicodedata
                    import numpy as np
                    from scipy import stats, special
                    from sklearn import preprocessing
                    import statistics
                    from collections import Counter, defaultdict
                    import itertools
                    import functools
                    
                    # Define allowed modules for import
                    import _strptime  # Pre-import to avoid issues with datetime.strptime
                    import time
                    
                    ALLOWED_MODULES = {
                        'math': math,
                        'datetime': datetime,
                        'timezone': timezone,
                        'timedelta': timedelta,
                        're': re,
                        'json': json,
                        'unicodedata': unicodedata,
                        '_strptime': _strptime,  # Internal module needed by datetime.strptime
                        'time': time,  # Needed by datetime internally
                        'numpy': np,
                        'np': np,  # Common alias
                        'stats': stats,  # scipy.stats
                        'special': special,  # scipy.special
                        'pandas': pd,
                        'pd': pd,  # Common alias
                        'preprocessing': preprocessing,  # sklearn.preprocessing
                        'statistics': statistics,
                        'Counter': Counter,
                        'defaultdict': defaultdict,
                        'itertools': itertools,
                        'functools': functools,
                    }
                    
                    # Create datetime module object for 'from datetime import ...'
                    import types
                    datetime_module = types.ModuleType('datetime')
                    datetime_module.datetime = datetime
                    datetime_module.timezone = timezone
                    datetime_module.timedelta = timedelta
                    
                    # Create scipy module object
                    scipy_module = types.ModuleType('scipy')
                    scipy_module.stats = stats
                    scipy_module.special = special
                    
                    # Create sklearn module object
                    sklearn_module = types.ModuleType('sklearn')
                    sklearn_module.preprocessing = preprocessing
                    
                    # Create collections module object
                    collections_module = types.ModuleType('collections')
                    collections_module.Counter = Counter
                    collections_module.defaultdict = defaultdict
                    
                    # Update ALLOWED_MODULES to use the module objects
                    ALLOWED_MODULES['datetime'] = datetime_module
                    ALLOWED_MODULES['scipy'] = scipy_module
                    ALLOWED_MODULES['sklearn'] = sklearn_module
                    ALLOWED_MODULES['collections'] = collections_module
                    
                    # Create a safe import function
                    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
                        """Safe import function that only allows specific modules"""
                        if name in ALLOWED_MODULES:
                            return ALLOWED_MODULES[name]
                        else:
                            raise ImportError(f"Import of module '{name}' is not allowed")
                    
                    safe_globals = {
                        '__builtins__': {
                            '__import__': safe_import,
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
                            'dict': dict,
                            'list': list,
                            'tuple': tuple,
                            'set': set,
                            'bool': bool,
                            'type': type,
                            'range': range,
                            'enumerate': enumerate,
                            'zip': zip,
                            'map': map,
                            'filter': filter,
                            'sorted': sorted,
                            'reversed': reversed,
                            'any': any,
                            'all': all,
                            'pow': pow,
                            'divmod': divmod,
                            'print': print,  # For debugging
                            # Add built-in exceptions
                            'Exception': Exception,
                            'ValueError': ValueError,
                            'TypeError': TypeError,
                            'KeyError': KeyError,
                            'IndexError': IndexError,
                            'AttributeError': AttributeError,
                            'ImportError': ImportError,
                            'RuntimeError': RuntimeError,
                            'ZeroDivisionError': ZeroDivisionError,
                            'NameError': NameError,
                            'FileNotFoundError': FileNotFoundError,
                            'NotImplementedError': NotImplementedError,
                            'StopIteration': StopIteration,
                        }
                    }
                    
                    # Add safe modules
                    safe_globals['math'] = math
                    safe_globals['datetime'] = datetime
                    safe_globals['timezone'] = timezone
                    safe_globals['timedelta'] = timedelta
                    
                    # Add pandas and numpy for numeric functions
                    if custom_func.function_type == 'numeric':
                        safe_globals['pd'] = pd
                        safe_globals['np'] = np
                    
                    # Execute the custom function code
                    exec(custom_func.code, safe_globals)
                    
                    if 'normalize' in safe_globals:
                        normalize_func = safe_globals['normalize']
                        
                        # Check if function returns multiple columns
                        if custom_func.new_columns and len(custom_func.new_columns) > 0:
                            # Function returns multiple columns
                            print(f"Custom function {custom_func.name} will create {len(custom_func.new_columns)} new columns")
                            
                            # Process in batches to avoid memory issues
                            batch_size = 1000
                            total_rows = len(df)
                            
                            # First, detect actual column names by running the function once
                            actual_column_names = None
                            if total_rows > 0:
                                try:
                                    sample_value = df[column].iloc[0]
                                    if custom_func.function_type == 'numeric':
                                        sample_result = normalize_func(sample_value, series=None)
                                    else:
                                        sample_result = normalize_func(sample_value)
                                    
                                    if isinstance(sample_result, dict):
                                        actual_column_names = list(sample_result.keys())
                                        print(f"Detected actual column names from function output: {actual_column_names}")
                                except Exception as e:
                                    print(f"Error detecting column names: {e}")
                            
                            # Use actual column names if detected, otherwise use saved names
                            column_names_to_use = actual_column_names if actual_column_names else custom_func.new_columns
                            results_dict = {col: [] for col in column_names_to_use}
                            
                            print(f"Processing {total_rows} rows in batches of {batch_size}")
                            
                            # Pre-compute series statistics if numeric to avoid recomputation
                            series_data = None
                            if custom_func.function_type == 'numeric':
                                # Create a lightweight series representation
                                series_data = {
                                    'min': df[column].min(),
                                    'max': df[column].max(),
                                    'mean': df[column].mean(),
                                    'std': df[column].std(),
                                    'median': df[column].median(),
                                    'count': df[column].count(),
                                    'sum': df[column].sum()
                                }
                                # Add the series data to safe_globals so the function can access it
                                safe_globals['series_stats'] = series_data
                                print(f"Pre-computed series statistics: {series_data}")
                            
                            for start_idx in range(0, total_rows, batch_size):
                                end_idx = min(start_idx + batch_size, total_rows)
                                batch = df[column].iloc[start_idx:end_idx]
                                
                                if start_idx % (batch_size * 10) == 0:  # Log progress every 10 batches
                                    print(f"Processing rows {start_idx} to {end_idx} ({(start_idx/total_rows)*100:.1f}% complete)")
                                
                                # Apply function to batch
                                if custom_func.function_type == 'numeric':
                                    # For numeric functions, DO NOT pass entire series to avoid memory issues
                                    for idx, value in batch.items():
                                        try:
                                            # Apply timeout for each value processing
                                            with time_limit(5):  # 5 second timeout per value
                                                # DO NOT pass entire series - it causes memory/performance issues
                                                result = normalize_func(value, series=None)
                                            
                                            if isinstance(result, dict):
                                                # Use actual keys from result if available
                                                for col_name in column_names_to_use:
                                                    results_dict[col_name].append(result.get(col_name, None))
                                            else:
                                                # If single value returned, use first column name
                                                results_dict[column_names_to_use[0]].append(result)
                                                for col_name in column_names_to_use[1:]:
                                                    results_dict[col_name].append(None)
                                        except TimeoutException:
                                            print(f"Timeout in normalize function at index {idx}")
                                            # Append None for all columns on timeout
                                            for col_name in column_names_to_use:
                                                results_dict[col_name].append(None)
                                        except Exception as e:
                                            print(f"Error in normalize function at index {idx}: {e}")
                                            # Append None for all columns on error
                                            for col_name in column_names_to_use:
                                                results_dict[col_name].append(None)
                                else:
                                    # For text functions, apply without series parameter
                                    for idx, value in batch.items():
                                        try:
                                            # Apply timeout for each value processing
                                            with time_limit(5):  # 5 second timeout per value
                                                result = normalize_func(value)
                                            
                                            if isinstance(result, dict):
                                                for col_name in column_names_to_use:
                                                    results_dict[col_name].append(result.get(col_name, ''))
                                            else:
                                                # If single value returned, use first column name
                                                results_dict[column_names_to_use[0]].append(result)
                                                for col_name in column_names_to_use[1:]:
                                                    results_dict[col_name].append('')
                                        except TimeoutException:
                                            print(f"Timeout in normalize function at index {idx}")
                                            # Append empty string for all columns on timeout
                                            for col_name in column_names_to_use:
                                                results_dict[col_name].append('')
                                        except Exception as e:
                                            print(f"Error in normalize function at index {idx}: {e}")
                                            # Append empty string for all columns on error
                                            for col_name in column_names_to_use:
                                                results_dict[col_name].append('')
                            
                            print(f"Finished processing all {total_rows} rows")
                            
                            # Create new columns from results
                            # Use the actual column names from results_dict
                            for new_col_name in results_dict.keys():
                                normalized_df[new_col_name] = results_dict[new_col_name]
                                print(f"Created new column: {new_col_name} with {len(results_dict[new_col_name])} values")
                            
                            # Remove original column if specified
                            if custom_func.remove_original_column:
                                normalized_df = normalized_df.drop(columns=[column])
                                print(f"Removed original column: {column}")
                        else:
                            # Traditional single column output
                            print(f"Applying traditional single-column normalization")
                            if custom_func.function_type == 'numeric':
                                # For numeric functions, DO NOT pass entire series
                                # Pre-compute statistics if needed
                                series_stats = {
                                    'min': df[column].min(),
                                    'max': df[column].max(),
                                    'mean': df[column].mean(),
                                    'std': df[column].std()
                                }
                                safe_globals['series_stats'] = series_stats
                                
                                # Apply function without passing series
                                normalized_df[column] = df[column].apply(
                                    lambda x: normalize_func(x, series=None)
                                )
                            else:
                                # For text functions, just provide the value
                                normalized_df[column] = df[column].apply(normalize_func)
                            print(f"Traditional normalization complete for column {column}")
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
        
        print(f"_apply_normalization completed. Final DataFrame shape: {normalized_df.shape}")
        print(f"Final columns: {list(normalized_df.columns)}")
        return normalized_df
    
    def _save_normalized_copy(self, original_dataset, normalized_df, 
                              copy_name, config, short_description='', long_description=''):
        """Save normalized dataset as a new copy"""
        print(f"_save_normalized_copy called with copy_name: {copy_name}")
        
        if not copy_name:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            copy_name = f"{original_dataset.name}_normalized_{timestamp}"
        
        print(f"Final copy name: {copy_name}")
        print(f"DataFrame shape before CSV generation: {normalized_df.shape}")
        
        # Generate CSV content
        try:
            print("Converting DataFrame to CSV...")
            csv_content = normalized_df.to_csv(index=False)
            print(f"CSV content generated. Size: {len(csv_content)} bytes")
        except Exception as e:
            print(f"Error generating CSV: {str(e)}")
            raise
        
        # Determine the root dataset ID
        # If the original dataset is already normalized, keep its root_dataset_id
        # If it's an original dataset, use its own ID
        if original_dataset.is_normalized and original_dataset.root_dataset_id:
            root_id = original_dataset.root_dataset_id
        elif original_dataset.is_normalized and not original_dataset.root_dataset_id:
            # This is a normalized dataset without root_dataset_id (shouldn't happen but handle it)
            root_id = original_dataset.id
        else:
            # This is an original dataset
            root_id = original_dataset.id
        
        # Create new dataset record
        print("Creating new dataset record...")
        try:
            new_dataset = Dataset.objects.create(
                name=copy_name,
                user=original_dataset.user,
                is_normalized=True,
                parent_dataset=original_dataset,
                parent_dataset_name=original_dataset.name,
                root_dataset_id=root_id,
                normalization_method=str(config),
                short_description=short_description or f"Normalized from {original_dataset.name}",
                long_description=long_description or f"Normalization applied: {config}"
            )
            print(f"Dataset record created with ID: {new_dataset.id}")
        except Exception as e:
            print(f"Error creating dataset record: {str(e)}")
            raise
        
        # Save file using Django's file storage
        print("Saving CSV file...")
        try:
            new_dataset.file.save(
                f"{copy_name}.csv",
                ContentFile(csv_content.encode('utf-8'))
            )
            print(f"CSV file saved successfully")
        except Exception as e:
            print(f"Error saving CSV file: {str(e)}")
            # Delete the dataset record if file save fails
            new_dataset.delete()
            raise
        
        print(f"Dataset saved successfully: {new_dataset.name} (ID: {new_dataset.id})")
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
                    # Check if this is a custom function with multiple outputs
                    method = normalization_config[column]
                    if method.startswith('CUSTOM_'):
                        try:
                            function_id = int(method.replace('CUSTOM_', ''))
                            custom_func = CustomNormalizationFunction.objects.get(id=function_id)
                            
                            if custom_func.new_columns and len(custom_func.new_columns) > 0:
                                # Detect actual column names in the normalized sample
                                # Look for columns that exist in normalized_sample but not in sample_df
                                original_columns = set(sample_df.columns)
                                normalized_columns = set(normalized_sample.columns)
                                new_columns_detected = list(normalized_columns - original_columns)
                                
                                # If custom function removed the original column, detect new columns differently
                                if custom_func.remove_original_column and column not in normalized_sample.columns:
                                    # All columns that didn't exist before are new
                                    actual_new_columns = new_columns_detected
                                else:
                                    # Filter to only truly new columns
                                    actual_new_columns = [col for col in new_columns_detected if col not in original_columns]
                                
                                # Use detected columns if available, otherwise fall back to saved names
                                columns_to_use = actual_new_columns if actual_new_columns else custom_func.new_columns
                                
                                # Handle multiple column output
                                comparison[column] = {
                                    'new_columns': columns_to_use,
                                    'remove_original': custom_func.remove_original_column
                                }
                                
                                # Add preview for each new column
                                for new_col in columns_to_use:
                                    if new_col in normalized_sample.columns:
                                        new_values = normalized_sample[new_col].dropna()
                                        comparison[column][new_col] = {
                                            'sample': new_values.head(50).tolist(),
                                            'stats': detect_column_type(normalized_sample[new_col]),
                                            'unique_count': normalized_sample[new_col].nunique()
                                        }
                                
                                # Add original column info
                                comparison[column]['original_stats'] = detect_column_type(df[column])
                                comparison[column]['original_unique_count'] = df[column].nunique()
                                continue
                        except:
                            pass
                    
                    # Traditional single column comparison
                    # Get unique values for better comparison
                    original_values = sample_df[column].dropna()
                    if column in normalized_sample.columns:
                        normalized_values = normalized_sample[column].dropna()
                    else:
                        # Column was removed, skip comparison
                        continue
                    
                    # For categorical/text columns, show unique value mapping
                    # Show mapping for all columns with reasonable number of unique values
                    if df[column].dtype == 'object' or df[column].nunique() < 1000:
                        unique_mapping = []
                        # Get ALL unique values from the full dataset, no limit
                        all_unique_values = df[column].dropna().unique()
                        
                        # Create a temporary dataframe with all unique values to normalize them
                        temp_df = pd.DataFrame({column: all_unique_values})
                        
                        # Apply the same normalization to all unique values
                        try:
                            view = DatasetNormalizationView()
                            normalized_temp = view._apply_normalization(temp_df.copy(), {column: method})
                            
                            # Build the mapping for all unique values
                            for idx, val in enumerate(all_unique_values):
                                if column in normalized_temp.columns:
                                    # Single column output
                                    normalized_val = normalized_temp.iloc[idx][column]
                                else:
                                    # Column was transformed or removed, try to find the new column
                                    new_cols = [col for col in normalized_temp.columns if col not in temp_df.columns]
                                    if new_cols:
                                        # Use the first new column for the mapping
                                        normalized_val = normalized_temp.iloc[idx][new_cols[0]]
                                    else:
                                        normalized_val = '[Columna eliminada]'
                                
                                unique_mapping.append({
                                    'original': val,
                                    'normalized': normalized_val
                                })
                        except Exception as e:
                            # If normalization fails, fall back to sample-based mapping
                            print(f"Error normalizing all unique values: {e}")
                            for val in all_unique_values:
                                if val in original_values.values:
                                    original_idx = original_values[original_values == val].index[0]
                                    if original_idx in normalized_values.index:
                                        unique_mapping.append({
                                            'original': val,
                                            'normalized': normalized_values.loc[original_idx]
                                        })
                                else:
                                    unique_mapping.append({
                                        'original': val,
                                        'normalized': '[Error al normalizar]'
                                    })
                        
                        comparison[column] = {
                            'original': {
                                'sample': original_values.head(50).tolist(),
                                'stats': detect_column_type(df[column]),  # Use full dataset for stats
                                'all_unique_count': df[column].nunique()  # Add count of all unique values
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
                                'stats': detect_column_type(df[column])  # Use full dataset for stats
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
    
    def get(self, request, pk=None):
        """List all custom normalization functions or get a specific one"""
        print(f"GET request - User: {request.user}, PK: {pk}")
        
        if pk:
            # Obtener una función específica
            try:
                if request.user.is_staff:
                    function = CustomNormalizationFunction.objects.get(pk=pk)
                else:
                    function = CustomNormalizationFunction.objects.get(pk=pk, user=request.user)
                serializer = CustomNormalizationFunctionSerializer(function)
                
                # Log para verificar el GET individual
                print(f"GET individual function {pk}:")
                print(f"  Name: {function.name}")
                print(f"  Code from DB (first 200 chars): {function.code[:200]}...")
                print(f"  Serialized code (first 200 chars): {serializer.data.get('code', '')[:200]}...")
                
                return Response(serializer.data)
            except CustomNormalizationFunction.DoesNotExist:
                return Response({'error': 'Función no encontrada'}, status=status.HTTP_404_NOT_FOUND)
        else:
            # Listar todas las funciones
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
            
            # Imprimir los datos validados
            print(f"Request data keys: {list(request.data.keys())}")
            print(f"Code in request: {request.data.get('code', 'NO CODE')[:200]}...")
            
            serializer = CustomNormalizationFunctionSerializer(function, data=request.data, partial=True)
            if serializer.is_valid():
                print(f"Validated data keys: {list(serializer.validated_data.keys())}")
                print(f"Code in validated_data: {'code' in serializer.validated_data}")
                
                # Obtener el código directamente de request.data si no está en validated_data
                if 'code' in request.data:
                    code = request.data['code']
                    print(f"Using code from request.data")
                else:
                    code = function.code
                    print(f"Using existing code")
                    
                print(f"New code to save (first 200 chars): {code[:200]}...")
                print(f"New code length: {len(code)}")
                
                try:
                    compile(code, '<string>', 'exec')
                except SyntaxError as e:
                    return error_response(f"Syntax error in function code: {str(e)}")
                
                # Actualizar manualmente todos los campos directamente desde request.data
                if 'name' in request.data:
                    function.name = request.data['name']
                if 'description' in request.data:
                    function.description = request.data['description']
                if 'function_type' in request.data:
                    function.function_type = request.data['function_type']
                if 'code' in request.data:
                    function.code = request.data['code']
                    print(f"Code updated to: {function.code[:50]}...{function.code[-50:]}")
                if 'remove_original_column' in request.data:
                    function.remove_original_column = request.data['remove_original_column']
                if 'new_columns' in request.data:
                    function.new_columns = request.data['new_columns']
                
                function.save()
                
                # Recargar desde la DB para asegurar
                function.refresh_from_db()
                
                print(f"Function updated successfully:")
                print(f"  Name: {function.name}")
                print(f"  Type: {function.function_type}")
                print(f"  Code length: {len(function.code)}")
                print(f"  Code after save (first 200 chars): {function.code[:200]}...")
                
                # Verificar directamente en la DB
                db_function = CustomNormalizationFunction.objects.get(pk=pk)
                print(f"  Code in DB (first 200 chars): {db_function.code[:200]}...")
                
                # Usar el objeto recién obtenido de la DB para la respuesta para evitar cualquier problema de caché
                # Serializar de nuevo para asegurar que devolvemos los datos actualizados
                response_serializer = CustomNormalizationFunctionSerializer(db_function)
                
                # Log adicional para verificar el contenido de la respuesta
                response_data = response_serializer.data
                print(f"Response data being sent:")
                print(f"  ID: {response_data.get('id')}")
                print(f"  Name: {response_data.get('name')}")
                print(f"  Code in response (first 200 chars): {response_data.get('code', '')[:200]}...")
                print(f"  Full response keys: {list(response_data.keys())}")
                
                return Response(response_data)
            
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
        import math
        from datetime import datetime, timezone, timedelta
        import re
        import json
        import unicodedata
        import numpy as np
        from scipy import stats, special
        import pandas as pd
        from sklearn import preprocessing
        import statistics
        from collections import Counter, defaultdict
        import itertools
        import functools
        
        # Define allowed modules for import
        import _strptime  # Pre-import to avoid issues with datetime.strptime
        import time
        
        ALLOWED_MODULES = {
            'math': math,
            'datetime': datetime,
            'timezone': timezone,
            'timedelta': timedelta,
            're': re,
            'json': json,
            'unicodedata': unicodedata,
            '_strptime': _strptime,  # Internal module needed by datetime.strptime
            'time': time,  # Needed by datetime internally
            'numpy': np,
            'np': np,  # Common alias
            'stats': stats,  # scipy.stats
            'special': special,  # scipy.special
            'pandas': pd,
            'pd': pd,  # Common alias
            'preprocessing': preprocessing,  # sklearn.preprocessing
            'statistics': statistics,
            'Counter': Counter,
            'defaultdict': defaultdict,
            'itertools': itertools,
            'functools': functools,
        }
        
        # Create datetime module object for 'from datetime import ...'
        import types
        datetime_module = types.ModuleType('datetime')
        datetime_module.datetime = datetime
        datetime_module.timezone = timezone
        datetime_module.timedelta = timedelta
        
        # Create scipy module object
        scipy_module = types.ModuleType('scipy')
        scipy_module.stats = stats
        scipy_module.special = special
        
        # Create sklearn module object
        sklearn_module = types.ModuleType('sklearn')
        sklearn_module.preprocessing = preprocessing
        
        # Create collections module object
        collections_module = types.ModuleType('collections')
        collections_module.Counter = Counter
        collections_module.defaultdict = defaultdict
        
        # Update ALLOWED_MODULES to use the module objects
        ALLOWED_MODULES['datetime'] = datetime_module
        ALLOWED_MODULES['scipy'] = scipy_module
        ALLOWED_MODULES['sklearn'] = sklearn_module
        ALLOWED_MODULES['collections'] = collections_module
        
        # Create a safe import function
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Safe import function that only allows specific modules"""
            if name in ALLOWED_MODULES:
                return ALLOWED_MODULES[name]
            else:
                raise ImportError(f"Import of module '{name}' is not allowed")
        
        safe_globals = {
            '__builtins__': {
                '__import__': safe_import,
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
                'dict': dict,
                'list': list,
                'tuple': tuple,
                'set': set,
                'bool': bool,
                'type': type,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                # Add built-in exceptions
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'AttributeError': AttributeError,
                'ImportError': ImportError,
                'RuntimeError': RuntimeError,
                'ZeroDivisionError': ZeroDivisionError,
                'NameError': NameError,
                'FileNotFoundError': FileNotFoundError,
                'NotImplementedError': NotImplementedError,
                'StopIteration': StopIteration,
            },
            # Add safe modules
            'math': math,
            'datetime': datetime,
            'timezone': timezone,
            'timedelta': timedelta,
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
            
            # Detect column names if result is a dictionary
            detected_columns = []
            if isinstance(result, dict):
                detected_columns = list(result.keys())
            
            return success_response({
                'result': str(result),
                'input': str(test_value),
                'function_type': function_type,
                'detected_columns': detected_columns
            })
            
        except Exception as e:
            # Capture the full traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            
            return error_response(f"Error executing function: {str(e)}\n\nTraceback:\n{tb_str}")