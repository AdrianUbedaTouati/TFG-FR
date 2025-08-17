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
from ..normalization_compatibility import (
    get_method_io_type, NORMALIZATION_IO_TYPES, 
    validate_method_chain, detect_column_data_type
)
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


def map_type_for_frontend(detailed_type):
    """Map detailed types to frontend expected types"""
    numeric_types = {'integer', 'float', 'numeric', 'int', 'double', 'decimal'}
    text_types = {'text', 'string', 'str', 'varchar', 'char'}
    
    if detailed_type.lower() in numeric_types:
        return 'numeric'
    elif detailed_type.lower() in text_types:
        return 'text'
    elif detailed_type.lower() in {'boolean', 'bool'}:
        return 'numeric'  # Booleans are treated as numeric (0/1)
    elif detailed_type.lower() in {'datetime', 'date', 'time', 'timestamp'}:
        return 'text'  # Dates are treated as text for compatibility
    else:
        return 'text'  # Default to text for unknown types


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
                    'is_custom': True,
                    'output_type': 'mixed' if func.new_columns else func.function_type,  # Multi-output functions have mixed types
                    'is_multi_output': bool(func.new_columns)
                }
                
                if func.function_type == 'numeric':
                    custom_numeric_functions.append(func_data)
                else:  # text
                    custom_text_functions.append(func_data)
            
            print(f"Custom numeric functions: {len(custom_numeric_functions)}")
            print(f"Custom text functions: {len(custom_text_functions)}")
            
            # Import conversion options
            from ..type_conversions import TYPE_CONVERSIONS
            
            # Analizar cada columna para determinar métodos de normalización disponibles
            normalization_info = {}
            
            for col in df.columns:
                col_type = str(df[col].dtype)
                
                # Use the more sophisticated type detection
                detected_data_type = detect_column_data_type(df[col])
                
                if detected_data_type == 'numeric' or df[col].dtype in ['int64', 'float64']:
                    # Columna numérica
                    col_dtype = str(df[col].dtype)
                    normalization_info[col] = {
                        'type': 'numeric',
                        'dtype': col_dtype,  # Add specific dtype
                        'primary_methods': [
                            {'value': 'MIN_MAX', 'label': 'Min-Max [0, 1]', 'description': 'Escala valores al rango [0, 1]', 'output_type': 'numeric', 'output_dtype': 'float64'},
                            {'value': 'Z_SCORE', 'label': 'Z-Score', 'description': 'Estandariza a media 0 y desviación 1', 'output_type': 'numeric', 'output_dtype': 'float64'},
                            {'value': 'LSTM_TCN', 'label': 'LSTM/TCN [-1, 1]', 'description': 'Escala al rango [-1, 1] para RNN/TCN', 'output_type': 'numeric', 'output_dtype': 'float64'},
                            {'value': 'CNN', 'label': 'CNN (Z-Score)', 'description': 'Normalización Z-Score para CNN', 'output_type': 'numeric', 'output_dtype': 'float64'},
                            {'value': 'TRANSFORMER', 'label': 'Transformer (Robust)', 'description': 'RobustScaler resistente a outliers', 'output_type': 'numeric', 'output_dtype': 'float64'},
                            {'value': 'TREE', 'label': 'Tree (Sin cambios)', 'description': 'Sin transformación (modelos de árbol)', 'output_type': 'numeric', 'output_dtype': col_dtype}
                        ] + custom_numeric_functions,  # Agregar funciones personalizadas numéricas
                        'secondary_methods': [
                            {'value': 'LOWER', 'label': 'Minúsculas', 'description': 'Convierte a minúsculas (si son texto)', 'output_type': 'text', 'output_dtype': 'object'},
                            {'value': 'STRIP', 'label': 'Eliminar espacios', 'description': 'Elimina espacios al inicio/final', 'output_type': 'text', 'output_dtype': 'object'},
                            {'value': 'ONE_HOT', 'label': 'One-Hot Encoding', 'description': 'Convierte categorías a códigos numéricos (0, 1, 2...)', 'output_type': 'numeric', 'output_dtype': 'Int64'}
                        ] + custom_text_functions,  # Agregar funciones personalizadas de texto
                        'stats': {
                            'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                            'std': float(df[col].std()) if not df[col].isna().all() else None,
                            'min': float(df[col].min()) if not df[col].isna().all() else None,
                            'max': float(df[col].max()) if not df[col].isna().all() else None
                        }
                    }
                elif detected_data_type == 'text' or df[col].dtype == 'object':
                    # Columna de texto o objeto
                    unique_values = df[col].nunique()
                    sample_values = df[col].dropna().head(20).tolist()
                    
                    normalization_info[col] = {
                        'type': 'text',
                        'dtype': 'object',  # Add specific dtype
                        'primary_methods': [
                            {'value': 'LOWER', 'label': 'Minúsculas', 'description': 'Convierte todo a minúsculas', 'output_type': 'text', 'output_dtype': 'object'},
                            {'value': 'STRIP', 'label': 'Eliminar espacios', 'description': 'Elimina espacios al inicio/final', 'output_type': 'text', 'output_dtype': 'object'},
                            {'value': 'ONE_HOT', 'label': 'One-Hot Encoding', 'description': f'Convierte {unique_values} categorías a códigos numéricos (0, 1, 2...)', 'output_type': 'numeric', 'output_dtype': 'Int64'}
                        ] + custom_text_functions,  # Agregar funciones personalizadas de texto
                        'secondary_methods': [
                            {'value': 'MIN_MAX', 'label': 'Min-Max [0, 1]', 'description': 'Escala valores (si son números como texto)', 'output_type': 'numeric', 'output_dtype': 'float64'},
                            {'value': 'Z_SCORE', 'label': 'Z-Score', 'description': 'Estandariza (si son números como texto)', 'output_type': 'numeric', 'output_dtype': 'float64'},
                            {'value': 'LSTM_TCN', 'label': 'LSTM/TCN [-1, 1]', 'description': 'Escala [-1, 1] (si son números)', 'output_type': 'numeric', 'output_dtype': 'float64'},
                            {'value': 'CNN', 'label': 'CNN (Z-Score)', 'description': 'Z-Score (si son números)', 'output_type': 'numeric', 'output_dtype': 'float64'},
                            {'value': 'TRANSFORMER', 'label': 'Transformer (Robust)', 'description': 'RobustScaler (si son números)', 'output_type': 'numeric', 'output_dtype': 'float64'},
                            {'value': 'TREE', 'label': 'Tree (Sin cambios)', 'description': 'Sin transformación', 'output_type': 'numeric', 'output_dtype': 'object'}
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
                            {'value': 'MIN_MAX', 'label': 'Min-Max [0, 1]', 'description': 'Intenta escalar al rango [0, 1]', 'output_type': 'numeric'},
                            {'value': 'Z_SCORE', 'label': 'Z-Score', 'description': 'Intenta estandarizar', 'output_type': 'numeric'},
                            {'value': 'LSTM_TCN', 'label': 'LSTM/TCN [-1, 1]', 'description': 'Intenta escalar a [-1, 1]', 'output_type': 'numeric'},
                            {'value': 'CNN', 'label': 'CNN (Z-Score)', 'description': 'Intenta Z-Score', 'output_type': 'numeric'},
                            {'value': 'TRANSFORMER', 'label': 'Transformer (Robust)', 'description': 'Intenta RobustScaler', 'output_type': 'numeric'},
                            {'value': 'TREE', 'label': 'Tree (Sin cambios)', 'description': 'Sin transformación', 'output_type': 'numeric'},
                            {'value': 'LOWER', 'label': 'Minúsculas', 'description': 'Intenta convertir a minúsculas', 'output_type': 'text'},
                            {'value': 'STRIP', 'label': 'Eliminar espacios', 'description': 'Intenta eliminar espacios', 'output_type': 'text'},
                            {'value': 'ONE_HOT', 'label': 'One-Hot Encoding', 'description': 'Convierte categorías a códigos numéricos', 'output_type': 'numeric'}
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
                'conversion_options': TYPE_CONVERSIONS,  # Add conversion options
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
            
            # Check if normalized_df is valid
            if not isinstance(normalized_df, pd.DataFrame):
                error_msg = f"_apply_normalization returned invalid type: {type(normalized_df)}, value: {normalized_df}"
                print(error_msg)
                raise ValueError(error_msg)
            
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
                
                response_data = {
                    'dataset_id': new_dataset.id,
                    'dataset_name': new_dataset.name,
                    'normalization_applied': normalization_config,
                    'new_shape': list(normalized_df.shape)
                }
                
                # Add warnings if any
                if hasattr(self, 'conversion_warnings') and self.conversion_warnings:
                    response_data['warnings'] = self.conversion_warnings
                
                return success_response(response_data, message=SUCCESS_NORMALIZATION_COMPLETE)
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
            import traceback
            error_msg = str(e)
            
            # Check if the error message is just "-1" and provide more context
            if error_msg == "-1":
                error_msg = "Unknown error occurred during normalization process"
                print("DEBUG: Exception with message '-1' caught")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception args: {e.args}")
                print(f"Full traceback:\n{traceback.format_exc()}")
            
            error_details = {
                'error': error_msg,
                'type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
            print(f"Normalization error details: {error_details}")
            return error_response(ERROR_NORMALIZATION_FAILED.format(error_msg))
    
    def _apply_normalization(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply normalization based on configuration"""
        print(f"\n=== _apply_normalization started ===")
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Config keys: {list(config.keys())}")
        
        normalized_df = df.copy()
        self.conversion_warnings = []  # Store warnings for type conversions
        
        for column, method_config in config.items():
            print(f"\nProcessing column: {column}")
            if column not in df.columns:
                print(f"WARNING: Column '{column}' not found in DataFrame. Available columns: {list(df.columns)[:10]}...")
                continue
            
            # Extract output conversion if present
            output_conversion = None
            output_conversion_params = None
            output_conversions = None  # For multi-column outputs
            actual_config = method_config
            
            # Check if the config has the new structure with layers and output_conversion
            if isinstance(method_config, dict) and 'layers' in method_config:
                actual_config = method_config['layers']
                output_conversion = method_config.get('output_conversion', None)
                output_conversion_params = method_config.get('output_conversion_params', None)
                output_conversions = method_config.get('output_conversions', None)  # Multi-column conversions
            elif isinstance(method_config, list):
                # Config is directly an array - check for attached properties
                actual_config = method_config
                # In JavaScript, arrays can have properties, but in Python they come as separate keys
                # Check parent config for these properties
                if isinstance(config.get(column), dict):
                    output_conversion = config[column].get('output_conversion', None)
                    output_conversion_params = config[column].get('output_conversion_params', None) 
                    output_conversions = config[column].get('output_conversions', None)
            
            # Handle both old format (string) and new format (dict/list)
            if isinstance(actual_config, list):
                # New format: list of normalization steps (chained normalization)
                
                # Extract method names for validation
                method_names = [step.get('method', '') for step in actual_config]
                
                # Detect initial column type
                initial_type = detect_column_data_type(df[column])
                
                # Validate the method chain - now it only logs warnings
                is_valid, error_msg, final_type = validate_method_chain(method_names, initial_type)
                
                # Don't raise error on validation issues anymore
                if not is_valid and error_msg:
                    print(f"Warning in column '{column}': {error_msg}")
                
                current_column = column
                # Track what happens at each step for column management
                step_info = []
                
                for step_index, step in enumerate(actual_config):
                    method = step.get('method', '')
                    keep_original = step.get('keep_original', False)
                    
                    # Check if this step has a specific input column (for multiple outputs)
                    if 'input_column' in step:
                        print(f"Step {step_index + 1}: input_column found: '{step['input_column']}'")
                        print(f"Available columns after step {step_index}: {list(normalized_df.columns)[:20]}...")
                        if step['input_column'] in normalized_df.columns:
                            current_column = step['input_column']
                            print(f"Step {step_index + 1}: Using input column '{current_column}'")
                        else:
                            error_msg = f"Input column '{step['input_column']}' not found in dataframe for step {step_index + 1}"
                            print(f"ERROR: {error_msg}")
                            print(f"Available columns: {list(normalized_df.columns)}")
                            # This might be causing the -1 error
                            if step['input_column'] == 'trend':
                                print("DEBUG: This appears to be the source of the -1 error - 'trend' column not found")
                            raise ValueError(error_msg)
                    else:
                        print(f"Step {step_index + 1}: No input_column specified, using '{current_column}'")
                    
                    # IMPORTANT: Apply conversion from PREVIOUS step before processing current step
                    if step_index > 0:
                        prev_step = actual_config[step_index - 1]
                        prev_conversion = prev_step.get('conversion', None)
                        
                        if prev_conversion:
                            # Determine which column to apply the conversion to
                            # CRITICAL: If current step has input_column, apply conversion to THAT column
                            # This handles multi-output transformations correctly
                            conversion_target = None
                            
                            if 'input_column' in step and step['input_column'] in normalized_df.columns:
                                # Current step specifies which column it wants - apply conversion to that
                                conversion_target = step['input_column']
                                print(f"Step {step_index + 1}: Will apply conversion to input_column '{conversion_target}'")
                            else:
                                # No specific input column, use current_column
                                conversion_target = current_column
                                print(f"Step {step_index + 1}: Will apply conversion to current column '{conversion_target}'")
                            
                            if conversion_target in normalized_df.columns:
                                from ..conversion_functions import apply_conversion, get_conversion_warnings
                                try:
                                    print(f"Step {step_index + 1}: Applying conversion '{prev_conversion}' from previous step to column '{conversion_target}'")
                                    
                                    # Get warning if any
                                    warning_msg = get_conversion_warnings(
                                        normalized_df[conversion_target].dtype, prev_conversion
                                    )
                                    
                                    # Apply conversion with parameters
                                    conversion_params = prev_step.get('conversion_params', None)
                                    normalized_df[conversion_target] = apply_conversion(
                                        normalized_df[conversion_target], 
                                        prev_conversion,
                                        params=conversion_params
                                    )
                                
                                    # Add conversion warning
                                    if hasattr(self, 'conversion_warnings'):
                                        base_msg = f"Conversión de tipo aplicada en columna '{conversion_target}': {prev_conversion}"
                                        if warning_msg:
                                            base_msg += f" - ⚠️ {warning_msg}"
                                        self.conversion_warnings.append({
                                            'column': conversion_target,
                                            'method': prev_conversion,
                                            'warning': base_msg
                                        })
                                except Exception as e:
                                    error_msg = f"Error aplicando {prev_conversion} a columna {conversion_target}: {str(e)}"
                                    print(error_msg)
                                    raise ValueError(error_msg)
                    
                    # Store the input column for this step
                    step_input = current_column
                    
                    # Track columns BEFORE normalization
                    cols_before_this_step = set(normalized_df.columns)
                    
                    # Apply single normalization step
                    try:
                        normalized_df = self._apply_single_normalization(
                            normalized_df, current_column, method, keep_original, 
                            step_index=step_index, total_steps=len(actual_config)
                        )
                    except Exception as e:
                        error_msg = f"Error in step {step_index + 1} for column '{current_column}' with method '{method}': {str(e)}"
                        print(error_msg)
                        import traceback
                        print("Traceback:", traceback.format_exc())
                        raise ValueError(error_msg)
                    
                    # Get columns AFTER normalization
                    cols_after_this_step = set(normalized_df.columns)
                    new_cols_created = cols_after_this_step - cols_before_this_step
                    
                    # Determine outputs
                    step_outputs = []
                    primary_output = current_column  # Default
                    
                    if new_cols_created:
                        # New columns were created
                        step_outputs = list(new_cols_created)
                        
                        # Sort outputs to ensure consistent ordering
                        step_outputs.sort()
                        
                        # For single-layer normalization or when column was replaced
                        if current_column not in normalized_df.columns and step_outputs:
                            # Column was replaced - find the most appropriate output
                            # Look for a column with the expected suffix pattern
                            expected_suffix = f"_step{step_index + 1}"
                            primary_output = None
                            
                            # First try to find column with expected suffix
                            for output in step_outputs:
                                if output.endswith(expected_suffix):
                                    primary_output = output
                                    break
                            
                            # If not found, use the first output
                            if not primary_output:
                                primary_output = step_outputs[0]
                            
                            current_column = primary_output
                        elif step_outputs:
                            # New columns created alongside original
                            # For multi-output functions, use the column that follows the naming pattern
                            expected_suffix = f"_step{step_index + 1}"
                            primary_output = None
                            
                            # Look for the main output column (usually has the step suffix)
                            for output in step_outputs:
                                if output.endswith(expected_suffix) and not any(
                                    output.endswith(f"_{suffix}") for suffix in 
                                    ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'dayofyear', 'weekofyear']
                                ):
                                    primary_output = output
                                    break
                            
                            # If not found, use the first output
                            if not primary_output:
                                primary_output = step_outputs[0]
                            
                            if not keep_original:
                                current_column = primary_output
                    else:
                        # In-place transformation or no new columns
                        step_outputs = [current_column]
                        primary_output = current_column
                    
                    # Save step information with ALL outputs
                    step_info.append({
                        'index': step_index,
                        'method': method,
                        'keep_original': keep_original,
                        'input': step_input,
                        'primary_output': primary_output,
                        'all_outputs': step_outputs
                    })
                    
                    print(f"Step {step_index + 1} complete: input='{step_input}', outputs={step_outputs}, keep_original={keep_original}")
                    print(f"Columns available after step {step_index + 1}: {list(normalized_df.columns)[:30]}...")
                    
                    # Check if 'trend' column exists after this step
                    if 'trend' in normalized_df.columns:
                        print(f"DEBUG: 'trend' column found after step {step_index + 1}")
                    else:
                        print(f"DEBUG: 'trend' column NOT found after step {step_index + 1}")
                
                # Apply conversion from the LAST step if it exists
                if len(actual_config) > 0:
                    last_step = actual_config[-1]
                    last_conversion = last_step.get('conversion', None)
                    if last_conversion:
                        # Determine which column to apply the final conversion to
                        # If there's a next step that would use a specific input_column, we should have applied it already
                        # For the last step, we apply to its output
                        final_conversion_target = current_column
                        
                        # But if the last step had keep_original=True, we need to find the actual output
                        if last_step.get('keep_original', False):
                            # Find the step output column
                            expected_suffix = f"_step{len(method_config)}"
                            for col in normalized_df.columns:
                                if col.startswith(current_column) and col.endswith(expected_suffix):
                                    final_conversion_target = col
                                    break
                        
                        if final_conversion_target in normalized_df.columns:
                            from ..conversion_functions import apply_conversion, get_conversion_warnings
                            try:
                                print(f"Applying final conversion '{last_conversion}' to column '{final_conversion_target}'")
                                conversion_params = last_step.get('conversion_params', None)
                                normalized_df[final_conversion_target] = apply_conversion(
                                    normalized_df[final_conversion_target], 
                                    last_conversion,
                                    params=conversion_params
                                )
                                
                                # Add warning if any
                                warning_msg = get_conversion_warnings(
                                    normalized_df[final_conversion_target].dtype, last_conversion
                                )
                                if hasattr(self, 'conversion_warnings') and warning_msg:
                                    self.conversion_warnings.append({
                                        'column': final_conversion_target,
                                        'method': last_conversion,
                                        'warning': f"Conversión de tipo aplicada en columna '{final_conversion_target}': {last_conversion} - ⚠️ {warning_msg}"
                                    })
                            except Exception as e:
                                error_msg = f"Error aplicando conversión final {last_conversion} a columna {final_conversion_target}: {str(e)}"
                                print(error_msg)
                                raise ValueError(error_msg)
                
                # After all layers, manage column retention using the step_info we collected
                # IMPORTANT: This happens AFTER all steps have been processed
                try:
                    if isinstance(actual_config, list) and len(actual_config) > 0 and step_info:
                        print(f"\n=== Column retention management for '{column}' ===")
                        print("Step information collected:")
                        for info in step_info:
                            outputs_str = ', '.join(info['all_outputs']) if info['all_outputs'] else 'none'
                            print(f"  Step {info['index'] + 1}: {info['input']} -> [{outputs_str}] (keep_original={info['keep_original']})")
                    
                    # Determine which columns to keep
                    columns_to_keep = set()
                    
                    # For the final step, determine what is the actual final output
                    if step_info:
                        last_step = step_info[-1]
                        # If the last step has a specific input_column in its config, that might be the final output
                        # Fix: step_index is from outer loop, use len(step_info) - 1 instead
                        if len(actual_config) > 0 and 'input_column' in actual_config[-1]:
                            # The last step processed a specific column
                            final_input = actual_config[-1]['input_column']
                            # Find any outputs from the last step
                            for output in last_step['all_outputs']:
                                if output.startswith(final_input):
                                    columns_to_keep.add(output)
                                    print(f"\nKeeping final output from specific input: {output}")
                                    break
                            else:
                                # Fallback to primary output
                                columns_to_keep.add(last_step['primary_output'])
                                print(f"\nKeeping final primary output: {last_step['primary_output']}")
                        else:
                            # Keep the primary output of the last step
                            columns_to_keep.add(last_step['primary_output'])
                            print(f"\nKeeping final output: {last_step['primary_output']}")
                    
                    # Check each step's keep_original preference
                    for info in step_info:
                        if info['keep_original']:
                            columns_to_keep.add(info['input'])
                            print(f"Step {info['index'] + 1} wants to keep its input: {info['input']}")
                    
                    # Determine what to remove
                    # Only consider for removal:
                    # 1. Input columns that were used by layers (and not marked to keep)
                    # 2. The original column if not marked to keep
                    
                    columns_to_remove = []
                    
                    # Check each step's input column
                    for info in step_info:
                        input_col = info['input']
                        if input_col in normalized_df.columns and input_col not in columns_to_keep:
                            # This input column is not marked to keep
                            # Only remove it if it's not a final output of any step
                            is_someones_output = False
                            for other_info in step_info:
                                if input_col in other_info['all_outputs']:
                                    # It's an output of some step
                                    # Check if it's used as input by another step or is the final output
                                    is_used_later = False
                                    for later_info in step_info[other_info['index'] + 1:]:
                                        if later_info['input'] == input_col:
                                            is_used_later = True
                                            break
                                    
                                    # If it's an output that's used later, we can remove it (unless keep_original)
                                    # If it's an output that's NOT used later, keep it
                                    if is_used_later:
                                        is_someones_output = True
                                    break
                            
                            # Only remove if it's an intermediate result that was used
                            if input_col == column or is_someones_output:
                                columns_to_remove.append(input_col)
                                print(f"Will remove: {input_col}")
                    
                    # Special check: outputs that are not used anywhere should be kept
                    all_outputs = set()
                    for info in step_info:
                        all_outputs.update(info['all_outputs'])
                    
                    # Remove any outputs from removal list if they're not used as inputs
                    for output in all_outputs:
                        if output in columns_to_remove:
                            # Check if this output is used as input somewhere
                            used_as_input = any(info['input'] == output for info in step_info)
                            if not used_as_input and output != column:
                                # This output is not used, keep it
                                columns_to_remove.remove(output)
                                print(f"Keeping unused output: {output}")
                    
                    # Remove the columns
                    print(f"\nRemoving {len(columns_to_remove)} columns...")
                    for col in columns_to_remove:
                        if col in normalized_df.columns:
                            print(f"Removing: {col}")
                            try:
                                normalized_df = normalized_df.drop(columns=[col])
                            except Exception as e:
                                print(f"Warning: Could not remove column {col}: {e}")
                                # Check if this is causing the -1 error
                                if str(e) == "-1":
                                    print(f"DEBUG: -1 error occurred while dropping column {col}")
                                    print(f"DataFrame shape before drop: {normalized_df.shape}")
                                    print(f"Columns before drop: {list(normalized_df.columns)}")
                                    raise ValueError(f"Failed to remove column '{col}' - pandas returned error code -1")
                        else:
                            print(f"Warning: Column {col} not found in dataframe")
                    
                    print(f"\nFinal columns: {sorted(normalized_df.columns)}")
                    print(f"Final DataFrame shape: {normalized_df.shape}")
                except Exception as e:
                    print(f"ERROR in column retention management: {str(e)}")
                    print(f"Exception type: {type(e).__name__}")
                    if str(e) == "-1":
                        print("DEBUG: -1 error occurred in column retention management")
                        import traceback
                        print(f"Traceback:\n{traceback.format_exc()}")
                        raise ValueError("Column retention failed with error code -1")
                    else:
                        raise
            elif isinstance(actual_config, dict):
                # Single normalization (backward compatibility)
                method = actual_config.get('method', '')
                keep_original = actual_config.get('keep_original', False)
                normalized_df = self._apply_single_normalization(
                    normalized_df, column, method, keep_original
                )
            else:
                # Backward compatibility: if it's just a string, treat as method
                method = actual_config
                keep_original = False  # Default for old format
                normalized_df = self._apply_single_normalization(
                    normalized_df, column, method, keep_original
                )
            
            # Apply output conversion if specified (already extracted above)
            if output_conversions:
                # Multi-column output conversions
                from ..conversion_functions import apply_conversion, get_conversion_warnings
                
                print(f"Applying multi-column output conversions: {output_conversions}")
                
                for conv in output_conversions:
                    target_column = conv.get('column', '')
                    conversion_method = conv.get('conversion', '')
                    conversion_params = conv.get('params', None)
                    
                    if not target_column or not conversion_method:
                        continue
                    
                    if target_column in normalized_df.columns:
                        try:
                            print(f"Applying output conversion '{conversion_method}' to column '{target_column}'")
                            
                            # Apply conversion
                            normalized_df[target_column] = apply_conversion(
                                normalized_df[target_column],
                                conversion_method,
                                params=conversion_params
                            )
                            
                            # Add warning if any
                            warning_msg = get_conversion_warnings(
                                normalized_df[target_column].dtype, conversion_method
                            )
                            if hasattr(self, 'conversion_warnings') and warning_msg:
                                self.conversion_warnings.append({
                                    'column': target_column,
                                    'method': conversion_method,
                                    'warning': f"Conversión de salida aplicada en columna '{target_column}': {conversion_method} - ⚠️ {warning_msg}"
                                })
                        except Exception as e:
                            print(f"Error applying output conversion to {target_column}: {str(e)}")
                            raise ValueError(f"Error aplicando conversión de salida a columna '{target_column}': {str(e)}")
                    else:
                        print(f"Warning: Target column '{target_column}' not found for output conversion")
                        
            elif output_conversion:
                # Single output conversion (backward compatibility)
                # Find all columns that were created/modified by this normalization
                output_columns = []
                
                # Check for columns with step suffixes
                for col in normalized_df.columns:
                    if col.startswith(column) and ('_step' in col or '_normalized' in col):
                        output_columns.append(col)
                
                # If no step columns found, check if the original column still exists
                if not output_columns and column in normalized_df.columns:
                    output_columns = [column]
                
                # Apply conversion to all output columns
                from ..conversion_functions import apply_conversion, get_conversion_warnings
                for output_col in output_columns:
                    try:
                        print(f"Applying output conversion '{output_conversion}' to column '{output_col}'")
                        
                        # Check data type to determine if it's numeric or text
                        is_numeric = pd.api.types.is_numeric_dtype(normalized_df[output_col])
                        
                        # Apply conversion
                        normalized_df[output_col] = apply_conversion(
                            normalized_df[output_col],
                            output_conversion,
                            params=output_conversion_params
                        )
                        
                        # Add warning if any
                        warning_msg = get_conversion_warnings(
                            normalized_df[output_col].dtype, output_conversion
                        )
                        if hasattr(self, 'conversion_warnings') and warning_msg:
                            self.conversion_warnings.append({
                                'column': output_col,
                                'method': output_conversion,
                                'warning': f"Conversión de salida aplicada en columna '{output_col}': {output_conversion} - ⚠️ {warning_msg}"
                            })
                    except Exception as e:
                        print(f"Error applying output conversion {output_conversion} to column {output_col}: {str(e)}")
                        # Continue with other columns
        
        # Final validation before returning
        if not isinstance(normalized_df, pd.DataFrame):
            raise ValueError(f"Normalization process failed - expected DataFrame but got {type(normalized_df)}")
        
        if normalized_df.empty:
            raise ValueError("Normalization process resulted in empty DataFrame")
            
        print(f"_apply_normalization returning DataFrame with shape: {normalized_df.shape}")
        return normalized_df
    
    def _apply_single_normalization(self, df: pd.DataFrame, column: str, method: str, 
                                   keep_original: bool, step_index: int = 0, total_steps: int = 1) -> pd.DataFrame:
        """Apply a single normalization step to a column"""
        print(f"\n=== _apply_single_normalization ===")
        print(f"Method: {method}")
        print(f"Column: {column}")
        print(f"Step: {step_index + 1}/{total_steps}")
        print(f"Available columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        
        normalized_df = df.copy()
        
        if column not in normalized_df.columns:
            print(f"ERROR: Column '{column}' not found in dataframe!")
            print(f"Available columns: {list(normalized_df.columns)}")
            # Check if this might be the -1 error source
            if column == 'trend' or column == '-1':
                print(f"DEBUG: Potential -1 error source - column '{column}' not found")
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        # Generate suffix for the new column
        if total_steps > 1:
            # For multi-layer, always use step number for consistency
            suffix = f"_step{step_index + 1}"
        else:
            suffix = "_normalized"
        
        # Check if it's a custom function
        if method.startswith('CUSTOM_'):
            try:
                function_id = int(method.replace('CUSTOM_', ''))
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
                                            # If the result has different keys than expected, adapt
                                            result_keys = list(result.keys())
                                            if result_keys != column_names_to_use:
                                                # Update column names to match actual output
                                                print(f"Updating column names from {column_names_to_use} to {result_keys}")
                                                # Create new results_dict with actual keys
                                                new_results_dict = {key: [] for key in result_keys}
                                                # Copy existing values
                                                for i, key in enumerate(result_keys):
                                                    if i < len(column_names_to_use):
                                                        old_key = column_names_to_use[i]
                                                        if old_key in results_dict:
                                                            new_results_dict[key] = results_dict[old_key].copy()
                                                    else:
                                                        new_results_dict[key] = []
                                                # Fill with previous values
                                                for key in result_keys:
                                                    while len(new_results_dict[key]) < idx - start_idx:
                                                        new_results_dict[key].append(None)
                                                results_dict = new_results_dict
                                                column_names_to_use = result_keys
                                            
                                            # Now append the values
                                            for col_name in result.keys():
                                                if col_name not in results_dict:
                                                    results_dict[col_name] = [None] * (idx - start_idx)
                                                results_dict[col_name].append(result[col_name])
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
                                        error_msg = f"Error en función personalizada '{custom_func.name}' at index {idx}: {e}"
                                        print(error_msg)
                                        # Track errors for reporting
                                        if not hasattr(self, 'custom_function_errors'):
                                            self.custom_function_errors = []
                                        self.custom_function_errors.append({
                                            'index': idx,
                                            'value': value,
                                            'error': str(e),
                                            'function': custom_func.name
                                        })
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
                                            # If the result has different keys than expected, adapt
                                            result_keys = list(result.keys())
                                            if result_keys != column_names_to_use:
                                                # Update column names to match actual output
                                                print(f"Updating column names from {column_names_to_use} to {result_keys}")
                                                # Create new results_dict with actual keys
                                                new_results_dict = {key: [] for key in result_keys}
                                                # Copy existing values
                                                for i, key in enumerate(result_keys):
                                                    if i < len(column_names_to_use):
                                                        old_key = column_names_to_use[i]
                                                        if old_key in results_dict:
                                                            new_results_dict[key] = results_dict[old_key].copy()
                                                    else:
                                                        new_results_dict[key] = []
                                                # Fill with previous values
                                                for key in result_keys:
                                                    while len(new_results_dict[key]) < idx - start_idx:
                                                        new_results_dict[key].append('')
                                                results_dict = new_results_dict
                                                column_names_to_use = result_keys
                                            
                                            # Now append the values
                                            for col_name in result.keys():
                                                if col_name not in results_dict:
                                                    results_dict[col_name] = [''] * (idx - start_idx)
                                                results_dict[col_name].append(result[col_name])
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
                                        error_msg = f"Error en función personalizada '{custom_func.name}' at index {idx}: {e}"
                                        print(error_msg)
                                        # Track errors for reporting
                                        if not hasattr(self, 'custom_function_errors'):
                                            self.custom_function_errors = []
                                        self.custom_function_errors.append({
                                            'index': idx,
                                            'value': value,
                                            'error': str(e),
                                            'function': custom_func.name
                                        })
                                        # Append empty string for all columns on error
                                        for col_name in column_names_to_use:
                                            results_dict[col_name].append('')
                        
                        print(f"Finished processing all {total_rows} rows")
                        
                        # Create new columns from results and detect their types
                        # Use the actual column names from results_dict
                        detected_column_types = {}
                        for new_col_name in results_dict.keys():
                            normalized_df[new_col_name] = results_dict[new_col_name]
                            
                            # Detect the actual data type of the created column
                            col_series = pd.Series(results_dict[new_col_name])
                            
                            # Initialize default values
                            detected_type = 'text'
                            detected_dtype = 'string'
                            
                            # Try to infer the best dtype with more specific types
                            # First check if it's already a numeric dtype
                            if pd.api.types.is_numeric_dtype(col_series):
                                if pd.api.types.is_integer_dtype(col_series):
                                    detected_type = 'integer'
                                    detected_dtype = str(col_series.dtype)
                                else:
                                    detected_type = 'float'
                                    detected_dtype = str(col_series.dtype)
                            elif col_series.notna().any():  # If there's at least one non-null value
                                # Get non-null values for analysis
                                non_null_values = col_series.dropna()
                                
                                # Check if all values are boolean
                                if all(isinstance(v, bool) or v in [True, False, 'True', 'False', 'true', 'false'] for v in non_null_values):
                                    detected_dtype = 'bool'
                                    detected_type = 'boolean'
                                    normalized_df[new_col_name] = col_series.astype('bool')
                                else:
                                    # Try to convert to numeric
                                    try:
                                        numeric_series = pd.to_numeric(col_series, errors='coerce')
                                        if numeric_series.notna().sum() == col_series.notna().sum():
                                            # All non-null values are numeric
                                            normalized_df[new_col_name] = numeric_series
                                            
                                            # Check if all values are integers
                                            if all(float(v).is_integer() for v in non_null_values if pd.notna(v)):
                                                # Convert to integer type
                                                if numeric_series.min() >= -2147483648 and numeric_series.max() <= 2147483647:
                                                    detected_dtype = 'int32'
                                                else:
                                                    detected_dtype = 'int64'
                                                detected_type = 'integer'
                                                # Use nullable integer if there are NaN values
                                                if col_series.isna().any():
                                                    normalized_df[new_col_name] = numeric_series.astype('Int64')
                                                else:
                                                    normalized_df[new_col_name] = numeric_series.astype('int64')
                                            else:
                                                # It's a float
                                                detected_dtype = str(numeric_series.dtype)
                                                detected_type = 'float'
                                        else:
                                            # Some values couldn't be converted to numeric
                                            # Check if it's a date/datetime
                                            try:
                                                date_series = pd.to_datetime(col_series, errors='coerce')
                                                if date_series.notna().sum() == col_series.notna().sum():
                                                    detected_dtype = 'datetime64'
                                                    detected_type = 'datetime'
                                                    normalized_df[new_col_name] = date_series
                                                else:
                                                    detected_dtype = 'string'
                                                    detected_type = 'text'
                                            except:
                                                detected_dtype = 'string'
                                                detected_type = 'text'
                                    except:
                                        detected_dtype = 'string'
                                        detected_type = 'text'
                            else:
                                detected_dtype = 'string'
                                detected_type = 'text'
                            
                            detected_column_types[new_col_name] = {
                                'type': detected_type,
                                'dtype': detected_dtype
                            }
                            
                            print(f"Created new column: {new_col_name} with {len(results_dict[new_col_name])} values - Type: {detected_type} ({detected_dtype})")
                        
                        # Store detected column types and names for later use
                        if hasattr(self, '_custom_function_column_types'):
                            if column not in self._custom_function_column_types:
                                self._custom_function_column_types[column] = {}
                            self._custom_function_column_types[column] = detected_column_types
                        else:
                            self._custom_function_column_types = {column: detected_column_types}
                        
                        # Also store the actual column names detected from the function output
                        if hasattr(self, '_custom_function_detected_columns'):
                            self._custom_function_detected_columns[column] = list(results_dict.keys())
                        else:
                            self._custom_function_detected_columns = {column: list(results_dict.keys())}
                        
                        # For multi-layer, don't remove here - it will be handled at the end
                        # For single layer, remove if keep_original is False
                        if total_steps == 1 and not keep_original:
                            normalized_df = normalized_df.drop(columns=[column])
                            print(f"Removed original column: {column}")
                        
                        # Debug: Show all columns after custom function
                        print(f"Columns after custom function execution: {list(normalized_df.columns)}")
                        print(f"New columns created: {list(results_dict.keys())}")
                    else:
                        # Traditional single column output
                        print(f"Applying traditional single-column normalization")
                        
                        # For multi-layer normalizations, always create new columns
                        if total_steps > 1:
                            new_column_name = f"{column}{suffix}"
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
                                normalized_df[new_column_name] = df[column].copy().apply(
                                    lambda x: normalize_func(x, series=None)
                                )
                            else:
                                # For text functions, just provide the value
                                normalized_df[new_column_name] = df[column].copy().apply(normalize_func)
                        else:
                            # Single normalization - respect keep_original setting
                            if keep_original:
                                # User wants to keep original - create new column
                                new_column_name = f"{column}{suffix}"
                                if custom_func.function_type == 'numeric':
                                    series_stats = {
                                        'min': df[column].min(),
                                        'max': df[column].max(),
                                        'mean': df[column].mean(),
                                        'std': df[column].std()
                                    }
                                    safe_globals['series_stats'] = series_stats
                                    normalized_df[new_column_name] = df[column].copy().apply(
                                        lambda x: normalize_func(x, series=None)
                                    )
                                else:
                                    normalized_df[new_column_name] = df[column].copy().apply(normalize_func)
                            else:
                                # Replace in place
                                if custom_func.function_type == 'numeric':
                                    series_stats = {
                                        'min': df[column].min(),
                                        'max': df[column].max(),
                                        'mean': df[column].mean(),
                                        'std': df[column].std()
                                    }
                                    safe_globals['series_stats'] = series_stats
                                    normalized_df[column] = df[column].apply(
                                        lambda x: normalize_func(x, series=None)
                                    )
                                else:
                                    normalized_df[column] = df[column].apply(normalize_func)
                        print(f"Traditional normalization complete for column {column}")
                    return normalized_df
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
        
        # Check if column exists
        if column not in normalized_df.columns:
            print(f"Column {column} not found in dataframe")
            return normalized_df
            
        # Detect the current data type of the column using our enhanced detection
        current_type = detect_column_data_type(normalized_df[column])
        
        # Get method information
        method_info = get_method_io_type(method)
        
        # Check compatibility - now just log warnings instead of blocking
        if method_info['input'] not in ['any', 'unknown'] and method_info['input'] != current_type:
            # Special case: ONE_HOT can convert text to numeric, which is its purpose
            if not (method == 'ONE_HOT' and current_type == 'text'):
                warning_msg = f"Warning: Método {method} expects input type '{method_info['input']}' but column '{column}' is type '{current_type}'"
                print(warning_msg)
                # Don't raise error anymore - let it try to process
        
        # Apply the normalization based on method type
        try:
            # Handle numeric methods
            if method in ['MIN_MAX', 'Z_SCORE', 'LSTM_TCN', 'CNN', 'TRANSFORMER', 'TREE']:
                norm_enum = get_numeric_enum(method)
                if norm_enum in DISPATCH_NUM:
                    func = DISPATCH_NUM[norm_enum]
                    
                    # Check if we need type conversion warning
                    if pd.api.types.is_integer_dtype(normalized_df[column]) and method != 'TREE':
                        if hasattr(self, 'conversion_warnings'):
                            self.conversion_warnings.append({
                                'column': column,
                                'method': method,
                                'warning': f"La columna '{column}' será convertida de entero a decimal para la normalización {method}"
                            })
                    
                    # For multi-layer normalizations, always create new columns
                    # The original will be removed at the end if no layer has keep_original=True
                    if total_steps > 1:
                        new_column_name = f"{column}{suffix}"
                        normalized_df[new_column_name] = func(normalized_df[column].copy())
                    else:
                        # Single normalization - use the original keep_original logic
                        if keep_original:
                            new_column_name = f"{column}{suffix}"
                            normalized_df[new_column_name] = func(normalized_df[column].copy())
                        else:
                            normalized_df[column] = func(normalized_df[column])
            
            # Handle text methods
            elif method in ['LOWER', 'STRIP', 'ONE_HOT']:
                text_enum = get_text_enum(method)
                if text_enum in DISPATCH_TEXT:
                    func = DISPATCH_TEXT[text_enum]
                    # For multi-layer normalizations, always create new columns
                    # The original will be removed at the end if no layer has keep_original=True
                    if total_steps > 1:
                        new_column_name = f"{column}{suffix}"
                        normalized_df[new_column_name] = func(normalized_df[column].copy())
                    else:
                        # Single normalization - use the original keep_original logic
                        if keep_original:
                            new_column_name = f"{column}{suffix}"
                            normalized_df[new_column_name] = func(normalized_df[column].copy())
                        else:
                            normalized_df[column] = func(normalized_df[column])
            
            else:
                print(f"Unknown method: {method}")
                
        except Exception as e:
            error_msg = f"Error aplicando {method} a columna {column}: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg)
        
        # Final check before returning
        if not isinstance(normalized_df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame but got {type(normalized_df)}: {normalized_df}")
        
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
        show_steps = request.data.get('show_steps', False)  # New parameter to show transformation steps
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Take sample
        sample_df = df.head(sample_size)
        
        try:
            # First validate all normalization chains
            # Note: We're being less strict now, allowing any method combination
            for column, method_config in normalization_config.items():
                if column not in df.columns:
                    continue
                    
                if isinstance(method_config, list):
                    # Extract method names for validation
                    method_names = [step.get('method', '') for step in method_config]
                    
                    # Detect initial column type
                    initial_type = detect_column_data_type(df[column])
                    
                    # For steps with input_column, we should detect the type of that column
                    # But for now, we'll just proceed without strict validation
                    
                    # Validate the method chain - now it only logs warnings
                    is_valid, error_msg, final_type = validate_method_chain(method_names, initial_type)
                    
                    # Don't block on validation errors anymore
                    if not is_valid and error_msg:
                        print(f"Warning in column '{column}': {error_msg}")
            
            # Apply normalization to sample
            view = DatasetNormalizationView()
            
            # If show_steps is true and we have chained normalizations, show step-by-step
            transformation_steps = {}
            
            if show_steps:
                for column, method_config in normalization_config.items():
                    if isinstance(method_config, list) and len(method_config) > 0:
                        # Show transformation for each step (even if it's just one step)
                        steps_preview = []
                        current_df = sample_df.copy()
                        current_column = column  # Track the current column name as it may change
                        
                        for step_index, step in enumerate(method_config):
                            method = step.get('method', '')
                            keep_original = step.get('keep_original', False)
                            
                            print(f"Preview - Step {step_index + 1} config: {step}")
                            print(f"Preview - Current columns in dataframe: {list(current_df.columns)}")
                            
                            # IMPORTANT: Apply conversion from PREVIOUS step before processing current step
                            if step_index > 0:
                                prev_step = method_config[step_index - 1]
                                prev_conversion = prev_step.get('conversion', None)
                                if prev_conversion:
                                    # Determine which column to apply the conversion to
                                    # CRITICAL: If current step has input_column, that's the column that needs conversion
                                    if 'input_column' in step and step['input_column'] in current_df.columns:
                                        conversion_target = step['input_column']
                                        print(f"Preview - Step {step_index + 1}: Will apply conversion to input_column '{conversion_target}'")
                                    else:
                                        conversion_target = current_column
                                        print(f"Preview - Step {step_index + 1}: Will apply conversion to current column '{conversion_target}'")
                                    
                                    if conversion_target in current_df.columns:
                                        from ..conversion_functions import apply_conversion, get_conversion_warnings
                                        try:
                                            print(f"Preview - Step {step_index + 1}: Applying conversion '{prev_conversion}' from previous step to column '{conversion_target}'")
                                            print(f"Column dtype before conversion: {current_df[conversion_target].dtype}")
                                            print(f"Sample values before conversion: {current_df[conversion_target].head(3).tolist()}")
                                            
                                            # Apply the conversion with parameters if available
                                            conversion_params = prev_step.get('conversion_params', None)
                                            current_df[conversion_target] = apply_conversion(
                                                current_df[conversion_target], 
                                                prev_conversion,
                                                params=conversion_params
                                            )
                                            
                                            print(f"Column dtype after conversion: {current_df[conversion_target].dtype}")
                                            print(f"Sample values after conversion: {current_df[conversion_target].head(3).tolist()}")
                                            
                                        except Exception as e:
                                            error_msg = f"Error aplicando {prev_conversion} a columna {conversion_target}: {str(e)}"
                                            print(error_msg)
                                            return Response({
                                                'success': False,
                                                'error': error_msg
                                            }, status=status.HTTP_400_BAD_REQUEST)
                            
                            # Check if this step has a specific input column (for multiple outputs)
                            if 'input_column' in step and step['input_column'] in current_df.columns:
                                current_column = step['input_column']
                                print(f"Preview - Step {step_index + 1}: Using input column '{current_column}'")
                            else:
                                print(f"Preview - Step {step_index + 1}: No input_column or not found, using '{current_column}'")
                            
                            # Get sample values before transformation
                            before_values = current_df[current_column].tolist() if current_column in current_df.columns else []
                            
                            # Apply single step
                            current_df = view._apply_single_normalization(
                                current_df, current_column, method, keep_original,
                                step_index=step_index, total_steps=len(method_config)
                            )
                            
                            # Get sample values after transformation
                            # Find the transformed column (might have a new name)
                            transformed_col = current_column
                            
                            # For methods that create new columns with _stepN suffix
                            if not keep_original and method not in ['date_extraction', 'custom'] and not method.startswith('CUSTOM_'):
                                # Standard normalization methods that transform in-place
                                transformed_col = current_column
                            elif keep_original or method in ['date_extraction'] or method.startswith('CUSTOM_'):
                                # Methods that create new columns
                                if keep_original:
                                    # Find the new column with _stepN suffix
                                    new_cols = [col for col in current_df.columns if col == f"{current_column}_step{step_index + 1}"]
                                    if new_cols:
                                        transformed_col = new_cols[0]
                                # For date_extraction and custom functions that create multiple columns,
                                # the transformed_col should remain as current_column if input_column was specified
                                elif 'input_column' in step:
                                    transformed_col = current_column  # Keep using the input column
                            
                            after_values = current_df[transformed_col].tolist() if transformed_col in current_df.columns else []
                            
                            # Note: Conversion is now applied BEFORE the transformation (see above)
                            # This ensures conversions happen between layers, not after
                            conversion = step.get('conversion', None)
                            if conversion:
                                print(f"Preview - Step {step_index + 1}: This step has conversion '{conversion}' which will be applied before the NEXT step")
                            
                            # Detect output type for this step
                            output_type = 'text'
                            if transformed_col in current_df.columns:
                                col_series = current_df[transformed_col]
                                if pd.api.types.is_numeric_dtype(col_series):
                                    output_type = 'numeric'
                                elif detect_column_data_type(col_series) == 'numeric':
                                    output_type = 'numeric'
                            
                            steps_preview.append({
                                'step': step_index + 1,
                                'method': method,
                                'conversion': conversion,  # Add conversion info
                                'keep_original': keep_original,
                                'before': before_values,
                                'after': after_values,
                                'column_name': transformed_col,
                                'output_type': output_type
                            })
                            
                            # Update column for next step if not keeping original
                            if not keep_original:
                                current_column = transformed_col
                        
                        # Apply conversion from the LAST step if it exists
                        if len(method_config) > 0:
                            last_step = method_config[-1]
                            last_conversion = last_step.get('conversion', None)
                            if last_conversion:
                                # Apply to the final output column
                                final_column = current_column
                                if not last_step.get('keep_original', False):
                                    # If not keeping original, the column name is current_column
                                    final_column = current_column
                                else:
                                    # If keeping original, look for the step column
                                    step_col = f"{current_column}_step{len(method_config)}"
                                    if step_col in current_df.columns:
                                        final_column = step_col
                                
                                if final_column in current_df.columns:
                                    from ..conversion_functions import apply_conversion
                                    try:
                                        print(f"Preview - Applying final conversion '{last_conversion}' to column '{final_column}'")
                                        conversion_params = last_step.get('conversion_params', None)
                                        current_df[final_column] = apply_conversion(
                                            current_df[final_column], 
                                            last_conversion,
                                            params=conversion_params
                                        )
                                    except Exception as e:
                                        print(f"Error applying final conversion: {e}")
                        
                        transformation_steps[column] = steps_preview
            
            # Apply full normalization
            normalized_sample = view._apply_normalization(sample_df, normalization_config)
            
            # Detect types for ALL columns in normalized_sample (including new columns from custom functions)
            all_column_types = {}
            for col in normalized_sample.columns:
                col_series = normalized_sample[col]
                detected_type = 'text'
                detected_dtype = 'string'
                
                # First check if it's already a numeric dtype
                if pd.api.types.is_numeric_dtype(col_series):
                    if pd.api.types.is_integer_dtype(col_series):
                        detected_type = 'integer'
                        detected_dtype = str(col_series.dtype)
                    else:
                        detected_type = 'numeric'
                        detected_dtype = str(col_series.dtype)
                elif col_series.notna().any():
                    non_null_values = col_series.dropna()
                    
                    # Check if all values are boolean
                    if all(isinstance(v, bool) or v in [True, False, 'True', 'False', 'true', 'false'] for v in non_null_values):
                        detected_dtype = 'bool'
                        detected_type = 'boolean'
                    else:
                        try:
                            numeric_series = pd.to_numeric(col_series, errors='coerce')
                            if numeric_series.notna().sum() == col_series.notna().sum():
                                # All non-null values are numeric
                                if all(float(v).is_integer() for v in non_null_values if pd.notna(v)):
                                    detected_type = 'integer'
                                    if numeric_series.min() >= -2147483648 and numeric_series.max() <= 2147483647:
                                        detected_dtype = 'int32'
                                    else:
                                        detected_dtype = 'int64'
                                else:
                                    detected_type = 'numeric'
                                    detected_dtype = str(numeric_series.dtype)
                            else:
                                # Check if it's a date/datetime
                                try:
                                    date_series = pd.to_datetime(col_series, errors='coerce')
                                    if date_series.notna().sum() == col_series.notna().sum():
                                        detected_dtype = 'datetime64'
                                        detected_type = 'datetime'
                                    else:
                                        detected_dtype = 'string'
                                        detected_type = 'text'
                                except:
                                    detected_dtype = 'string'
                                    detected_type = 'text'
                        except:
                            pass
                
                all_column_types[col] = {
                    'name': col,
                    'type': detected_type,
                    'dtype': detected_dtype
                }
            
            # Prepare comparison
            comparison = {}
            for column in normalization_config.keys():
                if column in df.columns:
                    # Check if this is a custom function with multiple outputs
                    method_config = normalization_config[column]
                    
                    # Extract method from config (handle both old and new formats)
                    if isinstance(method_config, list):
                        # For chained normalization, check if any step is custom
                        # Check the last step to see if it's a custom function
                        if method_config:
                            last_step = method_config[-1]
                            method = last_step.get('method', '') if isinstance(last_step, dict) else last_step
                        else:
                            method = None
                    elif isinstance(method_config, dict):
                        method = method_config.get('method', '')
                    else:
                        method = method_config
                    
                    if method and isinstance(method, str) and method.startswith('CUSTOM_'):
                        try:
                            function_id = int(method.replace('CUSTOM_', ''))
                            custom_func = CustomNormalizationFunction.objects.get(id=function_id)
                            
                            if custom_func.new_columns and len(custom_func.new_columns) > 0:
                                # Detect actual column names in the normalized sample
                                # Look for columns that exist in normalized_sample but not in sample_df
                                original_columns = set(sample_df.columns)
                                normalized_columns = set(normalized_sample.columns)
                                new_columns_detected = list(normalized_columns - original_columns)
                                
                                # Detect new columns
                                if column not in normalized_sample.columns:
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
                                    'remove_original': False  # Ahora se controla desde la normalización
                                }
                                
                                # Add preview for each new column and detect types
                                column_types_info = []
                                for new_col in columns_to_use:
                                    if new_col in normalized_sample.columns:
                                        new_values = normalized_sample[new_col].dropna()
                                        
                                        # Detect the actual data type
                                        col_series = normalized_sample[new_col]
                                        
                                        # Try to infer the best dtype with more specific types
                                        detected_type = 'text'
                                        detected_dtype = 'string'
                                        
                                        # First check if it's already a numeric dtype
                                        if pd.api.types.is_numeric_dtype(col_series):
                                            if pd.api.types.is_integer_dtype(col_series):
                                                detected_type = 'integer'
                                                detected_dtype = str(col_series.dtype)
                                            else:
                                                detected_type = 'float'
                                                detected_dtype = str(col_series.dtype)
                                        elif col_series.notna().any():
                                            non_null_values = col_series.dropna()
                                            
                                            # Check if all values are boolean
                                            if all(isinstance(v, bool) or v in [True, False, 'True', 'False', 'true', 'false'] for v in non_null_values):
                                                detected_dtype = 'bool'
                                                detected_type = 'boolean'
                                            else:
                                                try:
                                                    numeric_series = pd.to_numeric(col_series, errors='coerce')
                                                    if numeric_series.notna().sum() == col_series.notna().sum():
                                                        # All non-null values are numeric
                                                        # Check if all values are integers
                                                        if all(float(v).is_integer() for v in non_null_values if pd.notna(v)):
                                                            detected_type = 'integer'
                                                            if numeric_series.min() >= -2147483648 and numeric_series.max() <= 2147483647:
                                                                detected_dtype = 'int32'
                                                            else:
                                                                detected_dtype = 'int64'
                                                        else:
                                                            detected_type = 'float'
                                                            detected_dtype = str(numeric_series.dtype)
                                                    else:
                                                        # Check if it's a date/datetime
                                                        try:
                                                            date_series = pd.to_datetime(col_series, errors='coerce')
                                                            if date_series.notna().sum() == col_series.notna().sum():
                                                                detected_dtype = 'datetime64'
                                                                detected_type = 'datetime'
                                                            else:
                                                                detected_dtype = 'string'
                                                                detected_type = 'text'
                                                        except:
                                                            detected_dtype = 'string'
                                                            detected_type = 'text'
                                                except:
                                                    pass
                                        
                                        column_types_info.append({
                                            'name': new_col,
                                            'type': map_type_for_frontend(detected_type),
                                            'dtype': detected_dtype
                                        })
                                        
                                        comparison[column][new_col] = {
                                            'sample': new_values.head(50).tolist(),
                                            'stats': detect_column_type(normalized_sample[new_col]),
                                            'unique_count': normalized_sample[new_col].nunique(),
                                            'detected_type': detected_type,
                                            'detected_dtype': detected_dtype
                                        }
                                
                                comparison[column]['column_types_info'] = column_types_info
                                
                                # If we have stored column types from the actual execution, use those
                                if hasattr(view, '_custom_function_column_types') and column in view._custom_function_column_types:
                                    stored_types = view._custom_function_column_types[column]
                                    # Update column_types_info with the stored information
                                    updated_types_info = []
                                    for col_name, type_info in stored_types.items():
                                        updated_types_info.append({
                                            'name': col_name,
                                            'type': map_type_for_frontend(type_info['type']),
                                            'dtype': type_info['dtype']
                                        })
                                    comparison[column]['column_types_info'] = updated_types_info
                                
                                # Also update the new_columns list with the actual detected columns
                                if hasattr(view, '_custom_function_detected_columns') and column in view._custom_function_detected_columns:
                                    comparison[column]['new_columns'] = view._custom_function_detected_columns[column]
                                    comparison[column]['detected_columns'] = True
                                
                                # Add error information if any
                                if hasattr(view, 'custom_function_errors') and view.custom_function_errors:
                                    # Group errors by unique error message
                                    error_summary = {}
                                    for error in view.custom_function_errors:
                                        if error['function'] == custom_func.name:
                                            error_msg = error['error']
                                            if error_msg not in error_summary:
                                                error_summary[error_msg] = {
                                                    'count': 0,
                                                    'example_values': []
                                                }
                                            error_summary[error_msg]['count'] += 1
                                            if len(error_summary[error_msg]['example_values']) < 5:
                                                error_summary[error_msg]['example_values'].append(error['value'])
                                    
                                    if error_summary:
                                        comparison[column]['custom_function_errors'] = {
                                            'function_name': custom_func.name,
                                            'total_errors': len(view.custom_function_errors),
                                            'error_summary': error_summary
                                        }
                                
                                # Add original column info
                                comparison[column]['original_stats'] = detect_column_type(df[column])
                                comparison[column]['original_unique_count'] = df[column].nunique()
                                continue
                        except:
                            pass
                    
                    # For chained transformations, we need to track the actual column being transformed
                    target_column = column
                    
                    # If we have a chain, check if the last step has an input_column
                    if isinstance(method_config, list) and len(method_config) > 0:
                        last_step = method_config[-1]
                        if isinstance(last_step, dict) and 'input_column' in last_step:
                            target_column = last_step['input_column']
                            print(f"Preview: Using input_column '{target_column}' for comparison")
                    
                    # Traditional single column comparison
                    # Get unique values for better comparison
                    original_values = sample_df[column].dropna()
                    
                    # Look for the target column in the normalized sample
                    if target_column in normalized_sample.columns:
                        normalized_values = normalized_sample[target_column].dropna()
                    elif column in normalized_sample.columns:
                        normalized_values = normalized_sample[column].dropna()
                    else:
                        # Try to find a column with a suffix
                        possible_cols = [col for col in normalized_sample.columns if col.startswith(target_column) and col != target_column and len(col) > len(target_column)]
                        if possible_cols:
                            normalized_values = normalized_sample[possible_cols[0]].dropna()
                            print(f"Preview: Found transformed column '{possible_cols[0]}'")
                        else:
                            # Column was removed, skip comparison
                            print(f"Preview: Column '{target_column}' not found in normalized sample")
                            continue
                    
                    # For categorical/text columns, show unique value mapping
                    # Show mapping for all columns with reasonable number of unique values
                    if df[column].dtype == 'object' or df[column].nunique() < 1000:
                        unique_mapping = []
                        
                        # For layer-specific preview, we need to get the input data for that specific layer
                        if show_steps and isinstance(method_config, list) and len(method_config) > 1:
                            # We have a multi-layer transformation
                            # Apply all transformations except the last one to get the input for the last layer
                            try:
                                # Get all but the last transformation
                                previous_transformations = method_config[:-1]
                                last_transformation = method_config[-1]
                                
                                # Apply previous transformations to the full dataset
                                view = DatasetNormalizationView()
                                intermediate_df = view._apply_normalization(df.copy(), {column: previous_transformations})
                                
                                # Find the output column from the previous transformations
                                # Track the column name as it changes through transformations
                                intermediate_column = column
                                for step_idx, step in enumerate(previous_transformations):
                                    if step.get('keep_original', False):
                                        # Column was kept, look for the new column with step suffix
                                        step_col = f"{intermediate_column}_step{step_idx + 1}"
                                        if step_col in intermediate_df.columns:
                                            intermediate_column = step_col
                                    # If keep_original is False, the column name stays the same
                                
                                # Check if the last transformation expects a different input column
                                if 'input_column' in last_transformation and last_transformation['input_column'] in intermediate_df.columns:
                                    # Use the specified input column instead
                                    intermediate_column = last_transformation['input_column']
                                    print(f"Layer preview: Last transformation specifies input_column: '{intermediate_column}'")
                                
                                # IMPORTANT: Apply conversion from the last previous transformation if it exists
                                # This is needed for layer preview to show correct values
                                if len(previous_transformations) > 0:
                                    last_prev_step = previous_transformations[-1]
                                    if 'conversion' in last_prev_step and intermediate_column in intermediate_df.columns:
                                        from ..conversion_functions import apply_conversion
                                        try:
                                            print(f"Layer preview: Applying conversion '{last_prev_step['conversion']}' to column '{intermediate_column}'")
                                            conversion_params = last_prev_step.get('conversion_params', None)
                                            intermediate_df[intermediate_column] = apply_conversion(
                                                intermediate_df[intermediate_column],
                                                last_prev_step['conversion'],
                                                params=conversion_params
                                            )
                                            print(f"Layer preview: Conversion applied successfully")
                                        except Exception as e:
                                            print(f"Layer preview: Error applying conversion: {e}")
                                
                                # Get unique values from the intermediate column
                                if intermediate_column in intermediate_df.columns:
                                    all_unique_values = intermediate_df[intermediate_column].dropna().unique()
                                    
                                    # Limit the number of unique values for performance and display
                                    MAX_UNIQUE_FOR_PREVIEW = 100
                                    if len(all_unique_values) > MAX_UNIQUE_FOR_PREVIEW:
                                        print(f"Layer preview: Too many unique values ({len(all_unique_values)}), sampling {MAX_UNIQUE_FOR_PREVIEW}")
                                        # Sample evenly across the unique values
                                        step = len(all_unique_values) // MAX_UNIQUE_FOR_PREVIEW
                                        all_unique_values = all_unique_values[::step][:MAX_UNIQUE_FOR_PREVIEW]
                                    
                                    temp_column_name = intermediate_column
                                    print(f"Layer preview: Using intermediate column '{intermediate_column}' with {len(all_unique_values)} unique values")
                                    print(f"Layer preview: Sample values from intermediate: {list(all_unique_values)[:5]}")
                                    
                                    # Check if values look truncated (if conversion was TRUNCATE)
                                    if len(previous_transformations) > 0 and previous_transformations[-1].get('conversion') == 'TRUNCATE':
                                        decimals = previous_transformations[-1].get('conversion_params', {}).get('decimals', 2)
                                        print(f"Layer preview: Values should be truncated to {decimals} decimals")
                                else:
                                    # Fallback to original column
                                    all_unique_values = df[column].dropna().unique()
                                    temp_column_name = column
                                    print(f"Layer preview: WARNING - Using original column '{column}' instead of intermediate")
                                
                                # Create temp dataframe with intermediate values
                                temp_df = pd.DataFrame({temp_column_name: all_unique_values})
                                
                                # Prepare the transformation config for the last step
                                # We need to update it to use the correct column name
                                last_transformation_adjusted = last_transformation.copy()
                                if 'input_column' in last_transformation_adjusted:
                                    # The transformation expects a specific column, but our temp_df only has temp_column_name
                                    # So we need to either rename the column or adjust the config
                                    if last_transformation_adjusted['input_column'] != temp_column_name:
                                        # Remove the input_column since we're applying to temp_column_name
                                        del last_transformation_adjusted['input_column']
                                
                                # Apply only the last transformation
                                print(f"Layer preview: Applying transformation {last_transformation_adjusted['method']} to column '{temp_column_name}'")
                                print(f"Layer preview: Input shape: {temp_df.shape}, columns: {list(temp_df.columns)}")
                                normalized_temp = view._apply_normalization(temp_df.copy(), {temp_column_name: last_transformation_adjusted})
                                print(f"Layer preview: Applied transformation, result columns: {list(normalized_temp.columns)}")
                                print(f"Layer preview: Result shape: {normalized_temp.shape}")
                                
                                # Check if transformation produced any non-null values
                                if temp_column_name in normalized_temp.columns:
                                    non_null_count = normalized_temp[temp_column_name].notna().sum()
                                    print(f"Layer preview: Non-null values in result: {non_null_count}/{len(normalized_temp)}")
                                
                            except Exception as e:
                                print(f"Error applying intermediate transformations: {e}")
                                # Fallback to using sample data
                                last_step = transformation_steps[column][-1]
                                all_unique_values = pd.Series(last_step['before']).dropna().unique()
                                temp_column_name = column
                                temp_df = pd.DataFrame({temp_column_name: all_unique_values})
                                view = DatasetNormalizationView()
                                last_method_config = method_config[-1].copy() if isinstance(method_config[-1], dict) else method_config[-1]
                                
                                # Adjust the config if it has an input_column that doesn't match
                                if isinstance(last_method_config, dict) and 'input_column' in last_method_config:
                                    if last_method_config['input_column'] != temp_column_name:
                                        del last_method_config['input_column']
                                
                                normalized_temp = view._apply_normalization(temp_df.copy(), {temp_column_name: last_method_config})
                        else:
                            # Single transformation or no show_steps
                            # Get ALL unique values from the full dataset, no limit
                            all_unique_values = df[column].dropna().unique()
                            
                            # Limit the number of unique values for performance and display
                            MAX_UNIQUE_FOR_PREVIEW = 100
                            if len(all_unique_values) > MAX_UNIQUE_FOR_PREVIEW:
                                print(f"Layer preview: Too many unique values ({len(all_unique_values)}), sampling {MAX_UNIQUE_FOR_PREVIEW}")
                                # Sample evenly across the unique values
                                step = len(all_unique_values) // MAX_UNIQUE_FOR_PREVIEW
                                all_unique_values = all_unique_values[::step][:MAX_UNIQUE_FOR_PREVIEW]
                            
                            temp_column_name = column
                            temp_df = pd.DataFrame({temp_column_name: all_unique_values})
                            
                            # Apply the normalization
                            view = DatasetNormalizationView()
                            normalized_temp = view._apply_normalization(temp_df.copy(), {column: method_config})
                        
                        # Build the mapping for all unique values
                        try:
                            # First, let's check if we have valid data
                            if len(normalized_temp) == 0:
                                print(f"Layer preview: WARNING - normalized_temp is empty")
                            
                            for idx, val in enumerate(all_unique_values):
                                # Make sure idx is within bounds
                                if idx >= len(normalized_temp):
                                    print(f"Layer preview: WARNING - Index {idx} out of bounds for normalized_temp with length {len(normalized_temp)}")
                                    normalized_val = '[Error: índice fuera de rango]'
                                else:
                                    # When dealing with layer preview, look for the appropriate column
                                    if temp_column_name in normalized_temp.columns:
                                        # Direct match with temp column name
                                        normalized_val = normalized_temp.iloc[idx][temp_column_name]
                                    elif target_column in normalized_temp.columns:
                                        # Found target column directly
                                        normalized_val = normalized_temp.iloc[idx][target_column]
                                    elif column in normalized_temp.columns:
                                        # Single column output
                                        normalized_val = normalized_temp.iloc[idx][column]
                                    else:
                                        # Column was transformed or removed, try to find the new column
                                        # First try columns with target_column prefix
                                        target_cols = [col for col in normalized_temp.columns if col.startswith(target_column) and col != target_column and len(col) > len(target_column)]
                                        if target_cols:
                                            normalized_val = normalized_temp.iloc[idx][target_cols[0]]
                                        else:
                                            # Try columns with temp_column_name prefix
                                            temp_cols = [col for col in normalized_temp.columns if col.startswith(temp_column_name) and col != temp_column_name]
                                            if temp_cols:
                                                normalized_val = normalized_temp.iloc[idx][temp_cols[0]]
                                            else:
                                                # Try any new columns
                                                new_cols = [col for col in normalized_temp.columns if col not in temp_df.columns]
                                                if new_cols:
                                                    # Use the first new column for the mapping
                                                    normalized_val = normalized_temp.iloc[idx][new_cols[0]]
                                                    if idx == 0:  # Debug first value only
                                                        print(f"Layer preview: Using new column '{new_cols[0]}' for mapping")
                                                else:
                                                    normalized_val = '[Columna eliminada]'
                                                    if idx == 0:  # Debug first value only
                                                        print(f"Layer preview: No matching column found. Available: {list(normalized_temp.columns)}")
                                
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
                        
                        # Calculate the actual unique count based on the data source
                        if show_steps and isinstance(method_config, list) and len(method_config) > 1:
                            # For multi-layer preview, show the count from intermediate data
                            if 'intermediate_df' in locals() and intermediate_column in intermediate_df.columns:
                                actual_unique_count = intermediate_df[intermediate_column].nunique()
                            else:
                                actual_unique_count = df[column].nunique()
                        else:
                            actual_unique_count = df[column].nunique()
                        
                        comparison[column] = {
                            'original': {
                                'sample': original_values.head(50).tolist(),
                                'stats': detect_column_type(df[column]),  # Use full dataset for stats
                                'all_unique_count': actual_unique_count,  # Show actual count
                                'mapping_sample_size': len(unique_mapping) if len(unique_mapping) < actual_unique_count else None  # Indicate if we're showing a sample
                            },
                            'normalized': {
                                'sample': normalized_values.head(50).tolist(),
                                'stats': detect_column_type(normalized_values)  # Use the actual normalized values
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
                                'stats': detect_column_type(normalized_values)  # Use the actual normalized values
                            },
                            'is_categorical': False
                        }
            
            # Clean NaN and Inf values before sending response
            def clean_for_json(obj):
                if isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_for_json(item) for item in obj]
                elif isinstance(obj, float):
                    if pd.isna(obj) or np.isinf(obj):
                        return None
                    return obj
                return obj
            
            response_data = {
                'preview': clean_for_json(comparison),
                'sample_size': len(sample_df),
                'all_column_types': clean_for_json(all_column_types)
            }
            
            # Add transformation steps if available
            if transformation_steps:
                response_data['transformation_steps'] = clean_for_json(transformation_steps)
            
            # Add warnings if any
            if hasattr(view, 'conversion_warnings') and view.conversion_warnings:
                response_data['warnings'] = view.conversion_warnings
            
            return success_response(response_data)
            
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
                # Campo remove_original_column eliminado - ahora se controla desde la normalización
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


class CustomNormalizationFunctionInfoView(APIView):
    """Get info about a custom normalization function"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        """Get information about a custom normalization function"""
        try:
            # Verificar que la función pertenezca al usuario o sea admin
            if request.user.is_staff:
                function = CustomNormalizationFunction.objects.get(pk=pk)
            else:
                function = CustomNormalizationFunction.objects.get(pk=pk, user=request.user)
            
            # Return simplified info for frontend
            return Response({
                'success': True,
                'function': {
                    'id': function.id,
                    'name': function.name,
                    'function_type': function.function_type,
                    'new_columns': function.new_columns or [],
                    # 'remove_original_column' campo eliminado
                    'description': function.description
                }
            })
        except CustomNormalizationFunction.DoesNotExist:
            return Response({
                'success': False,
                'error': 'Función no encontrada'
            }, status=status.HTTP_404_NOT_FOUND)


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