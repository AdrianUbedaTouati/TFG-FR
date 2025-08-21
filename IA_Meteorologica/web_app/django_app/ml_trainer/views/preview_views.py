"""
Vistas para vista previa de datos con división aplicada
"""
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from ..models import Dataset
from ..data_splitter import DataSplitter


@csrf_exempt
@require_http_methods(["POST"])
def preview_split_data(request):
    """
    Vista previa de datos con división aplicada
    """
    try:
        # Parsear datos del request
        data = json.loads(request.body)
        
        dataset_id = data.get('dataset_id')
        split_type = data.get('split_type')  # 'train', 'val', 'test'
        split_method = data.get('split_method')
        split_config = data.get('split_config', {})
        train_size = data.get('train_size', 0.7)
        val_size = data.get('val_size', 0.15)
        test_size = data.get('test_size', 0.15)
        random_state = data.get('random_state')
        target_columns = data.get('target_columns', [])
        
        # Debug logging
        print(f"DEBUG preview_views: Received sizes - train: {train_size}, val: {val_size}, test: {test_size}")
        print(f"DEBUG preview_views: Sum = {train_size + val_size + test_size}")
        predictor_columns = data.get('predictor_columns', [])
        preview_rows = data.get('preview_rows', 20)
        
        # Obtener dataset
        dataset = Dataset.objects.get(id=dataset_id)
        df = pd.read_csv(dataset.file.path)
        
        # Preparar configuración de división
        split_config.update({
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'random_state': random_state
        })
        
        # Crear divisor de datos
        data_splitter = DataSplitter.create_splitter(
            strategy=split_method,
            config=split_config
        )
        
        # Obtener columnas a usar
        all_columns = list(set(predictor_columns + target_columns))
        if not all_columns:
            all_columns = df.columns.tolist()
        
        # Preparar datos para división
        X = df[all_columns].values
        y = df[target_columns[0]].values if target_columns else np.zeros(len(df))
        
        # Para estrategias que no sean 'sequential', necesitamos obtener índices de manera diferente
        if split_method == 'sequential':
            # Para división secuencial, los índices son consecutivos
            n = len(df)
            train_end = int(n * train_size)
            val_end = int(n * (train_size + val_size))
            
            if split_type == 'train':
                indices = list(range(0, train_end))
                total_rows = train_end
            elif split_type == 'val':
                indices = list(range(train_end, val_end))
                total_rows = val_end - train_end
            else:  # test
                indices = list(range(val_end, n))
                total_rows = n - val_end
                
        else:
            # Para otras estrategias, necesitamos rastrear los índices reales
            # Crear índices originales
            original_indices = np.arange(len(df))
            
            # Aplicar la misma división a los índices
            if split_method == 'group' and 'group_column' in split_config:
                groups = df[split_config['group_column']].values
                idx_train, _, idx_val, _, idx_test, _ = data_splitter.split(
                    original_indices.reshape(-1, 1), y, groups=groups
                )
            else:
                idx_train, _, idx_val, _, idx_test, _ = data_splitter.split(
                    original_indices.reshape(-1, 1), y
                )
            
            # Extraer los índices según el tipo
            if split_type == 'train':
                indices = idx_train.flatten().tolist()
                total_rows = len(idx_train)
            elif split_type == 'val':
                indices = idx_val.flatten().tolist()
                total_rows = len(idx_val)
            else:  # test
                indices = idx_test.flatten().tolist()
                total_rows = len(idx_test)
        
        # Obtener datos de vista previa
        preview_indices = indices[:preview_rows]
        preview_data = df.iloc[preview_indices][all_columns]
        
        # Calcular rango de índices
        if len(indices) > 0:
            start_idx = min(indices)
            end_idx = max(indices) + 1
        else:
            start_idx = 0
            end_idx = 0
        
        # Preparar datos para respuesta
        preview_dict = preview_data.to_dict('records')
        
        # Añadir índices originales
        for i, (idx, row) in enumerate(zip(preview_indices, preview_dict)):
            row['_index'] = int(idx)
        
        # Calcular estadísticas básicas
        stats = {}
        for col in all_columns:
            if col in preview_data.columns:
                try:
                    col_data = preview_data[col].dropna()
                    if pd.api.types.is_numeric_dtype(col_data):
                        stats[col] = {
                            'count': int(col_data.count()),
                            'mean': float(col_data.mean()),
                            'min': float(col_data.min()),
                            'max': float(col_data.max()),
                            'std': float(col_data.std())
                        }
                    else:
                        stats[col] = {
                            'count': int(col_data.count()),
                            'unique': int(col_data.nunique()),
                            'top': str(col_data.mode()[0]) if len(col_data.mode()) > 0 else 'N/A'
                        }
                except:
                    stats[col] = {'error': 'No se pudieron calcular estadísticas'}
        
        # Preparar respuesta
        response_data = {
            'success': True,
            'split_type': split_type,
            'split_method': split_method,
            'total_rows': total_rows,
            'start_index': int(start_idx),
            'end_index': int(end_idx),
            'columns': all_columns,
            'data': preview_dict,
            'statistics': stats,
            'preview_rows': len(preview_dict)
        }
        
        return JsonResponse(response_data)
        
    except Dataset.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Dataset no encontrado'
        }, status=404)
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error al generar vista previa: {str(e)}'
        }, status=500)