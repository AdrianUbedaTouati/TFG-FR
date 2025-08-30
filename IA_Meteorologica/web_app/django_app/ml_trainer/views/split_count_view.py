"""
Vista para obtener conteos exactos de la división de datos
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
def get_split_counts(request):
    """
    Obtiene los conteos exactos de muestras para cada conjunto (train/val/test)
    """
    try:
        # Parsear datos del request
        data = json.loads(request.body)
        
        dataset_id = data.get('dataset_id')
        split_method = data.get('split_method', 'random')
        split_config = data.get('split_config', {})
        train_size = data.get('train_size', 0.7)
        val_size = data.get('val_size', 0.15)
        test_size = data.get('test_size', 0.15)
        
        # Asegurar que random_state sea entero o None
        random_state = data.get('random_state')
        if random_state is not None:
            try:
                random_state = int(random_state)
            except (ValueError, TypeError):
                random_state = None
        
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
        
        # Obtener índices para cada conjunto
        n_samples = len(df)
        
        if split_method == 'sequential':
            # Para división secuencial, los índices son consecutivos
            train_end = int(n_samples * train_size)
            val_end = int(n_samples * (train_size + val_size))
            
            train_count = train_end
            val_count = val_end - train_end
            test_count = n_samples - val_end
            
        else:
            # Para otras estrategias, necesitamos calcular la división real
            original_indices = np.arange(n_samples)
            y_dummy = np.zeros(n_samples)
            
            # Aplicar la división
            if split_method == 'group' and 'group_column' in split_config:
                groups = df[split_config['group_column']].values
                idx_train, _, idx_val, _, idx_test, _ = data_splitter.split(
                    original_indices.reshape(-1, 1), y_dummy, groups=groups
                )
            else:
                idx_train, _, idx_val, _, idx_test, _ = data_splitter.split(
                    original_indices.reshape(-1, 1), y_dummy
                )
            
            train_count = len(idx_train)
            val_count = len(idx_val)
            test_count = len(idx_test)
        
        # Usar los porcentajes configurados por el usuario
        # Convertir a porcentajes si vienen como decimales
        train_pct = int(train_size * 100) if train_size <= 1 else int(train_size)
        val_pct = int(val_size * 100) if val_size <= 1 else int(val_size)
        test_pct = int(test_size * 100) if test_size <= 1 else int(test_size)
        
        # Asegurar que los porcentajes sumen 100
        total_pct = train_pct + val_pct + test_pct
        if total_pct != 100:
            # Ajustar el test_pct para que sume 100
            test_pct = 100 - train_pct - val_pct
        
        # Preparar respuesta
        response_data = {
            'success': True,
            'total_samples': n_samples,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count,
            'train_percentage': train_pct,
            'val_percentage': val_pct,
            'test_percentage': test_pct
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
            'error': f'Error al calcular división: {str(e)}'
        }, status=500)