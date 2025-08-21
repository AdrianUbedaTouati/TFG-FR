"""
Vista para descargar conjuntos de datos divididos en formato CSV o Excel
"""
import pandas as pd
import numpy as np
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import io
from ..models import Dataset
from ..data_splitter import DataSplitter


@csrf_exempt
@require_http_methods(["POST"])
def download_split_data(request):
    """
    Descarga un conjunto específico (train/val/test) en formato CSV o Excel
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
        predictor_columns = data.get('predictor_columns', [])
        format_type = data.get('format', 'csv')  # 'csv' o 'excel'
        
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
            elif split_type == 'val':
                indices = list(range(train_end, val_end))
            else:  # test
                indices = list(range(val_end, n))
                
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
            elif split_type == 'val':
                indices = idx_val.flatten().tolist()
            else:  # test
                indices = idx_test.flatten().tolist()
        
        # Obtener datos del conjunto seleccionado
        split_data = df.iloc[indices][all_columns]
        
        # Añadir columna de índice original
        split_data.insert(0, 'original_index', indices)
        
        # Preparar respuesta según el formato
        if format_type == 'csv':
            # Crear CSV
            output = io.StringIO()
            split_data.to_csv(output, index=False)
            output.seek(0)
            
            response = HttpResponse(output.getvalue(), content_type='text/csv')
            filename = f"{dataset.name}_{split_type}_{split_method}.csv"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            
        else:  # excel
            # Crear Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                split_data.to_excel(writer, sheet_name=f'{split_type}_data', index=False)
                
                # Añadir hoja con información de la división
                info_data = {
                    'Dataset': [dataset.name],
                    'Split Type': [split_type],
                    'Split Method': [split_method],
                    'Total Rows': [len(split_data)],
                    'Train Size': [f"{train_size*100:.0f}%"],
                    'Validation Size': [f"{val_size*100:.0f}%"],
                    'Test Size': [f"{test_size*100:.0f}%"],
                    'Random State': [random_state if random_state else 'None'],
                    'Target Columns': [', '.join(target_columns) if target_columns else 'None'],
                    'Predictor Columns': [', '.join(predictor_columns) if predictor_columns else 'All']
                }
                info_df = pd.DataFrame(info_data)
                info_df.to_excel(writer, sheet_name='Split_Info', index=False)
                
            output.seek(0)
            
            response = HttpResponse(
                output.getvalue(), 
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            filename = f"{dataset.name}_{split_type}_{split_method}.xlsx"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except Dataset.DoesNotExist:
        return HttpResponse('Dataset no encontrado', status=404)
        
    except Exception as e:
        return HttpResponse(f'Error al generar archivo: {str(e)}', status=500)