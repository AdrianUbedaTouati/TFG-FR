"""
Dataset normalization views
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.utils import timezone
import pandas as pd
import numpy as np
from datetime import datetime

from ..models import Dataset
from ..normalization_methods import NumNorm, TextNorm, Normalizador
from ..utils import (
    load_dataset, error_response, success_response,
    validate_dataframe, detect_column_type
)
from ..constants import (
    ERROR_PARSING_FAILED, ERROR_NORMALIZATION_FAILED,
    SUCCESS_NORMALIZATION_COMPLETE
)


class DatasetNormalizationView(APIView):
    """Apply normalization to a dataset"""
    
    def post(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        
        # Get normalization parameters
        normalization_config = request.data.get('normalization', {})
        create_copy = request.data.get('create_copy', True)
        copy_name = request.data.get('copy_name', '')
        
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
            
            # Save normalized dataset
            if create_copy:
                new_dataset = self._save_normalized_copy(
                    dataset, normalized_df, copy_name, normalization_config
                )
                
                return success_response({
                    'dataset_id': new_dataset.id,
                    'dataset_name': new_dataset.name,
                    'normalization_applied': normalization_config
                }, message=SUCCESS_NORMALIZATION_COMPLETE)
            else:
                # Overwrite original
                dataset.file.save(dataset.file.name, normalized_df.to_csv(index=False))
                dataset.is_normalized = True
                dataset.normalization_method = str(normalization_config)
                dataset.save()
                
                return success_response({
                    'dataset_id': dataset.id,
                    'normalization_applied': normalization_config
                }, message=SUCCESS_NORMALIZATION_COMPLETE)
                
        except Exception as e:
            return error_response(ERROR_NORMALIZATION_FAILED.format(str(e)))
    
    def _apply_normalization(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply normalization based on configuration"""
        normalized_df = df.copy()
        
        for column, method in config.items():
            if column not in df.columns:
                continue
            
            # Detect column type
            col_info = detect_column_type(df[column])
            
            if col_info.get('type') == 'numeric':
                # Apply numeric normalization
                normalizer = NumNorm(method)
                normalized_df[column] = normalizer.normalize(df[column])
            elif col_info.get('type') in ['categorical', 'text']:
                # Apply text normalization if applicable
                if method in ['onehot', 'label']:
                    normalizer = TextNorm(method)
                    normalized_df[column] = normalizer.normalize(df[column])
        
        return normalized_df
    
    def _save_normalized_copy(self, original_dataset, normalized_df, 
                              copy_name, config):
        """Save normalized dataset as a new copy"""
        if not copy_name:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            copy_name = f"{original_dataset.name}_normalized_{timestamp}"
        
        # Save to file
        file_path = f"media/datasets/{copy_name}.csv"
        normalized_df.to_csv(file_path, index=False)
        
        # Create new dataset record
        new_dataset = Dataset.objects.create(
            name=copy_name,
            file=file_path,
            user=original_dataset.user,
            is_normalized=True,
            parent_dataset=original_dataset,
            parent_dataset_name=original_dataset.name,
            root_dataset_id=original_dataset.root_dataset_id or original_dataset.id,
            normalization_method=str(config),
            short_description=f"Normalized from {original_dataset.name}",
            long_description=f"Normalization applied: {config}"
        )
        
        return new_dataset


class DatasetNormalizationPreviewView(APIView):
    """Preview normalization results without saving"""
    
    def post(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        
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
                    comparison[column] = {
                        'original': {
                            'sample': sample_df[column].head(10).tolist(),
                            'stats': detect_column_type(sample_df[column])
                        },
                        'normalized': {
                            'sample': normalized_sample[column].head(10).tolist(),
                            'stats': detect_column_type(normalized_sample[column])
                        }
                    }
            
            return success_response({
                'preview': comparison,
                'sample_size': len(sample_df)
            })
            
        except Exception as e:
            return error_response(f"Preview failed: {str(e)}")