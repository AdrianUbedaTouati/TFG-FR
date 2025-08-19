"""
Dataset-related views for ML Trainer
"""
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from django.http import HttpResponse
from django.core.files.base import ContentFile
from django.core.cache import cache
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from typing import Dict, Any, Optional

from ..models import Dataset
from ..serializers import DatasetSerializer
from ..utils import (
    load_dataset, detect_column_type, calculate_correlation_matrix,
    error_response, success_response, validate_dataframe,
    get_memory_usage
)
from ..utils_helpers import safe_to_list, safe_float, safe_dict_values
from ..constants import (
    HISTOGRAM_BINS, DEFAULT_FIGURE_SIZE, HEATMAP_FIGURE_SIZE,
    ERROR_DATASET_NOT_FOUND, ERROR_PARSING_FAILED,
    SUCCESS_DATASET_UPLOADED
)


class DatasetListCreateView(generics.ListCreateAPIView):
    """List all datasets or create a new one"""
    serializer_class = DatasetSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        # Los usuarios normales solo ven sus datasets
        # Los admin/staff ven todos
        if self.request.user.is_staff:
            return Dataset.objects.all()
        return Dataset.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        # Asociar el dataset al usuario actual
        serializer.save(user=self.request.user)


class DatasetDetailView(generics.RetrieveDestroyAPIView):
    """Retrieve or delete a specific dataset"""
    serializer_class = DatasetSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        if self.request.user.is_staff:
            return Dataset.objects.all()
        return Dataset.objects.filter(user=self.request.user)


class DatasetUpdateInfoView(APIView):
    """Update dataset metadata (name, descriptions)"""
    permission_classes = [IsAuthenticated]
    
    def put(self, request, dataset_id):
        try:
            # Obtener el dataset
            if request.user.is_staff:
                dataset = get_object_or_404(Dataset, id=dataset_id)
            else:
                dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
            
            # Obtener los nuevos valores
            name = request.data.get('name', '').strip()
            short_description = request.data.get('short_description', '').strip()
            long_description = request.data.get('long_description', '').strip()
            
            # Validar que el nombre no esté vacío
            if not name:
                return error_response('El nombre del dataset es obligatorio')
            
            # Actualizar los campos
            dataset.name = name
            dataset.short_description = short_description if short_description else None
            dataset.long_description = long_description if long_description else None
            dataset.save()
            
            # Retornar la información actualizada
            return success_response({
                'id': dataset.id,
                'name': dataset.name,
                'short_description': dataset.short_description,
                'long_description': dataset.long_description
            })
            
        except Exception as e:
            return error_response(f'Error al actualizar la información del dataset: {str(e)}')


class DatasetColumnsView(APIView):
    """Get column information for a dataset"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Load dataset (sin usar caché para asegurar datos frescos)
        df = load_dataset(dataset.file.path, use_cache=False)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Get columns and dtypes - ensure they are strings
        columns = [str(col) for col in df.columns if col and str(col).strip()]
        dtypes = {str(col): str(df[col].dtype) for col in columns}
        
        # Generate stats for each column
        stats = {}
        for col in columns:
            col_info = detect_column_type(df[col])
            # Add the real unique count to stats
            col_info['unique_count'] = int(df[col].nunique())
            stats[col] = col_info
        
        # Generate preview data (first 10 rows)
        preview = {}
        for col in columns:
            # Convert NaN values to None for JSON serialization
            col_data = df[col].head(10)
            preview[col] = safe_to_list(col_data)
        
        # Calculate total null count
        total_null_count = df.isnull().sum().sum()
        
        # Complete response data
        response_data = {
            'columns': columns,
            'dtypes': dtypes,
            'shape': [len(df), len(columns)],
            'stats': stats,
            'preview': preview,
            'total_null_count': int(total_null_count),
            'memory_usage': get_memory_usage(df)
        }
        
        return success_response(response_data)


class DatasetColumnDetailsView(APIView):
    """Get detailed information about a specific column"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk, column_name):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Clean and convert column names to strings
        df.columns = [str(col).strip() for col in df.columns]
        column_name = str(column_name).strip()
        
        # Check column exists
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found")
        
        column = df[column_name]
        
        # Get detailed stats
        stats = detect_column_type(column)
        
        # Get all value counts for the pie chart
        all_value_counts = column.value_counts()
        # For response, limit to 50 most frequent
        value_counts = all_value_counts.head(50).to_dict()
        # But also provide all counts if needed - limit to 1000 to avoid memory issues
        if len(all_value_counts) <= 1000:
            value_counts_complete = all_value_counts.to_dict()
        else:
            # For very large datasets, provide top 1000
            value_counts_complete = all_value_counts.head(1000).to_dict()
        
        # Get frequency data
        frequency_data = []
        total_count = len(column)
        
        for value, count in column.value_counts().items():
            frequency_data.append({
                'value': str(value) if pd.notna(value) else 'null',
                'count': int(count),
                'percentage': round((count / total_count) * 100, 2)
            })
        
        # Add null count if there are nulls
        null_count = column.isnull().sum()
        if null_count > 0:
            frequency_data.append({
                'value': 'null',
                'count': int(null_count),
                'percentage': round((null_count / total_count) * 100, 2)
            })
        
        response_data = {
            'column_name': column_name,
            'stats': stats,
            'value_counts': value_counts,
            'value_counts_complete': value_counts_complete,  # All value counts for pie chart
            'frequency_data': frequency_data[:100],  # Limit to 100 items
            'total_rows': int(total_count),
            'unique_count': int(column.nunique())  # Add the real unique count
        }
        
        return success_response(response_data)


class DatasetVariableAnalysisView(APIView):
    """Analyze a specific variable from a dataset"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk, column_name):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        column_name = str(column_name).strip()
        
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found")
        
        # Get analysis parameters
        analysis_type = request.data.get('analysis_type', 'histogram')
        include_outliers = request.data.get('include_outliers', True)
        num_bins = request.data.get('num_bins', 20)
        
        column = df[column_name]
        
        # Detect if numeric
        is_numeric = pd.api.types.is_numeric_dtype(column)
        
        try:
            if analysis_type == 'histogram' and is_numeric:
                plot_data = self._generate_histogram(
                    column, column_name, num_bins, include_outliers
                )
            elif analysis_type == 'boxplot' and is_numeric:
                plot_data = self._generate_boxplot(
                    column, column_name, include_outliers
                )
            elif analysis_type == 'outlier_map' and is_numeric:
                plot_data = self._generate_outlier_map(
                    column, column_name
                )
            else:
                return error_response("Invalid analysis type or non-numeric column")
            
            return success_response(plot_data)
            
        except Exception as e:
            return error_response(f"Analysis error: {str(e)}")
    
    def _generate_histogram(self, column, column_name, num_bins, include_outliers):
        """Generate histogram data and plot"""
        # Remove nulls
        data = column.dropna()
        
        if not include_outliers:
            # Remove outliers using IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        # Generate histogram
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        counts, bins, patches = ax.hist(data, bins=num_bins, edgecolor='black', alpha=0.7)
        
        ax.set_xlabel(column_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {column_name}')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = data.mean()
        std_val = data.std()
        mean_str = f'{mean_val:.2f}' if pd.notna(mean_val) else 'N/A'
        std_str = f'{std_val:.2f}' if pd.notna(std_val) else 'N/A'
        stats_text = f'Mean: {mean_str}\nStd: {std_str}\nCount: {len(data)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Convert plot to base64
        plot_base64 = self.generate_plot_base64(fig)
        
        return {
            'plot': plot_base64,
            'bins': safe_to_list(bins),
            'counts': safe_to_list(counts),
            'statistics': {
                'mean': safe_float(data.mean()),
                'std': safe_float(data.std()),
                'min': safe_float(data.min()),
                'max': safe_float(data.max()),
                'count': int(len(data))
            }
        }
    
    def _generate_boxplot(self, column, column_name, include_outliers):
        """Generate boxplot data and plot"""
        # Remove nulls
        data = column.dropna()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot([data], showfliers=include_outliers, labels=[column_name])
        ax.set_ylabel('Value')
        ax.set_title(f'Boxplot of {column_name}')
        ax.grid(True, alpha=0.3)
        
        # Calculate statistics
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        median_val = data.median()
        median_str = f'{median_val:.2f}' if pd.notna(median_val) else 'N/A'
        q1_str = f'{Q1:.2f}' if pd.notna(Q1) else 'N/A'
        q3_str = f'{Q3:.2f}' if pd.notna(Q3) else 'N/A'
        iqr_str = f'{IQR:.2f}' if pd.notna(IQR) else 'N/A'
        stats_text = f'Median: {median_str}\nQ1: {q1_str}\nQ3: {q3_str}\nIQR: {iqr_str}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plot_base64 = self.generate_plot_base64(fig)
        
        return {
            'plot': plot_base64,
            'statistics': {
                'median': safe_float(data.median()),
                'q1': safe_float(Q1),
                'q3': safe_float(Q3),
                'iqr': safe_float(IQR),
                'min': safe_float(data.min()),
                'max': safe_float(data.max()),
                'outlier_count': int(((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()) if pd.notna(Q1) and pd.notna(Q3) else 0
            }
        }
    
    def _generate_outlier_map(self, column, column_name):
        """Generate outlier visualization"""
        # Remove nulls
        data = column.dropna()
        
        # Calculate outlier bounds using IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        normal_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        
        # Plot normal data
        ax.scatter(range(len(normal_data)), normal_data, alpha=0.6, label='Normal', s=30)
        
        # Plot outliers
        outlier_indices = data[(data < lower_bound) | (data > upper_bound)].index
        ax.scatter(outlier_indices, outliers, color='red', alpha=0.8, label='Outliers', s=50, marker='^')
        
        # Add threshold lines
        ax.axhline(y=upper_bound, color='red', linestyle='--', alpha=0.5, label='Upper Bound')
        ax.axhline(y=lower_bound, color='red', linestyle='--', alpha=0.5, label='Lower Bound')
        
        ax.set_xlabel('Index')
        ax.set_ylabel(column_name)
        ax.set_title(f'Outlier Map for {column_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_base64 = self.generate_plot_base64(fig)
        
        return {
            'plot': plot_base64,
            'outlier_info': {
                'total_outliers': int(len(outliers)),
                'outlier_percentage': safe_float((len(outliers) / len(data)) * 100),
                'lower_bound': safe_float(lower_bound),
                'upper_bound': safe_float(upper_bound),
                'outlier_values': safe_to_list(outliers.head(20))  # First 20 outliers
            }
        }
    
    def generate_plot_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{plot_base64}"


class DatasetDownloadView(APIView):
    """Download a dataset as CSV"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Read the file
        try:
            with open(dataset.file.path, 'rb') as f:
                response = HttpResponse(f.read(), content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="{dataset.name}.csv"'
                return response
        except Exception as e:
            return error_response(f"Error downloading file: {str(e)}")


class DatasetReportView(APIView):
    """Generate analysis report for a dataset"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Implementation would generate a comprehensive report
        # This is a placeholder
        return success_response({
            'message': 'Report generation not yet implemented'
        })


class DatasetAnalysisView(APIView):
    """Comprehensive dataset analysis"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Perform comprehensive analysis
        analysis = {
            'basic_info': self._get_basic_info(df),
            'missing_data': self._analyze_missing_data(df),
            'correlations': self._analyze_correlations(df),
            'visualizations': {
                'correlation_heatmap': self._create_correlation_heatmap(df),
                'missing_data_chart': self._create_missing_data_chart(df)
            }
        }
        
        return success_response(analysis)
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            'shape': list(df.shape),
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': get_memory_usage(df),
            'duplicate_rows': int(df.duplicated().sum())
        }
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        return {
            'total_missing': int(missing.sum()),
            'columns_with_missing': missing[missing > 0].to_dict(),
            'missing_percentage': missing_pct[missing_pct > 0].to_dict()
        }
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric variables"""
        corr_matrix = calculate_correlation_matrix(df)
        
        # Find highly correlated pairs
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': safe_float(corr_val)
                    })
        
        return {
            'high_correlations': high_corr,
            'correlation_matrix': corr_matrix.to_dict()
        }
    
    def _create_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Create correlation heatmap"""
        corr_matrix = calculate_correlation_matrix(df)
        
        if corr_matrix.empty:
            return ""
        
        fig, ax = plt.subplots(figsize=HEATMAP_FIGURE_SIZE)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix Heatmap')
        
        return self.generate_plot_base64(fig)
    
    def _create_missing_data_chart(self, df: pd.DataFrame) -> str:
        """Create missing data visualization"""
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if missing.empty:
            return ""
        
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        missing.plot(kind='barh', ax=ax)
        ax.set_xlabel('Number of Missing Values')
        ax.set_title('Missing Data by Column')
        ax.grid(True, alpha=0.3)
        
        return self.generate_plot_base64(fig)


class DatasetDeleteColumnView(APIView):
    """Delete a column from a dataset"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get column name from request
        column_name = request.data.get('column_name')
        if not column_name:
            return error_response("Column name is required")
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Log initial state
        print(f"Dataset path: {dataset.file.path}")
        print(f"Initial columns: {list(df.columns)}")
        print(f"Column to delete: {column_name}")
        
        # Check if column exists
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found in dataset")
        
        # Remove the column
        df = df.drop(columns=[column_name])
        print(f"After drop - columns: {list(df.columns)}")
        
        # Save the modified dataset
        try:
            # Convert DataFrame to CSV string
            from io import StringIO
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Get the original file name
            original_name = os.path.basename(dataset.file.name)
            
            # Invalidar caché antes de eliminar el archivo antiguo
            old_path = dataset.file.path
            old_cache_key = f'dataset_{old_path}'
            cache.delete(old_cache_key)
            print(f"Old cache invalidated for key: {old_cache_key}")
            
            # Delete the old file
            dataset.file.delete(save=False)
            
            # Save the new content
            dataset.file.save(
                original_name,
                ContentFile(csv_content.encode('utf-8')),
                save=False
            )
            
            # Update dataset metadata if fields exist
            if hasattr(dataset, 'column_count'):
                dataset.column_count = len(df.columns)
            if hasattr(dataset, 'row_count'):
                dataset.row_count = len(df)
            
            # Force save to ensure database is updated
            dataset.save()
            
            # Invalidar el caché del dataset
            cache_key = f'dataset_{dataset.file.path}'
            cache.delete(cache_key)
            print(f"Cache invalidated for key: {cache_key}")
            
            # Verify the change was saved
            df_verify = pd.read_csv(dataset.file.path)
            print(f"After save - columns: {list(df_verify.columns)}")
            print(f"New file path: {dataset.file.path}")
            
            if column_name in df_verify.columns:
                return error_response(f"Column '{column_name}' was not deleted properly")
            
            return success_response({
                'message': f"Column '{column_name}' deleted successfully",
                'new_column_count': len(df_verify.columns),
                'columns': list(df_verify.columns),
                'file_path': dataset.file.path  # Para depuración
            })
        except Exception as e:
            print(f"Error in delete column: {str(e)}")
            import traceback
            traceback.print_exc()
            return error_response(f"Error saving dataset: {str(e)}")


class DatasetRenameColumnView(APIView):
    """Rename a column in a dataset"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get old and new column names from request
        old_name = request.data.get('old_name')
        new_name = request.data.get('new_name')
        
        if not old_name or not new_name:
            return error_response("Both old_name and new_name are required")
        
        # Validate new name
        new_name = new_name.strip()
        if not new_name:
            return error_response("New column name cannot be empty")
        
        if old_name == new_name:
            return error_response("New name must be different from old name")
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Check if old column exists
        if old_name not in df.columns:
            return error_response(f"Column '{old_name}' not found in dataset")
        
        # Check if new column name already exists
        if new_name in df.columns:
            return error_response(f"Column '{new_name}' already exists in dataset")
        
        # Rename the column
        df = df.rename(columns={old_name: new_name})
        
        # Save the modified dataset
        try:
            # Convert DataFrame to CSV string
            from io import StringIO
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Get the original file name
            original_name = os.path.basename(dataset.file.name)
            
            # Invalidar caché antes de eliminar el archivo antiguo
            old_path = dataset.file.path
            old_cache_key = f'dataset_{old_path}'
            cache.delete(old_cache_key)
            
            # Delete the old file
            dataset.file.delete(save=False)
            
            # Save the new content
            dataset.file.save(
                original_name,
                ContentFile(csv_content.encode('utf-8')),
                save=False
            )
            
            # Force save to ensure database is updated
            dataset.save()
            
            # Invalidar el caché del dataset
            cache_key = f'dataset_{dataset.file.path}'
            cache.delete(cache_key)
            
            # Verify the change was saved
            df_verify = pd.read_csv(dataset.file.path)
            
            if old_name in df_verify.columns:
                return error_response(f"Column '{old_name}' was not renamed properly")
            
            if new_name not in df_verify.columns:
                return error_response(f"Column '{new_name}' was not created properly")
            
            return success_response({
                'message': f"Column '{old_name}' renamed to '{new_name}' successfully",
                'columns': list(df_verify.columns)
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return error_response(f"Error saving dataset: {str(e)}")


class DatasetColumnDataView(APIView):
    """Get all data from a specific column for searching/filtering"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get column name from request
        column_name = request.data.get('column_name')
        if not column_name:
            return error_response("Column name is required")
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Check if column exists
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found in dataset")
        
        # Get column data
        column_data = df[column_name]
        
        # Convert to list, handling null values
        values = []
        for val in column_data:
            if pd.isna(val):
                values.append(None)
            else:
                values.append(str(val))
        
        return success_response({
            'column_name': column_name,
            'values': values,
            'total_count': len(values),
            'unique_count': column_data.nunique(),
            'null_count': column_data.isnull().sum()
        })


class DatasetFilterValuesView(APIView):
    """Filter dataset by removing rows with specific values in a column"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get parameters
        column_name = request.data.get('column_name')
        values_to_remove = request.data.get('values_to_remove', [])
        
        if not column_name:
            return error_response("Column name is required")
        
        if not values_to_remove:
            return error_response("Values to remove are required")
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Check if column exists
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found in dataset")
        
        # Count rows before filtering
        rows_before = len(df)
        
        # Convert values to the appropriate type based on the column dtype
        column_dtype = df[column_name].dtype
        converted_values = []
        
        for value in values_to_remove:
            if value == 'null' or value is None:
                # Handle null values separately
                continue
            try:
                # Try to convert to the column's dtype
                if pd.api.types.is_numeric_dtype(column_dtype):
                    # Try to convert to numeric
                    if pd.api.types.is_integer_dtype(column_dtype):
                        converted_values.append(int(float(value)))
                    else:
                        converted_values.append(float(value))
                elif pd.api.types.is_datetime64_any_dtype(column_dtype):
                    converted_values.append(pd.to_datetime(value))
                else:
                    # Keep as string for other types
                    converted_values.append(str(value))
            except (ValueError, TypeError):
                # If conversion fails, try keeping original value
                converted_values.append(value)
        
        # Filter out rows with specified values
        mask = df[column_name].isin(converted_values)
        
        # Also check for null values if 'null' was in the values to remove
        if 'null' in values_to_remove or None in values_to_remove:
            mask = mask | df[column_name].isnull()
        
        df_filtered = df[~mask]
        
        # Count rows removed
        rows_removed = rows_before - len(df_filtered)
        
        if rows_removed == 0:
            return error_response("No rows were removed. The specified values might not exist in the column.")
        
        # Save the modified dataset
        try:
            from io import StringIO
            csv_buffer = StringIO()
            df_filtered.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Get the original file name
            original_name = os.path.basename(dataset.file.name)
            
            # Invalidar caché
            old_path = dataset.file.path
            cache.delete(f'dataset_{old_path}')
            
            # Delete the old file
            dataset.file.delete(save=False)
            
            # Save the new content
            dataset.file.save(
                original_name,
                ContentFile(csv_content.encode('utf-8')),
                save=False
            )
            
            # Update metadata
            if hasattr(dataset, 'row_count'):
                dataset.row_count = len(df_filtered)
            
            dataset.save()
            
            # Invalidar el caché del nuevo archivo
            cache.delete(f'dataset_{dataset.file.path}')
            
            return success_response({
                'message': f"Successfully removed {rows_removed} rows",
                'rows_removed': rows_removed,
                'rows_remaining': len(df_filtered),
                'values_removed': len(values_to_remove)
            })
        except Exception as e:
            return error_response(f"Error saving dataset: {str(e)}")


class DatasetRemoveNullsView(APIView):
    """Remove rows with null values in a specific column"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get column name
        column_name = request.data.get('column_name')
        
        if not column_name:
            return error_response("Column name is required")
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Check if column exists
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found in dataset")
        
        # Count nulls before
        nulls_before = df[column_name].isnull().sum()
        rows_before = len(df)
        
        if nulls_before == 0:
            return error_response(f"Column '{column_name}' has no null values")
        
        # Remove rows with nulls in the specified column
        df_filtered = df[df[column_name].notna()]
        
        rows_removed = rows_before - len(df_filtered)
        
        # Save the modified dataset
        try:
            from io import StringIO
            csv_buffer = StringIO()
            df_filtered.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Get the original file name
            original_name = os.path.basename(dataset.file.name)
            
            # Invalidar caché
            old_path = dataset.file.path
            cache.delete(f'dataset_{old_path}')
            
            # Delete the old file
            dataset.file.delete(save=False)
            
            # Save the new content
            dataset.file.save(
                original_name,
                ContentFile(csv_content.encode('utf-8')),
                save=False
            )
            
            # Update metadata
            if hasattr(dataset, 'row_count'):
                dataset.row_count = len(df_filtered)
            
            dataset.save()
            
            # Invalidar el caché del nuevo archivo
            cache.delete(f'dataset_{dataset.file.path}')
            
            return success_response({
                'message': f"Successfully removed {rows_removed} rows with null values",
                'rows_removed': rows_removed,
                'rows_remaining': len(df_filtered)
            })
        except Exception as e:
            return error_response(f"Error saving dataset: {str(e)}")


class DatasetRemoveAllNullRowsView(APIView):
    """Remove all rows that contain any null value in any column"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Count rows before
        rows_before = len(df)
        
        # Remove rows with any null value
        df_filtered = df.dropna(how='any')
        
        rows_removed = rows_before - len(df_filtered)
        
        if rows_removed == 0:
            return error_response("No se encontraron filas con valores nulos")
        
        # Save the modified dataset
        try:
            from io import StringIO
            csv_buffer = StringIO()
            df_filtered.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Get the original file name
            original_name = os.path.basename(dataset.file.name)
            
            # Invalidar caché
            old_path = dataset.file.path
            cache.delete(f'dataset_{old_path}')
            
            # Delete the old file
            dataset.file.delete(save=False)
            
            # Save the new content
            dataset.file.save(
                original_name,
                ContentFile(csv_content.encode('utf-8')),
                save=False
            )
            
            # Update metadata
            if hasattr(dataset, 'row_count'):
                dataset.row_count = len(df_filtered)
            
            dataset.save()
            
            # Invalidar el caché del nuevo archivo
            cache.delete(f'dataset_{dataset.file.path}')
            
            return success_response({
                'message': f"Successfully removed {rows_removed} rows with null values",
                'rows_removed': rows_removed,
                'rows_remaining': len(df_filtered)
            })
        except Exception as e:
            return error_response(f"Error saving dataset: {str(e)}")


class DatasetFillNullsView(APIView):
    """Fill null values in a specific column"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get parameters
        column_name = request.data.get('column_name')
        method = request.data.get('method')
        custom_value = request.data.get('custom_value')
        
        if not column_name:
            return error_response("Column name is required")
        
        if not method:
            return error_response("Fill method is required")
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Check if column exists
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found in dataset")
        
        # Count nulls before
        nulls_before = df[column_name].isnull().sum()
        
        if nulls_before == 0:
            return error_response(f"Column '{column_name}' has no null values to fill")
        
        # Fill nulls based on method
        try:
            if method == 'mean':
                if not pd.api.types.is_numeric_dtype(df[column_name]):
                    return error_response("Mean can only be used for numeric columns")
                fill_value = df[column_name].mean()
                df[column_name] = df[column_name].fillna(fill_value)
            elif method == 'median':
                if not pd.api.types.is_numeric_dtype(df[column_name]):
                    return error_response("Median can only be used for numeric columns")
                fill_value = df[column_name].median()
                df[column_name] = df[column_name].fillna(fill_value)
            elif method == 'mode':
                mode_values = df[column_name].mode()
                if len(mode_values) == 0:
                    return error_response("No mode value found for this column")
                fill_value = mode_values[0]
                df[column_name] = df[column_name].fillna(fill_value)
            elif method == 'zero':
                fill_value = 0
                df[column_name] = df[column_name].fillna(fill_value)
            elif method == 'empty':
                fill_value = ''
                df[column_name] = df[column_name].fillna(fill_value)
            elif method == 'unknown':
                fill_value = 'Unknown'
                df[column_name] = df[column_name].fillna(fill_value)
            elif method == 'ffill':
                df[column_name] = df[column_name].fillna(method='ffill')
                fill_value = 'forward fill'
            elif method == 'bfill':
                df[column_name] = df[column_name].fillna(method='bfill')
                fill_value = 'backward fill'
            elif method == 'interpolate':
                if not pd.api.types.is_numeric_dtype(df[column_name]):
                    return error_response("Interpolation can only be used for numeric columns")
                df[column_name] = df[column_name].interpolate(method='linear')
                fill_value = 'interpolated values'
            elif method == 'custom':
                if custom_value is None:
                    return error_response("Custom value is required when using custom method")
                # Convert custom value to appropriate type
                if pd.api.types.is_numeric_dtype(df[column_name]):
                    try:
                        fill_value = float(custom_value)
                    except ValueError:
                        return error_response(f"Invalid numeric value: {custom_value}")
                else:
                    fill_value = str(custom_value)
                df[column_name] = df[column_name].fillna(fill_value)
            else:
                return error_response(f"Invalid fill method: {method}")
            
            # Verify nulls were filled
            nulls_after = df[column_name].isnull().sum()
            nulls_filled = nulls_before - nulls_after
            
            # Save the modified dataset
            from io import StringIO
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Get the original file name
            original_name = os.path.basename(dataset.file.name)
            
            # Invalidar caché
            old_path = dataset.file.path
            cache.delete(f'dataset_{old_path}')
            
            # Delete the old file
            dataset.file.delete(save=False)
            
            # Save the new content
            dataset.file.save(
                original_name,
                ContentFile(csv_content.encode('utf-8')),
                save=False
            )
            
            dataset.save()
            
            # Invalidar el caché del nuevo archivo
            cache.delete(f'dataset_{dataset.file.path}')
            
            return success_response({
                'message': f"Successfully filled {nulls_filled} null values with {fill_value}",
                'nulls_filled': nulls_filled,
                'fill_value': str(fill_value),
                'method': method
            })
        except Exception as e:
            return error_response(f"Error filling null values: {str(e)}")


class DatasetReplaceValuesView(APIView):
    """Replace values at specific indices in a dataset column"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get parameters
        column_name = request.data.get('column_name')
        indices = request.data.get('indices', [])
        new_value = request.data.get('new_value', '')
        char_replace = request.data.get('char_replace', False)
        char_to_find = request.data.get('char_to_find', '')
        char_to_replace = request.data.get('char_to_replace', '')
        partial_replace = request.data.get('partial_replace', False)
        partial_pattern = request.data.get('partial_pattern', [])
        partial_type = request.data.get('partial_type', 'complete')
        
        if not column_name:
            return error_response("Column name is required")
        
        if not indices:
            return error_response("Indices are required")
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Check if column exists
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found in dataset")
        
        # Get column type for validation
        column_dtype = df[column_name].dtype
        is_numeric = pd.api.types.is_numeric_dtype(column_dtype)
        
        # Replace values at specified indices
        try:
            # Convert indices to int and validate
            valid_indices = []
            for idx in indices:
                try:
                    idx_int = int(idx)
                    if 0 <= idx_int < len(df):
                        valid_indices.append(idx_int)
                except ValueError:
                    continue
            
            if not valid_indices:
                return error_response("No valid indices provided")
            
            # Process replacements based on mode
            if char_replace and char_to_find:
                # Character/substring replacement mode
                for idx in valid_indices:
                    current_value = df.loc[idx, column_name]
                    if pd.notna(current_value):
                        # Convert to string for replacement
                        str_value = str(current_value)
                        # Replace all occurrences of char_to_find with char_to_replace
                        new_str_value = str_value.replace(char_to_find, char_to_replace)
                        
                        # For numeric columns, try to convert back to numeric
                        if is_numeric:
                            try:
                                # Try to convert to float first
                                numeric_value = float(new_str_value)
                                # If original column was integer and value has no decimal part, convert to int
                                if pd.api.types.is_integer_dtype(column_dtype) and numeric_value.is_integer():
                                    df.loc[idx, column_name] = int(numeric_value)
                                else:
                                    df.loc[idx, column_name] = numeric_value
                            except (ValueError, TypeError):
                                # If conversion fails, keep the string value
                                # This might cause issues with numeric columns, so we'll warn
                                print(f"Warning: Could not convert '{new_str_value}' to numeric for row {idx}")
                                df.loc[idx, column_name] = new_str_value
                        else:
                            # For non-numeric columns, just use the string value
                            df.loc[idx, column_name] = new_str_value
            
            elif partial_replace and partial_pattern:
                # Partial replacement mode - replace specific characters at given indices
                for idx in valid_indices:
                    current_value = df.loc[idx, column_name]
                    if pd.notna(current_value):
                        # Convert to string for character manipulation
                        str_value = str(current_value)
                        
                        if partial_type == 'charByChar':
                            # Character by character replacement
                            # partial_pattern contains indices of characters to replace
                            # new_value contains the replacement string (char by char)
                            char_list = list(str_value)
                            replacement_chars = list(new_value) if new_value else []
                            
                            # Replace characters at specified indices
                            for i, char_idx in enumerate(partial_pattern):
                                if isinstance(char_idx, int) and 0 <= char_idx < len(char_list):
                                    # Use corresponding replacement character if available
                                    if i < len(replacement_chars):
                                        char_list[char_idx] = replacement_chars[i]
                                    else:
                                        # If not enough replacement chars, use empty string
                                        char_list[char_idx] = ''
                            
                            new_str_value = ''.join(char_list)
                        
                        else:  # partial_type == 'complete'
                            # Complete replacement at specific indices
                            # partial_pattern contains indices of characters to replace
                            # new_value is the complete replacement string
                            char_list = list(str_value)
                            
                            # Remove characters at specified indices (in reverse order to maintain indices)
                            for char_idx in sorted(partial_pattern, reverse=True):
                                if isinstance(char_idx, int) and 0 <= char_idx < len(char_list):
                                    char_list.pop(char_idx)
                            
                            # Insert the new value at the first index position
                            if partial_pattern and new_value:
                                insert_pos = min(partial_pattern)
                                # Insert each character of new_value starting at insert_pos
                                for i, char in enumerate(new_value):
                                    if insert_pos + i <= len(char_list):
                                        char_list.insert(insert_pos + i, char)
                            
                            new_str_value = ''.join(char_list)
                        
                        # For numeric columns, try to convert back to numeric
                        if is_numeric:
                            try:
                                # Try to convert to float first
                                numeric_value = float(new_str_value)
                                # If original column was integer and value has no decimal part, convert to int
                                if pd.api.types.is_integer_dtype(column_dtype) and numeric_value.is_integer():
                                    df.loc[idx, column_name] = int(numeric_value)
                                else:
                                    df.loc[idx, column_name] = numeric_value
                            except (ValueError, TypeError):
                                # If conversion fails, keep the string value
                                # This might cause issues with numeric columns, so we'll warn
                                print(f"Warning: Could not convert '{new_str_value}' to numeric for row {idx}")
                                df.loc[idx, column_name] = new_str_value
                        else:
                            # For non-numeric columns, just use the string value
                            df.loc[idx, column_name] = new_str_value
            
            else:
                # Direct value replacement
                # For numeric columns, validate the new value
                if is_numeric and new_value != '':
                    try:
                        # Try to convert new_value to numeric
                        numeric_value = float(new_value)
                        # If original column was integer and value has no decimal part, convert to int
                        if pd.api.types.is_integer_dtype(column_dtype) and numeric_value.is_integer():
                            processed_value = int(numeric_value)
                        else:
                            processed_value = numeric_value
                    except (ValueError, TypeError):
                        return error_response(f"Invalid numeric value: '{new_value}' for numeric column '{column_name}'")
                else:
                    # For non-numeric columns or empty values
                    processed_value = new_value if new_value != '' else np.nan
                
                # Apply the replacement
                df.loc[valid_indices, column_name] = processed_value
            
            # Verify no inadvertent NaN generation for critical operations
            if is_numeric and not char_replace:
                # Check if we introduced any NaN values where there weren't any before
                original_nan_count = df[column_name].iloc[valid_indices].isna().sum()
                new_nan_count = df.loc[valid_indices, column_name].isna().sum()
                if new_nan_count > original_nan_count and new_value != '':
                    print(f"Warning: Introduced {new_nan_count - original_nan_count} new NaN values")
            
            # Save the modified dataset
            from io import StringIO
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Get the original file name
            original_name = os.path.basename(dataset.file.name)
            
            # Invalidar caché
            old_path = dataset.file.path
            cache.delete(f'dataset_{old_path}')
            
            # Delete the old file
            dataset.file.delete(save=False)
            
            # Save the new content
            dataset.file.save(
                original_name,
                ContentFile(csv_content.encode('utf-8')),
                save=False
            )
            
            dataset.save()
            
            # Invalidar el caché del nuevo archivo
            cache.delete(f'dataset_{dataset.file.path}')
            
            # Prepare response message
            if char_replace:
                message = f"Successfully replaced '{char_to_find}' with '{char_to_replace}' in {len(valid_indices)} values"
            elif partial_replace:
                if partial_type == 'charByChar':
                    message = f"Successfully replaced characters at positions {partial_pattern} in {len(valid_indices)} values"
                else:
                    message = f"Successfully replaced content at positions {partial_pattern} with '{new_value}' in {len(valid_indices)} values"
            else:
                message = f"Successfully replaced {len(valid_indices)} values"
            
            # Determine the mode for response
            if char_replace:
                mode = 'char_replace'
            elif partial_replace:
                mode = f'partial_{partial_type}'
            else:
                mode = 'direct'
            
            return success_response({
                'message': message,
                'replaced_count': len(valid_indices),
                'mode': mode,
                'partial_info': {
                    'type': partial_type,
                    'pattern': partial_pattern
                } if partial_replace else None
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return error_response(f"Error replacing values: {str(e)}")


class DatasetTextManipulationPreviewView(APIView):
    """Preview text manipulation operations"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get parameters
        column_name = request.data.get('column_name')
        operation = request.data.get('operation')
        params = request.data.get('params', {})
        scope = request.data.get('scope', 'all')
        
        if not column_name or not operation:
            return error_response("Column name and operation are required")
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Check if column exists
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found in dataset")
        
        # Get sample data based on scope
        if scope == 'range':
            # Get range parameters
            range_start = params.get('range_start', 0)
            range_end = params.get('range_end', len(df) - 1)
            
            # Validate range
            try:
                range_start = int(range_start)
                range_end = int(range_end)
            except (ValueError, TypeError):
                range_start = 0
                range_end = min(19, len(df) - 1)  # Default to first 20 rows
            
            # Ensure range is valid
            if range_start < 0:
                range_start = 0
            if range_end >= len(df):
                range_end = len(df) - 1
            if range_start > range_end:
                range_start, range_end = 0, min(19, len(df) - 1)
            
            # Get up to 20 samples from the range
            sample_size = min(20, range_end - range_start + 1)
            sample_indices = list(range(range_start, min(range_start + sample_size, range_end + 1)))
            sample_data = df.loc[sample_indices, column_name].dropna()
        else:
            # Get sample data from entire column
            sample_data = df[column_name].dropna().head(20)
        
        preview = []
        
        try:
            for idx, value in sample_data.items():
                str_value = str(value)
                transformed = self._apply_text_operation(str_value, operation, params)
                preview.append({
                    'index': int(idx),
                    'original': str_value,
                    'transformed': transformed
                })
            
            return success_response({
                'preview': preview,
                'scope': scope,
                'range_info': {
                    'start': range_start,
                    'end': range_end
                } if scope == 'range' else None
            })
        except Exception as e:
            return error_response(f"Error generating preview: {str(e)}")
    
    def _apply_text_operation(self, value, operation, params):
        """Apply text operation to a single value"""
        if operation == 'truncate':
            length = params.get('length', 50)
            ellipsis = params.get('ellipsis', False)
            if len(value) > length:
                return value[:length] + ('...' if ellipsis else '')
            return value
            
        elif operation == 'extract':
            start = params.get('start', 0)
            length = params.get('length', 10)
            return value[start:start+length]
            
        elif operation == 'remove_chars':
            chars = params.get('chars', '')
            for char in chars:
                value = value.replace(char, '')
            return value
            
        elif operation == 'case':
            case_type = params.get('type', 'upper')
            if case_type == 'upper':
                return value.upper()
            elif case_type == 'lower':
                return value.lower()
            elif case_type == 'title':
                return value.title()
            elif case_type == 'capitalize':
                return value.capitalize()
            
        elif operation == 'trim':
            trim_type = params.get('type', 'both')
            if trim_type == 'both':
                return value.strip()
            elif trim_type == 'left':
                return value.lstrip()
            elif trim_type == 'right':
                return value.rstrip()
            
        elif operation == 'split':
            separator = params.get('separator', ',')
            part = params.get('part', 0)
            parts = value.split(separator)
            if part < 0:
                part = len(parts) + part
            if 0 <= part < len(parts):
                return parts[part].strip()
            return ''
            
        elif operation == 'pad':
            length = params.get('length', 10)
            char = params.get('char', '0')
            direction = params.get('direction', 'left')
            if direction == 'left':
                return value.rjust(length, char)
            else:
                return value.ljust(length, char)
        
        return value


class DatasetTextManipulationView(APIView):
    """Apply text manipulation operations to a dataset column"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get parameters
        column_name = request.data.get('column_name')
        operation = request.data.get('operation')
        params = request.data.get('params', {})
        scope = request.data.get('scope', 'all')  # Default to 'all' for backward compatibility
        
        if not column_name or not operation:
            return error_response("Column name and operation are required")
        
        # Limpiar caché antes de cargar el dataset
        old_path = dataset.file.path
        cache_key = f'dataset_{old_path}'
        cache.delete(cache_key)
        print(f"Cache cleared for key: {cache_key}")
        
        # Load dataset with no cache
        df = load_dataset(dataset.file.path, use_cache=False)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Check if column exists
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found in dataset")
        
        # Apply operation
        import tempfile
        temp_file = None
        
        try:
            preview_view = DatasetTextManipulationPreviewView()
            
            # Handle scope-based application
            if scope == 'range':
                # Get range parameters
                range_start = params.get('range_start', 0)
                range_end = params.get('range_end', len(df) - 1)
                
                # Validate range
                try:
                    range_start = int(range_start)
                    range_end = int(range_end)
                except (ValueError, TypeError):
                    return error_response("Invalid range values. range_start and range_end must be integers")
                
                # Ensure range is valid
                if range_start < 0:
                    range_start = 0
                if range_end >= len(df):
                    range_end = len(df) - 1
                if range_start > range_end:
                    return error_response("Invalid range: range_start must be less than or equal to range_end")
                
                # Apply operation only to the specified range
                # Create a copy of the column to modify
                modified_column = df[column_name].copy()
                
                # Apply the operation only to rows in the range (inclusive)
                for idx in range(range_start, range_end + 1):
                    if idx < len(df):
                        current_value = df.loc[idx, column_name]
                        modified_column.loc[idx] = preview_view._apply_text_operation(
                            str(current_value) if pd.notna(current_value) else '', 
                            operation, 
                            params
                        )
                
                # Replace the column with the modified version
                df[column_name] = modified_column
                
                rows_affected = range_end - range_start + 1
                message_suffix = f" to rows {range_start} to {range_end} ({rows_affected} rows)"
            else:
                # Apply to all rows (default behavior)
                df[column_name] = df[column_name].apply(
                    lambda x: preview_view._apply_text_operation(str(x) if pd.notna(x) else '', operation, params)
                )
                rows_affected = len(df)
                message_suffix = " to all rows"
            
            # Save the modified dataset to a temporary file first
            from io import StringIO
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            csv_buffer.close()  # Cerrar el buffer
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(csv_content)
                temp_file_path = temp_file.name
            
            # Get the original file name
            original_name = os.path.basename(dataset.file.name)
            
            # Invalidar todas las cachés relacionadas
            old_path = dataset.file.path
            cache.delete(f'dataset_{old_path}')
            cache.delete(f'dataset_preview_{pk}')
            cache.delete(f'dataset_columns_{pk}')
            
            # Ensure file handle is closed before deletion
            if hasattr(dataset.file, 'close'):
                dataset.file.close()
            
            # Small delay for Windows file system
            import time
            time.sleep(0.1)
            
            # Delete the old file with retry mechanism
            max_retries = 3
            for retry in range(max_retries):
                try:
                    dataset.file.delete(save=False)
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(0.2)
                        continue
                    else:
                        # If deletion fails, try to proceed anyway
                        print(f"Warning: Could not delete old file: {e}")
            
            # Read content from temporary file and save
            with open(temp_file_path, 'rb') as f:
                file_content = f.read()
            
            # Save the new content
            dataset.file.save(
                original_name,
                ContentFile(file_content),
                save=False
            )
            
            dataset.save()
            
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file: {e}")
            
            # Invalidar el caché del nuevo archivo
            new_cache_key = f'dataset_{dataset.file.path}'
            cache.delete(new_cache_key)
            
            # Verificar que el archivo se guardó correctamente
            if not os.path.exists(dataset.file.path):
                return error_response("Error: File was not saved properly")
            
            # Verificar el contenido leyendo el archivo guardado
            try:
                df_verify = pd.read_csv(dataset.file.path)
                print(f"Verification successful - rows: {len(df_verify)}, columns: {list(df_verify.columns)}")
            except Exception as verify_error:
                return error_response(f"Error verifying saved file: {str(verify_error)}")
            
            return success_response({
                'message': f"Successfully applied {operation} to column '{column_name}'{message_suffix}",
                'file_path': dataset.file.path,
                'rows': len(df),
                'columns': list(df.columns),
                'rows_affected': rows_affected,
                'scope': scope
            })
            
        except Exception as e:
            # Clean up temporary file if it exists
            if temp_file and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            import traceback
            traceback.print_exc()
            return error_response(f"Error applying text manipulation: {str(e)}")


class DatasetNumericTransformPreviewView(APIView):
    """Preview numeric transformation operations"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get parameters
        column_name = request.data.get('column_name')
        operation = request.data.get('operation')
        params = request.data.get('params', {})
        scope = request.data.get('scope', 'all')
        
        if not column_name or not operation:
            return error_response("Column name and operation are required")
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Check if column exists and is numeric
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found in dataset")
        
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return error_response(f"Column '{column_name}' is not numeric")
        
        # Get sample data based on scope
        if scope == 'range':
            # Get range parameters
            range_start = params.get('range_start', 0)
            range_end = params.get('range_end', len(df) - 1)
            
            # Validate range
            try:
                range_start = int(range_start)
                range_end = int(range_end)
            except (ValueError, TypeError):
                range_start = 0
                range_end = min(19, len(df) - 1)  # Default to first 20 rows
            
            # Ensure range is valid
            if range_start < 0:
                range_start = 0
            if range_end >= len(df):
                range_end = len(df) - 1
            if range_start > range_end:
                range_start, range_end = 0, min(19, len(df) - 1)
            
            # Get up to 20 samples from the range
            sample_size = min(20, range_end - range_start + 1)
            sample_indices = list(range(range_start, min(range_start + sample_size, range_end + 1)))
            sample_data = df.loc[sample_indices, column_name].dropna()
        else:
            # Get sample data from entire column
            sample_data = df[column_name].dropna().head(20)
        
        preview = []
        
        try:
            # Apply transformation to sample
            transformed_sample = []
            for idx, value in sample_data.items():
                transformed = self._apply_numeric_operation(value, operation, params)
                preview.append({
                    'index': int(idx),
                    'original': safe_float(value),
                    'transformed': safe_float(transformed)
                })
                transformed_sample.append(transformed)
            
            # Calculate stats for preview
            if transformed_sample:
                # Filter out None values for statistics calculation
                valid_transformed = [v for v in transformed_sample if pd.notna(v)]
                if valid_transformed:
                    mean_val = np.mean(valid_transformed)
                    median_val = np.median(valid_transformed)
                    std_val = np.std(valid_transformed)
                    min_val = np.min(valid_transformed)
                    max_val = np.max(valid_transformed)
                    
                    stats = {
                        'mean': safe_float(mean_val),
                        'median': safe_float(median_val),
                        'std': safe_float(std_val),
                        'min': safe_float(min_val),
                        'max': safe_float(max_val)
                    }
                else:
                    stats = {
                        'mean': None,
                        'median': None,
                        'std': None,
                        'min': None,
                        'max': None
                    }
            else:
                stats = None
            
            return success_response({
                'preview': preview,
                'stats': stats,
                'scope': scope,
                'range_info': {
                    'start': range_start,
                    'end': range_end
                } if scope == 'range' else None
            })
        except Exception as e:
            return error_response(f"Error generating preview: {str(e)}")
    
    def _apply_numeric_operation(self, value, operation, params):
        """Apply numeric operation to a single value"""
        if pd.isna(value):
            return value
            
        if operation == 'round':
            decimals = params.get('decimals', 2)
            return round(value, decimals)
            
        elif operation == 'floor':
            return np.floor(value)
            
        elif operation == 'ceil':
            return np.ceil(value)
            
        elif operation == 'scale':
            factor = params.get('factor', 1)
            return value * factor
            
        elif operation == 'clip':
            min_val = params.get('min', -np.inf)
            max_val = params.get('max', np.inf)
            return np.clip(value, min_val, max_val)
            
        elif operation == 'arithmetic':
            op = params.get('operation', 'add')
            op_value = params.get('value', 0)
            if op == 'add':
                return value + op_value
            elif op == 'subtract':
                return value - op_value
            elif op == 'multiply':
                return value * op_value
            elif op == 'divide':
                return value / op_value if op_value != 0 else np.nan
            
        elif operation == 'log':
            base = params.get('base', 'natural')
            if value <= 0:
                return np.nan
            if base == 'natural':
                return np.log(value)
            elif base == '10':
                return np.log10(value)
            elif base == '2':
                return np.log2(value)
            
        elif operation == 'power':
            exponent = params.get('exponent', 2)
            return np.power(value, exponent)
        
        return value


class DatasetNumericTransformView(APIView):
    """Apply numeric transformation operations to a dataset column"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        # Verificar permisos
        if request.user.is_staff:
            dataset = get_object_or_404(Dataset, pk=pk)
        else:
            dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
        
        # Get parameters
        column_name = request.data.get('column_name')
        operation = request.data.get('operation')
        params = request.data.get('params', {})
        scope = request.data.get('scope', 'all')  # Default to 'all' for backward compatibility
        
        if not column_name or not operation:
            return error_response("Column name and operation are required")
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Check if column exists and is numeric
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found in dataset")
        
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return error_response(f"Column '{column_name}' is not numeric")
        
        # Apply operation
        try:
            preview_view = DatasetNumericTransformPreviewView()
            
            # Handle scope-based application
            if scope == 'range':
                # Get range parameters
                range_start = params.get('range_start', 0)
                range_end = params.get('range_end', len(df) - 1)
                
                # Validate range
                try:
                    range_start = int(range_start)
                    range_end = int(range_end)
                except (ValueError, TypeError):
                    return error_response("Invalid range values. range_start and range_end must be integers")
                
                # Ensure range is valid
                if range_start < 0:
                    range_start = 0
                if range_end >= len(df):
                    range_end = len(df) - 1
                if range_start > range_end:
                    return error_response("Invalid range: range_start must be less than or equal to range_end")
                
                # Apply operation only to the specified range
                # Create a copy of the column to modify
                modified_column = df[column_name].copy()
                
                # Apply the operation only to rows in the range (inclusive)
                for idx in range(range_start, range_end + 1):
                    if idx < len(df):
                        current_value = df.loc[idx, column_name]
                        modified_column.loc[idx] = preview_view._apply_numeric_operation(
                            current_value, 
                            operation, 
                            params
                        )
                
                # Replace the column with the modified version
                df[column_name] = modified_column
                
                rows_affected = range_end - range_start + 1
                message_suffix = f" to rows {range_start} to {range_end} ({rows_affected} rows)"
            else:
                # Apply to all rows (default behavior)
                df[column_name] = df[column_name].apply(
                    lambda x: preview_view._apply_numeric_operation(x, operation, params)
                )
                rows_affected = len(df)
                message_suffix = " to all rows"
            
            # Save the modified dataset
            from io import StringIO
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Get the original file name
            original_name = os.path.basename(dataset.file.name)
            
            # Invalidar caché
            old_path = dataset.file.path
            cache.delete(f'dataset_{old_path}')
            
            # Delete the old file
            dataset.file.delete(save=False)
            
            # Save the new content
            dataset.file.save(
                original_name,
                ContentFile(csv_content.encode('utf-8')),
                save=False
            )
            
            dataset.save()
            
            # Invalidar el caché del nuevo archivo
            cache.delete(f'dataset_{dataset.file.path}')
            
            return success_response({
                'message': f"Successfully applied {operation} to column '{column_name}'{message_suffix}",
                'rows_affected': rows_affected,
                'scope': scope
            })
        except Exception as e:
            return error_response(f"Error applying numeric transformation: {str(e)}")