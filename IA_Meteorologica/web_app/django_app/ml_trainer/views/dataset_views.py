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
            preview[col] = df[col].head(10).tolist()
        
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
        stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nCount: {len(data)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Convert plot to base64
        plot_base64 = self.generate_plot_base64(fig)
        
        return {
            'plot': plot_base64,
            'bins': bins.tolist(),
            'counts': counts.tolist(),
            'statistics': {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
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
        
        stats_text = f'Median: {data.median():.2f}\nQ1: {Q1:.2f}\nQ3: {Q3:.2f}\nIQR: {IQR:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plot_base64 = self.generate_plot_base64(fig)
        
        return {
            'plot': plot_base64,
            'statistics': {
                'median': float(data.median()),
                'q1': float(Q1),
                'q3': float(Q3),
                'iqr': float(IQR),
                'min': float(data.min()),
                'max': float(data.max()),
                'outlier_count': int(((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum())
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
                'outlier_percentage': float((len(outliers) / len(data)) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_values': outliers.head(20).tolist()  # First 20 outliers
            }
        }
    
    def generate_plot_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return plot_base64


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
                        'correlation': float(corr_val)
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