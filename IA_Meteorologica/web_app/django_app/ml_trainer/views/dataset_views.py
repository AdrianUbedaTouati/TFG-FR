"""
Dataset-related views for ML Trainer
"""
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.http import HttpResponse
import pandas as pd
import numpy as np
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
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer


class DatasetDetailView(generics.RetrieveDestroyAPIView):
    """Retrieve or delete a specific dataset"""
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer


class DatasetColumnsView(APIView):
    """Get column information for a dataset"""
    
    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Get columns and dtypes - ensure they are strings
        columns = [str(col) for col in df.columns if col and str(col).strip()]
        dtypes = {str(col): str(df[col].dtype) for col in columns}
        
        # Simple response for column loading
        response_data = {
            'columns': columns,
            'dtypes': dtypes,
            'total_rows': len(df),
            'total_columns': len(columns)
        }
        
        # Add detailed analysis if requested
        if request.query_params.get('detailed', False):
            columns_info = []
            for col in columns:
                col_info = detect_column_type(df[col])
                col_info['name'] = col
                columns_info.append(col_info)
            
            response_data['detailed_info'] = columns_info
            response_data['memory_usage'] = get_memory_usage(df)
        
        return success_response(response_data)


class DatasetColumnDetailsView(APIView):
    """Get detailed information about a specific column"""
    
    def get(self, request, pk, column_name):
        dataset = get_object_or_404(Dataset, pk=pk)
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Validate column exists
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found")
        
        # Get column details
        series = df[column_name]
        details = detect_column_type(series)
        
        # Add histogram for numeric columns
        if details.get('type') == 'numeric':
            # Create a temporary base view instance to use its methods
            base_view = DatasetAnalysisBaseView()
            details['histogram'] = base_view.create_histogram(series)
        
        return success_response(details)
    


class DatasetDownloadView(APIView):
    """Download a dataset file"""
    
    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        
        try:
            with open(dataset.file.path, 'rb') as f:
                file_data = f.read()
            
            response = HttpResponse(
                file_data,
                content_type='text/csv'
            )
            response['Content-Disposition'] = f'attachment; filename="{dataset.name}.csv"'
            return response
            
        except Exception as e:
            return error_response(f"Error downloading file: {str(e)}")


class DatasetAnalysisBaseView(APIView):
    """Base class for dataset analysis views"""
    
    def load_and_validate_dataset(self, pk: int) -> Optional[pd.DataFrame]:
        """Load and validate dataset"""
        dataset = get_object_or_404(Dataset, pk=pk)
        df = load_dataset(dataset.file.path)
        
        if df is None:
            return None
            
        is_valid, error_msg = validate_dataframe(df)
        if not is_valid:
            return None
            
        return df
    
    def generate_plot_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def create_histogram(self, series: pd.Series, title: str = None, alpha: float = 0.7) -> str:
        """Create a histogram for a pandas Series"""
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        ax.hist(series.dropna(), bins=HISTOGRAM_BINS, edgecolor='black', alpha=alpha)
        ax.set_xlabel(series.name)
        ax.set_ylabel('Frequency')
        ax.set_title(title or f'Distribution of {series.name}')
        ax.grid(True, alpha=0.3)
        
        return self.generate_plot_base64(fig)


class DatasetVariableAnalysisView(DatasetAnalysisBaseView):
    """Analyze a specific variable in the dataset"""
    
    def get(self, request, pk, column_name):
        # Load dataset
        df = self.load_and_validate_dataset(pk)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Validate column
        if column_name not in df.columns:
            return error_response(f"Column '{column_name}' not found")
        
        # Generate analysis based on column type
        series = df[column_name]
        col_info = detect_column_type(series)
        
        analysis = {
            'column_name': column_name,
            'type_info': col_info,
            'visualizations': []
        }
        
        if col_info.get('type') == 'numeric':
            # Generate multiple visualizations for numeric data
            analysis['visualizations'].extend([
                self._create_histogram(series),
                self._create_boxplot(series),
                self._create_density_plot(series)
            ])
        elif col_info.get('type') == 'categorical':
            analysis['visualizations'].append(
                self._create_bar_chart(series)
            )
        
        return success_response(analysis)
    
    def _create_histogram(self, series: pd.Series) -> Dict[str, str]:
        """Create histogram visualization"""
        return {
            'type': 'histogram',
            'image': self.create_histogram(series, f'Histogram of {series.name}')
        }
    
    def _create_boxplot(self, series: pd.Series) -> Dict[str, str]:
        """Create boxplot visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(series.dropna())
        ax.set_ylabel(series.name)
        ax.set_title(f'Boxplot of {series.name}')
        ax.grid(True, alpha=0.3)
        
        return {
            'type': 'boxplot',
            'image': self.generate_plot_base64(fig)
        }
    
    def _create_density_plot(self, series: pd.Series) -> Dict[str, str]:
        """Create density plot visualization"""
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        series.dropna().plot(kind='density', ax=ax)
        ax.set_xlabel(series.name)
        ax.set_title(f'Density Plot of {series.name}')
        ax.grid(True, alpha=0.3)
        
        return {
            'type': 'density',
            'image': self.generate_plot_base64(fig)
        }
    
    def _create_bar_chart(self, series: pd.Series) -> Dict[str, str]:
        """Create bar chart for categorical data"""
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        value_counts = series.value_counts().head(20)
        value_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel(series.name)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {series.name}')
        plt.xticks(rotation=45, ha='right')
        
        return {
            'type': 'bar_chart',
            'image': self.generate_plot_base64(fig)
        }


class DatasetGeneralAnalysisView(DatasetAnalysisBaseView):
    """Generate general analysis for entire dataset"""
    
    def get(self, request, pk):
        # Load dataset
        df = self.load_and_validate_dataset(pk)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
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
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': get_memory_usage(df),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
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