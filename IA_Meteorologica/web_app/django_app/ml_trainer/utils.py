"""
Common utility functions for ML Trainer module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from django.core.cache import cache
from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response
import logging

from .constants import (
    DATE_FORMATS, MIN_SAMPLE_SIZE, MAX_CATEGORICAL_VALUES,
    CACHE_TIMEOUT, ERROR_PARSING_FAILED, OUTLIER_THRESHOLD
)
from .utils_helpers import safe_to_list, safe_float, safe_dict_values

logger = logging.getLogger(__name__)


def load_dataset(file_path: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Load dataset from file with caching support
    
    Args:
        file_path: Path to the CSV file
        use_cache: Whether to use cache
        
    Returns:
        DataFrame or None if loading fails
    """
    cache_key = f'dataset_{file_path}'
    
    if use_cache:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data
    
    try:
        df = pd.read_csv(file_path)
        if use_cache:
            cache.set(cache_key, df, CACHE_TIMEOUT)
        return df
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {str(e)}")
        return None


def detect_column_type(series: pd.Series) -> Dict[str, Any]:
    """
    Detect the type and characteristics of a pandas Series
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Dictionary with type information and statistics
    """
    null_count = int(series.isnull().sum())
    total_count = len(series)
    
    result = {
        'dtype': str(series.dtype),
        'null_count': null_count,
        'null_percentage': (null_count / total_count * 100) if total_count > 0 else 0,
        'unique_count': int(series.nunique()),
        'sample_values': safe_to_list(series.dropna().head(5))
    }
    
    # Skip if too many nulls
    if series.notna().sum() < MIN_SAMPLE_SIZE:
        result['type'] = 'insufficient_data'
        return result
    
    # Try to detect datetime
    if series.dtype == 'object':
        sample = series.dropna().head(MIN_SAMPLE_SIZE)
        for date_format in DATE_FORMATS:
            try:
                pd.to_datetime(sample, format=date_format, errors='coerce')
                if pd.to_datetime(sample, format=date_format, errors='coerce').notna().sum() / len(sample) > 0.8:
                    result['type'] = 'datetime'
                    result['date_format'] = date_format
                    return result
            except:
                continue
        
        # Check if it's numeric stored as string
        try:
            numeric_parsed = pd.to_numeric(sample, errors='coerce')
            if numeric_parsed.notna().sum() / len(sample) > 0.8:
                result['type'] = 'numeric_string'
                return result
        except:
            pass
        
        # It's categorical
        if series.nunique() < MAX_CATEGORICAL_VALUES:
            result['type'] = 'categorical'
            value_counts = series.value_counts().head(10)
            result['categories'] = value_counts.to_dict()
            result['top_values'] = {
                'values': safe_to_list(value_counts.index),
                'counts': safe_to_list(value_counts.values)
            }
        else:
            result['type'] = 'text'
            # Even for text, provide top values if possible
            value_counts = series.value_counts().head(10)
            if len(value_counts) > 0:
                result['top_values'] = {
                    'values': safe_to_list(value_counts.index),
                    'counts': safe_to_list(value_counts.values)
                }
    
    elif pd.api.types.is_numeric_dtype(series):
        result['type'] = 'numeric'
        
        # Calculate statistics, handling NaN values
        mean_val = series.mean()
        result['mean'] = float(mean_val) if pd.notna(mean_val) else None
        
        std_val = series.std()
        result['std'] = float(std_val) if pd.notna(std_val) else None
        
        min_val = series.min()
        result['min'] = float(min_val) if pd.notna(min_val) else None
        
        max_val = series.max()
        result['max'] = float(max_val) if pd.notna(max_val) else None
        
        q25_val = series.quantile(0.25)
        result['q25'] = float(q25_val) if pd.notna(q25_val) else None
        
        q50_val = series.quantile(0.50)
        result['q50'] = float(q50_val) if pd.notna(q50_val) else None
        
        q75_val = series.quantile(0.75)
        result['q75'] = float(q75_val) if pd.notna(q75_val) else None
        
        # Generate histogram data
        valid_data = series.dropna()
        if len(valid_data) > 0:
            hist, bins = np.histogram(valid_data, bins=20)
            result['histogram'] = {
                'bins': safe_to_list(bins),
                'counts': safe_to_list(hist)
            }
            
            # Detect outliers
            Q1 = result['q25']
            Q3 = result['q75']
            
            # Only calculate outliers if we have valid quartile values
            if Q1 is not None and Q3 is not None:
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
                outlier_count = len(outliers)
                
                result['histogram']['outliers_info'] = {
                    'outlier_count': outlier_count,
                    'outlier_percentage': (outlier_count / len(valid_data)) * 100,
                    'lower_bound': float(lower_bound) if pd.notna(lower_bound) else None,
                    'upper_bound': float(upper_bound) if pd.notna(upper_bound) else None,
                    'q1': float(Q1) if pd.notna(Q1) else None,
                    'q3': float(Q3) if pd.notna(Q3) else None,
                    'iqr': float(IQR) if pd.notna(IQR) else None
                }
            else:
                result['histogram']['outliers_info'] = {
                    'outlier_count': 0,
                    'outlier_percentage': 0,
                    'lower_bound': None,
                    'upper_bound': None,
                    'q1': None,
                    'q3': None,
                    'iqr': None
                }
            
            # Generate histogram without outliers if there are any
            if Q1 is not None and Q3 is not None and outlier_count > 0:
                data_no_outliers = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)]
                if len(data_no_outliers) > 0:
                    hist_no_outliers, bins_no_outliers = np.histogram(data_no_outliers, bins=20)
                    result['histogram']['bins_no_outliers'] = safe_to_list(bins_no_outliers)
                    result['histogram']['counts_no_outliers'] = safe_to_list(hist_no_outliers)
        
        result['outlier_count'] = len(detect_outliers(series))
        
    return result


def detect_outliers(series: pd.Series, threshold: float = OUTLIER_THRESHOLD) -> List[int]:
    """
    Detect outliers using IQR method
    
    Args:
        series: Numeric pandas Series
        threshold: Number of IQRs to consider as outlier
        
    Returns:
        List of outlier indices
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers.index.tolist()


def calculate_correlation_matrix(df: pd.DataFrame, 
                                numeric_only: bool = True) -> pd.DataFrame:
    """
    Calculate correlation matrix for DataFrame
    
    Args:
        df: Input DataFrame
        numeric_only: Whether to include only numeric columns
        
    Returns:
        Correlation matrix
    """
    if numeric_only:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].corr()
    return df.corr()


def error_response(message: str, 
                  status_code: int = status.HTTP_400_BAD_REQUEST) -> Response:
    """
    Create standardized error response
    
    Args:
        message: Error message
        status_code: HTTP status code
        
    Returns:
        Response object
    """
    logger.error(f"API Error: {message}")
    return Response({'error': message}, status=status_code)


def success_response(data: Dict[str, Any] = None, 
                    message: str = None,
                    status_code: int = status.HTTP_200_OK) -> Response:
    """
    Create standardized success response
    
    Args:
        data: Response data
        message: Success message
        status_code: HTTP status code
        
    Returns:
        Response object
    """
    response_data = {}
    if message:
        response_data['message'] = message
    if data:
        response_data.update(data)
    
    return Response(response_data, status=status_code)


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate DataFrame for basic requirements
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "Dataset is empty"
    
    if len(df) < 10:
        return False, "Dataset has too few rows (minimum 10)"
    
    if df.columns.empty:
        return False, "Dataset has no columns"
    
    # Check for duplicate column names
    if df.columns.duplicated().any():
        return False, "Dataset has duplicate column names"
    
    # Check if all columns are null
    if df.isnull().all().all():
        return False, "All values in dataset are null"
    
    return True, None


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_json_response(data: Any) -> JsonResponse:
    """
    Create JSON response handling NaN and Inf values
    
    Args:
        data: Data to serialize
        
    Returns:
        JsonResponse object
    """
    # Convert NaN and Inf to None for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        return obj
    
    cleaned_data = clean_for_json(data)
    return JsonResponse(cleaned_data, safe=False)


def paginate_dataframe(df: pd.DataFrame, 
                       page: int = 1, 
                       page_size: int = 100) -> Dict[str, Any]:
    """
    Paginate DataFrame results
    
    Args:
        df: DataFrame to paginate
        page: Page number (1-indexed)
        page_size: Number of rows per page
        
    Returns:
        Dictionary with paginated data and metadata
    """
    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size
    
    # Validate page number
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    return {
        'data': df.iloc[start_idx:end_idx].to_dict('records'),
        'pagination': {
            'page': page,
            'page_size': page_size,
            'total_rows': total_rows,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_previous': page > 1
        }
    }


def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get memory usage statistics for DataFrame
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with memory usage information
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    return {
        'total_bytes': int(total_memory),
        'total_mb': round(total_memory / (1024 * 1024), 2),
        'per_column': {
            col: int(memory_usage[col]) 
            for col in df.columns
        }
    }


def validate_columns(df: pd.DataFrame, 
                    required_columns: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate that DataFrame has required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    return True, None