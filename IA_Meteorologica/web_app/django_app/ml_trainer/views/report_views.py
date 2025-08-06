"""
Report generation views
"""
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from django.http import HttpResponse
from django.template.loader import render_to_string
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json

from ..models import Dataset
from ..utils import (
    load_dataset, detect_column_type, calculate_correlation_matrix,
    error_response, validate_dataframe, get_memory_usage
)
from ..constants import (
    ERROR_PARSING_FAILED, HISTOGRAM_BINS,
    DEFAULT_FIGURE_SIZE, HEATMAP_FIGURE_SIZE
)


class DatasetReportView(APIView):
    """Generate comprehensive HTML report for dataset"""
    
    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Validate dataset
        is_valid, error_msg = validate_dataframe(df)
        if not is_valid:
            return error_response(error_msg)
        
        # Generate report data
        report_data = self._generate_report_data(dataset, df)
        
        # Generate HTML report
        html_content = self._generate_html_report(report_data)
        
        # Return as HTML response
        response = HttpResponse(html_content, content_type='text/html')
        response['Content-Disposition'] = f'inline; filename="{dataset.name}_report.html"'
        
        return response
    
    def post(self, request, pk):
        """Generate report with custom chart data from frontend"""
        dataset = get_object_or_404(Dataset, pk=pk)
        
        # Load dataset
        df = load_dataset(dataset.file.path)
        if df is None:
            return error_response(ERROR_PARSING_FAILED)
        
        # Validate dataset
        is_valid, error_msg = validate_dataframe(df)
        if not is_valid:
            return error_response(error_msg)
        
        # Get chart data from request
        try:
            request_data = json.loads(request.body)
            charts_data = request_data.get('charts', {})
        except:
            charts_data = {}
        
        # Generate report data
        report_data = self._generate_report_data(dataset, df)
        
        # Add custom charts from frontend
        if charts_data:
            report_data['custom_charts'] = charts_data
        
        # Generate HTML report
        html_content = self._generate_html_report(report_data)
        
        # Return as HTML response
        response = HttpResponse(html_content, content_type='text/html')
        response['Content-Disposition'] = f'attachment; filename="{dataset.name}_report.html"'
        
        return response
    
    def _generate_report_data(self, dataset, df):
        """Generate all data needed for the report"""
        return {
            'dataset_info': {
                'name': dataset.name,
                'description': dataset.long_description or dataset.short_description,
                'uploaded_at': dataset.uploaded_at,
                'shape': df.shape,
                'memory_usage': get_memory_usage(df)
            },
            'basic_statistics': self._get_basic_statistics(df),
            'column_analysis': self._analyze_columns(df),
            'missing_data': self._analyze_missing_data(df),
            'correlations': self._analyze_correlations(df),
            'visualizations': self._generate_visualizations(df)
        }
    
    def _get_basic_statistics(self, df):
        """Get basic statistical summary"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {}
        
        stats = numeric_df.describe().to_dict()
        
        # Add additional statistics
        for col in numeric_df.columns:
            stats[col]['skewness'] = float(numeric_df[col].skew())
            stats[col]['kurtosis'] = float(numeric_df[col].kurtosis())
        
        return stats
    
    def _analyze_columns(self, df):
        """Analyze each column in detail"""
        columns_analysis = []
        
        for col in df.columns:
            col_analysis = detect_column_type(df[col])
            col_analysis['name'] = col
            columns_analysis.append(col_analysis)
        
        return columns_analysis
    
    def _analyze_missing_data(self, df):
        """Analyze missing data patterns"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        return {
            'total_missing': int(missing.sum()),
            'by_column': {
                col: {
                    'count': int(missing[col]),
                    'percentage': float(missing_pct[col])
                }
                for col in df.columns if missing[col] > 0
            }
        }
    
    def _analyze_correlations(self, df):
        """Analyze correlations between numeric variables"""
        corr_matrix = calculate_correlation_matrix(df)
        
        if corr_matrix.empty:
            return {}
        
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
            'matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr
        }
    
    def _generate_visualizations(self, df):
        """Generate visualizations for the report"""
        visualizations = {}
        
        # Correlation heatmap
        corr_matrix = calculate_correlation_matrix(df)
        if not corr_matrix.empty:
            visualizations['correlation_heatmap'] = self._create_heatmap(corr_matrix)
        
        # Missing data chart
        missing = df.isnull().sum()
        if missing.sum() > 0:
            visualizations['missing_data'] = self._create_missing_data_chart(missing)
        
        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6
        visualizations['distributions'] = []
        for col in numeric_cols:
            visualizations['distributions'].append({
                'column': col,
                'plot': self._create_distribution_plot(df[col])
            })
        
        return visualizations
    
    def _create_heatmap(self, corr_matrix):
        """Create correlation heatmap"""
        fig, ax = plt.subplots(figsize=HEATMAP_FIGURE_SIZE)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix')
        
        return self._fig_to_base64(fig)
    
    def _create_missing_data_chart(self, missing):
        """Create missing data visualization"""
        missing = missing[missing > 0].sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        missing.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Number of Missing Values')
        ax.set_title('Missing Data by Column')
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def _create_distribution_plot(self, series):
        """Create distribution plot for a series"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        ax1.hist(series.dropna(), bins=HISTOGRAM_BINS, edgecolor='black', alpha=0.7)
        ax1.set_xlabel(series.name)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Histogram of {series.name}')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(series.dropna())
        ax2.set_ylabel(series.name)
        ax2.set_title(f'Boxplot of {series.name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_html_report(self, report_data):
        """Generate HTML report from data"""
        # Use template if available, otherwise generate inline
        try:
            html = render_to_string('ml_trainer/dataset_report.html', report_data)
            return html
        except:
            # Fallback to inline HTML generation
            return self._generate_inline_html(report_data)
    
    def _generate_custom_charts_html(self, custom_charts):
        """Generate HTML for custom charts from frontend"""
        if not custom_charts:
            return ""
        
        html = "<h2>Analysis Charts</h2>"
        for chart_id, chart_data in custom_charts.items():
            variable = chart_data.get('variable', 'Unknown')
            chart_image = chart_data.get('chartImage', '')
            
            html += f'<div class="visualization">'
            html += f'<h3>Analysis: {variable}</h3>'
            if chart_image:
                html += f'<img src="{chart_image}" />'
            
            # Add outlier info if available
            outlier_info = chart_data.get('outlierInfo')
            if outlier_info:
                html += f'<p>Outliers detected: {outlier_info.get("outlier_count", 0)} '
                html += f'({outlier_info.get("outlier_percentage", 0):.2f}% of data)</p>'
            
            html += '</div>'
        
        return html
    
    def _generate_inline_html(self, data):
        """Generate HTML report inline (fallback)"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Report - {data['dataset_info']['name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .visualization {{ margin: 20px 0; text-align: center; }}
                .visualization img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Dataset Analysis Report</h1>
            <h2>Dataset: {data['dataset_info']['name']}</h2>
            <p>{data['dataset_info'].get('description', 'No description available')}</p>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{data['dataset_info']['shape'][0]}</div>
                    <div class="metric-label">Rows</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{data['dataset_info']['shape'][1]}</div>
                    <div class="metric-label">Columns</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{data['dataset_info']['memory_usage']['total_mb']} MB</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
            </div>
            
            <h2>Missing Data Analysis</h2>
            <p>Total missing values: {data['missing_data']['total_missing']}</p>
            {self._generate_missing_data_table(data['missing_data']['by_column'])}
            
            <h2>Visualizations</h2>
            {self._generate_visualizations_html(data['visualizations'])}
            
            {self._generate_custom_charts_html(data.get('custom_charts', {}))}
            
            <h2>Column Details</h2>
            {self._generate_columns_table(data['column_analysis'])}
            
            <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc;">
                <p style="color: #666;">Report generated on {data['dataset_info']['uploaded_at']}</p>
            </footer>
        </body>
        </html>
        """
        return html
    
    def _generate_missing_data_table(self, missing_data):
        """Generate HTML table for missing data"""
        if not missing_data:
            return "<p>No missing data found.</p>"
        
        html = "<table><tr><th>Column</th><th>Missing Count</th><th>Percentage</th></tr>"
        for col, info in missing_data.items():
            html += f"<tr><td>{col}</td><td>{info['count']}</td><td>{info['percentage']:.2f}%</td></tr>"
        html += "</table>"
        return html
    
    def _generate_visualizations_html(self, visualizations):
        """Generate HTML for visualizations"""
        html = ""
        
        if 'correlation_heatmap' in visualizations:
            html += f'<div class="visualization"><h3>Correlation Heatmap</h3>'
            html += f'<img src="{visualizations["correlation_heatmap"]}" /></div>'
        
        if 'missing_data' in visualizations:
            html += f'<div class="visualization"><h3>Missing Data</h3>'
            html += f'<img src="{visualizations["missing_data"]}" /></div>'
        
        if 'distributions' in visualizations:
            for dist in visualizations['distributions']:
                html += f'<div class="visualization"><h3>Distribution: {dist["column"]}</h3>'
                html += f'<img src="{dist["plot"]}" /></div>'
        
        return html
    
    def _generate_columns_table(self, columns):
        """Generate HTML table for column analysis"""
        html = "<table><tr><th>Column</th><th>Type</th><th>Non-Null</th><th>Unique</th></tr>"
        for col in columns:
            non_null = col.get('null_count', 0)
            unique = col.get('unique_count', 0)
            html += f"<tr><td>{col['name']}</td><td>{col.get('type', 'unknown')}</td>"
            html += f"<td>{non_null}</td><td>{unique}</td></tr>"
        html += "</table>"
        return html