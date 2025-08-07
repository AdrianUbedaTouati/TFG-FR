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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
                'description': dataset.short_description,
                'long_description': dataset.long_description,
                'uploaded_at': dataset.uploaded_at,
                'shape': df.shape,
                'memory_usage': get_memory_usage(df)
            },
            'basic_statistics': self._get_basic_statistics(df),
            'column_analysis': self._analyze_columns_enhanced(df),
            'missing_data': self._analyze_missing_data(df),
            'correlations': self._analyze_correlations(df),
            'visualizations': self._generate_visualizations(df),
            'variable_data': self._generate_variable_data(df),
            'data_preview': self._generate_data_preview(df)
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
    
    def _analyze_columns_enhanced(self, df):
        """Enhanced column analysis with statistics and outliers"""
        columns_analysis = []
        
        for col in df.columns:
            col_analysis = detect_column_type(df[col])
            col_analysis['name'] = col
            col_analysis['null_count'] = int(df[col].isnull().sum())
            col_analysis['null_percentage'] = float((df[col].isnull().sum() / len(df)) * 100)
            
            if col_analysis['type'] == 'numeric':
                # Calculate statistics
                col_analysis['stats'] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q1': float(df[col].quantile(0.25)),
                    'median': float(df[col].quantile(0.5)),
                    'q3': float(df[col].quantile(0.75))
                }
                
                # Calculate outliers
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                col_analysis['outliers_count'] = int(len(outliers))
                col_analysis['outliers_percentage'] = float((len(outliers) / len(df)) * 100)
            else:
                col_analysis['outliers_count'] = 0
                col_analysis['outliers_percentage'] = 0.0
            
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
        
        # Find strong correlations (|r| > 0.5)
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        # Sort by absolute correlation value
        strong_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr,
            'strong_correlations': strong_corr
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
        
        # PCA analysis
        pca_result = self._create_pca_analysis(df)
        if pca_result:
            visualizations['pca_analysis'] = pca_result['plot']
            visualizations['pca_statistics'] = pca_result.get('statistics', {})
        
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
        plt.style.use('dark_background')
        
        fig, ax = plt.subplots(figsize=HEATMAP_FIGURE_SIZE)
        fig.patch.set_facecolor('#0a0e27')
        ax.set_facecolor('#0a0e27')
        
        # Create custom colormap from blue to red
        colors = ['#0099ff', '#00d4ff', '#ffffff', '#ff00ff', '#ff0066']
        n_bins = 100
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap,
                   center=0, ax=ax, cbar_kws={'label': 'Corrélation'},
                   linewidths=0.5, linecolor='#0a0e27',
                   annot_kws={'size': 9, 'color': '#0a0e27', 'weight': 'bold'})
        
        ax.set_title('Matrice de Corrélation', color='#00d4ff', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', color='#f0f9ff')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color='#f0f9ff')
        
        # Style colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('#00d4ff')
        cbar.ax.tick_params(colors='#f0f9ff')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_missing_data_chart(self, missing):
        """Create missing data visualization"""
        missing = missing[missing > 0].sort_values(ascending=False)
        
        # Set dark style for consistency with report
        plt.style.use('dark_background')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0a0e27')
        ax.set_facecolor('#0a0e27')
        
        # Create bar plot with neon blue color
        bars = ax.barh(range(len(missing)), missing.values, color='#00d4ff', edgecolor='#0099ff', linewidth=1)
        
        # Set y-axis labels with column names
        ax.set_yticks(range(len(missing)))
        ax.set_yticklabels(missing.index, color='#f0f9ff')
        
        # Add value labels on bars
        for i, (col, value) in enumerate(missing.items()):
            ax.text(value + 10, i, str(int(value)), 
                   va='center', color='#f0f9ff', fontweight='bold')
        
        ax.set_xlabel('Nombre de valeurs manquantes', color='#00d4ff', fontsize=12)
        ax.set_title('Données Manquantes par Colonne', color='#00d4ff', fontsize=16, fontweight='bold', pad=20)
        
        # Style grid
        ax.grid(True, alpha=0.2, color='#00d4ff', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#00d4ff')
        ax.spines['left'].set_color('#00d4ff')
        
        # Set tick colors
        ax.tick_params(colors='#f0f9ff')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_distribution_plot(self, series):
        """Create distribution plot for a series"""
        plt.style.use('dark_background')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#0a0e27')
        ax1.set_facecolor('#0a0e27')
        ax2.set_facecolor('#0a0e27')
        
        # Histogram
        n, bins, patches = ax1.hist(series.dropna(), bins=HISTOGRAM_BINS, 
                                   color='#00d4ff', edgecolor='#0099ff', 
                                   alpha=0.8, linewidth=1)
        
        # Add gradient effect to bars
        for i, patch in enumerate(patches):
            patch.set_alpha(0.6 + 0.4 * (i / len(patches)))
        
        ax1.set_xlabel(series.name, color='#00d4ff', fontsize=12)
        ax1.set_ylabel('Fréquence', color='#00d4ff', fontsize=12)
        ax1.set_title(f'Histogramme - {series.name}', color='#00d4ff', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.2, color='#00d4ff', linestyle='--')
        ax1.tick_params(colors='#f0f9ff')
        
        # Box plot
        bp = ax2.boxplot(series.dropna(), patch_artist=True,
                        boxprops=dict(facecolor='#00d4ff', alpha=0.3, linewidth=2),
                        whiskerprops=dict(color='#00d4ff', linewidth=2),
                        capprops=dict(color='#00d4ff', linewidth=2),
                        medianprops=dict(color='#ff00ff', linewidth=3),
                        flierprops=dict(marker='o', markerfacecolor='#ff00ff', 
                                      markersize=8, alpha=0.7, markeredgecolor='#ff00ff'))
        
        ax2.set_ylabel(series.name, color='#00d4ff', fontsize=12)
        ax2.set_title(f'Boîte à moustaches - {series.name}', color='#00d4ff', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.2, color='#00d4ff', linestyle='--', axis='y')
        ax2.tick_params(colors='#f0f9ff')
        
        # Style spines
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#00d4ff')
            ax.spines['left'].set_color('#00d4ff')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_pca_analysis(self, df):
        """Create PCA analysis visualization"""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return None
        
        try:
            plt.style.use('dark_background')
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df.dropna())
            
            # Apply PCA
            pca = PCA()
            pca.fit(scaled_data)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.patch.set_facecolor('#0a0e27')
            ax1.set_facecolor('#0a0e27')
            ax2.set_facecolor('#0a0e27')
            
            # Scree plot
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            components = range(1, min(len(explained_variance_ratio) + 1, 11))  # Limit to 10 components
            
            bars = ax1.bar(components, explained_variance_ratio[:10], 
                          color='#00d4ff', edgecolor='#0099ff', 
                          alpha=0.8, linewidth=1, label='Individuelle')
            
            # Add gradient effect
            for i, bar in enumerate(bars):
                bar.set_alpha(0.4 + 0.6 * (1 - i / len(bars)))
            
            ax1.plot(components, cumulative_variance_ratio[:10], 
                    color='#ff00ff', marker='o', linewidth=3, 
                    markersize=8, label='Cumulative', markeredgecolor='#ff00ff')
            
            ax1.set_xlabel('Composantes Principales', color='#00d4ff', fontsize=12)
            ax1.set_ylabel('Ratio de Variance Expliquée', color='#00d4ff', fontsize=12)
            ax1.set_title('Analyse PCA - Variance Expliquée', color='#00d4ff', fontsize=14, fontweight='bold')
            ax1.legend(loc='center right', framealpha=0.9, facecolor='#0a0e27', edgecolor='#00d4ff')
            ax1.grid(True, alpha=0.2, color='#00d4ff', linestyle='--')
            ax1.tick_params(colors='#f0f9ff')
            
            # 2D projection on first two components
            if scaled_data.shape[0] > 0:
                pca_2d = PCA(n_components=2)
                transformed = pca_2d.fit_transform(scaled_data)
                
                scatter = ax2.scatter(transformed[:, 0], transformed[:, 1], 
                                    c=range(len(transformed)), cmap='plasma',
                                    alpha=0.6, s=50, edgecolors='#00d4ff', linewidth=0.5)
                
                ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)', 
                             color='#00d4ff', fontsize=12)
                ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)', 
                             color='#00d4ff', fontsize=12)
                ax2.set_title('Projection PCA 2D', color='#00d4ff', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.2, color='#00d4ff', linestyle='--')
                ax2.tick_params(colors='#f0f9ff')
            
            # Style spines
            for ax in [ax1, ax2]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#00d4ff')
                ax.spines['left'].set_color('#00d4ff')
            
            plt.tight_layout()
            plot_base64 = self._fig_to_base64(fig)
            
            # Calculate PCA statistics
            cumulative_variance = np.cumsum(explained_variance_ratio)
            n_components_95 = int(np.argmax(cumulative_variance >= 0.95) + 1)
            
            # Get component contributions
            components_info = []
            for i in range(min(3, len(pca.components_))):  # Top 3 components
                # Get top 5 contributing variables for this component
                loadings = pca.components_[i]
                contributions = [(numeric_df.columns[j], float(loadings[j])) 
                               for j in range(len(loadings))]
                contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                components_info.append({
                    'component': f'PC{i+1}',
                    'variance_explained': float(explained_variance_ratio[i]),
                    'top_contributors': contributions[:5]
                })
            
            statistics = {
                'total_variables': len(numeric_df.columns),
                'n_components_95_variance': n_components_95,
                'explained_variance_ratio': [float(x) for x in explained_variance_ratio[:10]],
                'cumulative_variance': [float(x) for x in cumulative_variance[:10]],
                'components_info': components_info
            }
            
            return {
                'plot': plot_base64,
                'statistics': statistics
            }
        except Exception as e:
            print(f"Error in PCA analysis: {e}")
            return None
    
    def _generate_variable_data(self, df):
        """Generate detailed data for each variable for JavaScript"""
        variable_data = {}
        
        for col in df.columns:
            col_data = {
                'name': col,
                'type': detect_column_type(df[col])['type'],
                'unique_count': int(df[col].nunique()),
                'total_count': int(len(df)),
                'null_count': int(df[col].isnull().sum())
            }
            
            # Get value counts (limit to top 100)
            value_counts = df[col].value_counts().head(100)
            col_data['value_counts'] = {str(k): int(v) for k, v in value_counts.to_dict().items()}
            
            # For numeric columns, add statistics and outliers
            if col_data['type'] == 'numeric':
                # Statistics
                col_data['stats'] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q1': float(df[col].quantile(0.25)),
                    'median': float(df[col].quantile(0.5)),
                    'q3': float(df[col].quantile(0.75))
                }
                
                # Outliers
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_df = df[outliers_mask]
                
                # Calculate below and above counts
                below_mask = df[col] < lower_bound
                above_mask = df[col] > upper_bound
                below_count = int(below_mask.sum())
                above_count = int(above_mask.sum())
                
                col_data['outliers'] = {
                    'count': int(outliers_mask.sum()),
                    'percentage': float((outliers_mask.sum() / len(df)) * 100),
                    'iqr': float(iqr),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'below_count': below_count,
                    'above_count': above_count,
                    'values': [
                        {'index': int(idx), 'value': float(val)}
                        for idx, val in outliers_df[col].head(50).items()
                    ]
                }
            
            variable_data[col] = col_data
        
        return variable_data
    
    def _generate_data_preview(self, df, rows=10):
        """Generate data preview for first N rows"""
        preview_df = df.head(rows)
        
        return {
            'columns': list(preview_df.columns),
            'rows': [
                [val if pd.notna(val) else None for val in row]
                for row in preview_df.values
            ]
        }
    
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
            # Convert variable_data to JSON for JavaScript
            context = report_data.copy()
            context['variable_data'] = json.dumps(report_data.get('variable_data', {}))
            
            html = render_to_string('ml_trainer/dataset_report.html', context)
            return html
        except Exception as e:
            print(f"Error rendering template: {e}")
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