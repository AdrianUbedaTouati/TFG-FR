"""
Vistas para análisis de datasets sin usar DRF para evitar problemas con NaN
"""
from django.http import JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .models import Dataset


class SafeJSONEncoder(json.JSONEncoder):
    """JSON Encoder que maneja valores NaN y tipos numpy"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self.default(obj.tolist())
        elif isinstance(obj, list):
            return [self.default(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.default(value) for key, value in obj.items()}
        elif pd.isna(obj):
            return None
        return super().default(obj)


@csrf_exempt
@require_http_methods(["GET"])
def dataset_analysis(request, pk):
    """Vista simple para análisis de dataset que evita DRF"""
    try:
        # Obtener dataset
        try:
            dataset = Dataset.objects.get(pk=pk)
        except Dataset.DoesNotExist:
            return JsonResponse({'error': 'Dataset not found'}, status=404)
        
        # Leer datos
        df = pd.read_csv(dataset.file.path)
        analysis_type = request.GET.get('type', 'correlation')
        
        if analysis_type == 'correlation':
            result = generate_correlation_analysis(df)
        elif analysis_type == 'pca':
            result = generate_pca_analysis(df)
        else:
            return JsonResponse({'error': f'Unknown analysis type: {analysis_type}'}, status=400)
        
        # Devolver respuesta usando el encoder seguro
        return JsonResponse(result, encoder=SafeJSONEncoder, safe=False)
        
    except Exception as e:
        import traceback
        print(f"Error in dataset_analysis: {e}")
        print(traceback.format_exc())
        return JsonResponse({
            'error': f'Error processing analysis: {str(e)}'
        }, status=500)


def generate_correlation_analysis(df):
    """Genera análisis de correlación"""
    # Seleccionar columnas numéricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        return {
            'error': 'At least 2 numeric columns are required for correlation matrix',
            'status': 'error'
        }
    
    # Calcular matriz de correlación
    corr_matrix = df[numeric_columns].corr()
    
    # Reemplazar NaN con 0
    corr_matrix = corr_matrix.fillna(0)
    
    # Crear visualización
    plt.style.use('default')  # Resetear estilo
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Crear máscara para triángulo superior
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Crear heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": .8}, ax=ax)
    
    ax.set_title('Matriz de Correlación', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Convertir a base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # Encontrar correlaciones fuertes
    strong_correlations = []
    for i in range(len(numeric_columns)):
        for j in range(i+1, len(numeric_columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.5:
                strong_correlations.append({
                    'var1': numeric_columns[i],
                    'var2': numeric_columns[j],
                    'correlation': float(corr_value)
                })
    
    strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return {
        'analysis_type': 'correlation_matrix',
        'image': f'data:image/png;base64,{image_base64}',
        'statistics': {
            'numeric_columns': numeric_columns,
            'column_count': len(numeric_columns),
            'strong_correlations': strong_correlations[:10]
        },
        'status': 'success'
    }


def generate_pca_analysis(df):
    """Genera análisis PCA"""
    # Seleccionar columnas numéricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 3:
        return {
            'error': 'At least 3 numeric columns are required for PCA',
            'status': 'error'
        }
    
    # Preparar datos
    X = df[numeric_columns].dropna()
    
    if len(X) < 10:
        return {
            'error': 'Not enough data points for PCA analysis',
            'status': 'error'
        }
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Crear visualizaciones
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Varianza explicada
    axes[0, 0].bar(range(1, min(len(pca.explained_variance_ratio_)+1, 11)), 
                   pca.explained_variance_ratio_[:10])
    axes[0, 0].set_xlabel('Componente Principal')
    axes[0, 0].set_ylabel('Varianza Explicada')
    axes[0, 0].set_title('Varianza Explicada por Componente')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Varianza acumulada
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    axes[0, 1].plot(range(1, min(len(cumsum)+1, 11)), cumsum[:10], 'bo-')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
    axes[0, 1].set_xlabel('Número de Componentes')
    axes[0, 1].set_ylabel('Varianza Acumulada')
    axes[0, 1].set_title('Varianza Acumulada')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Biplot PC1 vs PC2
    if X_pca.shape[1] >= 2:
        axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        axes[1, 0].set_title('PCA - Primeras 2 Componentes')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Loadings
    if len(pca.components_) >= 2:
        loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
        for i, (var, loading) in enumerate(zip(numeric_columns[:10], loadings[:10])):
            axes[1, 1].arrow(0, 0, loading[0], loading[1], 
                           head_width=0.05, head_length=0.05, alpha=0.7)
            axes[1, 1].text(loading[0]*1.1, loading[1]*1.1, var, fontsize=8)
        
        axes[1, 1].set_xlim(-1.2, 1.2)
        axes[1, 1].set_ylim(-1.2, 1.2)
        axes[1, 1].set_xlabel('PC1')
        axes[1, 1].set_ylabel('PC2')
        axes[1, 1].set_title('PCA Loadings')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # Convertir a base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # Componentes para 95% varianza
    n_components_95 = int(np.argmax(cumsum >= 0.95) + 1)
    
    return {
        'analysis_type': 'pca',
        'image': f'data:image/png;base64,{image_base64}',
        'statistics': {
            'total_variables': len(numeric_columns),
            'explained_variance_ratio': [float(x) for x in pca.explained_variance_ratio_[:10]],
            'cumulative_variance': [float(x) for x in cumsum[:10]],
            'n_components_95_variance': n_components_95,
            'principal_components': {
                f'PC{i+1}': {
                    'variance_explained': float(pca.explained_variance_ratio_[i]),
                    'top_contributors': [
                        (col, float(loading)) 
                        for col, loading in sorted(
                            zip(numeric_columns, pca.components_[i]),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )[:5]
                    ]
                }
                for i in range(min(3, len(pca.components_)))
            }
        },
        'status': 'success'
    }