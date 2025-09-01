"""
evaluate_architectures.py - Sistema mejorado de comparación de arquitecturas

Compara dos enfoques de clasificación meteorológica:
  A) Modelo único: Predice directamente las 4 clases (Cloudy, Rainy, Snowy, Sunny)
  B) Arquitectura jerárquica: 
     - Modelo general clasifica en grupos (Cloudy_Sunny vs Rainy_Snowy)
     - Modelos especialistas refinan la predicción dentro de cada grupo

Mejoras sobre la versión anterior:
- Carga automática de modelos desde sus directorios originales
- Mejor manejo de mapeo de clases y enrutamiento
- Visualizaciones mejoradas con análisis detallado
- Métricas comparativas más completas
- Análisis de errores y patrones de confusión
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import platform
import warnings
warnings.filterwarnings('ignore')

# =============================
# CONFIGURACIÓN
# =============================

# Detectar sistema operativo
IS_WINDOWS = platform.system() == 'Windows'

@dataclass
class EvalConfig:
    """Configuración para la evaluación comparativa"""
    
    # Rutas base - compatibles con Windows y Unix
    if IS_WINDOWS:
        BASE_DIR: str = r"C:\Users\andri\Desktop\TFG_FR\IA_Meteorologica\Réseaux\Data_base_summaty_class\LSTM"
    else:
        BASE_DIR: str = "/mnt/c/Users/andri/Desktop/TFG_FR/IA_Meteorologica/Réseaux/Data_base_summaty_class/LSTM"
    
    # Datos - buscar en la carpeta de data local
    CSV_PATH: str = field(default_factory=lambda: 
        os.path.join(os.path.dirname(__file__), "data", "weather_classification_normalized.csv")
    )
    
    # Columnas de características
    # NOTA: Algunos modelos pueden haber sido entrenados sin "Cloud Cover_clear"
    FEATURE_COLS: List[str] = field(default_factory=lambda: [
        "Temperature_normalized",
        "Humidity_normalized", 
        "Wind Speed_normalized",
        "Precipitation (%)_normalized",
        "Cloud Cover_clear",  # Esta característica podría no estar en todos los modelos
        "Cloud Cover_cloudy",
        "Cloud Cover_overcast",
        "Cloud Cover_partly cloudy",
        "Atmospheric Pressure_normalized",
        "UV Index_normalized",
        "Season_Autumn",
        "Season_Spring",
        "Season_Summer",
        "Season_Winter",
        "Visibility (km)_normalized",
        "Location_coastal",
        "Location_inland",
        "Location_mountain",
    ])
    
    # Características alternativas (sin Cloud Cover_clear) para modelos de 17 features
    FEATURE_COLS_17: List[str] = field(default_factory=lambda: [
        "Temperature_normalized",
        "Humidity_normalized", 
        "Wind Speed_normalized",
        "Precipitation (%)_normalized",
        # "Cloud Cover_clear",  # Omitida en algunos modelos
        "Cloud Cover_cloudy",
        "Cloud Cover_overcast",
        "Cloud Cover_partly cloudy",
        "Atmospheric Pressure_normalized",
        "UV Index_normalized",
        "Season_Autumn",
        "Season_Spring",
        "Season_Summer",
        "Season_Winter",
        "Visibility (km)_normalized",
        "Location_coastal",
        "Location_inland",
        "Location_mountain",
    ])
    
    # Etiquetas
    LABEL_COL_RAW: str = "Weather Type"
    SUMMARY_ONEHOT_PREFIX: str = "Weather Type_"
    
    # Modelos - rutas a los originales en sus carpetas respectivas
    if IS_WINDOWS:
        SINGLE_MODEL_PATH: str = field(default_factory=lambda:
            os.path.join(r"C:\Users\andri\Desktop\TFG_FR\IA_Meteorologica\Réseaux\Data_base_summaty_class\LSTM", "summary_4", "outputs", "lstm_summary_line_cls", "checkpoints", "4_global.pt")
        )
        MODEL_GENERAL_PATH: str = field(default_factory=lambda:
            os.path.join(r"C:\Users\andri\Desktop\TFG_FR\IA_Meteorologica\Réseaux\Data_base_summaty_class\LSTM", "summary_2_general", "outputs", "lstm_summary_line_cls", "checkpoints", "2_general.pt")
        )
        MODEL_SPEC_A_PATH: str = field(default_factory=lambda:
            os.path.join(r"C:\Users\andri\Desktop\TFG_FR\IA_Meteorologica\Réseaux\Data_base_summaty_class\LSTM", "summary_Cloudy_Sunny", "outputs", "lstm_summary_line_cls", "checkpoints", "cloudy_sunny.pt")
        )
        MODEL_SPEC_B_PATH: str = field(default_factory=lambda:
            os.path.join(r"C:\Users\andri\Desktop\TFG_FR\IA_Meteorologica\Réseaux\Data_base_summaty_class\LSTM", "summary_Rainy_Snowy", "outputs", "lstm_summary_line_cls", "checkpoints", "rain_snowy.pt")
        )
    else:
        SINGLE_MODEL_PATH: str = field(default_factory=lambda:
            os.path.join("/mnt/c/Users/andri/Desktop/TFG_FR/IA_Meteorologica/Réseaux/Data_base_summaty_class/LSTM", "summary_4", "outputs", "lstm_summary_line_cls", "checkpoints", "4_global.pt")
        )
        MODEL_GENERAL_PATH: str = field(default_factory=lambda:
            os.path.join("/mnt/c/Users/andri/Desktop/TFG_FR/IA_Meteorologica/Réseaux/Data_base_summaty_class/LSTM", "summary_2_general", "outputs", "lstm_summary_line_cls", "checkpoints", "2_general.pt")
        )
        MODEL_SPEC_A_PATH: str = field(default_factory=lambda:
            os.path.join("/mnt/c/Users/andri/Desktop/TFG_FR/IA_Meteorologica/Réseaux/Data_base_summaty_class/LSTM", "summary_Cloudy_Sunny", "outputs", "lstm_summary_line_cls", "checkpoints", "cloudy_sunny.pt")
        )
        MODEL_SPEC_B_PATH: str = field(default_factory=lambda:
            os.path.join("/mnt/c/Users/andri/Desktop/TFG_FR/IA_Meteorologica/Réseaux/Data_base_summaty_class/LSTM", "summary_Rainy_Snowy", "outputs", "lstm_summary_line_cls", "checkpoints", "rain_snowy.pt")
        )
    
    # Mapeo de clases basado en los archivos class_index.json
    # 4_global: {0: Cloudy, 1: Rainy, 2: Snowy, 3: Sunny}
    # 2_general: {0: Cloudy_Sunny, 1: Rainy_Snowy}
    # cloudy_sunny: {0: Cloudy, 1: Sunny}
    # rain_snowy: {0: Rainy, 1: Snowy}
    
    # Enrutamiento: modelo general -> especialista
    ROUTE_BY_GENERAL: Dict[int, str] = field(default_factory=lambda: {
        0: "A",  # Cloudy_Sunny -> especialista A
        1: "B"   # Rainy_Snowy -> especialista B
    })
    
    # Mapeo final: (especialista, índice) -> clase global
    FINAL_CLASS_MAP: Dict[Tuple[str, int], int] = field(default_factory=lambda: {
        ("A", 0): 0,  # Cloudy (del especialista A) -> 0 (Cloudy global)
        ("A", 1): 3,  # Sunny (del especialista A) -> 3 (Sunny global)
        ("B", 0): 1,  # Rainy (del especialista B) -> 1 (Rainy global)
        ("B", 1): 2,  # Snowy (del especialista B) -> 2 (Snowy global)
    })
    
    # Salidas
    OUTPUT_DIR: str = "outputs/comparison_results"
    
    # Configuración de evaluación
    TEST_RATIO: float = 0.15
    VAL_RATIO: float = 0.15
    BATCH_SIZE: int = 2048
    DEVICE: Optional[str] = None  # None = auto-detectar

# =============================
# MODELO BASE LSTM
# =============================

class LSTMLineClassifier(nn.Module):
    """Modelo LSTM para clasificación"""
    def __init__(self, in_features: int, num_classes: int, hidden_size: int = 128,
                 num_layers: int = 1, dropout: float = 0.2, bidirectional: bool = False,
                 head_hidden: Optional[int] = None):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        proj_in = hidden_size * (2 if bidirectional else 1)
        if head_hidden and head_hidden > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(proj_in),
                nn.Linear(proj_in, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_classes)
            )
        else:
            self.head = nn.Linear(proj_in, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        logits = self.head(h_last)
        return logits

# =============================
# CARGA Y EVALUACIÓN DE MODELOS
# =============================

class ModelWrapper:
    """Wrapper para cargar y usar modelos .pt o .ts"""
    def __init__(self, path: str, device: torch.device):
        self.path = path
        self.device = device
        self.model = None
        self.metadata = {}
        self._load()
        
    def _load(self):
        """Carga el modelo y sus metadatos"""
        if self.path.endswith('.ts'):
            self.model = torch.jit.load(self.path, map_location=self.device).eval()
        else:
            checkpoint = torch.load(self.path, map_location=self.device)
            
            # Extraer metadatos
            self.metadata = {
                'in_features': checkpoint.get('in_features', 18),
                'num_classes': checkpoint.get('num_classes', 4),
                'class_names': checkpoint.get('class_names', []),
                'cfg': checkpoint.get('cfg', {})
            }
            
            # Cargar class_index.json si existe
            model_dir = os.path.dirname(os.path.dirname(os.path.dirname(self.path)))
            class_index_path = os.path.join(model_dir, 'outputs', 'lstm_summary_line_cls', 'class_index.json')
            if os.path.exists(class_index_path):
                with open(class_index_path, 'r') as f:
                    class_index = json.load(f)
                    self.metadata['class_names'] = [class_index[str(i)] for i in range(len(class_index))]
                    self.metadata['num_classes'] = len(class_index)
            
            # Crear modelo
            cfg = self.metadata['cfg']
            self.model = LSTMLineClassifier(
                in_features=self.metadata['in_features'],
                num_classes=self.metadata['num_classes'],
                hidden_size=cfg.get('LSTM_HIDDEN_SIZE', 128),
                num_layers=cfg.get('LSTM_NUM_LAYERS', 1),
                dropout=cfg.get('LSTM_DROPOUT', 0.2),
                bidirectional=cfg.get('LSTM_BIDIRECTIONAL', False),
                head_hidden=cfg.get('LSTM_HEAD_HIDDEN', None)
            )
            
            # Cargar pesos
            state_dict = checkpoint.get('model_state', checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval()
    
    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """Predice clases y probabilidades"""
        # Ajustar dimensiones si es necesario
        if hasattr(self.model, 'lstm') and hasattr(self.model.lstm, 'input_size'):
            expected_features = self.model.lstm.input_size
            if X.shape[1] != expected_features:
                # Solo mostrar el mensaje la primera vez
                if not hasattr(self, '_adjustment_shown'):
                    print(f"    [AJUSTE] Modelo espera {expected_features} features, recibió {X.shape[1]}")
                    self._adjustment_shown = True
                
                if X.shape[1] > expected_features:
                    # Si espera 17 y recibe 18, probablemente es el modelo sin "Cloud Cover_clear"
                    if expected_features == 17 and X.shape[1] == 18:
                        # Omitir la columna 4 (Cloud Cover_clear)
                        # Las primeras 4 columnas: Temperature, Humidity, Wind Speed, Precipitation
                        # Columna 4 sería Cloud Cover_clear
                        # Resto de columnas desde la 5 en adelante
                        X = np.concatenate([X[:, :4], X[:, 5:]], axis=1)
                        if not hasattr(self, '_feature_info_shown'):
                            print(f"    Omitiendo 'Cloud Cover_clear' (columna 5)")
                            self._feature_info_shown = True
                    else:
                        # Caso general: tomar solo las primeras características
                        X = X[:, :expected_features]
                else:
                    # Padding con ceros (esto no debería ocurrir normalmente)
                    padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                    X = np.concatenate([X, padding], axis=1)
        
        all_logits = []
        
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
            batch = batch.unsqueeze(1)  # Agregar dimensión temporal
            logits = self.model(batch)
            all_logits.append(logits.cpu().numpy())
        
        logits = np.concatenate(all_logits, axis=0)
        probs = torch.softmax(torch.FloatTensor(logits), dim=1).numpy()
        preds = np.argmax(logits, axis=1)
        
        return preds, probs

# =============================
# FUNCIONES DE EVALUACIÓN
# =============================

def evaluate_single_model(X_test: np.ndarray, y_test: np.ndarray, 
                         model_path: str, device: torch.device) -> Dict:
    """Evalúa el modelo único de 4 clases"""
    print("\n[EVALUANDO] Modelo único (4 clases directas)")
    
    model = ModelWrapper(model_path, device)
    print(f"  - Modelo cargado: {model.metadata.get('num_classes', 4)} clases, {model.metadata.get('in_features', 'desconocido')} features")
    
    preds, probs = model.predict(X_test)
    
    # Calcular métricas
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, preds, average=None)
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    cm = confusion_matrix(y_test, preds)
    
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Macro F1: {macro_f1:.4f}")
    print(f"  - Weighted F1: {weighted_f1:.4f}")
    
    return {
        'predictions': preds,
        'probabilities': probs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'class_names': model.metadata.get('class_names', ['Cloudy', 'Rainy', 'Snowy', 'Sunny'])
    }

def evaluate_hierarchical_model(X_test: np.ndarray, y_test: np.ndarray,
                               general_path: str, spec_a_path: str, spec_b_path: str,
                               route_map: Dict, final_map: Dict, device: torch.device) -> Dict:
    """Evalúa la arquitectura jerárquica de 3 modelos"""
    print("\n[EVALUANDO] Arquitectura jerárquica (3 modelos)")
    
    # Cargar modelos
    print("  Cargando modelos:")
    general_model = ModelWrapper(general_path, device)
    print(f"    - General: {general_model.metadata.get('num_classes', 2)} clases, {general_model.metadata.get('in_features', 'desconocido')} features")
    
    spec_a_model = ModelWrapper(spec_a_path, device)
    print(f"    - Especialista A: {spec_a_model.metadata.get('num_classes', 2)} clases, {spec_a_model.metadata.get('in_features', 'desconocido')} features")
    
    spec_b_model = ModelWrapper(spec_b_path, device)
    print(f"    - Especialista B: {spec_b_model.metadata.get('num_classes', 2)} clases, {spec_b_model.metadata.get('in_features', 'desconocido')} features")
    
    # Predicción del modelo general
    print("\n  Ejecutando predicciones...")
    general_preds, general_probs = general_model.predict(X_test)
    
    # Inicializar predicciones finales
    final_preds = np.zeros_like(y_test)
    final_probs = np.zeros((len(y_test), 4))  # 4 clases finales
    
    # Estadísticas de enrutamiento
    route_counts = {'A': 0, 'B': 0}
    
    # Para cada muestra, enrutar al especialista correspondiente
    for i in range(len(X_test)):
        general_pred = general_preds[i]
        route = route_map.get(general_pred, 'A')
        route_counts[route] += 1
        
        if route == 'A':
            # Usar especialista A (Cloudy/Sunny)
            spec_pred, spec_prob = spec_a_model.predict(X_test[i:i+1])
            final_pred = final_map.get(('A', spec_pred[0]), 0)
            
            # Mapear probabilidades a las 4 clases finales
            for j, (k, v) in enumerate(final_map.items()):
                if k[0] == 'A':
                    final_probs[i, v] = spec_prob[0, k[1]]
                    
        else:
            # Usar especialista B (Rainy/Snowy)
            spec_pred, spec_prob = spec_b_model.predict(X_test[i:i+1])
            final_pred = final_map.get(('B', spec_pred[0]), 0)
            
            # Mapear probabilidades a las 4 clases finales
            for j, (k, v) in enumerate(final_map.items()):
                if k[0] == 'B':
                    final_probs[i, v] = spec_prob[0, k[1]]
        
        final_preds[i] = final_pred
    
    # Calcular métricas
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(y_test, final_preds)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, final_preds, average=None)
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    cm = confusion_matrix(y_test, final_preds)
    
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Macro F1: {macro_f1:.4f}")
    print(f"  - Weighted F1: {weighted_f1:.4f}")
    print(f"  - Enrutamiento: A={route_counts['A']}, B={route_counts['B']}")
    
    # Análisis detallado por componente
    # Evaluar modelo general
    y_test_general = np.zeros_like(y_test)
    for i, y in enumerate(y_test):
        if y in [0, 3]:  # Cloudy o Sunny -> grupo 0
            y_test_general[i] = 0
        else:  # Rainy o Snowy -> grupo 1
            y_test_general[i] = 1
    
    general_accuracy = accuracy_score(y_test_general, general_preds)
    
    return {
        'predictions': final_preds,
        'probabilities': final_probs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'class_names': ['Cloudy', 'Rainy', 'Snowy', 'Sunny'],
        'general_accuracy': general_accuracy,
        'general_predictions': general_preds,
        'route_counts': route_counts,
        'models': {
            'general': general_model.metadata,
            'spec_a': spec_a_model.metadata,
            'spec_b': spec_b_model.metadata
        }
    }

# =============================
# VISUALIZACIONES MEJORADAS
# =============================

def plot_confusion_matrices(results_single: Dict, results_hier: Dict, output_dir: str):
    """Crea visualizaciones comparativas de matrices de confusión"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    class_names = results_single['class_names']
    
    # Matriz modelo único
    sns.heatmap(results_single['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Número de predicciones'})
    axes[0].set_title('Modelo Único (4 clases)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicción', fontsize=12)
    axes[0].set_ylabel('Real', fontsize=12)
    
    # Matriz arquitectura jerárquica
    sns.heatmap(results_hier['confusion_matrix'], annot=True, fmt='d',
                cmap='Greens', xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Número de predicciones'})
    axes[1].set_title('Arquitectura Jerárquica (3 modelos)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicción', fontsize=12)
    axes[1].set_ylabel('Real', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Matrices normalizadas
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Normalizar por filas
    cm_single_norm = results_single['confusion_matrix'].astype('float') / results_single['confusion_matrix'].sum(axis=1)[:, np.newaxis]
    cm_hier_norm = results_hier['confusion_matrix'].astype('float') / results_hier['confusion_matrix'].sum(axis=1)[:, np.newaxis]
    
    # Convertir a porcentajes para la visualización
    cm_single_pct = cm_single_norm * 100
    cm_hier_pct = cm_hier_norm * 100
    
    sns.heatmap(cm_single_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Porcentaje (%)'})
    axes[0].set_title('Modelo Único (normalizado)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicción', fontsize=12)
    axes[0].set_ylabel('Real', fontsize=12)
    
    sns.heatmap(cm_hier_pct, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Porcentaje (%)'})
    axes[1].set_title('Arquitectura Jerárquica (normalizado)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicción', fontsize=12)
    axes[1].set_ylabel('Real', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(results_single: Dict, results_hier: Dict, output_dir: str):
    """Compara métricas entre ambos enfoques"""
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    single_values = [
        results_single['accuracy'],
        results_single['macro_f1'],
        results_single['weighted_f1']
    ]
    hier_values = [
        results_hier['accuracy'],
        results_hier['macro_f1'],
        results_hier['weighted_f1']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, single_values, width, label='Modelo Único', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, hier_values, width, label='Arquitectura Jerárquica', color='lightgreen', edgecolor='black')
    
    # Agregar valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Métricas', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Comparación de Métricas Globales', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_metrics(results_single: Dict, results_hier: Dict, output_dir: str):
    """Compara métricas por clase"""
    class_names = results_single['class_names']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    metrics = ['precision', 'recall', 'f1', 'support']
    metric_titles = ['Precisión', 'Recall', 'F1-Score', 'Soporte']
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        
        if metric == 'support':
            # El soporte es el mismo para ambos modelos
            values = results_single[metric]
            x = np.arange(len(class_names))
            bars = ax.bar(x, values, color='gray', edgecolor='black')
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                       f'{int(val)}', ha='center', va='bottom', fontweight='bold')
        else:
            single_values = results_single[metric]
            hier_values = results_hier[metric]
            
            x = np.arange(len(class_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, single_values, width, label='Modelo Único', 
                           color='skyblue', edgecolor='black')
            bars2 = ax.bar(x + width/2, hier_values, width, label='Arquitectura Jerárquica',
                           color='lightgreen', edgecolor='black')
            
            # Valores en barras
            for bars, vals in [(bars1, single_values), (bars2, hier_values)]:
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax.legend()
            ax.set_ylim(0, 1.15)
        
        ax.set_xlabel('Clases', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} por Clase', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_analysis(results_single: Dict, results_hier: Dict, y_test: np.ndarray, output_dir: str):
    """Analiza los errores de ambos modelos"""
    # Identificar errores
    errors_single = results_single['predictions'] != y_test
    errors_hier = results_hier['predictions'] != y_test
    
    # Errores únicos de cada modelo
    only_single_errors = errors_single & ~errors_hier
    only_hier_errors = ~errors_single & errors_hier
    both_errors = errors_single & errors_hier
    
    # Crear gráfico de Venn conceptual
    fig, ax = plt.subplots(figsize=(10, 8))
    
    categories = ['Solo Modelo\nÚnico', 'Ambos\nModelos', 'Solo Arquitectura\nJerárquica', 'Correctos\nAmbos']
    sizes = [
        np.sum(only_single_errors),
        np.sum(both_errors),
        np.sum(only_hier_errors),
        np.sum(~errors_single & ~errors_hier)
    ]
    colors = ['#ff9999', '#ffcc99', '#99ff99', '#99ccff']
    
    # Gráfico de pastel
    wedges, texts, autotexts = ax.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 11})
    
    # Agregar conteo absoluto
    for i, (wedge, text, autotext) in enumerate(zip(wedges, texts, autotexts)):
        autotext.set_text(f'{sizes[i]}\n({autotext.get_text()})')
        autotext.set_fontweight('bold')
    
    ax.set_title('Distribución de Errores y Aciertos', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Análisis detallado de mejoras
    improvements = only_single_errors  # Casos donde jerárquico acierta y único falla
    deteriorations = only_hier_errors  # Casos donde único acierta y jerárquico falla
    
    if np.sum(improvements) > 0:
        print(f"\n[MEJORAS] La arquitectura jerárquica corrige {np.sum(improvements)} errores del modelo único")
        
    if np.sum(deteriorations) > 0:
        print(f"[DETERIOROS] La arquitectura jerárquica introduce {np.sum(deteriorations)} nuevos errores")

def generate_detailed_report(results_single: Dict, results_hier: Dict, cfg: EvalConfig, output_dir: str):
    """Genera un reporte detallado en formato JSON y texto"""
    
    # Calcular mejoras
    improvement = {
        'accuracy': results_hier['accuracy'] - results_single['accuracy'],
        'macro_f1': results_hier['macro_f1'] - results_single['macro_f1'],
        'weighted_f1': results_hier['weighted_f1'] - results_single['weighted_f1']
    }
    
    # Mejoras por clase
    class_improvements = {}
    for i, class_name in enumerate(results_single['class_names']):
        class_improvements[class_name] = {
            'f1_improvement': results_hier['f1'][i] - results_single['f1'][i],
            'precision_improvement': results_hier['precision'][i] - results_single['precision'][i],
            'recall_improvement': results_hier['recall'][i] - results_single['recall'][i]
        }
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'models': {
                'single': cfg.SINGLE_MODEL_PATH,
                'general': cfg.MODEL_GENERAL_PATH,
                'specialist_a': cfg.MODEL_SPEC_A_PATH,
                'specialist_b': cfg.MODEL_SPEC_B_PATH
            },
            'data': cfg.CSV_PATH,
            'test_size': cfg.TEST_RATIO
        },
        'results': {
            'single_model': {
                'accuracy': float(results_single['accuracy']),
                'macro_f1': float(results_single['macro_f1']),
                'weighted_f1': float(results_single['weighted_f1']),
                'per_class': {
                    class_name: {
                        'precision': float(results_single['precision'][i]),
                        'recall': float(results_single['recall'][i]),
                        'f1': float(results_single['f1'][i]),
                        'support': int(results_single['support'][i])
                    }
                    for i, class_name in enumerate(results_single['class_names'])
                }
            },
            'hierarchical_model': {
                'accuracy': float(results_hier['accuracy']),
                'macro_f1': float(results_hier['macro_f1']),
                'weighted_f1': float(results_hier['weighted_f1']),
                'general_accuracy': float(results_hier['general_accuracy']),
                'routing_distribution': results_hier['route_counts'],
                'per_class': {
                    class_name: {
                        'precision': float(results_hier['precision'][i]),
                        'recall': float(results_hier['recall'][i]),
                        'f1': float(results_hier['f1'][i]),
                        'support': int(results_hier['support'][i])
                    }
                    for i, class_name in enumerate(results_hier['class_names'])
                }
            },
            'improvements': {
                'global': improvement,
                'per_class': class_improvements
            }
        }
    }
    
    # Guardar JSON
    with open(os.path.join(output_dir, 'detailed_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Crear reporte de texto (con encoding UTF-8 para soportar caracteres especiales)
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE COMPARACIÓN DE ARQUITECTURAS\n")
        f.write("="*80 + "\n\n")
        
        f.write("RESUMEN EJECUTIVO\n")
        f.write("-"*40 + "\n")
        
        if improvement['accuracy'] > 0:
            f.write(f"✓ La arquitectura jerárquica MEJORA la accuracy en {improvement['accuracy']*100:.2f}%\n")
        else:
            f.write(f"✗ La arquitectura jerárquica REDUCE la accuracy en {-improvement['accuracy']*100:.2f}%\n")
            
        if improvement['macro_f1'] > 0:
            f.write(f"✓ La arquitectura jerárquica MEJORA el macro F1 en {improvement['macro_f1']*100:.2f}%\n")
        else:
            f.write(f"✗ La arquitectura jerárquica REDUCE el macro F1 en {-improvement['macro_f1']*100:.2f}%\n")
        
        f.write("\nMÉTRICAS GLOBALES\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Métrica':<20} {'Modelo Único':>15} {'Jerárquico':>15} {'Diferencia':>15}\n")
        f.write(f"{'Accuracy':<20} {results_single['accuracy']:>15.4f} {results_hier['accuracy']:>15.4f} "
                f"{improvement['accuracy']:>+15.4f}\n")
        f.write(f"{'Macro F1':<20} {results_single['macro_f1']:>15.4f} {results_hier['macro_f1']:>15.4f} "
                f"{improvement['macro_f1']:>+15.4f}\n")
        f.write(f"{'Weighted F1':<20} {results_single['weighted_f1']:>15.4f} {results_hier['weighted_f1']:>15.4f} "
                f"{improvement['weighted_f1']:>+15.4f}\n")
        
        f.write("\nANÁLISIS POR CLASE\n")
        f.write("-"*40 + "\n")
        
        for i, class_name in enumerate(results_single['class_names']):
            f.write(f"\n{class_name}:\n")
            f.write(f"  F1-Score: {results_single['f1'][i]:.3f} → {results_hier['f1'][i]:.3f} "
                   f"({class_improvements[class_name]['f1_improvement']:+.3f})\n")
            f.write(f"  Precisión: {results_single['precision'][i]:.3f} → {results_hier['precision'][i]:.3f} "
                   f"({class_improvements[class_name]['precision_improvement']:+.3f})\n")
            f.write(f"  Recall: {results_single['recall'][i]:.3f} → {results_hier['recall'][i]:.3f} "
                   f"({class_improvements[class_name]['recall_improvement']:+.3f})\n")
            f.write(f"  Soporte: {results_single['support'][i]}\n")
        
        f.write("\nDETALLES DE LA ARQUITECTURA JERÁRQUICA\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy del modelo general: {results_hier['general_accuracy']:.4f}\n")
        f.write(f"Distribución de enrutamiento:\n")
        f.write(f"  - Especialista A (Cloudy/Sunny): {results_hier['route_counts']['A']} muestras\n")
        f.write(f"  - Especialista B (Rainy/Snowy): {results_hier['route_counts']['B']} muestras\n")
        
        f.write("\n" + "="*80 + "\n")

# =============================
# FUNCIÓN PRINCIPAL
# =============================

def main():
    """Función principal de evaluación"""
    cfg = EvalConfig()
    
    # Configurar dispositivo
    if cfg.DEVICE:
        device = torch.device(cfg.DEVICE)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[INFO] Usando dispositivo: {device}")
    
    # Verificar rutas de modelos
    print("\n[VERIFICANDO RUTAS]")
    print(f"  - CSV: {cfg.CSV_PATH}")
    print(f"  - Modelo único: {cfg.SINGLE_MODEL_PATH}")
    print(f"  - Modelo general: {cfg.MODEL_GENERAL_PATH}")
    print(f"  - Especialista A: {cfg.MODEL_SPEC_A_PATH}")
    print(f"  - Especialista B: {cfg.MODEL_SPEC_B_PATH}")
    
    # Verificar existencia de archivos
    for name, path in [
        ("CSV", cfg.CSV_PATH),
        ("Modelo único", cfg.SINGLE_MODEL_PATH),
        ("Modelo general", cfg.MODEL_GENERAL_PATH),
        ("Especialista A", cfg.MODEL_SPEC_A_PATH),
        ("Especialista B", cfg.MODEL_SPEC_B_PATH)
    ]:
        if os.path.exists(path):
            print(f"  ✓ {name} encontrado")
        else:
            print(f"  ✗ {name} NO encontrado en: {path}")
            raise FileNotFoundError(f"No se encontró {name} en: {path}")
    
    # Crear directorio de salida
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Cargar datos
    print("\n[CARGANDO DATOS]")
    df = pd.read_csv(cfg.CSV_PATH)
    print(f"  - Forma del dataset: {df.shape}")
    
    # Preparar características y etiquetas
    # Detectar si debemos usar 17 o 18 características
    feature_cols_18 = [col for col in cfg.FEATURE_COLS if col in df.columns]
    feature_cols_17 = [col for col in cfg.FEATURE_COLS_17 if col in df.columns]
    
    # Por defecto usar 18, pero verificar si algún modelo necesita 17
    print(f"\n[DETECTANDO CONFIGURACIÓN DE CARACTERÍSTICAS]")
    print(f"  - Features disponibles (18): {len(feature_cols_18)}")
    print(f"  - Features disponibles (17): {len(feature_cols_17)}")
    
    # Comprobar rápidamente los modelos para ver cuántas features esperan
    try:
        # Cargar solo los metadatos
        ckpt_single = torch.load(cfg.SINGLE_MODEL_PATH, map_location='cpu')
        ckpt_general = torch.load(cfg.MODEL_GENERAL_PATH, map_location='cpu')
        ckpt_spec_a = torch.load(cfg.MODEL_SPEC_A_PATH, map_location='cpu')
        ckpt_spec_b = torch.load(cfg.MODEL_SPEC_B_PATH, map_location='cpu')
        
        features_needed = [
            ckpt_single.get('in_features', 18),
            ckpt_general.get('in_features', 18),
            ckpt_spec_a.get('in_features', 18),
            ckpt_spec_b.get('in_features', 18)
        ]
        
        min_features = min(features_needed)
        max_features = max(features_needed)
        
        print(f"  - Modelos esperan entre {min_features} y {max_features} features")
        
        # IMPORTANTE: Usar siempre 18 características para no degradar el rendimiento
        # Los modelos que necesiten 17 harán el ajuste automáticamente
        feature_cols = feature_cols_18
        print(f"  - Usando configuración de 18 features (completa)")
        print(f"  - Los modelos que necesiten menos features se ajustarán automáticamente")
            
    except Exception as e:
        print(f"  - No se pudo detectar automáticamente, usando 18 features por defecto")
        feature_cols = feature_cols_18
    
    X = df[feature_cols].values.astype(np.float32)
    
    # Detectar etiquetas
    if cfg.SUMMARY_ONEHOT_PREFIX:
        label_cols = [col for col in df.columns if col.startswith(cfg.SUMMARY_ONEHOT_PREFIX)]
        y = df[label_cols].values.argmax(axis=1)
        class_names = [col.replace(cfg.SUMMARY_ONEHOT_PREFIX, '') for col in label_cols]
    else:
        # Mapear etiquetas de texto a índices
        y = pd.Categorical(df[cfg.LABEL_COL_RAW]).codes
        class_names = pd.Categorical(df[cfg.LABEL_COL_RAW]).categories.tolist()
    
    print(f"  - Características: {len(feature_cols)}")
    print(f"  - Clases: {class_names}")
    print(f"  - Distribución: {np.bincount(y)}")
    
    # División temporal de datos
    n_samples = len(X)
    train_end = int(n_samples * (1 - cfg.VAL_RATIO - cfg.TEST_RATIO))
    val_end = int(n_samples * (1 - cfg.TEST_RATIO))
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    print(f"\n[DIVISIÓN DE DATOS]")
    print(f"  - Total: {n_samples}")
    print(f"  - Test: {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")
    
    # Evaluar modelo único
    results_single = evaluate_single_model(X_test, y_test, cfg.SINGLE_MODEL_PATH, device)
    
    # Evaluar arquitectura jerárquica
    results_hier = evaluate_hierarchical_model(
        X_test, y_test,
        cfg.MODEL_GENERAL_PATH, cfg.MODEL_SPEC_A_PATH, cfg.MODEL_SPEC_B_PATH,
        cfg.ROUTE_BY_GENERAL, cfg.FINAL_CLASS_MAP, device
    )
    
    # Generar visualizaciones
    print("\n[GENERANDO VISUALIZACIONES]")
    plot_confusion_matrices(results_single, results_hier, cfg.OUTPUT_DIR)
    plot_metrics_comparison(results_single, results_hier, cfg.OUTPUT_DIR)
    plot_per_class_metrics(results_single, results_hier, cfg.OUTPUT_DIR)
    plot_error_analysis(results_single, results_hier, y_test, cfg.OUTPUT_DIR)
    
    # Generar reporte
    print("\n[GENERANDO REPORTE]")
    generate_detailed_report(results_single, results_hier, cfg, cfg.OUTPUT_DIR)
    
    print(f"\n[COMPLETADO] Resultados guardados en: {cfg.OUTPUT_DIR}")
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    print(f"Modelo Único:         Accuracy={results_single['accuracy']:.4f}, F1={results_single['macro_f1']:.4f}")
    print(f"Arquitectura Jerárquica: Accuracy={results_hier['accuracy']:.4f}, F1={results_hier['macro_f1']:.4f}")
    
    if results_hier['accuracy'] > results_single['accuracy']:
        print(f"\n[GANADOR] La arquitectura jerárquica es SUPERIOR (+{(results_hier['accuracy']-results_single['accuracy'])*100:.2f}%)")
    else:
        print(f"\n[GANADOR] El modelo único es SUPERIOR (+{(results_single['accuracy']-results_hier['accuracy'])*100:.2f}%)")
    
    return results_single, results_hier

if __name__ == "__main__":
    main()