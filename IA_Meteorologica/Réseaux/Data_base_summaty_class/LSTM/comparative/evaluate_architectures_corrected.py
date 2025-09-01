"""
evaluate_architectures_corrected.py - Evaluación con modelos especialistas corregidos

Este script evalúa la arquitectura jerárquica usando los modelos especialistas
reentrenados que ahora pueden manejar muestras fuera de dominio.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar funciones del evaluador original
from evaluate_architectures import (
    EvalConfig, ModelWrapper, evaluate_single_model,
    plot_confusion_matrices, plot_metrics_comparison,
    plot_per_class_metrics, plot_error_analysis,
    generate_detailed_report
)

class CorrectedModelWrapper:
    """Wrapper mejorado que maneja modelos con 3 clases (incluye 'Other')"""
    
    def __init__(self, model_path: str, device: torch.device):
        self.model_path = model_path
        self.device = device
        self.metadata = {}
        self._load()
    
    def _load(self):
        """Carga el modelo corregido"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Los modelos corregidos tienen estructura diferente
        if 'model_state_dict' in checkpoint:
            # Modelo corregido con metadata
            state_dict = checkpoint['model_state_dict']
            self.metadata = checkpoint.get('metadata', {})
        else:
            # Modelo original
            state_dict = checkpoint
            # Intentar extraer metadata del checkpoint
            self.metadata = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
        
        # Crear modelo basado en metadata
        in_features = self.metadata.get('in_features', 18)
        num_classes = self.metadata.get('num_classes', 3)
        
        # Importar la clase del modelo
        from retrain_specialists import LSTMLineClassifier as CorrectedLSTM
        
        self.model = CorrectedLSTM(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        
        # Cargar pesos
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predice y maneja la clase 'Other' apropiadamente"""
        self.model.eval()
        
        # Ajustar dimensiones si es necesario
        if X.shape[1] != self.metadata.get('in_features'):
            if X.shape[1] == 18 and self.metadata.get('in_features') == 17:
                # Eliminar 'Cloud Cover_clear' (índice 4)
                X = np.delete(X, 4, axis=1)
            elif X.shape[1] == 17 and self.metadata.get('in_features') == 18:
                # Agregar columna de ceros para 'Cloud Cover_clear'
                zeros_col = np.zeros((X.shape[0], 1))
                X = np.insert(X, 4, zeros_col, axis=1)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(1)
            
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
        
        return preds.cpu().numpy(), probs.cpu().numpy()

def evaluate_hierarchical_corrected(X_test: np.ndarray, y_test: np.ndarray,
                                  general_path: str, spec_a_path: str, spec_b_path: str,
                                  route_map: Dict, final_map: Dict, device: torch.device,
                                  use_corrected_models: bool = True) -> Dict:
    """
    Evalúa la arquitectura jerárquica con modelos corregidos
    
    Args:
        use_corrected_models: Si True, usa los modelos reentrenados
    """
    if use_corrected_models:
        print("\n[EVALUANDO] Arquitectura jerárquica con modelos CORREGIDOS")
    else:
        print("\n[EVALUANDO] Arquitectura jerárquica con modelos ORIGINALES")
    
    # Definir rutas de modelos corregidos
    spec_a_corrected = "outputs/retrained_a/a_corrected.pt"
    spec_b_corrected = "outputs/retrained_b/b_corrected.pt"
    
    # Determinar rutas de modelos
    if use_corrected_models:
        # Usar modelos reentrenados
        spec_a_corrected = "outputs/retrained_a/a_corrected.pt"
        spec_b_corrected = "outputs/retrained_b/b_corrected.pt"
        
        if os.path.exists(spec_a_corrected) and os.path.exists(spec_b_corrected):
            print("  ✓ Usando modelos especialistas CORREGIDOS")
            spec_a_path = spec_a_corrected
            spec_b_path = spec_b_corrected
        else:
            print("  ⚠️  No se encontraron modelos corregidos, usando originales")
    
    # Cargar modelos
    print("  Cargando modelos:")
    general_model = ModelWrapper(general_path, device)
    print(f"    - General: {general_model.metadata.get('num_classes', 2)} clases")
    
    # Usar wrapper apropiado según el tipo de modelo
    if use_corrected_models and os.path.exists(spec_a_corrected) and os.path.exists(spec_b_corrected):
        # Usar wrapper corregido para modelos reentrenados
        spec_a_model = CorrectedModelWrapper(spec_a_path, device)
        spec_b_model = CorrectedModelWrapper(spec_b_path, device)
    else:
        # Usar wrapper original para modelos originales
        spec_a_model = ModelWrapper(spec_a_path, device)
        spec_b_model = ModelWrapper(spec_b_path, device)
    
    print(f"    - Especialista A: {spec_a_model.metadata.get('num_classes', 2)} clases")
    print(f"    - Especialista B: {spec_b_model.metadata.get('num_classes', 2)} clases")
    
    # Verificar si son modelos corregidos (tienen 3 clases)
    is_corrected_a = spec_a_model.metadata.get('num_classes', 2) == 3
    is_corrected_b = spec_b_model.metadata.get('num_classes', 2) == 3
    
    if is_corrected_a and is_corrected_b:
        print("  ✓ Modelos con capacidad de manejar muestras fuera de dominio")
    
    # Predicción del modelo general
    print("\n  Ejecutando predicciones...")
    general_preds, general_probs = general_model.predict(X_test)
    
    # Preparar arrays para resultados
    final_preds = np.zeros_like(y_test)
    final_probs = np.zeros((len(y_test), 4))  # 4 clases finales
    confidence_scores = np.zeros(len(y_test))
    
    # Estadísticas
    route_counts = {'A': 0, 'B': 0}
    correct_routes = 0
    incorrect_routes = 0
    other_predictions = 0
    
    # Mapeo para verificar enrutamiento correcto
    correct_routing = {
        0: 'A',  # Cloudy -> A
        3: 'A',  # Sunny -> A
        1: 'B',  # Rainy -> B
        2: 'B'   # Snowy -> B
    }
    
    # Para cada muestra
    for i in range(len(X_test)):
        true_class = y_test[i]
        general_pred = general_preds[i]
        route = route_map.get(general_pred, 'A')
        route_counts[route] += 1
        
        # Verificar si el enrutamiento es correcto
        expected_route = correct_routing[true_class]
        is_correct_route = (route == expected_route)
        
        if is_correct_route:
            correct_routes += 1
        else:
            incorrect_routes += 1
        
        # Hacer predicción con el especialista correspondiente
        if route == 'A':
            spec_pred, spec_prob = spec_a_model.predict(X_test[i:i+1])
            
            if is_corrected_a and spec_pred[0] == 2:  # Clase "Other"
                other_predictions += 1
                # El modelo detectó que es una muestra fuera de dominio
                # Podemos usar una estrategia de fallback o confiar en la predicción general
                confidence_scores[i] = spec_prob[0, 2]  # Confianza en "Other"
                
                # Estrategia: usar predicción basada en probabilidades del general
                if general_probs[i, 0] > 0.5:  # Más probable Cloudy_Sunny
                    final_pred = 0  # Default a Cloudy
                else:
                    final_pred = 1  # Default a Rainy
            else:
                # Predicción normal dentro del dominio
                final_pred = final_map.get(('A', spec_pred[0]), 0)
                confidence_scores[i] = spec_prob[0, spec_pred[0]]
                
                # Mapear probabilidades
                for j, (k, v) in enumerate(final_map.items()):
                    if k[0] == 'A' and k[1] < 2:  # Solo clases objetivo
                        final_probs[i, v] = spec_prob[0, k[1]]
                        
        else:  # route == 'B'
            spec_pred, spec_prob = spec_b_model.predict(X_test[i:i+1])
            
            if is_corrected_b and spec_pred[0] == 2:  # Clase "Other"
                other_predictions += 1
                confidence_scores[i] = spec_prob[0, 2]
                
                # Estrategia de fallback
                if general_probs[i, 1] > 0.5:  # Más probable Rainy_Snowy
                    final_pred = 1  # Default a Rainy
                else:
                    final_pred = 0  # Default a Cloudy
            else:
                final_pred = final_map.get(('B', spec_pred[0]), 0)
                confidence_scores[i] = spec_prob[0, spec_pred[0]]
                
                for j, (k, v) in enumerate(final_map.items()):
                    if k[0] == 'B' and k[1] < 2:
                        final_probs[i, v] = spec_prob[0, k[1]]
        
        final_preds[i] = final_pred
    
    # Calcular métricas
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(y_test, final_preds)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, final_preds, average=None)
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    cm = confusion_matrix(y_test, final_preds)
    
    # Calcular accuracy del modelo general
    y_test_general = np.zeros_like(y_test)
    for i, y in enumerate(y_test):
        if y in [0, 3]:  # Cloudy o Sunny
            y_test_general[i] = 0
        else:  # Rainy o Snowy
            y_test_general[i] = 1
    
    general_accuracy = accuracy_score(y_test_general, general_preds)
    
    print(f"\n  RESULTADOS:")
    print(f"  - Accuracy global: {accuracy:.4f}")
    print(f"  - Macro F1: {macro_f1:.4f}")
    print(f"  - Weighted F1: {weighted_f1:.4f}")
    print(f"  - Accuracy del modelo general: {general_accuracy:.4f}")
    print(f"  - Enrutamiento correcto: {correct_routes}/{len(y_test)} ({correct_routes/len(y_test)*100:.1f}%)")
    print(f"  - Enrutamiento incorrecto: {incorrect_routes}/{len(y_test)} ({incorrect_routes/len(y_test)*100:.1f}%)")
    
    if is_corrected_a or is_corrected_b:
        print(f"  - Predicciones 'Other': {other_predictions}/{len(y_test)} ({other_predictions/len(y_test)*100:.1f}%)")
        print(f"  - Confianza promedio: {np.mean(confidence_scores):.3f}")
    
    return {
        'predictions': final_preds,
        'probabilities': final_probs,
        'confidence_scores': confidence_scores,
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
        'correct_routes': correct_routes,
        'incorrect_routes': incorrect_routes,
        'other_predictions': other_predictions,
        'models': {
            'general': general_model.metadata,
            'spec_a': spec_a_model.metadata,
            'spec_b': spec_b_model.metadata,
            'corrected': is_corrected_a and is_corrected_b
        }
    }

def plot_confidence_analysis(results_hier: Dict, output_dir: str):
    """Analiza y visualiza la confianza de las predicciones"""
    confidence = results_hier['confidence_scores']
    predictions = results_hier['predictions']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histograma de confianza
    ax1.hist(confidence, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(confidence), color='red', linestyle='--', 
                label=f'Media: {np.mean(confidence):.3f}')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Confidence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Confianza por clase
    class_names = results_hier['class_names']
    conf_by_class = [confidence[predictions == i] for i in range(len(class_names))]
    
    ax2.boxplot(conf_by_class, labels=class_names)
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Confidence by Predicted Class')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_report(results_single: Dict, results_hier_original: Dict,
                              results_hier_corrected: Dict, output_dir: str):
    """Genera un reporte comparando las tres arquitecturas"""
    
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE COMPARACIÓN: ARQUITECTURAS ORIGINAL vs CORREGIDA\n")
        f.write("="*80 + "\n\n")
        
        f.write("RESUMEN EJECUTIVO\n")
        f.write("-"*40 + "\n")
        
        # Tabla de comparación
        f.write("\n┌─────────────────────────┬──────────┬──────────┬──────────┐\n")
        f.write("│ Arquitectura           │ Accuracy │ Macro F1 │ Weighted │\n")
        f.write("├─────────────────────────┼──────────┼──────────┼──────────┤\n")
        f.write(f"│ Modelo Único           │  {results_single['accuracy']:.4f}  │  {results_single['macro_f1']:.4f}  │  {results_single['weighted_f1']:.4f}  │\n")
        f.write(f"│ Jerárquica Original    │  {results_hier_original['accuracy']:.4f}  │  {results_hier_original['macro_f1']:.4f}  │  {results_hier_original['weighted_f1']:.4f}  │\n")
        f.write(f"│ Jerárquica Corregida   │  {results_hier_corrected['accuracy']:.4f}  │  {results_hier_corrected['macro_f1']:.4f}  │  {results_hier_corrected['weighted_f1']:.4f}  │\n")
        f.write("└─────────────────────────┴──────────┴──────────┴──────────┘\n\n")
        
        # Análisis de mejora
        mejora_original = (results_hier_original['accuracy'] - results_single['accuracy']) * 100
        mejora_corregida = (results_hier_corrected['accuracy'] - results_single['accuracy']) * 100
        mejora_relativa = (results_hier_corrected['accuracy'] - results_hier_original['accuracy']) * 100
        
        f.write("ANÁLISIS DE MEJORA\n")
        f.write("-"*40 + "\n")
        f.write(f"Mejora Original vs Único: {mejora_original:+.2f}%\n")
        f.write(f"Mejora Corregida vs Único: {mejora_corregida:+.2f}%\n")
        f.write(f"Mejora Corregida vs Original: {mejora_relativa:+.2f}%\n\n")
        
        # Análisis de enrutamiento
        f.write("ANÁLISIS DE ENRUTAMIENTO\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy del modelo general: {results_hier_corrected['general_accuracy']:.4f}\n")
        f.write(f"Enrutamiento correcto: {results_hier_corrected['correct_routes']/len(results_hier_corrected['predictions'])*100:.1f}%\n")
        
        if results_hier_corrected.get('other_predictions', 0) > 0:
            f.write(f"Muestras detectadas como fuera de dominio: {results_hier_corrected['other_predictions']}\n")
        
        f.write("\nCONCLUSIONES\n")
        f.write("-"*40 + "\n")
        
        if results_hier_corrected['accuracy'] > results_single['accuracy']:
            f.write("✓ La arquitectura jerárquica corregida SUPERA al modelo único\n")
            f.write("✓ El entrenamiento con muestras fuera de dominio fue exitoso\n")
            f.write("✓ Los especialistas ahora son robustos a errores de enrutamiento\n")
        else:
            f.write("✗ La arquitectura jerárquica aún no supera al modelo único\n")
            f.write("  Posibles causas:\n")
            f.write("  - El modelo general necesita mayor precisión (>98%)\n")
            f.write("  - Los especialistas necesitan más entrenamiento\n")
            f.write("  - La arquitectura puede beneficiarse de ensemble\n")
        
        f.write("\nRECOMENDACIONES\n")
        f.write("-"*40 + "\n")
        f.write("1. Mejorar el modelo general para reducir errores de enrutamiento\n")
        f.write("2. Implementar ensemble voting entre arquitecturas\n")
        f.write("3. Ajustar pesos de clase para optimizar el balance\n")
        f.write("4. Considerar arquitectura con rechazo de baja confianza\n")

def main():
    """Función principal de evaluación con modelos corregidos"""
    cfg = EvalConfig()
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Usando dispositivo: {device}")
    
    # Crear directorio de salida
    output_dir = "outputs/comparison_corrected"
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datos
    print("\n[CARGANDO DATOS]")
    df = pd.read_csv(cfg.CSV_PATH)
    print(f"  - Forma del dataset: {df.shape}")
    
    # Preparar datos
    feature_cols = [col for col in cfg.FEATURE_COLS if col in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    
    label_cols = [col for col in df.columns if col.startswith(cfg.SUMMARY_ONEHOT_PREFIX)]
    y = df[label_cols].values.argmax(axis=1)
    
    # División de datos
    n_samples = len(X)
    val_end = int(n_samples * (1 - cfg.TEST_RATIO))
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    print(f"\n[EVALUACIÓN 1/3] Modelo único")
    results_single = evaluate_single_model(X_test, y_test, cfg.SINGLE_MODEL_PATH, device)
    
    print(f"\n[EVALUACIÓN 2/3] Arquitectura jerárquica original")
    results_hier_original = evaluate_hierarchical_corrected(
        X_test, y_test,
        cfg.MODEL_GENERAL_PATH, cfg.MODEL_SPEC_A_PATH, cfg.MODEL_SPEC_B_PATH,
        cfg.ROUTE_BY_GENERAL, cfg.FINAL_CLASS_MAP, device,
        use_corrected_models=False
    )
    
    print(f"\n[EVALUACIÓN 3/3] Arquitectura jerárquica corregida")
    results_hier_corrected = evaluate_hierarchical_corrected(
        X_test, y_test,
        cfg.MODEL_GENERAL_PATH, cfg.MODEL_SPEC_A_PATH, cfg.MODEL_SPEC_B_PATH,
        cfg.ROUTE_BY_GENERAL, cfg.FINAL_CLASS_MAP, device,
        use_corrected_models=True
    )
    
    # Generar visualizaciones
    print("\n[GENERANDO VISUALIZACIONES]")
    
    # Matrices de confusión
    plot_confusion_matrices(results_single, results_hier_corrected, output_dir)
    
    # Comparación de métricas
    plot_metrics_comparison(results_single, results_hier_corrected, output_dir)
    
    # Métricas por clase
    plot_per_class_metrics(results_single, results_hier_corrected, output_dir)
    
    # Análisis de confianza
    if 'confidence_scores' in results_hier_corrected:
        plot_confidence_analysis(results_hier_corrected, output_dir)
    
    # Generar reporte comparativo
    print("\n[GENERANDO REPORTE]")
    generate_comparison_report(results_single, results_hier_original, 
                              results_hier_corrected, output_dir)
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)
    print(f"\nModelo Único:")
    print(f"  - Accuracy: {results_single['accuracy']:.4f}")
    print(f"  - Macro F1: {results_single['macro_f1']:.4f}")
    
    print(f"\nJerárquica Original:")
    print(f"  - Accuracy: {results_hier_original['accuracy']:.4f}")
    print(f"  - Macro F1: {results_hier_original['macro_f1']:.4f}")
    
    print(f"\nJerárquica Corregida:")
    print(f"  - Accuracy: {results_hier_corrected['accuracy']:.4f}")
    print(f"  - Macro F1: {results_hier_corrected['macro_f1']:.4f}")
    
    mejora = (results_hier_corrected['accuracy'] - results_hier_original['accuracy']) * 100
    print(f"\n[MEJORA] {mejora:+.2f}% con modelos corregidos")
    
    if results_hier_corrected['accuracy'] > results_single['accuracy']:
        print("\n✅ ¡ÉXITO! La arquitectura jerárquica corregida supera al modelo único")
    else:
        print("\n⚠️  La arquitectura jerárquica aún no supera al modelo único")
        print("   Considere las recomendaciones en el reporte para mejorar el rendimiento")
    
    print(f"\n[COMPLETADO] Resultados guardados en: {output_dir}")

if __name__ == "__main__":
    main()