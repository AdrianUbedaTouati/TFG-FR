#!/usr/bin/env python3
"""
Script para ejecutar la comparación usando los modelos reentrenados
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_architectures import (
    EvalConfig, ModelWrapper, evaluate_single_model,
    plot_confusion_matrices, plot_metrics_comparison,
    plot_per_class_metrics, plot_error_analysis,
    generate_detailed_report
)
from evaluate_architectures_corrected import CorrectedModelWrapper

def evaluate_hierarchical_with_corrected(X_test, y_test, general_path, 
                                        route_map, final_map, device):
    """
    Evalúa la arquitectura jerárquica usando los modelos reentrenados
    """
    print("\n[EVALUANDO] Arquitectura jerárquica con modelos REENTRENADOS")
    
    # Rutas de modelos reentrenados
    spec_a_path = "outputs/retrained_a/a_corrected.pt"
    spec_b_path = "outputs/retrained_b/b_corrected.pt"
    
    # Verificar que existen
    if not os.path.exists(spec_a_path) or not os.path.exists(spec_b_path):
        print("⚠️  No se encontraron modelos reentrenados.")
        print("   Ejecute primero retrain_specialists.py")
        raise FileNotFoundError("Modelos reentrenados no encontrados")
    
    # Cargar modelos
    print("  Cargando modelos:")
    general_model = ModelWrapper(general_path, device)
    print(f"    - General: {general_model.metadata.get('num_classes', 2)} clases")
    
    spec_a_model = CorrectedModelWrapper(spec_a_path, device)
    print(f"    - Especialista A (reentrenado): {spec_a_model.metadata.get('num_classes', 3)} clases")
    print(f"      Clases objetivo: {spec_a_model.metadata.get('target_classes', 'N/A')}")
    
    spec_b_model = CorrectedModelWrapper(spec_b_path, device)
    print(f"    - Especialista B (reentrenado): {spec_b_model.metadata.get('num_classes', 3)} clases")
    print(f"      Clases objetivo: {spec_b_model.metadata.get('target_classes', 'N/A')}")
    
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
            
            if spec_pred[0] == 2:  # Clase "Other"
                other_predictions += 1
                confidence_scores[i] = spec_prob[0, 2]
                # Usar predicción por defecto basada en probabilidades
                final_pred = 0  # Default a Cloudy
            else:
                # Predicción normal dentro del dominio
                final_pred = final_map.get(('A', spec_pred[0]), 0)
                confidence_scores[i] = spec_prob[0, spec_pred[0]]
                
                # Mapear probabilidades
                for j, (k, v) in enumerate(final_map.items()):
                    if k[0] == 'A' and k[1] < 2:
                        final_probs[i, v] = spec_prob[0, k[1]]
                        
        else:  # route == 'B'
            spec_pred, spec_prob = spec_b_model.predict(X_test[i:i+1])
            
            if spec_pred[0] == 2:  # Clase "Other"
                other_predictions += 1
                confidence_scores[i] = spec_prob[0, 2]
                final_pred = 1  # Default a Rainy
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
            'corrected': True
        }
    }

def main():
    """Función principal de comparación con modelos reentrenados"""
    cfg = EvalConfig()
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Usando dispositivo: {device}")
    
    # Crear directorio de salida
    output_dir = "outputs/comparison_results_corrected"
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
    
    print(f"\n[DIVISIÓN DE DATOS]")
    print(f"  - Total: {n_samples}")
    print(f"  - Test: {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")
    
    # Evaluar modelo único
    print(f"\n[EVALUACIÓN 1/2] Modelo único")
    results_single = evaluate_single_model(X_test, y_test, cfg.SINGLE_MODEL_PATH, device)
    
    # Evaluar arquitectura jerárquica con modelos reentrenados
    print(f"\n[EVALUACIÓN 2/2] Arquitectura jerárquica con modelos reentrenados")
    results_hier = evaluate_hierarchical_with_corrected(
        X_test, y_test, cfg.MODEL_GENERAL_PATH,
        cfg.ROUTE_BY_GENERAL, cfg.FINAL_CLASS_MAP, device
    )
    
    # Generar visualizaciones
    print("\n[GENERANDO VISUALIZACIONES]")
    
    # Configuración temporal para el directorio de salida
    cfg_temp = EvalConfig()
    cfg_temp.OUTPUT_DIR = output_dir
    
    plot_confusion_matrices(results_single, results_hier, output_dir)
    plot_metrics_comparison(results_single, results_hier, output_dir)
    plot_per_class_metrics(results_single, results_hier, output_dir)
    plot_error_analysis(results_single, results_hier, y_test, output_dir)
    
    # Generar reporte
    print("\n[GENERANDO REPORTE]")
    generate_detailed_report(results_single, results_hier, cfg_temp, output_dir)
    
    # Reporte adicional específico para modelos corregidos
    with open(os.path.join(output_dir, 'corrected_models_report.txt'), 'w', encoding='utf-8') as f:
        f.write("REPORTE DE MODELOS REENTRENADOS\n")
        f.write("="*60 + "\n\n")
        
        f.write("INFORMACIÓN DE MODELOS ESPECIALISTAS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Especialista A:\n")
        f.write(f"  - Clases objetivo: {results_hier['models']['spec_a'].get('target_classes', 'N/A')}\n")
        f.write(f"  - Número de clases: {results_hier['models']['spec_a'].get('num_classes', 'N/A')}\n")
        f.write(f"  - Mejor accuracy validación: {results_hier['models']['spec_a'].get('best_val_acc', 0):.4f}\n")
        f.write(f"\nEspecialista B:\n")
        f.write(f"  - Clases objetivo: {results_hier['models']['spec_b'].get('target_classes', 'N/A')}\n")
        f.write(f"  - Número de clases: {results_hier['models']['spec_b'].get('num_classes', 'N/A')}\n")
        f.write(f"  - Mejor accuracy validación: {results_hier['models']['spec_b'].get('best_val_acc', 0):.4f}\n")
        
        f.write(f"\nESTADÍSTICAS DE PREDICCIÓN:\n")
        f.write("-"*40 + "\n")
        f.write(f"Muestras detectadas como 'Other': {results_hier['other_predictions']}\n")
        f.write(f"Porcentaje de 'Other': {results_hier['other_predictions']/len(y_test)*100:.1f}%\n")
        f.write(f"Confianza promedio: {np.mean(results_hier['confidence_scores']):.3f}\n")
    
    print(f"\n[COMPLETADO] Resultados guardados en: {output_dir}")
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS CON MODELOS REENTRENADOS")
    print("="*60)
    print(f"Modelo Único:                    Accuracy={results_single['accuracy']:.4f}, F1={results_single['macro_f1']:.4f}")
    print(f"Arquitectura Jerárquica (corregida): Accuracy={results_hier['accuracy']:.4f}, F1={results_hier['macro_f1']:.4f}")
    
    mejora = (results_hier['accuracy'] - results_single['accuracy']) * 100
    
    if results_hier['accuracy'] > results_single['accuracy']:
        print(f"\n✅ [ÉXITO] La arquitectura jerárquica SUPERA al modelo único (+{mejora:.2f}%)")
    else:
        print(f"\n❌ [ATENCIÓN] El modelo único sigue siendo superior (+{-mejora:.2f}%)")
    
    print(f"\nDetalles adicionales:")
    print(f"- Enrutamiento correcto: {results_hier['correct_routes']/len(y_test)*100:.1f}%")
    print(f"- Muestras fuera de dominio detectadas: {results_hier['other_predictions']}")
    
    return results_single, results_hier

if __name__ == "__main__":
    print("="*80)
    print("COMPARACIÓN DE ARQUITECTURAS CON MODELOS REENTRENADOS")
    print("="*80)
    print("\nComparando:")
    print("1. Modelo único (4 clases directas)")
    print("2. Arquitectura jerárquica con especialistas REENTRENADOS")
    print("\n" + "-"*80)
    
    try:
        results_single, results_hier = main()
        print("\n✓ Evaluación completada exitosamente")
        print(f"✓ Los resultados se han guardado en: outputs/comparison_results_corrected/")
        print("\nArchivos generados:")
        print("  - confusion_matrices_comparison.png")
        print("  - confusion_matrices_normalized.png")
        print("  - metrics_comparison.png")
        print("  - per_class_metrics_comparison.png")
        print("  - error_distribution.png")
        print("  - detailed_report.json")
        print("  - summary_report.txt")
        print("  - corrected_models_report.txt")
        
    except Exception as e:
        print(f"\n✗ Error durante la evaluación: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)