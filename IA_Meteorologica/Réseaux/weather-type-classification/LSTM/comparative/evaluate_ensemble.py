"""
Script principal para evaluar la arquitectura de Voting Ensemble
Combina las predicciones del modelo único y la arquitectura jerárquica
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import argparse

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_architectures import (
    EvalConfig, ModelWrapper, evaluate_single_model,
    plot_confusion_matrices, plot_metrics_comparison,
    plot_per_class_metrics
)
from evaluate_architectures_corrected import CorrectedModelWrapper
from ensemble_voting_architecture import (
    VotingEnsemble, EnsembleConfig,
    create_ensemble_visualizations, generate_ensemble_report
)


def evaluate_hierarchical_for_ensemble(X_test, y_test, general_path, 
                                     spec_a_path, spec_b_path,
                                     route_map, final_map, device,
                                     use_corrected=True):
    """
    Evalúa la arquitectura jerárquica y retorna información detallada para el ensemble
    """
    print("\n[EVALUANDO] Arquitectura jerárquica para ensemble")
    
    # Rutas de modelos corregidos
    spec_a_corrected = "outputs/retrained_a/a_corrected.pt"
    spec_b_corrected = "outputs/retrained_b/b_corrected.pt"
    
    # Cargar modelo general
    general_model = ModelWrapper(general_path, device)
    general_preds, general_probs = general_model.predict(X_test)
    
    # Decidir qué modelos usar
    if use_corrected and os.path.exists(spec_a_corrected) and os.path.exists(spec_b_corrected):
        print("  ✓ Usando modelos especialistas CORREGIDOS")
        spec_a_model = CorrectedModelWrapper(spec_a_corrected, device)
        spec_b_model = CorrectedModelWrapper(spec_b_corrected, device)
        is_corrected = True
    else:
        print("  ✓ Usando modelos especialistas ORIGINALES")
        spec_a_model = ModelWrapper(spec_a_path, device)
        spec_b_model = ModelWrapper(spec_b_path, device)
        is_corrected = False
    
    # Arrays para resultados
    final_preds = np.zeros_like(y_test)
    final_probs = np.zeros((len(y_test), 4))
    confidence_scores = np.zeros(len(y_test))
    general_confidence = np.zeros(len(y_test))
    
    # Estadísticas
    correct_routes = 0
    other_predictions = 0
    
    # Mapeo para verificar enrutamiento
    correct_routing = {0: 'A', 3: 'A', 1: 'B', 2: 'B'}
    
    # Procesar cada muestra
    for i in range(len(X_test)):
        true_class = y_test[i]
        general_pred = general_preds[i]
        route = route_map.get(general_pred, 'A')
        
        # Guardar confianza del modelo general
        general_confidence[i] = general_probs[i].max()
        
        # Verificar enrutamiento
        expected_route = correct_routing[true_class]
        if route == expected_route:
            correct_routes += 1
        
        # Predicción con especialista
        if route == 'A':
            spec_pred, spec_prob = spec_a_model.predict(X_test[i:i+1])
            
            if is_corrected and spec_pred[0] == 2:  # Clase "Other"
                other_predictions += 1
                confidence_scores[i] = spec_prob[0, 2]
                final_pred = 0  # Default
            else:
                final_pred = final_map.get(('A', spec_pred[0]), 0)
                confidence_scores[i] = spec_prob[0, spec_pred[0]]
                
                # Mapear probabilidades
                for j, (k, v) in enumerate(final_map.items()):
                    if k[0] == 'A' and (not is_corrected or k[1] < 2):
                        if k[1] < spec_prob.shape[1]:
                            final_probs[i, v] = spec_prob[0, k[1]]
        else:  # route == 'B'
            spec_pred, spec_prob = spec_b_model.predict(X_test[i:i+1])
            
            if is_corrected and spec_pred[0] == 2:
                other_predictions += 1
                confidence_scores[i] = spec_prob[0, 2]
                final_pred = 1  # Default
            else:
                final_pred = final_map.get(('B', spec_pred[0]), 0)
                confidence_scores[i] = spec_prob[0, spec_pred[0]]
                
                for j, (k, v) in enumerate(final_map.items()):
                    if k[0] == 'B' and (not is_corrected or k[1] < 2):
                        if k[1] < spec_prob.shape[1]:
                            final_probs[i, v] = spec_prob[0, k[1]]
        
        final_preds[i] = final_pred
        
        # Normalizar probabilidades si es necesario
        if final_probs[i].sum() > 0:
            final_probs[i] = final_probs[i] / final_probs[i].sum()
        else:
            # Si no hay probabilidades válidas, usar one-hot
            final_probs[i, int(final_pred)] = 1.0
    
    # Calcular métricas
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(y_test, final_preds)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, final_preds, average=None)
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    cm = confusion_matrix(y_test, final_preds)
    
    print(f"\n  RESULTADOS:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Macro F1: {macro_f1:.4f}")
    print(f"  - Enrutamiento correcto: {correct_routes/len(y_test)*100:.1f}%")
    if is_corrected:
        print(f"  - Predicciones 'Other': {other_predictions}")
    
    return {
        'predictions': final_preds,
        'probabilities': final_probs,
        'confidence': confidence_scores,
        'general_confidence': general_confidence,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'class_names': ['Cloudy', 'Rainy', 'Snowy', 'Sunny'],
        'is_corrected': is_corrected
    }


def evaluate_all_strategies(ensemble, y_test, results_single, results_hier, 
                          general_confidence, output_dir):
    """
    Evalúa todas las estrategias de voting y compara resultados
    """
    strategies = ['hard', 'soft', 'weighted', 'weighted_confidence', 'cascade']
    strategy_results = {}
    
    print("\n" + "="*60)
    print("EVALUANDO DIFERENTES ESTRATEGIAS DE VOTING")
    print("="*60)
    
    for strategy in strategies:
        print(f"\nEstrategia: {strategy}")
        
        # Configurar ensemble
        ensemble.config.voting_strategy = strategy
        
        # Realizar predicciones
        results = ensemble.batch_predict(results_single, results_hier, general_confidence)
        
        # Analizar rendimiento
        analysis = ensemble.analyze_performance(y_test, results, results_single, results_hier)
        
        # Guardar resultados
        strategy_results[strategy] = {
            'results': results,
            'analysis': analysis,
            'accuracy': analysis['accuracy']['ensemble']
        }
        
        print(f"  - Accuracy: {analysis['accuracy']['ensemble']:.4f}")
        print(f"  - Mejora sobre mejor individual: {analysis['improvements']['over_best_individual']:+.2f}%")
    
    # Encontrar mejor estrategia
    best_strategy = max(strategy_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✅ MEJOR ESTRATEGIA: {best_strategy[0]} con accuracy {best_strategy[1]['accuracy']:.4f}")
    
    # Crear visualización comparativa
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Accuracy por estrategia
    strategies_names = list(strategy_results.keys())
    accuracies = [strategy_results[s]['accuracy'] for s in strategies_names]
    
    bars = ax1.bar(strategies_names, accuracies, color='skyblue', edgecolor='black')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Voting Strategy')
    ax1.set_ylim(min(accuracies) - 0.01, max(accuracies) + 0.01)
    
    # Marcar la mejor
    best_idx = accuracies.index(max(accuracies))
    bars[best_idx].set_color('green')
    
    # Agregar valores
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{acc:.4f}', ha='center', va='bottom')
    
    # Gráfico 2: Mejora sobre mejor individual
    improvements = [strategy_results[s]['analysis']['improvements']['over_best_individual'] 
                   for s in strategies_names]
    
    bars2 = ax2.bar(strategies_names, improvements, 
                    color=['green' if x > 0 else 'red' for x in improvements],
                    edgecolor='black')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Improvement over Best Individual Model')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Agregar valores
    for bar, imp in zip(bars2, improvements):
        y_pos = bar.get_height() + 0.05 if imp > 0 else bar.get_height() - 0.1
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{imp:+.2f}%', ha='center', va='bottom' if imp > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_strategy[0], strategy_results


def main():
    """Función principal de evaluación del ensemble"""
    
    parser = argparse.ArgumentParser(description='Evaluación de Voting Ensemble')
    parser.add_argument('--strategy', type=str, default='cascade',
                       choices=['hard', 'soft', 'weighted', 'weighted_confidence', 'cascade', 'all'],
                       help='Estrategia de voting a usar (default: cascade)')
    parser.add_argument('--use-corrected', action='store_true',
                       help='Usar modelos especialistas corregidos')
    parser.add_argument('--evaluate-all', action='store_true',
                       help='Evaluar todas las estrategias y seleccionar la mejor')
    
    args = parser.parse_args()
    
    # Configuración
    cfg = EvalConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("EVALUACIÓN DE VOTING ENSEMBLE")
    print("="*80)
    print(f"\nDispositivo: {device}")
    print(f"Estrategia: {args.strategy}")
    print(f"Modelos corregidos: {'Sí' if args.use_corrected else 'No'}")
    
    # Crear directorio de salida
    output_dir = "outputs/ensemble_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datos
    print("\n[CARGANDO DATOS]")
    df = pd.read_csv(cfg.CSV_PATH)
    
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
    
    print(f"  - Muestras de test: {len(X_test)}")
    
    # Evaluar modelo único
    print(f"\n[EVALUACIÓN 1/3] Modelo único")
    results_single = evaluate_single_model(X_test, y_test, cfg.SINGLE_MODEL_PATH, device)
    
    # Evaluar arquitectura jerárquica
    print(f"\n[EVALUACIÓN 2/3] Arquitectura jerárquica")
    results_hier = evaluate_hierarchical_for_ensemble(
        X_test, y_test,
        cfg.MODEL_GENERAL_PATH, cfg.MODEL_SPEC_A_PATH, cfg.MODEL_SPEC_B_PATH,
        cfg.ROUTE_BY_GENERAL, cfg.FINAL_CLASS_MAP, device,
        use_corrected=args.use_corrected
    )
    
    # Crear ensemble
    print(f"\n[EVALUACIÓN 3/3] Voting Ensemble")
    
    # Configurar ensemble
    ensemble_config = EnsembleConfig(
        voting_strategy=args.strategy if not args.evaluate_all else 'cascade',
        single_model_weight=0.5,
        hierarchical_weight=0.5,
        confidence_threshold=0.85,
        use_calibration=True,
        use_meta_features=True,
        adaptive_weights=True
    )
    
    ensemble = VotingEnsemble(ensemble_config)
    
    # Extraer confianza del modelo general
    general_confidence = results_hier.get('general_confidence', None)
    
    if args.evaluate_all or args.strategy == 'all':
        # Evaluar todas las estrategias
        best_strategy, all_results = evaluate_all_strategies(
            ensemble, y_test, results_single, results_hier, 
            general_confidence, output_dir
        )
        
        # Usar la mejor estrategia
        ensemble.config.voting_strategy = best_strategy
        results_ensemble = all_results[best_strategy]['results']
        analysis = all_results[best_strategy]['analysis']
        
        # Guardar resultados de todas las estrategias
        with open(f'{output_dir}/all_strategies_results.json', 'w') as f:
            json.dump({
                strategy: {
                    'accuracy': res['accuracy'],
                    'improvements': res['analysis']['improvements']
                }
                for strategy, res in all_results.items()
            }, f, indent=2)
    else:
        # Usar estrategia especificada
        results_ensemble = ensemble.batch_predict(results_single, results_hier, general_confidence)
        analysis = ensemble.analyze_performance(y_test, results_ensemble, results_single, results_hier)
    
    # Generar visualizaciones
    print("\n[GENERANDO VISUALIZACIONES]")
    create_ensemble_visualizations(
        y_test, results_ensemble, results_single, results_hier,
        analysis, output_dir
    )
    
    # Calcular métricas adicionales para el ensemble
    from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, results_ensemble['predictions'], average=None
    )
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    cm = confusion_matrix(y_test, results_ensemble['predictions'])
    
    # Crear diccionario con resultados del ensemble para las funciones de plotting
    ensemble_results_dict = {
        'predictions': results_ensemble['predictions'],
        'probabilities': results_ensemble['probabilities'],
        'confidence': results_ensemble['confidence'],
        'agreements': results_ensemble['agreements'],
        'accuracy': analysis['accuracy']['ensemble'],
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm,
        'class_names': ['Cloudy', 'Rainy', 'Snowy', 'Sunny']
    }
    
    plot_confusion_matrices(results_single, ensemble_results_dict, output_dir)
    plot_metrics_comparison(results_single, ensemble_results_dict, output_dir)
    
    # Generar reporte
    print("\n[GENERANDO REPORTE]")
    generate_ensemble_report(analysis, ensemble.config, f'{output_dir}/ensemble_report.txt')
    
    # Guardar configuración y resultados
    with open(f'{output_dir}/ensemble_config.json', 'w') as f:
        json.dump({
            'strategy': ensemble.config.voting_strategy,
            'weights': {
                'single': ensemble.config.single_model_weight,
                'hierarchical': ensemble.config.hierarchical_weight
            },
            'thresholds': {
                'confidence': ensemble.config.confidence_threshold,
                'disagreement': ensemble.config.disagreement_threshold
            },
            'options': {
                'use_calibration': ensemble.config.use_calibration,
                'use_meta_features': ensemble.config.use_meta_features,
                'adaptive_weights': ensemble.config.adaptive_weights
            }
        }, f, indent=2)
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    print(f"\nEstrategia de Voting: {ensemble.config.voting_strategy}")
    print(f"\nAccuracy:")
    print(f"  - Modelo Único:              {analysis['accuracy']['single']:.4f}")
    print(f"  - Arquitectura Jerárquica:   {analysis['accuracy']['hierarchical']:.4f}")
    print(f"  - Voting Ensemble:           {analysis['accuracy']['ensemble']:.4f} ⭐")
    
    print(f"\nMejoras:")
    print(f"  - Sobre modelo único:        {analysis['improvements']['over_single']:+.2f}%")
    print(f"  - Sobre arquitectura jerárquica: {analysis['improvements']['over_hierarchical']:+.2f}%")
    print(f"  - Sobre mejor individual:    {analysis['improvements']['over_best_individual']:+.2f}%")
    
    print(f"\nAnálisis de Acuerdo:")
    print(f"  - Tasa de acuerdo:           {analysis['agreement_analysis']['agreement_rate']:.2%}")
    print(f"  - Accuracy cuando acuerdan:  {analysis['agreement_analysis']['accuracy_when_agree']:.4f}")
    print(f"  - Accuracy cuando discrepan: {analysis['agreement_analysis']['accuracy_when_disagree']:.4f}")
    
    print(f"\n[COMPLETADO] Resultados guardados en: {output_dir}/")
    
    return analysis


if __name__ == "__main__":
    main()