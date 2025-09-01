"""
Arquitectura de Voting Ensemble que combina:
1. Modelo único (4 clases directas)
2. Arquitectura jerárquica (3 modelos)

Implementa múltiples estrategias de voting para maximizar el rendimiento.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EnsembleConfig:
    """Configuración para el ensemble"""
    # Estrategias de voting
    voting_strategy: str = 'weighted_confidence'  # 'hard', 'soft', 'weighted', 'weighted_confidence'
    
    # Pesos para voting ponderado
    single_model_weight: float = 0.5
    hierarchical_weight: float = 0.5
    
    # Umbrales de confianza
    confidence_threshold: float = 0.85
    disagreement_threshold: float = 0.3
    
    # Opciones avanzadas
    use_calibration: bool = True
    use_meta_features: bool = True
    adaptive_weights: bool = True


class VotingEnsemble:
    """
    Ensemble que combina predicciones del modelo único y arquitectura jerárquica
    usando múltiples estrategias de voting.
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.calibration_params = None
        self.class_weights = None
        self.performance_history = []
        
    def calibrate_probabilities(self, probs: np.ndarray, method: str = 'platt') -> np.ndarray:
        """
        Calibra las probabilidades para mejorar la confiabilidad
        
        Args:
            probs: Probabilidades sin calibrar
            method: 'platt' o 'isotonic'
        """
        if method == 'platt':
            # Platt scaling simple
            # En producción, esto requeriría datos de validación para ajustar
            calibrated = 1 / (1 + np.exp(-np.log(probs / (1 - probs + 1e-8))))
            return calibrated / calibrated.sum(axis=1, keepdims=True)
        else:
            return probs
    
    def extract_meta_features(self, probs_single: np.ndarray, probs_hier: np.ndarray,
                            confidence_single: float, confidence_hier: float) -> np.ndarray:
        """
        Extrae meta-características para decisiones más inteligentes
        
        Returns:
            Array de meta-características
        """
        meta_features = []
        
        # 1. Niveles de confianza
        meta_features.extend([confidence_single, confidence_hier])
        
        # 2. Entropía de las distribuciones
        entropy_single = -np.sum(probs_single * np.log(probs_single + 1e-8))
        entropy_hier = -np.sum(probs_hier * np.log(probs_hier + 1e-8))
        meta_features.extend([entropy_single, entropy_hier])
        
        # 3. Diferencia entre top 2 clases
        sorted_single = np.sort(probs_single)[::-1]
        sorted_hier = np.sort(probs_hier)[::-1]
        margin_single = sorted_single[0] - sorted_single[1]
        margin_hier = sorted_hier[0] - sorted_hier[1]
        meta_features.extend([margin_single, margin_hier])
        
        # 4. Acuerdo entre modelos
        agreement = 1.0 if np.argmax(probs_single) == np.argmax(probs_hier) else 0.0
        meta_features.append(agreement)
        
        # 5. Distancia KL entre distribuciones
        kl_divergence = np.sum(probs_single * np.log((probs_single + 1e-8) / (probs_hier + 1e-8)))
        meta_features.append(kl_divergence)
        
        return np.array(meta_features)
    
    def hard_voting(self, pred_single: int, pred_hier: int, 
                   confidence_single: float, confidence_hier: float) -> int:
        """
        Voting duro: cada modelo vota por una clase
        """
        if pred_single == pred_hier:
            return pred_single
        
        # En caso de desacuerdo, usar el modelo con mayor confianza
        if confidence_single > confidence_hier:
            return pred_single
        else:
            return pred_hier
    
    def soft_voting(self, probs_single: np.ndarray, probs_hier: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Voting suave: promedia las probabilidades
        """
        # Promedio simple
        avg_probs = (probs_single + probs_hier) / 2
        
        # Calibrar si está habilitado
        if self.config.use_calibration:
            avg_probs = self.calibrate_probabilities(avg_probs.reshape(1, -1)).flatten()
        
        return np.argmax(avg_probs), avg_probs
    
    def weighted_voting(self, probs_single: np.ndarray, probs_hier: np.ndarray,
                       weight_single: float = None, weight_hier: float = None) -> Tuple[int, np.ndarray]:
        """
        Voting ponderado: promedio ponderado de probabilidades
        """
        w1 = weight_single or self.config.single_model_weight
        w2 = weight_hier or self.config.hierarchical_weight
        
        # Normalizar pesos
        total_weight = w1 + w2
        w1, w2 = w1 / total_weight, w2 / total_weight
        
        # Promedio ponderado
        weighted_probs = w1 * probs_single + w2 * probs_hier
        
        return np.argmax(weighted_probs), weighted_probs
    
    def weighted_confidence_voting(self, probs_single: np.ndarray, probs_hier: np.ndarray,
                                  confidence_single: float, confidence_hier: float,
                                  meta_features: np.ndarray = None) -> Tuple[int, np.ndarray]:
        """
        Voting ponderado por confianza: usa la confianza como peso
        """
        # Pesos base basados en confianza
        w1 = confidence_single ** 2  # Cuadrático para enfatizar alta confianza
        w2 = confidence_hier ** 2
        
        # Ajuste adaptativo basado en meta-características
        if self.config.adaptive_weights and meta_features is not None:
            # Ajustar pesos basado en entropía (menor entropía = mayor peso)
            entropy_factor_single = 1 / (1 + meta_features[2])  # entropy_single
            entropy_factor_hier = 1 / (1 + meta_features[3])    # entropy_hier
            
            w1 *= entropy_factor_single
            w2 *= entropy_factor_hier
            
            # Bonus por acuerdo
            if meta_features[6] > 0.5:  # agreement
                w1 *= 1.1
                w2 *= 1.1
        
        # Aplicar pesos de clase si están disponibles
        if self.class_weights is not None:
            pred_single = np.argmax(probs_single)
            pred_hier = np.argmax(probs_hier)
            w1 *= self.class_weights.get(pred_single, 1.0)
            w2 *= self.class_weights.get(pred_hier, 1.0)
        
        return self.weighted_voting(probs_single, probs_hier, w1, w2)
    
    def cascade_voting(self, probs_single: np.ndarray, probs_hier: np.ndarray,
                      confidence_single: float, confidence_hier: float,
                      general_confidence: float = None) -> Tuple[int, np.ndarray]:
        """
        Voting en cascada: usa diferentes estrategias según el contexto
        """
        # 1. Si ambos modelos están muy seguros y de acuerdo, usar soft voting
        if confidence_single > 0.9 and confidence_hier > 0.9:
            if np.argmax(probs_single) == np.argmax(probs_hier):
                return self.soft_voting(probs_single, probs_hier)
        
        # 2. Si el modelo jerárquico tiene problemas de enrutamiento, preferir el único
        if general_confidence is not None and general_confidence < 0.85:
            # Bajo confidence en enrutamiento, dar más peso al modelo único
            return self.weighted_voting(probs_single, probs_hier, 0.7, 0.3)
        
        # 3. Si hay mucho desacuerdo, usar meta-características
        kl_div = np.sum(probs_single * np.log((probs_single + 1e-8) / (probs_hier + 1e-8)))
        if kl_div > self.config.disagreement_threshold:
            # Alto desacuerdo, usar weighted confidence
            meta_features = self.extract_meta_features(
                probs_single, probs_hier, confidence_single, confidence_hier
            )
            return self.weighted_confidence_voting(
                probs_single, probs_hier, confidence_single, confidence_hier, meta_features
            )
        
        # 4. Caso por defecto: weighted voting estándar
        return self.weighted_voting(probs_single, probs_hier)
    
    def predict(self, probs_single: np.ndarray, probs_hier: np.ndarray,
                pred_single: int = None, pred_hier: int = None,
                confidence_single: float = None, confidence_hier: float = None,
                general_confidence: float = None) -> Dict:
        """
        Realiza la predicción del ensemble
        
        Args:
            probs_single: Probabilidades del modelo único [n_classes]
            probs_hier: Probabilidades de la arquitectura jerárquica [n_classes]
            pred_single: Predicción del modelo único (opcional)
            pred_hier: Predicción de la arquitectura jerárquica (opcional)
            confidence_single: Confianza del modelo único
            confidence_hier: Confianza de la arquitectura jerárquica
            general_confidence: Confianza del modelo general en la arquitectura jerárquica
            
        Returns:
            Dict con predicción, probabilidades y metadata
        """
        # Calcular predicciones si no se proporcionaron
        if pred_single is None:
            pred_single = np.argmax(probs_single)
        if pred_hier is None:
            pred_hier = np.argmax(probs_hier)
        
        # Calcular confianzas si no se proporcionaron
        if confidence_single is None:
            confidence_single = np.max(probs_single)
        if confidence_hier is None:
            confidence_hier = np.max(probs_hier)
        
        # Extraer meta-características
        meta_features = None
        if self.config.use_meta_features:
            meta_features = self.extract_meta_features(
                probs_single, probs_hier, confidence_single, confidence_hier
            )
        
        # Aplicar estrategia de voting
        if self.config.voting_strategy == 'hard':
            final_pred = self.hard_voting(pred_single, pred_hier, confidence_single, confidence_hier)
            final_probs = (probs_single + probs_hier) / 2  # Para consistencia
            
        elif self.config.voting_strategy == 'soft':
            final_pred, final_probs = self.soft_voting(probs_single, probs_hier)
            
        elif self.config.voting_strategy == 'weighted':
            final_pred, final_probs = self.weighted_voting(probs_single, probs_hier)
            
        elif self.config.voting_strategy == 'weighted_confidence':
            final_pred, final_probs = self.weighted_confidence_voting(
                probs_single, probs_hier, confidence_single, confidence_hier, meta_features
            )
            
        elif self.config.voting_strategy == 'cascade':
            final_pred, final_probs = self.cascade_voting(
                probs_single, probs_hier, confidence_single, confidence_hier, general_confidence
            )
        else:
            raise ValueError(f"Estrategia de voting no reconocida: {self.config.voting_strategy}")
        
        # Calcular confianza final
        final_confidence = np.max(final_probs)
        
        # Determinar si los modelos están de acuerdo
        agreement = pred_single == pred_hier
        
        # Guardar en historial para análisis
        self.performance_history.append({
            'pred_single': pred_single,
            'pred_hier': pred_hier,
            'pred_ensemble': final_pred,
            'confidence_single': confidence_single,
            'confidence_hier': confidence_hier,
            'confidence_ensemble': final_confidence,
            'agreement': agreement
        })
        
        return {
            'prediction': final_pred,
            'probabilities': final_probs,
            'confidence': final_confidence,
            'agreement': agreement,
            'meta_features': meta_features,
            'voting_weights': {
                'single': confidence_single if self.config.voting_strategy == 'weighted_confidence' else self.config.single_model_weight,
                'hierarchical': confidence_hier if self.config.voting_strategy == 'weighted_confidence' else self.config.hierarchical_weight
            }
        }
    
    def batch_predict(self, results_single: Dict, results_hier: Dict, 
                     general_confidence: np.ndarray = None) -> Dict:
        """
        Realiza predicciones en batch para todo el conjunto de test
        
        Args:
            results_single: Resultados del modelo único
            results_hier: Resultados de la arquitectura jerárquica
            general_confidence: Array de confianzas del modelo general
            
        Returns:
            Dict con todas las predicciones y métricas
        """
        n_samples = len(results_single['predictions'])
        ensemble_preds = np.zeros(n_samples, dtype=int)
        ensemble_probs = np.zeros((n_samples, 4))  # 4 clases
        ensemble_confidence = np.zeros(n_samples)
        agreements = np.zeros(n_samples, dtype=bool)
        
        # Procesar cada muestra
        for i in range(n_samples):
            # Preparar inputs
            probs_single = results_single['probabilities'][i]
            probs_hier = results_hier['probabilities'][i]
            
            # Confianza del modelo general si está disponible
            gen_conf = general_confidence[i] if general_confidence is not None else None
            
            # Predecir con ensemble
            result = self.predict(
                probs_single=probs_single,
                probs_hier=probs_hier,
                general_confidence=gen_conf
            )
            
            ensemble_preds[i] = result['prediction']
            ensemble_probs[i] = result['probabilities']
            ensemble_confidence[i] = result['confidence']
            agreements[i] = result['agreement']
        
        return {
            'predictions': ensemble_preds,
            'probabilities': ensemble_probs,
            'confidence': ensemble_confidence,
            'agreements': agreements,
            'agreement_rate': np.mean(agreements),
            'avg_confidence': np.mean(ensemble_confidence)
        }
    
    def analyze_performance(self, y_true: np.ndarray, results_ensemble: Dict,
                          results_single: Dict, results_hier: Dict) -> Dict:
        """
        Analiza el rendimiento del ensemble comparado con modelos individuales
        """
        # Calcular métricas para cada modelo
        acc_single = accuracy_score(y_true, results_single['predictions'])
        acc_hier = accuracy_score(y_true, results_hier['predictions'])
        acc_ensemble = accuracy_score(y_true, results_ensemble['predictions'])
        
        # Análisis de mejora
        improvement_over_single = (acc_ensemble - acc_single) * 100
        improvement_over_hier = (acc_ensemble - acc_hier) * 100
        improvement_over_best = (acc_ensemble - max(acc_single, acc_hier)) * 100
        
        # Análisis por tipo de muestra
        agreements = results_ensemble['agreements']
        
        # Precisión cuando los modelos están de acuerdo
        agree_mask = agreements
        acc_when_agree = accuracy_score(
            y_true[agree_mask], 
            results_ensemble['predictions'][agree_mask]
        ) if agree_mask.sum() > 0 else 0
        
        # Precisión cuando los modelos discrepan
        disagree_mask = ~agreements
        acc_when_disagree = accuracy_score(
            y_true[disagree_mask], 
            results_ensemble['predictions'][disagree_mask]
        ) if disagree_mask.sum() > 0 else 0
        
        # Análisis de confianza
        high_conf_mask = results_ensemble['confidence'] > self.config.confidence_threshold
        acc_high_conf = accuracy_score(
            y_true[high_conf_mask], 
            results_ensemble['predictions'][high_conf_mask]
        ) if high_conf_mask.sum() > 0 else 0
        
        return {
            'accuracy': {
                'single': acc_single,
                'hierarchical': acc_hier,
                'ensemble': acc_ensemble
            },
            'improvements': {
                'over_single': improvement_over_single,
                'over_hierarchical': improvement_over_hier,
                'over_best_individual': improvement_over_best
            },
            'agreement_analysis': {
                'agreement_rate': results_ensemble['agreement_rate'],
                'accuracy_when_agree': acc_when_agree,
                'accuracy_when_disagree': acc_when_disagree,
                'samples_agree': agree_mask.sum(),
                'samples_disagree': disagree_mask.sum()
            },
            'confidence_analysis': {
                'avg_confidence': results_ensemble['avg_confidence'],
                'high_confidence_samples': high_conf_mask.sum(),
                'high_confidence_accuracy': acc_high_conf
            }
        }


def create_ensemble_visualizations(y_true: np.ndarray, results_ensemble: Dict,
                                 results_single: Dict, results_hier: Dict,
                                 analysis: Dict, output_dir: str):
    """
    Crea visualizaciones comprehensivas del ensemble
    """
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Comparación de Accuracies
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfico de barras de accuracy
    models = ['Single', 'Hierarchical', 'Ensemble']
    accuracies = [
        analysis['accuracy']['single'],
        analysis['accuracy']['hierarchical'],
        analysis['accuracy']['ensemble']
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.8, 1.0)
    
    # Agregar valores en las barras
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Análisis de Acuerdo/Desacuerdo
    agreement_data = [
        analysis['agreement_analysis']['accuracy_when_agree'],
        analysis['agreement_analysis']['accuracy_when_disagree'],
        analysis['accuracy']['ensemble']
    ]
    agreement_labels = ['When Agree', 'When Disagree', 'Overall']
    
    bars2 = ax2.bar(agreement_labels, agreement_data, 
                    color=['#27ae60', '#e67e22', '#9b59b6'],
                    edgecolor='black', linewidth=2)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Ensemble Performance by Agreement', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.7, 1.0)
    
    # Agregar conteo de muestras
    sample_counts = [
        analysis['agreement_analysis']['samples_agree'],
        analysis['agreement_analysis']['samples_disagree'],
        len(y_true)
    ]
    
    for bar, acc, count in zip(bars2, agreement_data, sample_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.4f}\n(n={count})', ha='center', va='bottom', fontweight='bold')
    
    # 3. Distribución de Confianza
    confidence_ensemble = results_ensemble['confidence']
    
    ax3.hist(confidence_ensemble, bins=30, color='#3498db', alpha=0.7, 
             edgecolor='black', linewidth=1.5)
    ax3.axvline(analysis['confidence_analysis']['avg_confidence'], 
                color='red', linestyle='--', linewidth=2,
                label=f'Mean: {analysis["confidence_analysis"]["avg_confidence"]:.3f}')
    ax3.axvline(0.85, color='green', linestyle='--', linewidth=2,
                label='Threshold: 0.85')
    ax3.set_xlabel('Confidence Score', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Ensemble Confidence Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    
    # 4. Matriz de Confusión del Ensemble
    cm = confusion_matrix(y_true, results_ensemble['predictions'])
    class_names = ['Cloudy', 'Rainy', 'Snowy', 'Sunny']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax4)
    ax4.set_xlabel('Predicted', fontsize=12)
    ax4.set_ylabel('True', fontsize=12)
    ax4.set_title('Ensemble Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ensemble_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Gráfico de Mejora Detallado
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    improvements = analysis['improvements']
    imp_labels = ['vs Single Model', 'vs Hierarchical', 'vs Best Individual']
    imp_values = [
        improvements['over_single'],
        improvements['over_hierarchical'],
        improvements['over_best_individual']
    ]
    
    # Colores basados en si es mejora o no
    colors = ['green' if v > 0 else 'red' for v in imp_values]
    
    bars = ax.bar(imp_labels, imp_values, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=2)
    
    # Línea en y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Ensemble Performance Improvement', fontsize=14, fontweight='bold')
    
    # Agregar valores
    for bar, val in zip(bars, imp_values):
        y_pos = bar.get_height() + 0.1 if val > 0 else bar.get_height() - 0.3
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:+.2f}%', ha='center', va='bottom' if val > 0 else 'top',
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ensemble_improvements.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualizaciones guardadas en {output_dir}/")


def generate_ensemble_report(analysis: Dict, config: EnsembleConfig, output_path: str):
    """
    Genera un reporte detallado del ensemble
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE ENSEMBLE VOTING\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Estrategia de Voting: {config.voting_strategy}\n")
        f.write(f"Fecha: {np.datetime64('now')}\n\n")
        
        f.write("RESUMEN DE RESULTADOS\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy del Modelo Único:        {analysis['accuracy']['single']:.4f}\n")
        f.write(f"Accuracy de Arquitectura Jerárquica: {analysis['accuracy']['hierarchical']:.4f}\n")
        f.write(f"Accuracy del Ensemble:            {analysis['accuracy']['ensemble']:.4f} ⭐\n\n")
        
        f.write("ANÁLISIS DE MEJORA\n")
        f.write("-"*40 + "\n")
        f.write(f"Mejora sobre Modelo Único:        {analysis['improvements']['over_single']:+.2f}%\n")
        f.write(f"Mejora sobre Arquitectura Jerárquica: {analysis['improvements']['over_hierarchical']:+.2f}%\n")
        f.write(f"Mejora sobre Mejor Individual:    {analysis['improvements']['over_best_individual']:+.2f}%\n\n")
        
        f.write("ANÁLISIS DE ACUERDO ENTRE MODELOS\n")
        f.write("-"*40 + "\n")
        f.write(f"Tasa de Acuerdo:                  {analysis['agreement_analysis']['agreement_rate']:.2%}\n")
        f.write(f"Accuracy cuando están de acuerdo: {analysis['agreement_analysis']['accuracy_when_agree']:.4f}\n")
        f.write(f"Accuracy cuando discrepan:        {analysis['agreement_analysis']['accuracy_when_disagree']:.4f}\n")
        f.write(f"Muestras con acuerdo:             {analysis['agreement_analysis']['samples_agree']}\n")
        f.write(f"Muestras con desacuerdo:          {analysis['agreement_analysis']['samples_disagree']}\n\n")
        
        f.write("ANÁLISIS DE CONFIANZA\n")
        f.write("-"*40 + "\n")
        f.write(f"Confianza Promedio:               {analysis['confidence_analysis']['avg_confidence']:.4f}\n")
        f.write(f"Muestras con Alta Confianza:      {analysis['confidence_analysis']['high_confidence_samples']}\n")
        f.write(f"Accuracy en Alta Confianza:       {analysis['confidence_analysis']['high_confidence_accuracy']:.4f}\n\n")
        
        f.write("CONFIGURACIÓN DEL ENSEMBLE\n")
        f.write("-"*40 + "\n")
        f.write(f"Estrategia: {config.voting_strategy}\n")
        if config.voting_strategy == 'weighted':
            f.write(f"Peso Modelo Único: {config.single_model_weight}\n")
            f.write(f"Peso Arquitectura Jerárquica: {config.hierarchical_weight}\n")
        f.write(f"Umbral de Confianza: {config.confidence_threshold}\n")
        f.write(f"Usar Calibración: {config.use_calibration}\n")
        f.write(f"Usar Meta-features: {config.use_meta_features}\n")
        f.write(f"Pesos Adaptativos: {config.adaptive_weights}\n\n")
        
        f.write("CONCLUSIONES\n")
        f.write("-"*40 + "\n")
        
        if analysis['improvements']['over_best_individual'] > 0:
            f.write("✅ El ensemble MEJORA el rendimiento sobre los modelos individuales\n")
            f.write(f"   Ganancia: {analysis['improvements']['over_best_individual']:.2f}%\n\n")
        else:
            f.write("❌ El ensemble NO mejora sobre el mejor modelo individual\n\n")
        
        if analysis['agreement_analysis']['accuracy_when_disagree'] > 0.8:
            f.write("✅ El ensemble maneja bien los casos de desacuerdo\n")
        else:
            f.write("⚠️  Hay margen de mejora en el manejo de desacuerdos\n")
        
        if analysis['confidence_analysis']['avg_confidence'] > 0.9:
            f.write("✅ El ensemble muestra alta confianza en sus predicciones\n")
        else:
            f.write("⚠️  La confianza promedio podría ser mayor\n")


if __name__ == "__main__":
    print("Módulo de Voting Ensemble cargado exitosamente")
    print("\nEstrategias disponibles:")
    print("- hard: Voting duro basado en predicciones")
    print("- soft: Promedio de probabilidades")
    print("- weighted: Promedio ponderado con pesos fijos")
    print("- weighted_confidence: Ponderado por confianza dinámica")
    print("- cascade: Estrategia adaptativa en cascada")