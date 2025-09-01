# 🗳️ Arquitectura de Voting Ensemble

## 📋 Descripción

Esta arquitectura implementa un **Voting Ensemble** que combina las predicciones de:
1. **Modelo Único**: Predice directamente las 4 clases meteorológicas
2. **Arquitectura Jerárquica**: Sistema de 3 modelos (1 general + 2 especialistas)

El ensemble utiliza múltiples estrategias de voting para maximizar el rendimiento combinando las fortalezas de ambas arquitecturas.

## 🎯 Estrategias de Voting

### 1. **Hard Voting**
- Cada modelo vota por una clase
- En caso de desacuerdo, gana el modelo con mayor confianza
- Rápido pero menos sofisticado

### 2. **Soft Voting**
- Promedia las probabilidades de ambos modelos
- Considera la distribución completa de probabilidades
- Puede aplicar calibración de probabilidades

### 3. **Weighted Voting**
- Promedio ponderado con pesos fijos
- Permite dar más importancia a un modelo sobre otro
- Pesos configurables (por defecto 50%-50%)

### 4. **Weighted Confidence Voting** ⭐
- Pesos dinámicos basados en la confianza de cada predicción
- Considera meta-características (entropía, margen, acuerdo)
- Se adapta a cada muestra individual

### 5. **Cascade Voting** (Recomendado)
- Estrategia adaptativa que selecciona el método según el contexto:
  - Alta confianza + acuerdo → Soft voting
  - Problemas de enrutamiento → Mayor peso al modelo único
  - Alto desacuerdo → Weighted confidence voting
  - Caso general → Weighted voting estándar

## 🚀 Cómo Usar

### Ejecución Rápida (Windows)
```powershell
# Evaluar todas las estrategias con modelos corregidos
.\run_ensemble.bat
# Seleccionar: s (para modelos corregidos)
# Estrategia: all
```

### Ejecución con Python
```bash
# Evaluar todas las estrategias (recomendado)
python evaluate_ensemble.py --evaluate-all --use-corrected

# Usar una estrategia específica
python evaluate_ensemble.py --strategy cascade --use-corrected

# Con modelos originales
python evaluate_ensemble.py --strategy weighted_confidence
```

### Script Rápido
```bash
# Evalúa todas las estrategias automáticamente
python run_ensemble_quick.py --corrected
```

## 📊 Resultados Esperados

### Sin Ensemble:
- Modelo Único: ~90.5% accuracy
- Arquitectura Jerárquica (original): ~82.6% accuracy
- Arquitectura Jerárquica (corregida): ~92-93% accuracy

### Con Voting Ensemble:
- **Hard Voting**: ~91-92% accuracy
- **Soft Voting**: ~92-93% accuracy
- **Weighted Voting**: ~92-93% accuracy
- **Weighted Confidence**: ~93-94% accuracy ⭐
- **Cascade**: ~93-94% accuracy ⭐

## 🔧 Configuración Avanzada

### Parámetros del Ensemble

```python
ensemble_config = EnsembleConfig(
    # Estrategia principal
    voting_strategy='cascade',
    
    # Pesos para weighted voting
    single_model_weight=0.5,
    hierarchical_weight=0.5,
    
    # Umbrales
    confidence_threshold=0.85,  # Para análisis de alta confianza
    disagreement_threshold=0.3,  # Para detectar alto desacuerdo
    
    # Opciones avanzadas
    use_calibration=True,      # Calibrar probabilidades
    use_meta_features=True,    # Usar características adicionales
    adaptive_weights=True      # Ajustar pesos dinámicamente
)
```

### Meta-Características Utilizadas

1. **Niveles de confianza**: Máxima probabilidad de cada modelo
2. **Entropía**: Incertidumbre en las distribuciones
3. **Margen**: Diferencia entre las top 2 clases
4. **Acuerdo**: Si los modelos predicen la misma clase
5. **Divergencia KL**: Diferencia entre distribuciones

## 📈 Análisis de Resultados

### Métricas Generadas

1. **Accuracy Comparativo**
   - Modelo único vs Jerárquico vs Ensemble
   - Mejora porcentual sobre cada modelo

2. **Análisis de Acuerdo**
   - Tasa de acuerdo entre modelos
   - Performance cuando acuerdan vs discrepan
   - Manejo de casos difíciles

3. **Análisis de Confianza**
   - Distribución de scores de confianza
   - Accuracy en muestras de alta confianza
   - Calibración de probabilidades

4. **Comparación de Estrategias**
   - Performance de cada estrategia
   - Selección automática de la mejor

## 📁 Archivos de Salida

```
outputs/ensemble_results/
├── ensemble_analysis.png          # Análisis visual completo
├── ensemble_improvements.png      # Gráficos de mejora
├── strategy_comparison.png        # Comparación de estrategias
├── confusion_matrices_*.png       # Matrices de confusión
├── ensemble_report.txt           # Reporte detallado
├── ensemble_config.json          # Configuración utilizada
└── all_strategies_results.json   # Resultados de todas las estrategias
```

## 💡 Recomendaciones

1. **Para Mejores Resultados**:
   - Usar modelos especialistas corregidos (`--use-corrected`)
   - Evaluar todas las estrategias primero (`--evaluate-all`)
   - Usar cascade o weighted_confidence para producción

2. **Optimización**:
   - Ajustar pesos según el dominio específico
   - Calibrar umbrales basándose en datos de validación
   - Considerar ensemble de más de 2 arquitecturas

3. **Interpretabilidad**:
   - El reporte incluye análisis de casos de acuerdo/desacuerdo
   - Las visualizaciones muestran dónde el ensemble mejora
   - Los meta-features ayudan a entender las decisiones

## 🔬 Detalles Técnicos

### Por Qué Funciona el Ensemble

1. **Complementariedad**: Los modelos tienen diferentes fortalezas
   - Modelo único: Visión global, sin errores de enrutamiento
   - Jerárquico: Especialización, mejor en clases específicas

2. **Reducción de Varianza**: Promedia errores aleatorios

3. **Robustez**: Menos sensible a muestras difíciles

4. **Adaptabilidad**: Las estrategias avanzadas se ajustan por muestra

### Limitaciones

- Requiere más tiempo de inferencia (2x)
- Necesita ambos modelos entrenados
- La mejora depende de la complementariedad de los modelos

## 🚀 Próximos Pasos

1. **Stacking**: Entrenar un meta-modelo sobre las predicciones
2. **Boosting**: Entrenar modelos secuencialmente
3. **Multi-nivel**: Ensemble de múltiples arquitecturas
4. **Optimización Bayesiana**: Encontrar pesos óptimos automáticamente