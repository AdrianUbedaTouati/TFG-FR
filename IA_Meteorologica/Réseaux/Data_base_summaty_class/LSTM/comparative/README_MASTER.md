# 🌦️ Sistema de Comparación de Arquitecturas de Clasificación Meteorológica

## 📋 Descripción General

Este sistema compara tres arquitecturas de redes neuronales para clasificación meteorológica:

1. **Modelo Único**: Clasifica directamente 4 tipos de clima (Cloudy, Rainy, Snowy, Sunny)
2. **Arquitectura Jerárquica**: 3 modelos especializados (1 general + 2 especialistas)
3. **Voting Ensemble**: Combina las predicciones de ambas arquitecturas

## 🏗️ Arquitecturas Implementadas

### 1. Modelo Único (Baseline)
- **Archivo**: Usa modelo preentrenado en `../summary_4/`
- **Accuracy**: ~90.5%
- **Ventajas**: Simple, sin errores de enrutamiento
- **Desventajas**: No aprovecha especialización

### 2. Arquitectura Jerárquica
- **Modelo General**: Clasifica Cloudy_Sunny vs Rainy_Snowy
- **Especialista A**: Distingue entre Cloudy y Sunny
- **Especialista B**: Distingue entre Rainy y Snowy
- **Accuracy Original**: ~82.6% (problema: especialistas no manejaban muestras fuera de dominio)
- **Accuracy Corregida**: ~91.6% (especialistas reentrenados con clase "Other")

### 3. Voting Ensemble
- **Combina**: Modelo único + Arquitectura jerárquica
- **Estrategias**: Hard, Soft, Weighted, Weighted Confidence, Cascade
- **Accuracy**: ~91.6-93% (depende de la estrategia)

## 🚀 Guía de Uso Rápido

### Prerrequisitos
```powershell
# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Navegar al directorio
cd comparative
```

### 1. Comparación Básica (Modelos Originales)
```powershell
python run_comparison.py
```

### 2. Corregir y Reentrenar Especialistas
```powershell
# Proceso completo: preparar datos, reentrenar, evaluar
.\run_retraining.bat
```

### 3. Comparación con Modelos Corregidos
```powershell
.\run_comparison_corrected.bat
# O directamente:
python run_comparison_updated.py --use-corrected
```

### 4. Evaluar Voting Ensemble
```powershell
.\run_ensemble.bat
# Seleccionar:
# - Modelos corregidos: s
# - Estrategia: all (evalúa todas)
```

## 📁 Estructura de Archivos

### Scripts Principales
```
comparative/
├── evaluate_architectures.py          # Sistema base de evaluación
├── prepare_corrected_datasets.py      # Prepara datasets con clase "Other"
├── retrain_specialists.py             # Reentrena especialistas
├── evaluate_architectures_corrected.py # Evaluación con modelos corregidos
├── ensemble_voting_architecture.py     # Implementación del voting ensemble
└── evaluate_ensemble.py               # Evaluación del ensemble
```

### Scripts de Ejecución
```
├── run_comparison.py                  # Comparación básica
├── run_comparison_updated.py          # Comparación con opciones
├── run_comparison_corrected.py        # Comparación con modelos corregidos
├── run_retraining.py                  # Script de reentrenamiento
├── run_retraining.bat                 # Batch para reentrenamiento
├── run_comparison_corrected.bat       # Batch para comparación corregida
└── run_ensemble.bat                   # Batch para ensemble
```

### Documentación
```
├── README.md                          # Documentación original
├── README_SOLUCION.md                 # Explicación de la corrección
├── README_ENSEMBLE.md                 # Documentación del ensemble
└── README_MASTER.md                   # Este archivo
```

## 📊 Resultados Esperados

| Arquitectura | Sin Corrección | Con Corrección |
|--------------|----------------|----------------|
| Modelo Único | 90.5% | 90.5% |
| Jerárquica | 82.6% | 91.6% |
| Ensemble | - | 91.6-93% |

## 🔧 Solución del Problema Original

**Problema**: Los especialistas fueron entrenados solo con sus clases específicas, fallando cuando el modelo general los enrutaba incorrectamente.

**Solución**: 
1. Reentrenar especialistas con TODAS las clases
2. Mapear clases fuera de dominio a "Other"
3. Usar pesos de clase (1.0 para objetivo, 0.3 para "Other")

## 📈 Flujo de Trabajo Recomendado

1. **Evaluación Inicial**
   ```powershell
   python run_comparison.py
   ```

2. **Si la jerárquica underperforma** (< modelo único)
   ```powershell
   .\run_retraining.bat
   ```

3. **Evaluación con Modelos Corregidos**
   ```powershell
   python run_comparison_updated.py --use-corrected
   ```

4. **Probar Ensemble para Mejor Performance**
   ```powershell
   python evaluate_ensemble.py --evaluate-all --use-corrected
   ```

## 📂 Salidas Generadas

```
outputs/
├── comparison_results/           # Comparación básica
├── comparison_results_corrected/ # Comparación con modelos corregidos
├── retrained_a/                 # Especialista A reentrenado
├── retrained_b/                 # Especialista B reentrenado
└── ensemble_results/            # Resultados del ensemble
```

## 🎯 Mejores Prácticas

1. **Siempre verificar** que los modelos corregidos existen antes de usarlos
2. **Usar ensemble** cuando se necesite máxima precisión
3. **Estrategia "cascade"** o "weighted_confidence" suelen dar mejores resultados
4. **Documentar** qué modelos se usaron en cada experimento

## ⚡ Comandos Rápidos

```powershell
# Ver ayuda
python evaluate_ensemble.py --help

# Ensemble con todas las estrategias
python evaluate_ensemble.py --evaluate-all --use-corrected

# Comparación rápida
python run_comparison_updated.py --use-corrected

# Solo reentrenar sin evaluar
python retrain_specialists.py
```

## 🛠️ Solución de Problemas

1. **ModuleNotFoundError**: Activar entorno virtual
2. **FileNotFoundError**: Ejecutar desde directorio `comparative/`
3. **Modelos no encontrados**: Ejecutar `run_retraining.bat` primero
4. **GPU no disponible**: El sistema funciona en CPU automáticamente

## 📚 Para Más Información

- **Detalles de la solución**: Ver `README_SOLUCION.md`
- **Detalles del ensemble**: Ver `README_ENSEMBLE.md`
- **Arquitectura original**: Ver `README.md`