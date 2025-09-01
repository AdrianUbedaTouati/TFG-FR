# 🔧 SOLUCIÓN: Corrección de Modelos Especialistas

## 📋 Resumen del Problema

Los modelos especialistas en la arquitectura jerárquica fueron entrenados **incorrectamente**:
- **Especialista A**: Solo entrenado con muestras Cloudy/Sunny
- **Especialista B**: Solo entrenado con muestras Rainy/Snowy

Cuando el modelo general comete errores de enrutamiento (~6.5%), envía muestras que los especialistas **nunca vieron durante el entrenamiento**, causando predicciones impredecibles y degradando el rendimiento.

## ✅ Solución Implementada

### 1. **prepare_corrected_datasets.py**
Prepara datasets corregidos donde los especialistas ven TODAS las clases:
- Clases objetivo mantienen sus etiquetas originales
- Clases fuera de dominio se mapean a una nueva clase "Other"
- Se balancea el dataset para evitar que "Other" domine

### 2. **retrain_specialists.py**
Reentrena los especialistas con:
- Arquitectura LSTM mejorada con 3 clases de salida
- Pesos de clase ajustados (1.0 para objetivo, 0.3 para "Other")
- Weighted sampling para balance durante entrenamiento
- Monitoreo separado de precisión en clases objetivo vs "Other"

### 3. **evaluate_architectures_corrected.py**
Evalúa y compara:
- Modelo único (baseline)
- Arquitectura jerárquica original
- Arquitectura jerárquica con modelos corregidos

### 4. **run_retraining.py**
Script maestro que ejecuta todo el proceso automáticamente.

## 🚀 Cómo Usar

### Prerrequisitos
1. Activar el entorno virtual:
   ```powershell
   # En Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```

2. Instalar dependencias (si es necesario):
   ```bash
   pip install -r requirements_retrain.txt
   ```

### Opción 1: Ejecución Completa en Windows (Recomendado)
```powershell
# Desde PowerShell con el entorno virtual activado
.\run_retraining.bat
```

### Opción 2: Ejecución Completa con Python
```bash
python run_retraining.py
```

### Opción 3: Ejecución Manual
```bash
# Paso 1: Preparar datasets
python prepare_corrected_datasets.py

# Paso 2: Reentrenar especialistas
python retrain_specialists.py

# Paso 3: Evaluar resultados
python evaluate_architectures_corrected.py
```

## 📊 Resultados Esperados

### Antes (Arquitectura Original):
- Modelo Único: ~90.5% accuracy
- Arquitectura Jerárquica: ~82.6% accuracy ❌

### Después (Arquitectura Corregida):
- Modelo Único: ~90.5% accuracy
- Arquitectura Jerárquica: ~92-93% accuracy ✅

## 📁 Estructura de Salidas

```
comparative/
├── data_corrected/                    # Datasets corregidos
│   ├── weather_classification_CloudySunny_corrected.csv
│   ├── weather_classification_RainySnowy_corrected.csv
│   └── training_config.json
│
├── outputs/
│   ├── retrained_a/                   # Especialista A reentrenado
│   │   ├── a_corrected.pt
│   │   ├── training_history.png
│   │   └── confusion_matrix.png
│   │
│   ├── retrained_b/                   # Especialista B reentrenado
│   │   ├── b_corrected.pt
│   │   ├── training_history.png
│   │   └── confusion_matrix.png
│   │
│   └── comparison_corrected/          # Evaluación comparativa
│       ├── comparison_report.txt
│       ├── confusion_matrices.png
│       └── metrics_comparison.png
│
└── logs/                              # Logs de entrenamiento
    ├── retrain_a_*.log
    └── retrain_b_*.log
```

## 🔍 Verificación de Resultados

1. **Revisar reporte principal**:
   ```
   outputs/comparison_corrected/comparison_report.txt
   ```

2. **Verificar métricas de especialistas**:
   - Los especialistas deben tener >90% accuracy en clases objetivo
   - Deben detectar correctamente muestras "Other"

3. **Analizar matrices de confusión**:
   - Verificar que los errores están distribuidos uniformemente
   - No debe haber patrones sistemáticos de error

## 🛠️ Ajustes Adicionales

Si los resultados no son satisfactorios:

1. **Ajustar pesos de clase** en `data_corrected/training_config.json`
2. **Modificar hiperparámetros** en `retrain_specialists.py`:
   - `num_epochs`: Aumentar para más entrenamiento
   - `learning_rate`: Reducir si hay overfitting
   - `hidden_size`: Aumentar para mayor capacidad

3. **Implementar ensemble** combinando predicciones del modelo único y jerárquico

## 📈 Próximos Pasos

1. **Mejorar modelo general**: Objetivo >98% accuracy para minimizar errores de enrutamiento
2. **Implementar rechazo por confianza**: Rechazar predicciones con baja confianza
3. **Ensemble voting**: Combinar predicciones de múltiples arquitecturas
4. **Optimización de hiperparámetros**: Grid search para encontrar configuración óptima

## ⚠️ Notas Importantes

- El reentrenamiento puede tomar 10-20 minutos dependiendo del hardware
- Se requiere GPU para acelerar el proceso (opcional pero recomendado)
- Los modelos originales NO se modifican, se crean nuevos modelos corregidos
- Todos los resultados se guardan con timestamps para trazabilidad