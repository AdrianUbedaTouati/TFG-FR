# Sistema de Comparación de Arquitecturas - Versión Mejorada

Este sistema compara dos enfoques para la clasificación meteorológica:

1. **Modelo Único**: Un solo modelo LSTM que clasifica directamente en 4 clases (Cloudy, Rainy, Snowy, Sunny)
2. **Arquitectura Jerárquica**: Sistema de 3 modelos:
   - Modelo general: Clasifica en grupos (Cloudy_Sunny vs Rainy_Snowy)
   - Especialista A: Refina predicciones para Cloudy vs Sunny
   - Especialista B: Refina predicciones para Rainy vs Snowy

## Mejoras implementadas

- ✅ Carga automática de modelos desde sus ubicaciones originales
- ✅ Manejo correcto de mapeo de clases usando los archivos `class_index.json`
- ✅ Visualizaciones mejoradas y más informativas
- ✅ Análisis detallado de errores
- ✅ Métricas comparativas completas
- ✅ Reporte detallado en JSON y texto

## Estructura de archivos

```
comparative_improved/
├── evaluate_architectures.py   # Script principal de evaluación
├── run_comparison.py          # Script para ejecutar fácilmente
├── README.md                  # Este archivo
└── outputs/
    └── comparison_results/    # Resultados generados
        ├── confusion_matrices_comparison.png
        ├── confusion_matrices_normalized.png
        ├── metrics_comparison.png
        ├── per_class_metrics_comparison.png
        ├── error_distribution.png
        ├── detailed_report.json
        └── summary_report.txt
```

## Uso

### Ejecución simple:

```bash
cd comparative_improved
python run_comparison.py
```

### Ejecución con configuración personalizada:

Edita las rutas de los modelos en `evaluate_architectures.py` en la clase `EvalConfig`:

```python
SINGLE_MODEL_PATH: str = "ruta/a/tu/modelo_4_clases.pt"
MODEL_GENERAL_PATH: str = "ruta/a/tu/modelo_general.pt"
MODEL_SPEC_A_PATH: str = "ruta/a/tu/modelo_cloudy_sunny.pt"
MODEL_SPEC_B_PATH: str = "ruta/a/tu/modelo_rainy_snowy.pt"
```

## Mapeo de clases

El sistema detecta automáticamente el mapeo de clases, pero el esquema predeterminado es:

### Modelo único (4 clases):
- 0: Cloudy
- 1: Rainy
- 2: Snowy
- 3: Sunny

### Modelo general (2 grupos):
- 0: Cloudy_Sunny
- 1: Rainy_Snowy

### Especialista A (Cloudy_Sunny):
- 0: Cloudy
- 1: Sunny

### Especialista B (Rainy_Snowy):
- 0: Rainy
- 1: Snowy

## Interpretación de resultados

### Métricas globales
- **Accuracy**: Porcentaje de predicciones correctas
- **Macro F1**: Promedio no ponderado del F1-score de todas las clases
- **Weighted F1**: Promedio ponderado del F1-score según el soporte de cada clase

### Visualizaciones

1. **Matrices de confusión**: Muestran las predicciones vs realidad para cada modelo
2. **Métricas comparativas**: Gráfico de barras comparando métricas globales
3. **Métricas por clase**: Comparación detallada de precisión, recall y F1 por clase
4. **Análisis de errores**: Distribución de errores únicos y compartidos entre modelos

### Reportes

- `detailed_report.json`: Reporte completo en formato JSON con todas las métricas
- `summary_report.txt`: Resumen ejecutivo en texto plano con análisis de mejoras

## Requisitos

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Notas

- Los modelos deben estar en formato `.pt` (PyTorch checkpoint) o `.ts` (TorchScript)
- El dataset debe tener las mismas características que se usaron para entrenar
- La arquitectura jerárquica generalmente funciona mejor cuando los grupos tienen patrones distintivos