# Estructura y Arquitectura de las Redes Neuronales

Este documento detalla la arquitectura interna de cada tipo de red neuronal implementada en el proyecto meteorológico.

## 1. REDES LSTM (Long Short-Term Memory)

### Arquitectura Base
```
Input → LSTM Layers → Output Head
```

#### Componentes Principales:
- **Capas LSTM**: 2 capas bidireccionales (por defecto)
- **Unidades ocultas**: 256 por capa
- **Dropout**: 0.2 entre capas LSTM
- **Activación**: Tanh (interna LSTM) + GELU (en MLP head opcional)

### Variantes:

#### 1.1 LSTM para Regresión (`LSTMForecaster`)
- **Entrada**: Secuencia temporal de características meteorológicas
- **Salida**: Predicciones de temperatura para H pasos futuros
- **Función de pérdida**: Weighted Huber Loss
- **Head opcional**: MLP con GELU para proyección final

#### 1.2 LSTM para Clasificación (`LSTMClassifier`)
- **Diferencias con regresión**:
  - Head de salida: Linear → Softmax
  - Función de pérdida: Weighted CrossEntropy
  - Salida: Probabilidades para C clases

## 2. N-BEATS (Neural Basis Expansion Analysis)

### Arquitectura Base
```
Input → Stack de Bloques N-BEATS → Agregación → Forecast
```

#### Componentes de cada Bloque:
1. **MLP de entrada**: 
   - Profundidad configurable (default: 2 capas)
   - Ancho: 256 unidades
   - Activación: ReLU

2. **Proyecciones duales**:
   - Backcast: Para reconstruir el pasado
   - Forecast: Para predecir el futuro

3. **Conexiones residuales**: Entre bloques consecutivos

#### Configuración típica:
- **Bloques**: 6 (court) u 8 (long)
- **Hidden width**: 256
- **Depth por bloque**: 2-4 capas

### Diferencias según configuración:
- **Univariable**: Procesa solo temperatura
- **Multivariable**: Procesa 19+ características meteorológicas

## 3. N-HiTS (Neural Hierarchical Interpolation for Time Series)

### Arquitectura Base
```
Input → Multi-Scale Processing → Hierarchical Interpolation → Forecast
```

#### Componentes Principales:

1. **Pooling Multi-escala**:
   - Escalas: [1, 2, 4, 8] (diferentes resoluciones temporales)
   - Tipo: Average pooling

2. **Bloques N-HiTS** (por cada escala):
   - MLP con GELU y Dropout(0.1)
   - Linear head para predicciones gruesas
   - Interpolación a resolución objetivo

3. **Parámetros clave**:
   - Bloques por escala: 2
   - Hidden width: 256
   - Profundidad (D): 1 (univariable) o 21 (multivariable)

### Ventaja arquitectural:
Procesa la señal a múltiples resoluciones simultáneamente, capturando patrones de corto y largo plazo.

## 4. MLP (Multi-Layer Perceptron)

### 4.1 MLP para Clasificación Meteorológica

#### Arquitectura Profunda:
```
Input → [Linear(1536) → LayerNorm → GELU → Dropout] × 6 → Output(3)
```

#### Características especiales:
- **Conexiones residuales**: Cada 2 capas
- **Normalización**: LayerNorm después de cada capa
- **Regularización**: Dropout(0.3) agresivo

### 4.2 MLP para Cloud Cover

#### Arquitectura Flexible:
```
Input → Linear(512) → ReLU → BatchNorm → Linear(256) → ReLU → BatchNorm → Linear(128) → Output(4)
```

#### Diferencias con MLP meteorológico:
- Menos capas (3 vs 6)
- BatchNorm en lugar de LayerNorm
- Sin conexiones residuales
- Arquitectura más simple pero efectiva

## 5. OPTIMIZACIÓN Y ENTRENAMIENTO

### Configuraciones Comunes:

1. **Optimizador**: AdamW
   - Learning rate: 2e-3 a 3e-4
   - Weight decay: 1e-5 a 1e-6

2. **Técnicas de GPU**:
   - Mixed Precision (AMP)
   - TF32 para Ampere+
   - Non-blocking transfers

3. **Regularización**:
   - Early stopping (paciencia: 10-50 épocas)
   - Gradient clipping (norm: 1.0)
   - Learning rate scheduling

4. **Manejo de datos**:
   - Ventanas deslizantes para series temporales
   - Normalización Min-Max o Standard
   - Class weighting para desbalance

## 6. DIFERENCIAS CLAVE ENTRE ARQUITECTURAS

### LSTM vs N-BEATS/N-HiTS:
- LSTM: Procesamiento secuencial con memoria
- N-BEATS/N-HiTS: Procesamiento paralelo con bloques especializados

### Univariable vs Multivariable:
- **Univariable**: Arquitectura más simple, menos parámetros
- **Multivariable**: Mayor profundidad (D=21 vs D=1 en N-HiTS)

### Court vs Long:
- Principalmente difieren en configuración, no en arquitectura
- Long puede usar más bloques o mayor profundidad

## 7. SELECCIÓN DE ARQUITECTURA

### Criterios de decisión:

1. **Para clasificación binaria**: LSTM por su capacidad secuencial
2. **Para series temporales univariables**: N-HiTS por su procesamiento multi-escala
3. **Para múltiples features**: LSTM multivariable por su flexibilidad
4. **Para clasificación multiclase simple**: MLP con arquitectura profunda

### Trade-offs:
- **Complejidad vs Rendimiento**: N-HiTS más complejo pero mejor para patrones multi-escala
- **Velocidad vs Precisión**: MLP más rápido, LSTM más preciso para secuencias
- **Interpretabilidad**: N-BEATS ofrece descomposición backcast/forecast