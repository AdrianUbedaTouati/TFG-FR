# Análisis Completo de Redes Neuronales - Proyecto Meteorológico

## Resumen Ejecutivo

Este documento presenta un análisis exhaustivo de las diferentes arquitecturas de redes neuronales implementadas en el proyecto meteorológico. Se analizan 5 tipos principales de redes: LSTM (clasificación y regresión), N-BEATS, N-HiTS, MLP y redes de clasificación de cobertura nubosa.

## 1. REDES LSTM

### 1.1 LSTM para Clasificación

#### 1.1.1 Red Ventana (Clasificación de Resumen Meteorológico - 5 clases)
- **Configuración**: H=336, L=24, C=5
- **Variable objetivo**: Summary (5 clases: Partly Cloudy, Mostly Cloudy, Overcast, Clear, Foggy)
- **Precisión**: 26.93%
- **Características**: Baja precisión debido a la complejidad de distinguir entre 5 tipos de condiciones climáticas

#### 1.1.2 Red Rain (Clasificación Binaria Lluvia/Nieve)
- **Variable objetivo**: Tipo de precipitación (rain/snow)
- **Precisión**: 99.19%
- **F1 Score**: 0.9662
- **Características**: Excelente rendimiento en clasificación binaria, dataset desbalanceado (rain >> snow)

#### 1.1.3 Red Summary (Clasificación de Resumen - 3 clases)
- **Variable objetivo**: Summary (3 clases: Partly Cloudy, Mostly Cloudy, Overcast)
- **Precisión**: 56.77%
- **F1 Score**: 0.5566
- **Características**: Rendimiento moderado, versión simplificada de la red Ventana

### 1.2 LSTM Multivariable (Regresión)

#### 1.2.1 Court/Neutral (Predicción a corto plazo)
- **Configuración**: H=336, L=24 (14 días horizonte, 1 día lookback)
- **Variable objetivo**: Temperatura (°C) normalizada
- **Métricas normalizadas**: RMSE=0.250, MAE=0.193

#### 1.2.2 Long/Neutral (Predicción a largo plazo)
- **Configuración**: H=1440, L=120 (60 días horizonte, 5 días lookback)
- **Variable objetivo**: Temperatura (°C) normalizada
- **Métricas normalizadas**: RMSE=0.344, MAE=0.275

#### 1.2.3 Long/Ventanas
- **Configuración**: H=1440, L=120
- **Variable objetivo**: Temperatura (°C) normalizada
- **Métricas normalizadas**: RMSE=0.349, MAE=0.284
- **Características**: Rendimiento inferior al neutral, sugiere que la estrategia de ventanas no es óptima

## 2. REDES N-BEATS

### 2.1 N-BEATS Univariable

#### 2.1.1 Court (H=336, L=24)
- **Neutral**: RMSE=0.260, MAE=0.188
- **Grid**: RMSE=0.258, MAE=0.186 (mejor)
- **Random**: RMSE=0.258, MAE=0.191
- **Mejores hiperparámetros**: Hidden=256, Depth=4, Blocks=8

#### 2.1.2 Long (H=1440, L=120)
- **Neutral**: RMSE=0.407, MAE=0.308
- **Random**: RMSE=0.398, MAE=0.308 (mejor)
- **Mejores hiperparámetros**: Hidden=256, Depth=4, Blocks=4

### 2.2 N-BEATS Multivariable

#### 2.2.1 Court (H=336, L=24)
- **Neutral**: RMSE=0.395, MAE=0.319
- **Random**: RMSE=0.297, MAE=0.230 (mejor)
- **Características**: 19 features incluyendo condiciones meteorológicas

#### 2.2.2 Long (H=1440, L=120)
- **Random**: RMSE=0.425, MAE=0.338 

**Observación clave**: Para N-BEATS, los modelos univariables superan significativamente a los multivariables en predicciones a corto plazo.

## 3. REDES N-HiTS

### 3.1 Comparación Court (H=336, L=24)

| Tipo | Configuración | Val Loss | MAE | RMSE |
|------|---------------|----------|-----|------|
| Univariable/Neutral | D=1 | 0.0268 | 0.1764 | 0.2408 |
| Univariable/Random | D=1 | 0.0275 | 0.1759 | 0.2399 |
| Multivariable/Random | D=21 | 0.0268 | 0.2284 | 0.2863 |

### 3.2 Comparación Long (H=1440, L=120)

| Tipo | Configuración | Val Loss | MAE | RMSE |
|------|---------------|----------|-----|------|
| Univariable/Neutral | D=1 | 0.0657 | 0.3345 | 0.4169 |
| Univariable/Random | D=1 | 0.0632 | 0.3296 | 0.4115 |
| Multivariable/Neutral | D=21 | 0.0664 | 0.3195 | 0.4030 |

**Observación clave**: A diferencia de N-BEATS, en N-HiTS los modelos multivariables muestran ventaja para predicciones a largo plazo.

## 4. REDES MLP

### 4.1 MLP para Clasificación de Summary (3 clases)
- **Clases**: Cloudy, Clear, Foggy
- **Distribución**: Muy desbalanceada (Cloudy >> Clear > Foggy)
- **Arquitectura**: Red neuronal feedforward simple
- **Nota**: No se encontraron métricas de precisión en los archivos disponibles

### 4.2 MLP para Cloud Cover (4 clases)
- **Clases**: clear, cloudy, overcast, partly cloudy
- **Precisión (sin CV)**: 61.29%
- **F1 Score**: 0.580
- **Precisión (con CV)**: 62.12%
- **F1 Score CV**: 0.584
- **Características**: Cross-validation mejora ligeramente el rendimiento

## 5. COMPARACIONES Y SIMILITUDES

### 5.1 Patrones Arquitecturales

1. **LSTM vs N-BEATS/N-HiTS**: 
   - LSTM usa arquitectura recurrente tradicional
   - N-BEATS/N-HiTS usan bloques especializados para series temporales

2. **Configuraciones H/L estándar**:
   - Court: H=336, L=24 (14 días predicción, 1 día histórico)
   - Long: H=1440, L=120 (60 días predicción, 5 días histórico)

### 5.2 Rendimiento Comparativo

#### Para Clasificación:
1. **Binaria (rain/snow)**: LSTM 99.19% - Excelente
2. **3 clases (summary)**: LSTM 56.77% - Moderado
3. **4 clases (cloud cover)**: MLP 62.12% - Moderado
4. **5 clases (summary completo)**: LSTM 26.93% - Pobre

#### Para Regresión (Temperatura):

**RANKING CORTO PLAZO (H=336, L=24)**

**Univariable:**
1. N-HiTS: MAE=0.176, RMSE=0.240
2. N-BEATS: MAE=0.186, RMSE=0.258
3. (LSTM no tiene versión univariable)

**Multivariable:**
1. LSTM: MAE=0.193, RMSE=0.250 ⭐ MEJOR
2. N-HiTS: MAE=0.228, RMSE=0.286
3. N-BEATS: MAE=0.230, RMSE=0.297

**RANKING LARGO PLAZO (H=1440, L=120)**

**Univariable:**
1. N-BEATS: MAE=0.308, RMSE=0.398 ⭐ MEJOR
2. N-HiTS: MAE=0.330, RMSE=0.412

**Multivariable:**
1. LSTM: MAE=0.275, RMSE=0.344 ⭐ MEJOR
2. N-HiTS: MAE=0.320, RMSE=0.403
3. N-BEATS: MAE=0.338, RMSE=0.425

### 5.3 Recomendaciones Actualizadas

1. **Para clasificación binaria**: LSTM es la mejor opción (99.19% precisión)
2. **Para clasificación multiclase**: MLP con CV o reducir número de clases
3. **Para predicción de temperatura a corto plazo**:
   - Univariable: N-HiTS (MAE=0.176)
   - Multivariable: LSTM (MAE=0.193)
4. **Para predicción de temperatura a largo plazo**:
   - Univariable: N-BEATS (MAE=0.308)
   - Multivariable: LSTM (MAE=0.275)
5. **Conclusión importante**: LSTM multivariable es consistentemente el mejor modelo para regresión cuando se usan múltiples features

## 6. MÉTRICAS UTILIZADAS Y SU JUSTIFICACIÓN

### 6.1 Para Problemas de Clasificación

#### Precisión (Accuracy)
- **Qué mide**: Porcentaje de predicciones correctas sobre el total
- **Cuándo usarla**: Cuando las clases están balanceadas
- **Limitación**: Puede ser engañosa con clases desbalanceadas

#### F1 Score
- **Qué mide**: Media armónica entre precisión y recall
- **Por qué se usa**: Balancea falsos positivos y falsos negativos
- **Ventaja**: Mejor métrica para datasets desbalanceados (ej: rain/snow)
- **Rango**: 0-1, donde 1 es perfecto

### 6.2 Para Problemas de Regresión

#### MAE (Mean Absolute Error)
- **Qué mide**: Promedio de errores absolutos
- **Por qué se usa**: Fácil de interpretar, robusto a outliers
- **Interpretación**: Error promedio en las mismas unidades que la variable objetivo
- **Ejemplo**: MAE=0.193 significa error promedio del 19.3% del rango normalizado

#### RMSE (Root Mean Square Error)
- **Qué mide**: Raíz del promedio de errores cuadrados
- **Por qué se usa**: Penaliza más los errores grandes
- **Ventaja**: Útil cuando los errores grandes son especialmente problemáticos
- **Comparación**: Siempre RMSE ≥ MAE; si RMSE >> MAE, hay errores grandes ocasionales

### 6.3 Métricas Normalizadas vs No Normalizadas
- **Normalizadas (0-1)**: Permiten comparar entre diferentes modelos y datasets
- **No normalizadas (°C)**: Más interpretables para el usuario final
- **Conversión**: Se puede estimar el error real multiplicando por el rango de la variable

## 7. TÉCNICAS PARA MEJORAR LA PRECISIÓN

### 7.1 Normalización de Datos
- **Por qué**: Los algoritmos de deep learning convergen mejor cuando los datos están en rangos similares (0-1)
- **Cómo**: Min-Max scaling o StandardScaler para llevar todas las variables al mismo rango
- **Impacto**: Esencial para el entrenamiento estable y la comparación entre métricas

### 7.2 Ingeniería de Características Temporales
- **Codificación cíclica**: Usar sin/cos para hora del día y día del año
- **Por qué**: Captura la naturaleza cíclica del tiempo (ej: 23:59 está cerca de 00:00)
- **Ejemplo**: hour_sin = sin(2π * hour/24), hour_cos = cos(2π * hour/24)

### 7.3 Búsqueda de Hiperparámetros
- **Grid Search**: Prueba sistemática de combinaciones predefinidas
- **Random Search**: Exploración aleatoria del espacio de hiperparámetros
- **Por qué**: Encuentra la configuración óptima de learning rate, capas ocultas, bloques, etc.
- **Resultado**: Mejoras del 5-15% en precisión típicamente

### 7.4 Arquitecturas Especializadas
- **N-BEATS/N-HiTS**: Diseñadas específicamente para series temporales
- **Bloques residuales**: Permiten entrenar redes más profundas sin degradación
- **Atención temporal**: Permite al modelo enfocarse en momentos relevantes del pasado

### 7.5 Estrategias de Entrenamiento
- **Early Stopping**: Detiene el entrenamiento cuando la validación no mejora
- **Learning Rate Scheduling**: Reduce la tasa de aprendizaje gradualmente
- **Regularización**: Weight decay (L2) previene sobreajuste

### 7.6 Selección de Features
- **Univariable vs Multivariable**: 
  - Corto plazo: Menos features pueden reducir ruido
  - Largo plazo: Más features capturan patrones complejos
- **Feature Importance**: Identifica qué variables contribuyen más a las predicciones

### 7.7 Manejo de Datos Desbalanceados
- **Problema**: En clasificación rain/snow hay muchos más datos de lluvia
- **Solución**: Class weighting o sampling estratificado
- **Resultado**: Mejor rendimiento en clases minoritarias

### 7.8 Validación Cruzada (Cross-Validation)
- **K-fold CV**: Divide datos en K partes, entrena K modelos
- **Por qué**: Estimación más robusta del rendimiento real
- **Ejemplo**: MLP cloud cover mejoró de 61.29% a 62.12% con CV

## 8. CONCLUSIONES FINALES

1. **LSTM multivariable domina en regresión**: Contrario a lo esperado, LSTM supera a N-BEATS y N-HiTS en configuraciones multivariables
2. **Para univariable**: N-HiTS es mejor a corto plazo, N-BEATS a largo plazo
3. **La complejidad importa**: El rendimiento decrece con más clases (99%→57%→27%)
4. **Horizonte temporal**: Los errores aumentan ~40% al pasar de corto a largo plazo
5. **La estrategia de "ventanas" en LSTM no mostró mejoras**
6. **LSTM es versátil**: Excelente en clasificación binaria (99%) y líder en regresión multivariable
7. **No existe arquitectura universal**: La elección depende del tipo de problema y configuración