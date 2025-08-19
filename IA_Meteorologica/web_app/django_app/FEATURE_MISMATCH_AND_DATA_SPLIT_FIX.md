# Corrección de Error de Features y Configuración de División de Datos

## Problemas Resueltos

### 1. Error: "X has 15 features, but RandomForestClassifier is expecting 27 features"

#### Causa
El preprocesamiento (one-hot encoding, features cíclicas) se aplicaba solo dentro de `train_sklearn_model` pero los datos procesados no se devolvían para usar en la evaluación. Esto causaba que:
- El modelo se entrenaba con 27 features (después del preprocesamiento)
- La validación/predicción recibía solo 15 features (sin preprocesar)

#### Solución Implementada

1. **Modificación de `train_sklearn_model`**:
   - Ahora acepta `X_test` y `y_test` como parámetros
   - Aplica el preprocessing pipeline a todos los conjuntos de datos (train, val, test)
   - Devuelve un diccionario con los datos procesados:
   ```python
   return {
       'model': model,
       'history': history,
       'preprocessing_pipeline': preprocessing_pipeline,
       'X_train_processed': X_train,
       'X_val_processed': X_val,
       'X_test_processed': X_test if X_test is not None else None
   }
   ```

2. **Actualización en `train_model`**:
   - Recibe el resultado como diccionario
   - Usa los datos procesados para la evaluación final
   ```python
   result = train_sklearn_model(session, session.model_type, session.hyperparameters,
                               X_train, y_train, X_val, y_val, X_test, y_test)
   model = result['model']
   history = result['history']
   preprocessing_pipeline = result.get('preprocessing_pipeline')
   
   # Use processed data for evaluation
   if preprocessing_pipeline:
       X_test = result['X_test_processed']
   ```

### 2. Configuración de División de Datos Train/Validation/Test

#### Problema
No había forma de configurar los porcentajes de train/validation/test al crear o entrenar modelos.

#### Soluciones Implementadas

1. **En la página de Dashboard** (`templates/dashboard.html`):
   - Agregados 3 campos para configurar porcentajes
   - Cálculo automático del porcentaje de test
   - Validación en tiempo real que suma sea 100%
   - Función `updateDataSplits()` para validación

2. **En la página de Modelos** (`templates/models.html`):
   - Modal de creación actualizado con los mismos campos
   - Nuevo modal `trainConfigModal` para configurar antes de entrenar
   - Función `trainModel()` modificada para mostrar configuración
   - Función `confirmTraining()` que envía los datos con los porcentajes

3. **Envío de datos al backend**:
   ```javascript
   train_split: trainSize / 100,
   val_split: valSize / 100,
   test_split: testSize / 100,
   test_size: testSize / 100  // Para compatibilidad
   ```

## Archivos Modificados

1. **`ml_trainer/ml_utils.py`**:
   - Función `train_sklearn_model` modificada para devolver datos procesados
   - Función `train_model` actualizada para usar los datos procesados

2. **`templates/dashboard.html`**:
   - Agregados campos de división de datos
   - Función `updateDataSplits()` para validación
   - Actualizado envío de datos de entrenamiento

3. **`templates/models.html`**:
   - Agregados campos de división en modal de creación
   - Nuevo modal `trainConfigModal` para configuración de entrenamiento
   - Funciones `trainModel()`, `confirmTraining()`, `updateDataSplitsConfig()`

## Resultado

✅ **Error de features corregido**: Los datos ahora se procesan consistentemente en todas las etapas
✅ **División de datos configurable**: Los usuarios pueden ajustar train/validation/test splits
✅ **Disponible en todos los modelos**: La configuración aparece para todos los tipos de modelo
✅ **Validación robusta**: Los porcentajes se validan para sumar 100%

## Uso

1. Al crear un modelo o entrenar desde la página de modelos, aparecerá la configuración de división
2. Ajusta los porcentajes de Entrenamiento y Validación
3. El porcentaje de Test se calcula automáticamente
4. El sistema valida que la suma sea 100%
5. Los datos se procesan consistentemente usando el pipeline guardado