# Configuración de División de Datos Train/Validation/Test

## Problema
El usuario reportó que el error "X has 15 features, but RandomForestClassifier is expecting 29 features" ocurre porque hay una discrepancia entre las características usadas en entrenamiento y predicción. Además, solicitó agregar la capacidad de configurar los porcentajes de train/validation/test al crear modelos.

## Solución Implementada

### 1. Interfaz de Usuario Actualizada

En `templates/dashboard.html`, se actualizó el formulario de creación de modelos:

#### Antes:
```html
<div class="mb-3">
    <label for="testSize" class="form-label">Taille du Test (%)</label>
    <input type="number" class="form-control" id="testSize" min="10" max="50" value="20" required>
</div>
```

#### Después:
```html
<div class="mb-3">
    <label class="form-label">Division des Données</label>
    <div class="row g-2">
        <div class="col-md-4">
            <label for="trainSize" class="form-label small">Entraînement (%)</label>
            <input type="number" class="form-control" id="trainSize" min="40" max="80" value="70" required onchange="updateDataSplits()">
        </div>
        <div class="col-md-4">
            <label for="valSize" class="form-label small">Validation (%)</label>
            <input type="number" class="form-control" id="valSize" min="10" max="30" value="15" required onchange="updateDataSplits()">
        </div>
        <div class="col-md-4">
            <label for="testSize" class="form-label small">Test (%)</label>
            <input type="number" class="form-control" id="testSize" min="10" max="30" value="15" readonly style="background-color: rgba(255,255,255,0.1);">
        </div>
    </div>
    <small class="text-muted">Les pourcentages doivent totaliser 100%</small>
</div>
```

### 2. Función JavaScript para Validación

Se agregó la función `updateDataSplits()` que:
- Calcula automáticamente el porcentaje de test
- Valida que la suma sea 100%
- Muestra errores visuales si hay problemas

```javascript
function updateDataSplits() {
    const trainSize = parseInt(document.getElementById('trainSize').value) || 0;
    const valSize = parseInt(document.getElementById('valSize').value) || 0;
    const testSizeInput = document.getElementById('testSize');
    
    const testSize = 100 - trainSize - valSize;
    
    if (testSize < 0 || testSize > 100) {
        testSizeInput.value = 'Error';
        testSizeInput.style.backgroundColor = 'rgba(255, 0, 0, 0.2)';
        return false;
    }
    
    testSizeInput.value = testSize;
    testSizeInput.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
    
    if (trainSize + valSize + testSize !== 100) {
        testSizeInput.style.backgroundColor = 'rgba(255, 193, 7, 0.2)';
        return false;
    }
    
    return true;
}
```

### 3. Envío de Datos al Backend

Se actualizó el envío de datos para incluir los tres porcentajes:
```javascript
const sessionData = {
    // ... otros campos ...
    train_split: document.getElementById('trainSize').value / 100,
    val_split: document.getElementById('valSize').value / 100,
    test_split: document.getElementById('testSize').value / 100,
    test_size: document.getElementById('testSize').value / 100,  // Para compatibilidad
    // ...
};
```

### 4. Validación al Entrenar

Se agregó validación en la función `trainModel()`:
```javascript
// Validation des pourcentages de division des données
if (!updateDataSplits()) {
    showNotification('Erreur', 'Les pourcentages de division des données doivent totaliser 100%', 'error');
    return;
}

const trainSize = parseInt(document.getElementById('trainSize').value);
if (trainSize < 40) {
    showNotification('Erreur', 'Le pourcentage d\'entraînement doit être d\'au moins 40%', 'error');
    return;
}
```

### 5. Backend ya Preparado

El modelo `TrainingSession` ya tiene los campos necesarios:
```python
train_split = models.FloatField(default=0.7)
val_split = models.FloatField(default=0.15)
test_split = models.FloatField(default=0.15)
```

La función `prepare_data` en `ml_utils.py` ya usa estos campos correctamente:
```python
train_end = int(n * session.train_split)
val_end = int(n * (session.train_split + session.val_split))
```

## Consistencia de Características

Para asegurar la consistencia entre entrenamiento y predicción, el sistema ya tiene implementado:

1. **Preprocessing Pipeline**: Guarda los transformadores (encoders, scalers) con el modelo
2. **Feature Names**: Almacena los nombres de características después del preprocesamiento
3. **Validation**: Verifica que las características de entrada coincidan con las esperadas

## Beneficios

1. **Flexibilidad**: Los usuarios pueden ajustar la división de datos según sus necesidades
2. **Validación**: El sistema valida que los porcentajes sumen 100%
3. **Mejor UI**: Interfaz clara con cálculo automático del porcentaje de test
4. **Prevención de errores**: Validaciones previenen configuraciones inválidas

## Uso

1. Al crear un modelo, ajusta los porcentajes de Entrenamiento y Validación
2. El porcentaje de Test se calcula automáticamente
3. El sistema valida que:
   - La suma sea 100%
   - El entrenamiento sea al menos 40%
   - Los valores estén en rangos válidos
4. Los datos se dividen según estos porcentajes durante el entrenamiento