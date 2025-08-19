# Botones de Restablecimiento y Optimización para Random Forest

## Nuevas Funcionalidades

### 1. Botón "Restablecer" para Configuración Predefinida
- **Ubicación**: Junto al selector de "Configuración predefinida"
- **Icono**: ↻ (arrow-counterclockwise)
- **Función**: Restablece la configuración al preset "Balanceado"
- **Comportamiento**: 
  - Resetea el estado de modificaciones del usuario
  - Aplica el preset balanceado
  - Muestra notificación de confirmación

### 2. Botón "Optimizar" para Tipo de Problema
- **Ubicación**: Junto al selector de "Tipo de problema"
- **Icono**: ✨ (magic)
- **Color**: Azul primario (más destacado)
- **Función**: Aplica configuración óptima según el tipo de problema
- **Comportamiento**:
  - Requiere seleccionar primero Clasificación o Regresión
  - Aplica valores óptimos predefinidos
  - Permite modificación posterior

## Configuraciones Óptimas

### Para Clasificación
```javascript
{
    n_estimators: 500,           // Más árboles para mejor precisión
    max_depth: 20,               // Profundidad limitada para evitar overfitting
    max_features: 'sqrt',        // Óptimo para clasificación
    min_samples_split: 2,        // Permite divisiones finas
    min_samples_leaf: 1,         // Hojas pequeñas permitidas
    criterion: 'gini',           // Criterio estándar para clasificación
    bootstrap: true,             // Muestreo con reemplazo
    oob_score: true,            // Evaluación Out-of-Bag
    class_weight: 'balanced',    // Manejo de clases desbalanceadas
    validation_method: 'stratified_cv'  // Validación estratificada
}
```

### Para Regresión
```javascript
{
    n_estimators: 500,           // Más árboles para suavizar predicciones
    max_depth: 25,               // Más profundidad permitida
    max_features: 1.0,           // Usar todas las características
    min_samples_split: 5,        // Evitar divisiones muy pequeñas
    min_samples_leaf: 2,         // Hojas más grandes para estabilidad
    criterion: 'squared_error',  // MSE para regresión
    bootstrap: true,             // Muestreo con reemplazo
    oob_score: true,            // Evaluación Out-of-Bag
    validation_method: 'cv'      // Validación cruzada estándar
}
```

## Uso

### Caso 1: Restablecer Configuración
1. Click en botón "Restablecer" (↻)
2. Se aplica preset "Balanceado"
3. Se resetean las modificaciones del usuario

### Caso 2: Optimizar para Clasificación
1. Seleccionar "Clasificación" en tipo de problema
2. Click en botón "Optimizar" (✨)
3. Se aplican valores óptimos para clasificación
4. Se puede modificar cualquier valor después

### Caso 3: Optimizar para Regresión
1. Seleccionar "Regresión" en tipo de problema
2. Click en botón "Optimizar" (✨)
3. Se aplican valores óptimos para regresión
4. Se puede modificar cualquier valor después

## Características Especiales

1. **No sobrescribe número de árboles con presets**: Como se solicitó, los presets normales no cambian n_estimators
2. **Optimizar sí cambia todo**: El botón "Optimizar" aplica una configuración completa óptima
3. **Resetea estado de modificación**: Después de optimizar, los valores pueden ser cambiados por presets
4. **Validación inteligente**: Si tipo es "auto", pide seleccionar específicamente

## Beneficios

1. **Configuración rápida**: Un click para configuración óptima
2. **Educativo**: Los usuarios aprenden qué valores son buenos para cada tipo
3. **Flexible**: Todo sigue siendo modificable
4. **Visual**: Botones claramente diferenciados por función

## Estilo Visual

- **Restablecer**: Botón secundario gris, más discreto
- **Optimizar**: Botón primario azul, más prominente
- **Tooltips**: Explican la función al pasar el mouse
- **Iconos**: Ayudan a identificar la función rápidamente