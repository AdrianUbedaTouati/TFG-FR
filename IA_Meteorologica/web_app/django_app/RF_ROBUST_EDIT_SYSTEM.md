# Sistema Robusto de Edición para Random Forest

## Problemas Resueltos

1. **Pérdida de valores personalizados**: El criterio de división y otros valores se perdían al cambiar el tipo de problema
2. **Sobrescritura por presets**: Los presets sobrescribían valores modificados por el usuario
3. **Falta de persistencia**: No había un sistema para rastrear qué valores fueron modificados por el usuario

## Nueva Arquitectura: Sistema de Estado Global

### 1. Estado Global (RFUserState)
```javascript
const RFUserState = {
    userModified: {
        n_estimators: false,
        criterion: false,
        max_depth: false,
        // etc...
    },
    currentValues: {},
    markAsUserModified: function(field) {...},
    isUserModified: function(field) {...}
}
```

**Ventajas:**
- Rastrea qué campos fueron modificados por el usuario
- Mantiene los valores actuales
- Fácil de extender con nuevos campos

### 2. Event Listeners con Tracking
Cada input ahora marca cuando el usuario lo modifica:
```javascript
nEstimatorsInput.addEventListener('input', function() {
    RFUserState.markAsUserModified('n_estimators');
    RFUserState.saveValue('n_estimators', this.value);
});
```

### 3. Presets Inteligentes
Los presets ahora respetan las modificaciones del usuario:
```javascript
function applyRandomForestPreset() {
    // Solo aplica valores NO modificados por el usuario
    if (!RFUserState.isUserModified('n_estimators')) {
        // NO actualiza número de árboles según petición
    }
}
```

**Comportamiento especial para número de árboles:**
- Los presets NO cambian el número de árboles
- El usuario puede modificarlo libremente
- Se mantiene el valor modificado

### 4. Actualización de Criterio Preservando Selección
```javascript
function updateCriterionOptionsPreserving(problemType, config) {
    // Guarda valor actual
    const userModifiedValue = RFUserState.getValue('criterion');
    
    // Reconstruye opciones
    // ...
    
    // Restaura valor si es válido para el nuevo tipo
    if (RFUserState.isUserModified('criterion') && 
        config.criteriaOptions.includes(userModifiedValue)) {
        criterionSelect.value = userModifiedValue;
    }
}
```

### 5. Carga de Configuración Mejorada
Al cargar un modelo existente:
```javascript
function loadRandomForestConfiguration(hyperparams) {
    // Reset estado de modificación
    RFUserState.resetModificationState();
    
    // Carga todos los valores
    // ...
    
    // Guarda valores cargados
    RFUserState.currentValues = { ...hyperparams };
}
```

## Beneficios del Nuevo Sistema

1. **Persistencia Real**: Los valores modificados por el usuario se mantienen
2. **Presets No Invasivos**: Los presets no sobrescriben valores del usuario
3. **Validación Inteligente**: El criterio se valida según el tipo de problema
4. **Fácil Mantenimiento**: Agregar nuevos campos es simple
5. **Debug Mejorado**: El estado global facilita el debugging

## Cómo Extender el Sistema

Para agregar un nuevo campo persistente:

1. Agregar al estado de modificación:
```javascript
userModified: {
    nuevo_campo: false,
    // ...
}
```

2. Agregar event listener con tracking:
```javascript
nuevoInput.addEventListener('change', function() {
    RFUserState.markAsUserModified('nuevo_campo');
    RFUserState.saveValue('nuevo_campo', this.value);
});
```

3. Actualizar la función de preset:
```javascript
if (!RFUserState.isUserModified('nuevo_campo')) {
    // Aplicar valor del preset
}
```

## Uso

1. El archivo `random-forest-config-v2.js` reemplaza completamente el anterior
2. El sistema se inicializa automáticamente cuando se detecta el elemento Random Forest
3. Todos los valores se preservan correctamente durante la edición

## Testing

Para verificar el funcionamiento:
1. Crear un modelo Random Forest
2. Cambiar el número de árboles a 500
3. Cambiar el criterio a "Entropie"
4. Cambiar el preset - el número de árboles debe mantenerse en 500
5. Cambiar el tipo de problema - el criterio debe mantenerse si es válido
6. Guardar y recargar - todos los valores deben persistir