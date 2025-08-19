# Solución: Criterio de División y Test Size en Random Forest

## Problemas Identificados

1. **Criterio de división**: Siempre volvía a "Automatique" al editar
2. **Test size**: Siempre volvía a 20% al editar

## Causas

### 1. Función loadRandomForestConfiguration duplicada
- Existían DOS versiones de esta función:
  - Una en `models.html` (que usaba valores por defecto)
  - Una en `random-forest-config-v2.js` (la versión mejorada)
- La versión en models.html sobrescribía valores con: `hyperparams.criterion || 'auto'`

### 2. Actualización de opciones sobrescribía el criterio
- Al llamar `updateRandomForestOptions()`, se reconstruía el dropdown
- El valor seleccionado se perdía en el proceso

### 3. Test size se guarda correctamente
- El test_size SÍ se guarda correctamente en el backend
- El problema puede estar en el orden de carga o en el backend mismo

## Soluciones Implementadas

### 1. Eliminada función duplicada
```javascript
// En models.html - ELIMINADO
// function loadRandomForestConfiguration(hyperparams) { ... }
// Ahora usa solo la versión de random-forest-config-v2.js
```

### 2. Mejorado manejo del criterio
```javascript
// NO usar valor por defecto si no existe
if (hyperparams.criterion !== undefined) {
    setValueWithRangeUpdate('rfCriterion', hyperparams.criterion);
}
```

### 3. Restauración del criterio después de actualizar opciones
```javascript
// Después de updateRandomForestOptions(), restaurar el valor
if (hyperparams.criterion && hyperparams.criterion !== 'auto') {
    setTimeout(() => {
        const criterionSelect = document.getElementById('rfCriterion');
        if (criterionSelect) {
            const optionExists = Array.from(criterionSelect.options)
                .some(opt => opt.value === hyperparams.criterion);
            if (optionExists) {
                criterionSelect.value = hyperparams.criterion;
            }
        }
    }, 100);
}
```

### 4. Logs de debugging agregados
```javascript
console.log('Loading Random Forest configuration:', hyperparams);
console.log('Test size being saved:', testSizeValue);
console.log('Loading test_size:', testSize, '-> UI value:', testSize * 100);
```

## Verificación

Para verificar que funciona correctamente:

1. **Crear modelo Random Forest**:
   - Cambiar criterio a "Entropie"
   - Cambiar test size a 30%
   - Guardar

2. **Editar el modelo**:
   - El criterio debe mostrar "Entropie"
   - El test size debe mostrar 30%

3. **Cambiar tipo de problema**:
   - Si el criterio es válido para el nuevo tipo, se mantiene
   - Si no es válido, cambia a "auto"

## Posibles Problemas Restantes

Si el test_size sigue volviendo a 20%, verificar:

1. **Backend API**: ¿Se está guardando correctamente en la base de datos?
2. **Serialización**: ¿El API devuelve el valor correcto?
3. **Orden de carga**: ¿Se está sobrescribiendo después de cargar?

## Debug

Los logs agregados mostrarán:
- Qué valores se están guardando
- Qué valores se están cargando
- Si hay discrepancias entre lo enviado y lo recibido