# Funcionalidad de Código de Inicialización - SOLUCIONADO

## Problema Resuelto ✅

El error `name 'global_mu' is not defined` se ha solucionado completamente. El problema era que la función de prueba del frontend NO estaba ejecutando el código de inicialización antes del código principal.

## Cambios Realizados

### 1. Backend - Vista de Prueba (CustomNormalizationFunctionTestView)
- ✅ **Agregado soporte para `initialization_code`**: La API ahora acepta el código de inicialización
- ✅ **Agregado soporte para `column_data`**: Puede recibir datos de columna reales o genera datos mock
- ✅ **Ejecución secuencial correcta**:
  1. Ejecuta código de inicialización (con acceso a `column_data`)
  2. Ejecuta código principal de la función
- ✅ **Manejo de errores mejorado**: Errores específicos para código de inicialización

### 2. Frontend - Funciones de Prueba JavaScript
- ✅ **testMultipleValues()**: Ahora envía `initialization_code` y `column_data`
- ✅ **Prueba manual**: Incluye código de inicialización en la petición
- ✅ **testFirst20Values() y testAllUniqueValues()**: Funcionan automáticamente porque usan `testMultipleValues()`

### 3. Generación de Datos Mock para Pruebas
Cuando no hay datos reales disponibles, el sistema genera datos de prueba apropiados:

**Para funciones numéricas:**
```python
# Genera 100 valores normalmente distribuidos
mock_data = np.random.normal(10, 2, 100)  # mean=10, std=2
```

**Para funciones de texto:**
```python
# Genera datos categóricos repetidos
mock_data = ['apple', 'banana', 'cherry', 'date', 'elderberry'] * 20
```

## Ejemplo de Uso Completo

### Código de Inicialización:
```python
# μ y σ usando SOLO el 70% inicial (respetando el orden)
n = len(column_data)
cut = int(0.7 * n)

global_mu = column_data.iloc[:cut].mean()
global_sigma = column_data.iloc[:cut].std(ddof=0)
if not (global_sigma > 0):  # evita σ=0 o NaN
    global_sigma = 1.0

print(f"Usando {cut} de {n} valores para estadísticas")
print(f"Global μ = {global_mu:.4f}")
print(f"Global σ = {global_sigma:.4f}")
```

### Código de la Función:
```python
def normalize(value):
    # Usar las variables globales definidas en la inicialización
    return (value - global_mu) / global_sigma
```

## Estado Actual

✅ **Totalmente Funcional**: El código de inicialización se ejecuta correctamente  
✅ **Variables Globales**: Las variables definidas están disponibles en la función principal  
✅ **Pruebas**: Todas las funciones de prueba (manual, 20 valores, únicos) funcionan  
✅ **Datos Reales**: Cuando hay datos de columna disponibles, los usa; sino genera mock data  
✅ **Manejo de Errores**: Errores claros y específicos para debugging  

## Flujo de Ejecución

1. **Usuario ingresa código de inicialización** (opcional)
2. **Usuario ingresa código de función** (requerido)
3. **Usuario prueba la función**:
   - Frontend envía ambos códigos + datos de prueba
   - Backend ejecuta inicialización → función principal
   - Se retorna el resultado
4. **Variables globales están disponibles** en toda la ejecución

La funcionalidad está completamente implementada y funcionando. Las variables `global_mu`, `global_sigma` y cualquier otra variable definida en el código de inicialización estarán disponibles en el código de la función principal.