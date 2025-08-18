# DatasetReplaceValuesView - Implementación Mejorada

## Resumen de Cambios

Se ha mejorado el endpoint `DatasetReplaceValuesView` para manejar correctamente los reemplazos de valores, especialmente en columnas numéricas, con las siguientes características:

### 1. Verificación del Tipo de Datos

- Antes de realizar cualquier reemplazo, el sistema ahora detecta si la columna es numérica usando `pd.api.types.is_numeric_dtype()`
- Distingue entre columnas enteras y de punto flotante para preservar los tipos de datos

### 2. Validación de Valores Numéricos

Para columnas numéricas:
- Valida que el nuevo valor sea convertible a número antes de aplicarlo
- Si la conversión falla, retorna un error descriptivo en lugar de generar NaN inadvertidamente
- Preserva el tipo de dato original (int vs float) cuando es posible

### 3. Manejo de Reemplazo de Caracteres (char_replace)

Implementación completa del modo `char_replace`:
- Permite reemplazar subcadenas dentro de los valores
- Para columnas numéricas, intenta convertir el resultado de vuelta a número
- Si la conversión falla (ej: reemplazar '.' con ','), mantiene el valor como string con advertencia
- Soporta reemplazo vacío para eliminar caracteres

### 4. Prevención de NaN Inadvertidos

- Solo genera NaN cuando el usuario explícitamente proporciona un valor vacío
- Registra advertencias cuando se introducen nuevos valores NaN
- Maneja correctamente los valores existentes que ya son NaN

## Parámetros del Endpoint

```python
{
    'column_name': str,           # Nombre de la columna (requerido)
    'indices': list[int],         # Índices de filas a modificar (requerido)
    'new_value': str,            # Nuevo valor para reemplazo directo
    'char_replace': bool,        # Activar modo de reemplazo de caracteres
    'char_to_find': str,         # Carácter/subcadena a buscar
    'char_to_replace': str,      # Carácter/subcadena de reemplazo
    'partial_replace': bool,     # Modo de reemplazo parcial (no implementado)
    'partial_pattern': list,     # Patrón para reemplazo parcial
    'partial_type': str          # Tipo de reemplazo parcial
}
```

## Ejemplos de Uso

### 1. Reemplazo Directo en Columna Numérica
```json
{
    "column_name": "temperature",
    "indices": [0, 5, 10],
    "new_value": "25.5"
}
```

### 2. Reemplazo de Caracteres
```json
{
    "column_name": "price",
    "indices": [0, 1, 2],
    "char_replace": true,
    "char_to_find": ".",
    "char_to_replace": ","
}
```

### 3. Eliminación de Caracteres
```json
{
    "column_name": "product_code",
    "indices": [0, 1, 2],
    "char_replace": true,
    "char_to_find": "-",
    "char_to_replace": ""
}
```

## Comportamiento Esperado

1. **Columnas Numéricas + Valor No Numérico**: Retorna error
2. **Columnas Numéricas + char_replace**: Intenta mantener tipo numérico si es posible
3. **Valor Vacío**: Genera NaN (intencional)
4. **Índices Inválidos**: Son ignorados silenciosamente
5. **Columna No Existente**: Retorna error

## Mejoras de Seguridad

- Validación estricta de tipos de datos
- Manejo robusto de excepciones con trazabilidad
- Prevención de corrupción de datos numéricos
- Logging de advertencias para operaciones potencialmente problemáticas