# Fix para Valores NaN No Serializables en JSON

## Problema
El endpoint `/api/datasets/{id}/columns/` estaba fallando con el error:
```
Out of range float values are not JSON compliant
```

Esto ocurría porque los valores NaN (Not a Number) de pandas/numpy no son serializables directamente a JSON.

## Solución Implementada

### 1. En `dataset_views.py` - DatasetColumnsView (líneas 94-99)

**Antes:**
```python
# Generate preview data (first 10 rows)
preview = {}
for col in columns:
    preview[col] = df[col].head(10).tolist()
```

**Después:**
```python
# Generate preview data (first 10 rows)
preview = {}
for col in columns:
    # Convert NaN values to None for JSON serialization
    col_data = df[col].head(10)
    preview[col] = [None if pd.isna(val) else val for val in col_data]
```

### 2. En `dataset_views.py` - DatasetVariableAnalysisView._generate_outlier_map (línea 378)

**Antes:**
```python
'outlier_values': outliers.head(20).tolist()  # First 20 outliers
```

**Después:**
```python
'outlier_values': [None if pd.isna(val) else float(val) for val in outliers.head(20)]  # First 20 outliers
```

## Explicación Técnica

- Los valores NaN de numpy/pandas no son compatibles con el estándar JSON
- Al usar `.tolist()` directamente, los NaN se mantienen como valores float especiales que causan el error
- La solución convierte explícitamente cada valor NaN a `None`, que es serializable en JSON como `null`
- Se usa `pd.isna()` para detectar valores NaN de manera robusta

## Otras Consideraciones

El código ya maneja correctamente los NaN en otros lugares:
- `detect_column_type` en `utils.py` usa `dropna()` antes de `tolist()` para `sample_values`
- Los histogramas y estadísticas ya filtran valores NaN antes de procesarlos
- Los value_counts excluyen NaN por defecto en pandas

## Pruebas

Se creó un script de prueba `test_nan_serialization.py` que:
1. Crea un dataset con valores faltantes
2. Llama al endpoint `/api/datasets/{id}/columns/`
3. Verifica que la respuesta sea JSON válido
4. Confirma que no haya valores NaN, solo None/null

## Impacto

Este cambio garantiza que:
- El endpoint siempre devuelve JSON válido
- Los valores faltantes se representan consistentemente como `null` en JSON
- No se pierden datos - los valores faltantes siguen siendo identificables
- La aplicación frontend puede manejar los datos correctamente