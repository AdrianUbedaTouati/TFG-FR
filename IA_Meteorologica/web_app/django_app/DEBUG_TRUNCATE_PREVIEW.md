# Debug: TRUNCATE Preview Error

## Problema
Cuando se tiene una configuración de normalización como:
```json
{
  "Formatted Date": [
    {
      "method": "CUSTOM_1234", // Función que crea múltiples columnas numéricas
      "keep_original": true
    },
    {
      "method": "one_hot",
      "input_column": "trend", // Una de las columnas numéricas creadas
      "conversion": "TRUNCATE",
      "conversion_params": {"decimals": 2}
    }
  ]
}
```

El error muestra: "Error aplicando TRUNCATE a columna Formatted Date: La conversión de truncamiento requiere datos numéricos"

## Análisis
1. La conversión TRUNCATE se guarda en la configuración del paso 1 (índice 0)
2. Cuando se genera el preview para el paso 2:
   - Se aplican las transformaciones anteriores (paso 1)
   - Se obtienen los valores únicos de la columna intermedia ("trend")
   - Se crea un DataFrame temporal con esos valores
   - Se aplica la transformación del paso 2

El problema es que el nombre de la columna en el error muestra "Formatted Date" cuando debería ser "trend".

## Posibles causas
1. El DataFrame temporal podría estar usando el nombre de columna incorrecto
2. La conversión podría estar aplicándose antes de cambiar al input_column correcto
3. El mensaje de error podría estar mostrando el nombre de la columna original en lugar del real

## Solución propuesta
Verificar que cuando se procesa un paso con input_column:
1. Primero se cambie a la columna especificada
2. Luego se aplique cualquier conversión del paso anterior
3. Finalmente se aplique la transformación del paso actual