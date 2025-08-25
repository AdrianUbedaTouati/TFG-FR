# Ejemplos de Código de Inicialización

La nueva funcionalidad de **Código de Inicialización** permite definir variables globales y realizar cálculos que se ejecutan una sola vez antes de procesar cualquier valor de la columna.

## ¿Cómo funciona?

1. **Código de Inicialización**: Se ejecuta UNA SOLA VEZ al comenzar la normalización
2. **Variable `column_data`**: Tienes acceso a todos los valores de la columna como una pandas Series
3. **Variables Globales**: Cualquier variable que definas estará disponible en la función principal

## Ejemplos de Uso

### Ejemplo 1: Normalización Z-Score con Estadísticas Globales

**Código de Inicialización:**
```python
# Calcular estadísticas globales una sola vez
global_mean = column_data.mean()
global_std = column_data.std()
global_min = column_data.min()
global_max = column_data.max()

print(f"Estadísticas calculadas: mean={global_mean:.2f}, std={global_std:.2f}")
```

**Código de la Función:**
```python
def normalize(value):
    # Usar las variables globales calculadas en la inicialización
    if global_std == 0:
        return 0  # Evitar división por cero
    return (value - global_mean) / global_std
```

### Ejemplo 2: Mapeo de Categorías

**Código de Inicialización:**
```python
# Crear mapeo de categorías únicas
unique_categories = sorted(column_data.unique())
category_map = {cat: i for i, cat in enumerate(unique_categories)}
reverse_map = {i: cat for cat, i in category_map.items()}

print(f"Encontradas {len(unique_categories)} categorías únicas")
print(f"Mapeo: {category_map}")
```

**Código de la Función:**
```python
def normalize(value):
    # Usar el mapeo global
    return category_map.get(value, -1)  # -1 para categorías desconocidas
```

### Ejemplo 3: Normalización Min-Max Personalizada

**Código de Inicialización:**
```python
# Calcular rangos personalizados
data_min = column_data.min()
data_max = column_data.max()
data_range = data_max - data_min

# Definir rango objetivo
target_min = 10
target_max = 100
target_range = target_max - target_min

print(f"Rango original: [{data_min}, {data_max}]")
print(f"Rango objetivo: [{target_min}, {target_max}]")
```

**Código de la Función:**
```python
def normalize(value):
    # Normalizar al rango personalizado
    if data_range == 0:
        return target_min
    
    # Escalar de [data_min, data_max] a [target_min, target_max]
    normalized = (value - data_min) / data_range
    return target_min + (normalized * target_range)
```

### Ejemplo 4: Función Multi-Columna con Análisis de Fechas

**Código de Inicialización:**
```python
import pandas as pd

# Convertir a datetime y analizar patrones
date_series = pd.to_datetime(column_data)
year_range = (date_series.dt.year.min(), date_series.dt.year.max())
month_counts = date_series.dt.month.value_counts()
most_common_month = month_counts.idxmax()

print(f"Rango de años: {year_range}")
print(f"Mes más común: {most_common_month}")
```

**Código de la Función:**
```python
def normalize(value):
    import pandas as pd
    
    dt = pd.to_datetime(value)
    
    # Usar las variables globales para crear múltiples columnas
    year_normalized = (dt.year - year_range[0]) / max(1, year_range[1] - year_range[0])
    
    return {
        'fecha_año_norm': year_normalized,
        'fecha_mes': dt.month,
        'fecha_dia': dt.day,
        'fecha_es_mes_comun': dt.month == most_common_month,
        'fecha_trimestre': (dt.month - 1) // 3 + 1
    }
```

### Ejemplo 5: Análisis de Texto con Vocabulario Global

**Código de Inicialización:**
```python
import re
from collections import Counter

# Analizar todo el texto para crear vocabulario
all_words = []
for text in column_data:
    if pd.notna(text):
        words = re.findall(r'\b\w+\b', str(text).lower())
        all_words.extend(words)

word_counts = Counter(all_words)
vocabulary = {word: idx for idx, (word, count) in enumerate(word_counts.most_common(100))}
vocab_size = len(vocabulary)

print(f"Vocabulario creado con {vocab_size} palabras")
print(f"Palabras más comunes: {list(word_counts.most_common(5))}")
```

**Código de la Función:**
```python
def normalize(value):
    import re
    
    if pd.isna(value):
        return 0
    
    # Tokenizar y mapear usando el vocabulario global
    words = re.findall(r'\b\w+\b', str(value).lower())
    word_indices = [vocabulary.get(word, -1) for word in words]
    
    # Calcular métricas usando el vocabulario
    known_words = sum(1 for idx in word_indices if idx != -1)
    
    return {
        'texto_longitud': len(str(value)),
        'texto_palabras': len(words),
        'texto_palabras_conocidas': known_words,
        'texto_ratio_conocidas': known_words / len(words) if words else 0
    }
```

## Ventajas del Código de Inicialización

1. **Eficiencia**: Los cálculos costosos se realizan una sola vez
2. **Consistencia**: Todas las transformaciones usan las mismas estadísticas/mapeos
3. **Flexibilidad**: Puedes crear cualquier variable global que necesites
4. **Análisis Global**: Acceso completo a todos los datos para análisis previo

## Consejos de Uso

- El código de inicialización es **opcional**
- Usa `column_data` para acceder a todos los valores de la columna
- Las variables que definas estarán disponibles en la función principal
- Ideal para estadísticas, mapeos, vocabularios, etc.
- Perfecto para funciones que necesitan análisis previo de los datos