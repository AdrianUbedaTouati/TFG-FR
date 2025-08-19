# Corrección: Error OneHotEncoder sparse

## Problema
Al entrenar un modelo Random Forest, aparecía el error:
```
TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'
```

## Causa
En versiones recientes de scikit-learn (>= 1.2), el parámetro `sparse` fue renombrado a `sparse_output` para mayor claridad.

## Solución
Cambiar en `ml_utils.py` línea 1621:

**Antes:**
```python
encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
```

**Después:**
```python
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
```

## Compatibilidad
- **scikit-learn < 1.2**: Usa `sparse`
- **scikit-learn >= 1.2**: Usa `sparse_output`

## Verificación
El cambio es compatible con las versiones más recientes de scikit-learn y debería resolver el error de entrenamiento para Random Forest y cualquier otro modelo que use codificación categórica.

## Impacto
Este cambio afecta a la codificación One-Hot de variables categóricas durante el entrenamiento de modelos de Machine Learning (Random Forest, XGBoost, Decision Tree, etc.).

El parámetro `sparse_output=False` asegura que la salida sea una matriz densa (numpy array) en lugar de una matriz dispersa, lo cual es importante para la compatibilidad con el resto del código.