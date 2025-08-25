from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Dict, Union, Tuple, Optional
import numpy as np
try:
    from .type_conversion import ensure_numeric_compatibility, TypeConversionWarning
except ImportError:
    # Fallback if module structure is different
    from type_conversion import ensure_numeric_compatibility, TypeConversionWarning

# ────────────────────────────────
# 1. Definir los Enum permitidos
# ────────────────────────────────
class NumNorm(Enum):
    """Métodos de normalización numérica."""
    MIN_MAX = auto()        # Rango [0, 1]
    Z_SCORE = auto()        # Media 0, desviación típica 1
    LSTM_TCN = auto()       # Rango [-1, 1] para RNN/TCN (tanh)
    CNN = auto()            # Z‑score enfocado a CNN
    TRANSFORMER = auto()    # RobustScaler resistente a outliers
    TREE = auto()           # Sin transformación (modelos de árboles)

class TextNorm(Enum):
    LOWER = auto()          # Minúsculas
    STRIP = auto()          # Eliminar espacios en blanco en extremos
    LABEL_ENCODING = auto() # Codificación de etiquetas (0, 1, 2...)
    ONE_HOT = auto()        # Codificación one‑hot real (columnas binarias)

# ────────────────────────────────
# 2. Implementar las funciones de normalización numérica
# ────────────────────────────────
def min_max_numeric(serie: pd.Series) -> pd.Series:
    # Manejar valores NaN
    mask = serie.notna()
    if mask.sum() == 0:  # Todos son NaN
        return serie
    
    # Asegurar compatibilidad numérica
    result, warning = ensure_numeric_compatibility(serie)
    
    esc = MinMaxScaler()
    result.loc[mask] = esc.fit_transform(result.loc[mask].to_frame()).ravel()
    return result

def z_score_numeric(serie: pd.Series) -> pd.Series:
    mask = serie.notna()
    if mask.sum() == 0:
        return serie
    
    # Asegurar compatibilidad numérica
    result, warning = ensure_numeric_compatibility(serie)
    
    esc = StandardScaler()
    result.loc[mask] = esc.fit_transform(result.loc[mask].to_frame()).ravel()
    return result

def lstm_tcn_norm(serie: pd.Series) -> pd.Series:
    mask = serie.notna()
    if mask.sum() == 0:
        return serie
    
    # Asegurar compatibilidad numérica
    result, warning = ensure_numeric_compatibility(serie)
    
    esc = MinMaxScaler(feature_range=(-1, 1))
    result.loc[mask] = esc.fit_transform(result.loc[mask].to_frame()).ravel()
    return result

def cnn_norm(serie: pd.Series) -> pd.Series:
    return z_score_numeric(serie)

def transformer_norm(serie: pd.Series) -> pd.Series:
    mask = serie.notna()
    if mask.sum() == 0:
        return serie
    
    # Asegurar compatibilidad numérica
    result, warning = ensure_numeric_compatibility(serie)
    
    esc = RobustScaler()
    result.loc[mask] = esc.fit_transform(result.loc[mask].to_frame()).ravel()
    return result

def tree_norm(serie: pd.Series) -> pd.Series:
    return serie.copy()

# ────────────────────────────────
# 3. Implementar las funciones de normalización de texto
# ────────────────────────────────
def lower_text(serie: pd.Series) -> pd.Series:
    return serie.str.lower()

def strip_text(serie: pd.Series) -> pd.Series:
    return serie.str.strip()

def label_encoding_text(serie: pd.Series) -> pd.Series:
    # Convertir a códigos categóricos (0, 1, 2, etc.)
    if serie.dtype == 'object' or serie.dtype.name == 'category':
        # Crear un mapeo de categorías a números
        unique_vals = serie.dropna().unique()
        mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
        # Aplicar el mapeo, mantener NaN como NaN
        result = serie.map(mapping)
        # Asegurar que los valores sean enteros (0, 1, 2, etc.)
        return result.astype('Int64')  # Int64 permite NaN
    return serie

def one_hot_text(serie: pd.Series) -> pd.DataFrame:
    # Crear verdadero one-hot encoding (columnas binarias)
    if serie.dtype == 'object' or serie.dtype.name == 'category':
        # Obtener el nombre de la columna original
        col_name = serie.name if serie.name else 'column'
        
        # Crear one-hot encoding usando pd.get_dummies
        # prefix usa el nombre de la columna original
        one_hot_df = pd.get_dummies(serie, prefix=col_name, dummy_na=False)
        
        # Asegurar que los valores sean enteros (0 o 1)
        return one_hot_df.astype('int64')
    
    # Si no es texto/categoría, devolver como DataFrame de una columna
    return pd.DataFrame({serie.name: serie})

# ────────────────────────────────
# 4. Configurar el Dispatch Table
# ────────────────────────────────
DISPATCH_NUM = {
    NumNorm.MIN_MAX: min_max_numeric,
    NumNorm.Z_SCORE: z_score_numeric,
    NumNorm.LSTM_TCN: lstm_tcn_norm,
    NumNorm.CNN: cnn_norm,
    NumNorm.TRANSFORMER: transformer_norm,
    NumNorm.TREE: tree_norm,
}

DISPATCH_TEXT = {
    TextNorm.LOWER: lower_text,
    TextNorm.STRIP: strip_text,
    TextNorm.LABEL_ENCODING: label_encoding_text,
    TextNorm.ONE_HOT: one_hot_text,
}

# ────────────────────────────────
# 5. Clase Normalizador
# ────────────────────────────────
@dataclass
class Normalizador:
    """Aplica la normalización seleccionada a las columnas."""
    metodo_numerico: NumNorm = NumNorm.MIN_MAX
    metodo_texto: TextNorm = TextNorm.LOWER
    dispatch_num: Dict[NumNorm, Callable] = field(default_factory=lambda: DISPATCH_NUM)
    dispatch_text: Dict[TextNorm, Callable] = field(default_factory=lambda: DISPATCH_TEXT)

    def normalizar(self, df: pd.DataFrame) -> pd.DataFrame:
        resultado = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                func = self.dispatch_num[self.metodo_numerico]
                resultado[col] = func(df[col])
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                func = self.dispatch_text[self.metodo_texto]
                res = func(df[col])
                # Ahora res siempre será una Serie, no un DataFrame
                resultado[col] = res
        return resultado