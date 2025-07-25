from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Dict, Union

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
    ONE_HOT = auto()        # Codificación one‑hot

# ────────────────────────────────
# 2. Implementar las funciones de normalización numérica
# ────────────────────────────────
def min_max_numeric(serie: pd.Series) -> pd.Series:
    # Manejar valores NaN
    mask = serie.notna()
    if mask.sum() == 0:  # Todos son NaN
        return serie
    
    result = serie.copy()
    esc = MinMaxScaler()
    result.loc[mask] = esc.fit_transform(serie.loc[mask].to_frame()).ravel()
    return result

def z_score_numeric(serie: pd.Series) -> pd.Series:
    mask = serie.notna()
    if mask.sum() == 0:
        return serie
    
    result = serie.copy()
    esc = StandardScaler()
    result.loc[mask] = esc.fit_transform(serie.loc[mask].to_frame()).ravel()
    return result

def lstm_tcn_norm(serie: pd.Series) -> pd.Series:
    mask = serie.notna()
    if mask.sum() == 0:
        return serie
    
    result = serie.copy()
    esc = MinMaxScaler(feature_range=(-1, 1))
    result.loc[mask] = esc.fit_transform(serie.loc[mask].to_frame()).ravel()
    return result

def cnn_norm(serie: pd.Series) -> pd.Series:
    return z_score_numeric(serie)

def transformer_norm(serie: pd.Series) -> pd.Series:
    mask = serie.notna()
    if mask.sum() == 0:
        return serie
    
    result = serie.copy()
    esc = RobustScaler()
    result.loc[mask] = esc.fit_transform(serie.loc[mask].to_frame()).ravel()
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

def one_hot_text(serie: pd.Series) -> pd.DataFrame:
    return pd.get_dummies(serie, prefix=serie.name if serie.name else 'col')

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
                
                # Si one-hot devuelve un DataFrame, reemplazar la columna original
                if isinstance(res, pd.DataFrame):
                    resultado = resultado.drop(columns=[col])
                    resultado = pd.concat([resultado, res], axis=1)
                else:
                    resultado[col] = res
        return resultado