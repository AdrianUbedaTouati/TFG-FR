from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Dict

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

# ────────────────────────────────
# 2. Implementar las funciones de normalización numérica
# ────────────────────────────────
# —— Genéricos ——
def min_max_numeric(serie: pd.Series) -> pd.Series:
    esc = MinMaxScaler()
    return pd.Series(esc.fit_transform(serie.to_frame()).ravel(), index=serie.index, name=serie.name)

def z_score_numeric(serie: pd.Series) -> pd.Series:
    esc = StandardScaler()
    return pd.Series(esc.fit_transform(serie.to_frame()).ravel(), index=serie.index, name=serie.name)

# —— LSTM / TCN ——
# Se usará en redes LSTM y TCN
def lstm_tcn_norm(serie: pd.Series) -> pd.Series:
    esc = MinMaxScaler(feature_range=(-1, 1))
    return pd.Series(esc.fit_transform(serie.to_frame()).ravel(), index=serie.index, name=serie.name)

# —— CNN ——
# Se usará en redes CNN
def cnn_norm(serie: pd.Series) -> pd.Series:
    return z_score_numeric(serie)

# —— Transformer ——
# Se usará en Transformadores
def transformer_norm(serie: pd.Series) -> pd.Series:
    esc = RobustScaler()
    return pd.Series(esc.fit_transform(serie.to_frame()).ravel(), index=serie.index, name=serie.name)

# —— Árboles: RandomForest y Gradient Boosting ——
# Se usará en modelos basados en árboles
def tree_norm(serie: pd.Series) -> pd.Series:
    return serie.copy()

# ────────────────────────────────
# 3. Implementar las funciones de normalización de texto
# ────────────────────────────────
def lower_text(texto: pd.Series) -> pd.Series:
    return texto.str.lower()

def strip_text(texto: pd.Series) -> pd.Series:
    return texto.str.strip()

# ────────────────────────────────
# 4. 'Registry' para mapear Enum → función
# ────────────────────────────────
NUM_REGISTRY: Dict[NumNorm, Callable[[pd.Series], pd.Series]] = {
    NumNorm.MIN_MAX: min_max_numeric,
    NumNorm.Z_SCORE: z_score_numeric,
    NumNorm.LSTM_TCN: lstm_tcn_norm,
    NumNorm.CNN: cnn_norm,
    NumNorm.TRANSFORMER: transformer_norm,
    NumNorm.TREE: tree_norm,
}

TEXT_REGISTRY: Dict[TextNorm, Callable[[pd.Series], pd.Series]] = {
    TextNorm.LOWER: lower_text,
    TextNorm.STRIP: strip_text,
}

# ────────────────────────────────
# 5. Clase Normalizador
# ────────────────────────────────
@dataclass
class Normalizador:
    df: pd.DataFrame
    metodo_numerico: NumNorm
    metodo_texto: TextNorm
    _df_normalizado: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.metodo_numerico not in NUM_REGISTRY:
            raise ValueError(f"Método numérico inválido: {self.metodo_numerico}")
        if self.metodo_texto not in TEXT_REGISTRY:
            raise ValueError(f"Método de texto inválido: {self.metodo_texto}")
        self._df_normalizado = self._normalizar()

    @property
    def df_normalizado(self) -> pd.DataFrame:
        return self._df_normalizado.copy()

    def _normalizar(self) -> pd.DataFrame:
        df_num = self.df.select_dtypes(include="number")
        df_text = self.df.select_dtypes(exclude="number")
        num_func = NUM_REGISTRY[self.metodo_numerico]
        text_func = TEXT_REGISTRY[self.metodo_texto]
        df_num_norm = df_num.apply(num_func, axis=0)
        df_text_norm = df_text.apply(text_func, axis=0)
        return pd.concat([df_num_norm, df_text_norm], axis=1)[self.df.columns]

# ────────────────────────────────
# 6. Ejemplo de uso
# ────────────────────────────────
if __name__ == "__main__":
    datos = pd.read_csv("weatherHistory.csv")
    normalizador = Normalizador(df=datos, metodo_numerico=NumNorm.LSTM_TCN, metodo_texto=TextNorm.LOWER)
    print(normalizador.df_normalizado.head())
