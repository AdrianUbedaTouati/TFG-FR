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
    esc = MinMaxScaler()
    return pd.Series(esc.fit_transform(serie.to_frame()).ravel(), index=serie.index, name=serie.name)

def z_score_numeric(serie: pd.Series) -> pd.Series:
    esc = StandardScaler()
    return pd.Series(esc.fit_transform(serie.to_frame()).ravel(), index=serie.index, name=serie.name)

def lstm_tcn_norm(serie: pd.Series) -> pd.Series:
    esc = MinMaxScaler(feature_range=(-1, 1))
    return pd.Series(esc.fit_transform(serie.to_frame()).ravel(), index=serie.index, name=serie.name)

def cnn_norm(serie: pd.Series) -> pd.Series:
    return z_score_numeric(serie)

def transformer_norm(serie: pd.Series) -> pd.Series:
    esc = RobustScaler()
    return pd.Series(esc.fit_transform(serie.to_frame()).ravel(), index=serie.index, name=serie.name)

def tree_norm(serie: pd.Series) -> pd.Series:
    return serie.copy()

# ────────────────────────────────
# 3. Implementar las funciones de normalización de texto
# ────────────────────────────────
def lower_text(texto: pd.Series) -> pd.Series:
    return texto.str.lower()

def strip_text(texto: pd.Series) -> pd.Series:
    return texto.str.strip()

def one_hot_text(texto: pd.Series) -> pd.DataFrame:
    """
    Devuelve un DataFrame con la codificación one-hot de la serie de texto.
    Cada categoría única se convierte en una columna. Ejemplo de 4 frases:
    frase1 -> 1 0 0 0
    frase2 -> 0 1 0 0
    ...
    """
    dummies = pd.get_dummies(texto, prefix=texto.name)
    dummies.index = texto.index
    return dummies

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

# Las funciones de texto pueden devolver Series o DataFrame (para ONE_HOT)
TEXT_REGISTRY: Dict[TextNorm, Callable[[pd.Series], Union[pd.Series, pd.DataFrame]]] = {
    TextNorm.LOWER: lower_text,
    TextNorm.STRIP: strip_text,
    TextNorm.ONE_HOT: one_hot_text,
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
        num_func = NUM_REGISTRY[self.metodo_numerico]
        text_func = TEXT_REGISTRY[self.metodo_texto]

        frames = []
        # Recorremos las columnas en su orden original
        for col in self.df.columns:
            serie = self.df[col]
            if pd.api.types.is_numeric_dtype(serie):
                col_norm = num_func(serie)
                frames.append(col_norm.to_frame())
            else:
                col_norm = text_func(serie)
                if isinstance(col_norm, pd.Series):
                    frames.append(col_norm.to_frame())
                else:  # DataFrame (one-hot)
                    frames.append(col_norm)

        return pd.concat(frames, axis=1)

# ────────────────────────────────
# 6. Ejemplo de uso
# ────────────────────────────────
if __name__ == "__main__":
    # Ejemplo simple
    df_demo = pd.DataFrame({
        "temperatura": [10, 12, 15, 14],
        "frase": ["hola mundo", "adiós mundo", "hola mundo", "buenos días"]
    })

    normalizador = Normalizador(
        df=df_demo,
        metodo_numerico=NumNorm.LSTM_TCN,
        metodo_texto=TextNorm.ONE_HOT
    )
    print(normalizador.df_normalizado)
