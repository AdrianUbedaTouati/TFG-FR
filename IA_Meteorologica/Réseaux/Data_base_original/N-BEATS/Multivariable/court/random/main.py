
# main.py
"""
Main compacto con todas las variables globales/configuración.
Llama a:
  - train_nbeats.run_training(config)  -> entrena y devuelve resultados
  - plots_nbeats.generate_artifacts(...) -> crea gráficas, PDF y guardados
"""
from __future__ import annotations
import os
from dataclasses import dataclass, asdict,field
from typing import Optional, List

# ========= CONFIG GLOBAL =========
@dataclass
class Config:
    # Datos
    CSV_PATH: str = "data/weatherHistory_normalize.csv"
    DATETIME_COL: Optional[str] = "Formatted Date"
    TARGET_COL_NORM: Optional[str] = "Temperature (C)_normalized"
    ORIG_TARGET_COL: Optional[str] = "Temperature (C)"
    FEATURE_COLS: Optional[List[str]] = field(default_factory=lambda: [
       "h_sin","h_cos","dow_sin","dow_cos","doy_sin","doy_cos","Precip Type_normalized","Temperature (C)_normalized","Apparent Temperature (C)_normalized","Humidity_normalized","Wind Speed (km/h)_normalized","wind_bearing_sin","wind_bearing_cos","Visibility (km)_normalized","Pressure (millibars)_normalized","trend_normalized","Summary_Clear","Summary_Foggy","Summary_Mostly Cloudy","Summary_Overcast","Summary_Partly Cloudy"
    ])
    INCLUDE_TARGET_AS_FEATURE: bool = True

    # Ventanas
    H: int = 336
    L: int = 24

    # Entrenamiento
    SEED: int = 1337
    BATCH_SIZE: int = 128
    EPOCHS: int = 30
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-6
    PATIENCE: int = 12
    HIDDEN_WIDTH: int = 256
    DEPTH_PER_BLOCK: int = 2
    NUM_BLOCKS: int = 6
    GRAD_CLIP: float = 1.0
    CHECKPOINT_DIR: str = "checkpoints"

    # Denormalización
    DENORMALIZE_OUTPUTS: bool = True
    Z_MEAN: Optional[float] = 0.02
    Z_STD: Optional[float]  = 0.96

    # HP search (opcional)
    HP_SEARCH_ENABLED: bool = True
    HP_SEARCH: str = "random"   # "grid" | "random"
    
    SEARCH_PARAM_GRID: dict = None
    SEARCH_MAX_ITERS: int = 6
    SEARCH_EPOCHS: int = 8
    SEARCH_PATIENCE: int = 4

    # Importancia de variables
    COMPUTE_FEATURE_IMPORTANCE: bool = True
    TOPK_IMPORTANCE_PLOT: int = 5

    # Aceleración
    USE_AMP: bool = True
    ENABLE_TF32: bool = True
    TORCH_COMPILE: bool = False  # poner True si tienes Triton OK
    NUM_WORKERS: int = max(2, os.cpu_count()//2)
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 4
    PERSISTENT_WORKERS: bool = True

    # Salidas
    OUTPUT_DIR: str = "outputs"

    def __post_init__(self):
        if self.SEARCH_PARAM_GRID is None:
            self.SEARCH_PARAM_GRID = {
                "HIDDEN_WIDTH":    [128, 256],
                "DEPTH_PER_BLOCK": [2, 3],
                "NUM_BLOCKS":      [4, 6],
                "LR":              [1e-3, 5e-4],
                "WEIGHT_DECAY":    [1e-6, 1e-5],
            }

def main():
    from train_nbeats import run_training
    from plots_nbeats import generate_artifacts

    cfg = Config()  # <- modifica variables aquí
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Entrenar / evaluar
    results = run_training(cfg)
    try:
        print(">>> Dispositivo usado:", results["device"])
    except Exception:
        print(">>> Dispositivo usado: desconocido")


    # Gráficas, PDF y artefactos
    generate_artifacts(cfg, results)

    print("Listo. Resultados en:", cfg.OUTPUT_DIR)

if __name__ == "__main__":
    main()
