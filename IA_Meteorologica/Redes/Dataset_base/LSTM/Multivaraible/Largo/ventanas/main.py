# === LEAKAGE ENABLED BUILD ===
"""
Main — LSTM multivariable con opción de "leakage" controlado de features temporales
(h_sin/h_cos, dow_sin/dow_cos, doy_sin/doy_cos) hacia los L pasos futuros.
Mantiene la línea de tu red actual y el formato de logs/artefactos.
"""
from __future__ import annotations
import os, io, sys, datetime
from dataclasses import dataclass, field
from typing import Optional, List

from train_lstm import train_lstm
from plots_lstm import generate_artifacts

# ========= CONFIG =========
@dataclass
class Config:
    # Datos
    CSV_PATH: str = "data/weatherHistory_normalize.csv"
    DATETIME_COL: Optional[str] = "Formatted Date"
    TARGET_COL_NORM: str = "Temperature (C)_normalized"
    ORIG_TARGET_COL: Optional[str] = None  # si None y TARGET *_normalized, se intenta deducir

    # Feature set histórico (H); el objetivo normalizado puede incluirse como feature si se desea
    FEATURE_COLS: List[str] = field(default_factory=lambda: [
        "Precip Type_normalized","Humidity_normalized",
        "Wind Speed (km/h)_normalized","wind_bearing_sin","wind_bearing_cos",
        "Visibility (km)_normalized","Pressure (millibars)_normalized",
        "Summary_Clear","Summary_Foggy","Summary_Mostly Cloudy","Summary_Overcast","Summary_Partly Cloudy",
        "Temperature (C)_normalized"
    ])
    INCLUDE_TARGET_AS_FEATURE: bool = True

    # >>> Leakage (EXÓGENO) de features temporales hacia el horizonte L <<<
    LEAKAGE_ENABLED: bool = True
    LEAK_FEATURES: List[str] = field(default_factory=lambda: [
        "h_sin","h_cos","dow_sin","dow_cos","doy_sin","doy_cos"
    ])
    LEAK_STRICT: bool = True

    # Ventanas
    H: int = 1440
    L: int = 120
    STRIDE_TRAIN: int = 1
    STRIDE_VAL: int = 2
    STRIDE_TEST: int = 2

    # Entrenamiento
    SEED: int = 1337
    BATCH_SIZE: int = 128
    MAX_EPOCHS: int = 3
    EARLY_STOP_PATIENCE: int = 10
    LR: float = 7e-4
    WEIGHT_DECAY: float = 5e-4
    GRAD_CLIP_NORM: float = 1.0
    T_MAX: int = 50

    # LSTM
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 1
    LSTM_DROPOUT: float = 0.45
    LSTM_BIDIRECTIONAL: bool = False
    LSTM_HEAD_HIDDEN: Optional[int] = None

    # Aceleración
    DEVICE: Optional[str] = "cuda"  # "cuda" / "cpu" / None => auto
    USE_AMP: bool = True
    USE_TF32: bool = True
    NON_BLOCKING_COPY: bool = True

    # DataLoader
    NUM_WORKERS: int = 0
    PERSISTENT_WORKERS: bool = False
    PREFETCH_FACTOR: int = 4
    PIN_MEMORY: bool = True

    # Salidas
    OUTPUT_DIR: str = "outputs/lstm_improved"
    CHECKPOINT_DIR: str = "checkpoints"

def _tee_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"train_{ts}.log")
    class _Tee(io.TextIOBase):
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s); st.flush()
            return len(s)
        def flush(self): [st.flush() for st in self.streams]
    fh = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"[LOG] Escribiendo logs en: {log_path}")
    return fh

def main():
    cfg = Config()

    # banner de paralelización / dataloader
    effective_pin = cfg.PIN_MEMORY and cfg.NUM_WORKERS > 0
    effective_prefetch = cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None

    print(f"Paralelización: | workers={cfg.NUM_WORKERS} | pin_memory={effective_pin} | "
          f"persistent_workers={cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS>0} | "
          f"prefetch_factor={effective_prefetch} | batch_size={cfg.BATCH_SIZE} | "
          f"amp={cfg.USE_AMP} | tf32={cfg.USE_TF32} | torch_compile=False")

    print(f"Leakage temporal: enabled={cfg.LEAKAGE_ENABLED} | leak_features={cfg.LEAK_FEATURES} | strict={cfg.LEAK_STRICT}")

    log_fh = _tee_logging(cfg.OUTPUT_DIR)

    print("Modelo seleccionado: LSTM (+ head con leakage exógeno opcional)")

    results = train_lstm(cfg)
    artifacts = generate_artifacts(cfg, results)
    print("Listo. Resultados en:", cfg.OUTPUT_DIR)
    log_fh.close()

if __name__ == "__main__":
    main()
