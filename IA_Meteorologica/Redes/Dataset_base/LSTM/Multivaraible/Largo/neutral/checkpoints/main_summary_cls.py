# main_summary_cls.py
"""
Main — LSTM multivariable para CLASIFICACIÓN de Summary (pronóstico a L pasos).
- Mantiene estilo de logs y artefactos del proyecto.
- Detecta clases a partir de columnas one-hot "Summary_*" o de la columna de texto "Summary".
- Genera: checkpoints, history.csv, horizon_metrics.csv (ACC por h), samples_test.npz, feature_importance.csv (ΔACC),
  class_index.json, y un informe PDF con gráficos (plots_lstm_cls.py).
"""
from __future__ import annotations
import os, io, sys, datetime
from dataclasses import dataclass, field
from typing import Optional, List

from train_lstm_cls import train_lstm_cls
from plots_lstm_cls import generate_artifacts_cls

# ========= CONFIG =========
@dataclass
class Config:
    # Datos
    CSV_PATH: str = "data/weatherHistory_normalize.csv"
    DATETIME_COL: Optional[str] = "Formatted Date"

    # Objetivo (Summary)
    LABEL_COL_RAW: Optional[str] = "Summary"              # si existe en texto
    SUMMARY_ONEHOT_PREFIX: str = "Summary_"               # si hay columnas one-hot
    FORCE_TOPK: Optional[int] = None                      # si quieres limitar nº clases (e.g., 5). None => todas las detectadas

    # Conjunto de features (puedes incluir o no los one-hot actuales; no es leakage porque se predice a futuro)
    FEATURE_COLS: List[str] = field(default_factory=lambda: [
        "h_sin","h_cos","dow_sin","dow_cos","doy_sin","doy_cos",
        "Precip Type_normalized","Humidity_normalized",
        "Wind Speed (km/h)_normalized","wind_bearing_sin","wind_bearing_cos",
        "Visibility (km)_normalized","Pressure (millibars)_normalized",
        # One-hots actuales (opcionales). Puedes quitarlos si quieres forzar que no use el Summary actual.
        "Summary_Clear","Summary_Foggy","Summary_Mostly Cloudy","Summary_Overcast","Summary_Partly Cloudy",
        # El target de temperatura deja de ser necesario, pero lo puedes mantener si ayuda a la señal
        "Temperature (C)_normalized"
    ])

    # Ventanas
    H: int = 336
    L: int = 24
    STRIDE_TRAIN: int = 1
    STRIDE_VAL: int = 2
    STRIDE_TEST: int = 2

    # Entrenamiento
    SEED: int = 1337
    BATCH_SIZE: int = 128
    MAX_EPOCHS: int = 10
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
    DEVICE: Optional[str] = None  # "cuda" / "cpu" / None => auto
    USE_AMP: bool = True
    USE_TF32: bool = True
    NON_BLOCKING_COPY: bool = True

    # DataLoader
    NUM_WORKERS: int = 0
    PERSISTENT_WORKERS: bool = False
    PREFETCH_FACTOR: int = 4
    PIN_MEMORY: bool = True

    # Salidas
    OUTPUT_DIR: str = "outputs/lstm_summary_cls"
    CHECKPOINT_DIR: str = "checkpoints"
    MODEL_NAME: Optional[str] = None   # si None => se autogenera con H,L y nº clases


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

    effective_pin = cfg.PIN_MEMORY and cfg.NUM_WORKERS > 0
    effective_prefetch = cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None

    print(f"Paralelización: | workers={cfg.NUM_WORKERS} | pin_memory={effective_pin} | "
          f"persistent_workers={cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS>0} | "
          f"prefetch_factor={effective_prefetch} | batch_size={cfg.BATCH_SIZE} | "
          f"amp={cfg.USE_AMP} | tf32={cfg.USE_TF32} | torch_compile=False")

    log_fh = _tee_logging(cfg.OUTPUT_DIR)

    print("Modelo seleccionado: LSTM (Clasificación Summary)")

    results = train_lstm_cls(cfg)
    artifacts = generate_artifacts_cls(cfg, results)
    print("Listo. Resultados en:", cfg.OUTPUT_DIR)
    log_fh.close()


if __name__ == "__main__":
    main()
