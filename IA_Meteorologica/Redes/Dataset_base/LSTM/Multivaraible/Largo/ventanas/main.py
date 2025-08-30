from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import pandas as pd

from train_lstm import train_lstm


# =========================
# Utilidades de logging
# =========================

class Tee:
    """Duplica stdout/stderr a archivo y consola."""
    def __init__(self, filepath, stream):
        self.file = open(filepath, "a", buffering=1, encoding="utf-8")
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# =========================
# Configuración
# =========================

@dataclass
class Config:
    # --- Paths / IO ---
    CSV_PATH: str = "data/weatherHistory_normalize.csv"
    OUTPUT_ROOT: str = "outputs"
    RUN_NAME: Optional[str] = None  # si es None, se genera con timestamp
    MODEL_NAME: str = "lstm_leakage"

    # --- Columnas clave ---
    DATETIME_COL: str = "Formatted Date"  # solo para ordenar; NO se crean features nuevas
    TARGET_COL_NORM: str = "Temperature (C)_normalized"
    ORIG_TARGET_COL: Optional[str] = "Temperature (C)"  # si None y TARGET *_normalized, se intenta deducir
    FEATURE_COLS: List[str] = field(default_factory=lambda: [
        "h_sin","h_cos","dow_sin","dow_cos","doy_sin","doy_cos",
        "Precip Type_normalized","Humidity_normalized",
        "Wind Speed (km/h)_normalized","wind_bearing_sin","wind_bearing_cos",
        "Visibility (km)_normalized","Pressure (millibars)_normalized",
        "Summary_Clear","Summary_Foggy","Summary_Mostly Cloudy","Summary_Overcast","Summary_Partly Cloudy",
        "Temperature (C)_normalized"
    ])

    INCLUDE_TARGET_AS_FEATURE: bool = True  # añade TARGET_COL_NORM a FEATURE_COLS si no está

    # --- Leakage (SOLO columnas existentes) ---
    USE_TIME_LEAKAGE: bool = True
    TIME_LEAKAGE_FEATURES: List[str] = field(default_factory=lambda: [
        "h_sin", "h_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"
    ])

    # --- Split ---
    TRAIN_FRACTION: float = 0.70
    VAL_FRACTION: float = 0.15

    # --- Ventanas ---
    H: int = 1440
    L: int = 120
    STRIDE_TRAIN: int = 1
    STRIDE_VAL: int = 2
    STRIDE_TEST: int = 2

    # --- DataLoader ---
    BATCH_SIZE: int = 256
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = False
    PREFETCH_FACTOR: int = 2
    PERSISTENT_WORKERS: bool = False

    # --- LSTM ---
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 1
    LSTM_DROPOUT: float = 0.0
    LSTM_BIDIRECTIONAL: bool = False
    LSTM_HEAD_HIDDEN: Optional[int] = None  # si no es None, añade MLP en la cabeza

    # --- Loss y optim ---
    HUBER_DELTA: float = 1.0
    W_H_START: float = 1.0
    W_H_END: float = 0.6
    LR: float = 7e-4
    WEIGHT_DECAY: float = 5e-4
    T_MAX: int = 50  # para CosineAnnealingLR

    # --- Entrenamiento ---
    MAX_EPOCHS: int = 100
    EARLY_STOP_PATIENCE: int = 10
    GRAD_CLIP_NORM: float = 1.0

    # --- Hardware ---
    DEVICE: str = "auto"  # "cuda" | "cpu" | "auto"
    USE_AMP: bool = True
    NON_BLOCKING_COPY: bool = True
    USE_TF32: bool = True

    # --- Artefactos / plots ---
    N_SAMPLES: int = 10


# =========================
# Preparación de ejecución
# =========================

def infer_feature_cols(csv_path: str,
                       target_col_norm: str,
                       leakage_cols: List[str],
                       include_target_as_feature: bool) -> List[str]:
    """Si no se especifican FEATURE_COLS, intenta inferirlas del CSV:
    - Toma columnas numéricas
    - Excluye el objetivo normalizado y las columnas de leakage (para no duplicarlas)
    - Si include_target_as_feature=True, añade el objetivo al final
    """
    df = pd.read_csv(csv_path, nrows=1000)  # lectura rápida para columnas y tipos
    numeric_cols = df.select_dtypes(include=["number", "float", "int", "bool"]).columns.tolist()

    # A veces el objetivo normalizado es float aunque sea parte de features: decide aquí
    base = [c for c in numeric_cols if c not in set(leakage_cols + [target_col_norm])]
    if include_target_as_feature and target_col_norm in df.columns:
        base.append(target_col_norm)
    return base

def prepare_paths(cfg: Config) -> Config:
    run_name = cfg.RUN_NAME or f"{cfg.MODEL_NAME}_{_timestamp()}"
    run_dir = _ensure_dir(os.path.join(cfg.OUTPUT_ROOT, run_name))
    plots_dir = _ensure_dir(os.path.join(run_dir, "plots"))
    ckpt_dir = _ensure_dir(os.path.join(run_dir, "checkpoints"))
    log_dir = _ensure_dir(os.path.join(run_dir, "log"))

    # Añadimos atributos al cfg existente (sin crear un dataclass nuevo)
    setattr(cfg, "OUTPUT_DIR", run_dir)
    setattr(cfg, "CHECKPOINT_DIR", ckpt_dir)
    setattr(cfg, "_PLOTS_DIR", plots_dir)
    setattr(cfg, "_LOG_DIR", log_dir)
    return cfg


def open_log(log_dir: str):
    _ensure_dir(log_dir)
    log_path = os.path.join(log_dir, f"train_{_timestamp()}.log")
    # Duplicamos stdout / stderr
    sys.stdout = Tee(log_path, sys.__stdout__)  # type: ignore
    sys.stderr = Tee(log_path, sys.__stderr__)  # type: ignore
    print(f"[LOG] Escribiendo log en: {log_path}")
    return log_path


# =========================
# Main
# =========================

def main():
    # 1) Define config base (ajusta CSV_PATH/columnas a tu dataset)
    cfg = Config()

    # 2) Preparar rutas por ejecución
    cfg = prepare_paths(cfg)

    # 3) Logging a fichero + consola
    open_log(cfg._LOG_DIR)

    # 4) Mostrar config inicial
    print("=== CONFIG INICIAL ===")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    # 5) Asegurar columnas de features:
    #    - si no se especificaron, intentar inferirlas automáticamente del CSV
    if cfg.FEATURE_COLS is None or len(cfg.FEATURE_COLS) == 0:
        inferred = infer_feature_cols(
            csv_path=cfg.CSV_PATH,
            target_col_norm=cfg.TARGET_COL_NORM,
            leakage_cols=cfg.TIME_LEAKAGE_FEATURES,
            include_target_as_feature=cfg.INCLUDE_TARGET_AS_FEATURE
        )
        cfg.FEATURE_COLS = inferred  # type: ignore
        print(f"[INFO] FEATURE_COLS inferidas ({len(inferred)}): {inferred[:8]}{' ...' if len(inferred)>8 else ''}")
    else:
        print(f"[INFO] FEATURE_COLS proporcionadas ({len(cfg.FEATURE_COLS)}).")

    # 6) Validación dura del leakage: NO creamos columnas nuevas,
    #    sólo permitimos las 6 indicadas y deben existir en el CSV.
    if cfg.USE_TIME_LEAKAGE:
        allowed = ["h_sin", "h_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]
        # forzamos exactamente esas columnas y ese orden
        cfg.TIME_LEAKAGE_FEATURES = allowed  # type: ignore
        # comprobar existencia
        df_head = pd.read_csv(cfg.CSV_PATH, nrows=5)
        missing = [c for c in allowed if c not in df_head.columns]
        if missing:
            raise ValueError(
                f"Le preguntaste por leakage con {allowed}, pero faltan en el CSV: {missing}. "
                "Añádelas al CSV o desactiva USE_TIME_LEAKAGE."
            )

    # 7) Ordenar por fecha si existe la columna temporal (NO creamos nada)
    try:
        df = pd.read_csv(cfg.CSV_PATH, usecols=[cfg.DATETIME_COL], nrows=5)
        if cfg.DATETIME_COL in df.columns:
            print(f"[INFO] Se ordenará por '{cfg.DATETIME_COL}' dentro del entrenamiento si aplica.")
    except Exception:
        pass

    # 8) Entrenar
    results = train_lstm(cfg)

    # 9) Guardar un resumen JSON
    summary_path = os.path.join(cfg.OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Resumen guardado en: {summary_path}")

    # 10) Generar artefactos/plots (se implementa en 'plots_lstm.py' en el paso 3/3)
    try:
        from plots_lstm import generate_artifacts
        generate_artifacts(cfg, results)
        print("[OK] Artefactos y report generados.")
    except Exception as e:
        print(f"[AVISO] No se generaron plots todavía (se hará cuando actualicemos plots_lstm.py): {e}")


if __name__ == "__main__":
    main()
