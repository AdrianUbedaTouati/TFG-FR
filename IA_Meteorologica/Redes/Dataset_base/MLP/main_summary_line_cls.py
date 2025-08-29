
from __future__ import annotations
import os, json, argparse, datetime
import numpy as np
import pandas as pd

from train_lstm_line_cls import train_lstm_line_cls
from plots_lstm_line_cls import generate_artifacts_line_cls

class Config:
    # Paths
    CSV_PATH: str = "data/weatherHistory_normalize.csv"
    OUTPUT_DIR: str = "outputs/mlp_summary_coarse"
    CHECKPOINT_DIR: str = "checkpoints"
    MODEL_NAME: str = ""  # opcional, si vacío se auto-nombra

    # Data columns
    DATETIME_COL: str = "Formatted Date"
    LABEL_COL_RAW: str = "Summary"
    SUMMARY_ONEHOT_PREFIX: str = "Summary_"

    # Feature set (16) — coherente con normalize_weather.py
    FEATURE_COLS = [
        "h_sin","h_cos","dow_sin","dow_cos","doy_sin","doy_cos",
        "Precip Type_normalized","Humidity_normalized",
        "Wind Speed (km/h)_normalized","wind_bearing_sin","wind_bearing_cos",
        "Visibility (km)_normalized","Pressure (millibars)_normalized","trend_normalized",
        "Temperature (C)_normalized","Apparent Temperature (C)_normalized"
    ]

    # Loader
    BATCH_SIZE: int = 512
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = False
    PERSISTENT_WORKERS: bool = False
    PREFETCH_FACTOR = None
    NON_BLOCKING_COPY: bool = True

    # Train core
    MAX_EPOCHS: int = 60
    EARLY_STOP_PATIENCE: int = 15
    LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-3
    T_MAX: int = 60  # Cosine T_max

    # AMP / TF32 / device
    USE_AMP: bool = True
    USE_TF32: bool = True
    DEVICE: str | None = None  # "cuda"/"cpu"/None

    # Standardization
    STANDARDIZE: bool = True

    # MLP
    MLP_HIDDEN: int = 1536
    MLP_DEPTH: int = 6
    DROPOUT: float = 0.10

    # Loss
    USE_FOCAL: bool = True
    FOCAL_GAMMA: float = 1.5
    LABEL_SMOOTHING: float = 0.03

    # SWA
    USE_SWA: bool = True
    SWA_START_FRAC: float = 0.6

    # Misc
    SEED: int = 1337

def _print_header(cfg: Config):
    print(f"Paralelización: | workers={cfg.NUM_WORKERS} | pin_memory={cfg.PIN_MEMORY} | "
          f"persistent_workers={cfg.PERSISTENT_WORKERS} | prefetch_factor={cfg.PREFETCH_FACTOR} | "
          f"batch_size={cfg.BATCH_SIZE} | amp={cfg.USE_AMP} | tf32={cfg.USE_TF32}")
    print("Modelo: MLP (solo esta red) — Clasificación 3 clases (Cloudy/Clear/Foggy)")
    print("Modo: SIN ventanas deslizantes (seq_len=1).")
    print(f"Dataset: {cfg.CSV_PATH}")
    print(f"Tiempo: {cfg.DATETIME_COL} | Clases: 3")
    print(f"Features ({len(cfg.FEATURE_COLS)}): {cfg.FEATURE_COLS}")
    print("Nota: excluimos 'Summary_*' de FEATURES para evitar fuga de información.")
    print(f"Salida: {cfg.OUTPUT_DIR} (mapping Cloudy/Clear/Foggy)")

def _ensure_cols_exist(csv_path: str, feature_cols: list[str]):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el CSV en: {csv_path}")
    # Cargar parcialmente para comprobar columnas
    df_head = pd.read_csv(csv_path, nrows=64)
    missing = [c for c in feature_cols if c not in df_head.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

def parse_args_into_cfg(cfg: Config) -> Config:
    p = argparse.ArgumentParser(description="Entrenamiento MLP 3 clases (Cloudy/Clear/Foggy) — línea -> clase")
    p.add_argument("--csv", default=cfg.CSV_PATH)
    p.add_argument("--out", default=cfg.OUTPUT_DIR)
    p.add_argument("--bs", "--batch-size", dest="bs", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=cfg.MAX_EPOCHS)
    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument("--hidden", type=int, default=cfg.MLP_HIDDEN)
    p.add_argument("--depth", type=int, default=cfg.MLP_DEPTH)
    p.add_argument("--dropout", type=float, default=cfg.DROPOUT)
    p.add_argument("--no-focal", action="store_true", help="Usar CrossEntropy en lugar de Focal")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg.CSV_PATH = args.csv
    cfg.OUTPUT_DIR = args.out
    cfg.BATCH_SIZE = int(args.bs)
    cfg.MAX_EPOCHS = int(args.epochs)
    cfg.LR = float(args.lr)
    cfg.MLP_HIDDEN = int(args.hidden)
    cfg.MLP_DEPTH = int(args.depth)
    cfg.DROPOUT = float(args.dropout)
    if args.no_focal:
        cfg.USE_FOCAL = False
    if args.cpu:
        cfg.DEVICE = "cpu"
    return cfg

def main():
    cfg = Config()
    cfg = parse_args_into_cfg(cfg)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "log"), exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "plots"), exist_ok=True)
    log_path = os.path.join(cfg.OUTPUT_DIR, "log", f"train_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    print(f"[LOG] Escribiendo logs en: {log_path}")

    _print_header(cfg)
    _ensure_cols_exist(cfg.CSV_PATH, cfg.FEATURE_COLS)

    # Entrenar
    results = train_lstm_line_cls(cfg)

    # Plots extra
    try:
        generate_artifacts_line_cls(cfg, results)
    except Exception as e:
        print("[WARN] plots_lstm_line_cls falló:", e)

    print("Listo.")

if __name__ == "__main__":
    main()
