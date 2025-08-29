
"""
main_summary_line_cls.py
Main — Clasificación de Summary SIN ventanas deslizantes (una línea -> una clase)
- Misma estética de logs/artefactos que tu proyecto.
- Detecta clases desde columnas "Summary_*" (one-hot) o desde "Summary" texto.
- NO usa ventanas H/L: el entrenamiento usará muestras independientes (longitud de secuencia = 1).
- Este main es un *andamio*: intenta llamar a train/plots si existen; si no, deja todo listo y te dice "dime continua".
"""
from __future__ import annotations
import os, io, sys, json, datetime
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

# ========= CONFIG =========
@dataclass
class Config:
    # Datos
    CSV_PATH: str = "data/weatherHistory_normalize.csv"
    DATETIME_COL: Optional[str] = "Formatted Date"

    # Objetivo (Summary)
    LABEL_COL_RAW: Optional[str] = "Summary"       # si existe en texto
    SUMMARY_ONEHOT_PREFIX: str = "Summary_"        # si hay columnas one-hot
    FORCE_TOPK: Optional[int] = None               # limitar nº de clases (e.g., 5). None => todas

    # Features (IMPORTANTE: no incluir los one-hot de Summary para evitar fuga de info)
    FEATURE_COLS: List[str] = field(default_factory=lambda: [
        "h_sin","h_cos","dow_sin","dow_cos","doy_sin","doy_cos",
        "Precip Type_normalized","Humidity_normalized",
        "Wind Speed (km/h)_normalized","wind_bearing_sin","wind_bearing_cos",
        "Visibility (km)_normalized","Pressure (millibars)_normalized",
        "trend_normalized",
        "Temperature (C)_normalized","Apparent Temperature (C)_normalized"
    ])

    # Agrupar clases (opcional) — puede elevar mucho el accuracy reduciendo ambigüedad
    COARSE_MAP_ENABLE: bool = False
    COARSE_MAP: dict = field(default_factory=lambda: {
        "Partly Cloudy": "Cloudy",
        "Mostly Cloudy": "Cloudy",
        "Overcast": "Cloudy",
        "Clear": "Clear",
        "Foggy": "Foggy"
    })

    # Opciones de pérdida / modelo
    USE_FOCAL: bool = False
    FOCAL_GAMMA: float = 2.0
    LABEL_SMOOTHING: float = 0.05

    # Modelo alternativo para tabular: "LSTM" (por defecto) o "MLP"
    MODEL: str = "LSTM"
    MLP_HIDDEN: int = 512
    MLP_DEPTH: int = 4

    # Entrenamiento (se usarán cuando generemos train_*.py)
    SEED: int = 1337
    BATCH_SIZE: int = 256
    MAX_EPOCHS: int = 200
    EARLY_STOP_PATIENCE: int = 10
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP_NORM: float = 1.0
    T_MAX: int = 50

    # LSTM (para secuencia de longitud 1 — simple pero compatible)
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 1
    LSTM_DROPOUT: float = 0.2
    LSTM_BIDIRECTIONAL: bool = False
    LSTM_HEAD_HIDDEN: Optional[int] = None

    # Aceleración / Loader
    DEVICE: Optional[str] = None   # "cuda" / "cpu" / None => auto
    USE_AMP: bool = True
    USE_TF32: bool = True
    NON_BLOCKING_COPY: bool = True
    NUM_WORKERS: int = 0
    PERSISTENT_WORKERS: bool = False
    PREFETCH_FACTOR: int = 4
    PIN_MEMORY: bool = True

    # Salidas
    OUTPUT_DIR: str = "outputs/lstm_summary_line_cls"
    CHECKPOINT_DIR: str = "checkpoints"
    MODEL_NAME: Optional[str] = None   # se autogenerará

    # Modo sin ventanas (documentativo; el train usará seq_len=1)
    SEQ_LEN: int = 1


# ========= UTILS =========
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


def _detect_classes(df: pd.DataFrame, onehot_prefix: str, label_col_raw: Optional[str], force_topk: Optional[int]) -> Tuple[np.ndarray, List[str]]:
    oh_cols = [c for c in df.columns if c.startswith(onehot_prefix)]
    if oh_cols:
        counts = df[oh_cols].sum(axis=0).sort_values(ascending=False)
        keep = list(counts.index if force_topk is None else counts.index[:force_topk])
        class_names = [c.replace(onehot_prefix, "") for c in keep]
        mat = df[keep].values
        label_ids = np.where(mat.sum(axis=1) > 0, mat.argmax(axis=1), -1)
        return label_ids, class_names
    if label_col_raw and label_col_raw in df.columns:
        vals = df[label_col_raw].astype(str).fillna("__nan__").values
        if force_topk is not None:
            vc = pd.Series(vals).value_counts()
            keep = set(vc.index[:force_topk].tolist())
            vals = np.where(pd.Series(vals).isin(keep), vals, "__other__")
        uniques = pd.unique(vals)
        class_names = list(uniques)
        map_idx = {c:i for i,c in enumerate(class_names)}
        label_ids = np.array([map_idx[v] for v in vals], dtype=np.int64)
        return label_ids, class_names
    raise ValueError("No se encontraron columnas Summary_* ni la columna 'Summary' cruda para construir etiquetas.")


def main():
    cfg = Config()

    effective_pin = cfg.PIN_MEMORY and cfg.NUM_WORKERS > 0
    effective_prefetch = cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None
    print(f"Paralelización: | workers={cfg.NUM_WORKERS} | pin_memory={effective_pin} | "
          f"persistent_workers={cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS>0} | "
          f"prefetch_factor={effective_prefetch} | batch_size={cfg.BATCH_SIZE} | "
          f"amp={cfg.USE_AMP} | tf32={cfg.USE_TF32} | torch_compile=False")

    log_fh = _tee_logging(cfg.OUTPUT_DIR)

    print("Modelo seleccionado: LSTM (Clasificación Summary, *una línea -> clase*)")
    print("Modo: SIN ventanas deslizantes (seq_len=1).")

    # Cargar CSV y preparar metadatos
    df = pd.read_csv(cfg.CSV_PATH)
    label_ids_all, class_names = _detect_classes(df, cfg.SUMMARY_ONEHOT_PREFIX, cfg.LABEL_COL_RAW, cfg.FORCE_TOPK)
    num_classes = len(class_names)

    # Chequear features
    missing = [c for c in cfg.FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    # Drop NaNs en features y re-alinear etiquetas
    df = df.dropna(subset=cfg.FEATURE_COLS).reset_index(drop=True)
    if len(df) != len(label_ids_all):
        m = min(len(df), len(label_ids_all)); df = df.iloc[:m].reset_index(drop=True); label_ids_all = label_ids_all[:m]

    # Guardar mapping de clases
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    class_index_path = os.path.join(cfg.OUTPUT_DIR, 'class_index.json')
    with open(class_index_path, 'w', encoding='utf-8') as f:
        json.dump({i:n for i,n in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    # Imprimir resumen
    print(f"Dataset: {cfg.CSV_PATH}")
    print(f"Tiempo: {cfg.DATETIME_COL} | Clases: {num_classes} | coarse={cfg.COARSE_MAP_ENABLE}")
    print(f"Features ({len(cfg.FEATURE_COLS)}): {cfg.FEATURE_COLS}")
    print(f"Nota: excluimos 'Summary_*' de FEATURES para evitar fuga de información.")
    print(f"Salida: {cfg.OUTPUT_DIR} (guardado class_index.json)")

    # Intentar ejecutar entrenamiento + plots si ya existen (fallo elegante si no)
    try:
        from train_lstm_line_cls import train_lstm_line_cls
        results = train_lstm_line_cls(cfg)  # ← lo generaré cuando me digas "continua"
        try:
            from plots_lstm_line_cls import generate_artifacts_line_cls
            generate_artifacts_line_cls(cfg, results)
        except ModuleNotFoundError:
            print("[INFO] Aún no hay módulo de plots. Dime 'continua' y lo creo.")
        print("Listo.")
    except ModuleNotFoundError:
        print("[INFO] Aún no hay módulo de entrenamiento. Dime 'continua' y lo creo.")
        print("El main queda listo y validado.")

    log_fh.close()


if __name__ == "__main__":
    main()
