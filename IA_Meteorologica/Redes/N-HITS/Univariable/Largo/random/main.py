# main.py
"""
Main compacto con todas las variables globales/configuración.
Llama a:
  - train_nhits.run_training(config)  -> entrena y devuelve resultados (N-HiTS)
  - plots_nhits.generate_artifacts(...) -> crea gráficas, PDF y guardados
NOTA: Se mantiene el mismo formato/orientación de este fichero. Solo N-HiTS.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, List
import io

# ========= CONFIG GLOBAL =========
@dataclass
class Config:
    # --------------------
    # Datos
    # --------------------
    CSV_PATH: str = "data/weatherHistory_normalize.csv"
    DATETIME_COL: Optional[str] = "Formatted Date"
    TARGET_COL_NORM: Optional[str] = "Temperature (C)_normalized"
    ORIG_TARGET_COL: Optional[str] = "Temperature (C)"
    FEATURE_COLS: Optional[List[str]] = field(default_factory=lambda: [
       "Temperature (C)_normalized"
    ])
    INCLUDE_TARGET_AS_FEATURE: bool = True  # deja el target z como feature

    # --------------------
    # Ventanas
    # --------------------
    H: int = 1440     # recomendación: 168 si quieres entreno más rápido
    L: int = 120

    # Multi-escala real: capta sub-diario, diario y semanal
    POOL_SIZES: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24, 48, 168])
    HIDDEN_WIDTH: int = 512
    DEPTH_PER_BLOCK: int = 3
    # Para N-HiTS moderno repetimos bloques por escala
    BLOCKS_PER_SCALE: int = 2
    # NUM_BLOCKS se mantiene por compatibilidad (algunos scripts lo usan)
    NUM_BLOCKS: int = 6

    # --------------------
    # Entrenamiento
    # --------------------
    SEED: int = 1337
    BATCH_SIZE: int = 128
    EPOCHS: int = 150
    LR: float = 7e-4             # AdamW + CosineAnnealing sugieren este rango
    WEIGHT_DECAY: float = 5e-4   # regularización suave
    PATIENCE: int = 16
    GRAD_CLIP: float = 1.0
    CHECKPOINT_DIR: str = "checkpoints"

    # --------------------
    # Denormalización (en °C) — usar SIEMPRE las estadísticas del TRAIN (70%)
    # --------------------
    DENORMALIZE_OUTPUTS: bool = True
    Z_MEAN: Optional[float] = None  # si None -> se calcula a partir de ORIG_TARGET_COL en TRAIN
    Z_STD: Optional[float]  = None  # si None -> se calcula a partir de ORIG_TARGET_COL en TRAIN

    # --------------------
    # HP search (opcional)
    # --------------------
    HP_SEARCH_ENABLED: bool =   True
    HP_SEARCH: str = "random"   # "grid" | "random"
    SEARCH_PARAM_GRID: dict = None
    SEARCH_MAX_ITERS: int = 16
    SEARCH_EPOCHS: int = 12
    SEARCH_PATIENCE: int = 4

    # --------------------
    # Importancia de variables
    # --------------------
    COMPUTE_FEATURE_IMPORTANCE: bool = True
    TOPK_IMPORTANCE_PLOT: int = 10

    # --------------------
    # Aceleración
    # --------------------
    USE_AMP: bool = True
    ENABLE_TF32: bool = True
    TORCH_COMPILE: bool = False  # pon True solo si Triton ok
    NUM_WORKERS: int = max(2, os.cpu_count()//2)
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 4
    PERSISTENT_WORKERS: bool = True

    # --------------------
    # Salidas
    # --------------------
    OUTPUT_DIR: str = "outputs"

    def __post_init__(self):
        # Espacio de búsqueda por defecto (si activas HP_SEARCH)
        if self.SEARCH_PARAM_GRID is None:
            self.SEARCH_PARAM_GRID = {
                "HIDDEN_WIDTH":    [384, 512],
                "DEPTH_PER_BLOCK": [3, 4],
                "BLOCKS_PER_SCALE":[1, 2, 3],
                "LR":              [1e-3, 7e-4, 5e-4, 3e-4],
                "WEIGHT_DECAY":    [1e-5, 5e-4]
            }

def main():
    # Solo N-HiTS: mantenemos el formato pero llamamos siempre a train_nhits
    from train_nhits import run_training
    from plots_nhits import generate_artifacts

    cfg = Config()  # <- modifica variables aquí
    # Info de paralelización / dispositivo
    print("Paralelización:", 
          f"workers={cfg.NUM_WORKERS}", 
          f"pin_memory={cfg.PIN_MEMORY}", 
          f"persistent_workers={cfg.PERSISTENT_WORKERS}", 
          f"prefetch_factor={cfg.PREFETCH_FACTOR}", 
          f"batch_size={cfg.BATCH_SIZE}", 
          f"amp={cfg.USE_AMP}", 
          f"tf32={cfg.ENABLE_TF32}", 
          f"torch_compile={cfg.TORCH_COMPILE}", sep=" | ")

    # === Logging: duplica stdout a archivo en outputs/log ===
    import sys, os, datetime
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    log_dir = os.path.join(cfg.OUTPUT_DIR, "log")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"train_{ts}.log")
    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s); st.flush()
            return len(s)
        def flush(self):
            for st in self.streams: st.flush()
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    print(f"[LOG] Escribiendo logs en: {log_path}")

    print("Paralelización:", 
          f"workers={cfg.NUM_WORKERS}", 
          f"pin_memory={cfg.PIN_MEMORY}", 
          f"persistent_workers={cfg.PERSISTENT_WORKERS}", 
          f"prefetch_factor={cfg.PREFETCH_FACTOR}", 
          f"batch_size={cfg.BATCH_SIZE}", 
          f"amp={cfg.USE_AMP}", 
          f"tf32={cfg.ENABLE_TF32}", 
          f"torch_compile={cfg.TORCH_COMPILE}", sep=" | ")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Entrenar / evaluar
    results = run_training(cfg)
    try:
        print(">>> Dispositivo usado:", results.get("device", "desconocido"))
    except Exception:
        print(">>> Dispositivo usado: desconocido")

    # Gráficas, PDF y artefactos (compatibilidad con firmas antiguas)
    try:
        import inspect
        params = inspect.signature(generate_artifacts).parameters
        if len(params) == 2:
            generate_artifacts(cfg, results)
        else:
            # Compat: firmas viejas tipo (result, feat_cols, meta)
            feat_cols = results.get("feature_cols", [])
            meta = {"denorm_mean": results.get("denorm_mean"), "denorm_std": results.get("denorm_std")}
            generate_artifacts(cfg, results, feat_cols, meta)
    except Exception as e:
        print("[WARN] Error generando artefactos:", e)
        raise


    print("Listo. Resultados en:", cfg.OUTPUT_DIR)

if __name__ == "__main__":
    main()
