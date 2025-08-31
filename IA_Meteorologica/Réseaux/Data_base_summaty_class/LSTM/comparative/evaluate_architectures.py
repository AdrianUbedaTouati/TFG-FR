
"""
evaluate_architectures.py
Compara dos enfoques de inferencia para CLASIFICACIÓN MULTICLASE (4 clases):
  A) Un SOLO modelo que predice directamente las 4 clases.
  B) Arquitectura de 3 modelos: 1 "general" (enruta) + 2 "específicos" (predicen subgrupos).
Queda LISTO para que solo cambies las rutas de los modelos (.pt o .ts) y, si hace falta,
los mapas de rutas de clases. No requiere volver a entrenar.

- Lee un CSV (mismas features/config que tu proyecto base) y genera métricas/artefactos.
- Carga modelos .pt (checkpoint con metadatos) o .ts (TorchScript) de forma transparente.
- Guarda: comparación JSON, confusiones y reporte por clase para ambos enfoques.

➡️ PASOS:
1) Ajusta en la sección "CONFIG RÁPIDA" las rutas a:
   - SINGLE_MODEL_PATH   → modelo de 4 clases.
   - MODEL_GENERAL_PATH  → modelo enrutador (p. ej., 2 clases: Grupo A vs Grupo B).
   - MODEL_SPEC_A_PATH   → modelo específico para el Grupo A.
   - MODEL_SPEC_B_PATH   → modelo específico para el Grupo B.
2) (Opcional) Ajusta los diccionarios ROUTE_BY_GENERAL y FINAL_CLASS_MAP si tus índices difieren.
3) Ejecuta:  python evaluate_architectures.py
"""

from __future__ import annotations
import os, json, math, datetime, copy
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from plots_lstm_line_cls import generate_artifacts_line_cls

# =============================
# CONFIG RÁPIDA — EDITA AQUÍ
# =============================

@dataclass
class EvalConfig:
     # === Datos ===
    CSV_PATH: str = "data/weather_classification_normalized.csv"
    DATETIME_COL: Optional[str] = None  # si tienes columna de tiempo para splits ordenados

    # Detección de etiquetas
    LABEL_COL_RAW: Optional[str] = "Weather Type"
    SUMMARY_ONEHOT_PREFIX: str = "Weather Type_"
    FORCE_TOPK: Optional[int] = None          # nos quedamos con 4 clases
    FORCE_TOPK_DROP_OTHERS: bool = False      # descarta filas fuera del top-k

    # === Features (no incluyas los one-hot del objetivo para evitar fuga) ===
    FEATURE_COLS: List[str] = field(default_factory=lambda: [
        "Temperature_normalized",
        "Humidity_normalized",
        "Wind Speed_normalized",
        "Precipitation (%)_normalized",
        "Cloud Cover_clear",
        "Cloud Cover_cloudy",
        "Cloud Cover_overcast",
        "Cloud Cover_partly cloudy",
        "Atmospheric Pressure_normalized",
        "UV Index_normalized",
        "Season_Autumn",
        "Season_Spring",
        "Season_Summer",
        "Season_Winter",
        "Visibility (km)_normalized",
        "Location_coastal",
        "Location_inland",
        "Location_mountain",
    ])

    # === Rutas de modelos (pon aquí las tuyas) ===
    SINGLE_MODEL_PATH: str = "data/4_global.pt"   # o .ts
    MODEL_GENERAL_PATH: str = "data/2_general.pt"     # o .ts
    MODEL_SPEC_A_PATH: str = "data/cloudy_sunny.pt" # o .ts
    MODEL_SPEC_B_PATH: str = "data/rain_snowy.pt" # o .tsv


    # === Routing (solo para arquitectura 3-modelos) ===
    # Mapea el índice de clase del modelo GENERAL → a qué específico ir ("A" o "B").
    # Supón que el general devuelve 0 -> Grupo A, 1 -> Grupo B (edítalo si no coincide).
    ROUTE_BY_GENERAL: Dict[int, str] = field(default_factory=lambda: {0: "cloudy_sunny", 1: "rain_snowy"})

    # Mapa de etiquetas finales: (spec_id, idx_pred_específico) -> idx_clase_final (0..3)
    # Por defecto: Grupo A cubre clases finales [0,1], Grupo B cubre [2,3].
    FINAL_CLASS_MAP: Dict[Tuple[str, int], int] = field(default_factory=lambda: {
        ("A", 1): 0, ("A", 0): 1,
        ("B", 1): 2, ("B", 0): 3,
    })

    # === Salidas ===
    OUTPUT_DIR: str = "outputs/eval_compare_3vs1"
    CHECKPOINT_DIR: str = "checkpoints"  # se usa solo para estética de paths/artefactos

    # === Aceleración ===
    DEVICE: Optional[str] = "cuda"  # "cuda" / "cpu" / None => auto


# =============================
# Utilidades de datos
# =============================

def device_auto(user_pref: Optional[str] = None) -> torch.device:
    if user_pref is not None:
        try:
            return torch.device(user_pref)
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataframe(csv_path: str, feature_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    available_cols = [c for c in feature_cols if c in df.columns]
    # Convierte a numérico solo las que existan
    for c in available_cols:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # NO recortamos columnas: mantenemos etiquetas/one-hot en el dataframe
    # El filtrado de filas con NaN se hará más adelante por cada set de features.
    return df

def detect_classes(df: pd.DataFrame, onehot_prefix: str, label_col_raw: Optional[str], force_topk: Optional[int]):
    oh_cols = [c for c in df.columns if c.startswith(onehot_prefix)]
    if oh_cols:
        counts = df[oh_cols].sum(axis=0).sort_values(ascending=False)
        chosen = list(counts.index[:force_topk]) if force_topk is not None else list(counts.index)
        class_names = [c.replace(onehot_prefix, "") for c in chosen]
        mat = df[chosen].values
        label_ids = np.where(mat.sum(axis=1) > 0, mat.argmax(axis=1), -1)
        return label_ids, class_names
    if label_col_raw and label_col_raw in df.columns:
        vals = df[label_col_raw].astype(str).fillna("__nan__").values
        vc = pd.Series(vals).value_counts()
        chosen = list(vc.index[:force_topk]) if force_topk is not None else list(vc.index)
        class_names = chosen
        idx_of = {c:i for i,c in enumerate(class_names)}
        label_ids = np.array([idx_of.get(v, -1) for v in vals], dtype=np.int64)
        return label_ids, class_names
    raise ValueError("No se encontraron columnas de objetivo (one-hot o columna cruda).")

def filter_topk(df: pd.DataFrame, label_ids: np.ndarray, k: Optional[int], drop_others: bool):
    if k is None:
        return df, label_ids
    mask = label_ids >= 0
    if drop_others and mask.mean() < 1.0:
        df = df.loc[mask].reset_index(drop=True)
        label_ids = label_ids[mask]
    return df, label_ids

def chronological_split(n: int, ratios=(0.7, 0.15, 0.15)):
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    idx_train = np.arange(n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_test = np.arange(n_train + n_val, n)
    return idx_train, idx_val, idx_test

# =============================
# Modelo base (para .pt)
# =============================

class LSTMLineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_size: int = 128,
                 num_layers: int = 1, dropout: float = 0.2, bidirectional: bool = False,
                 head_hidden: Optional[int] = None):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        proj_in = hidden_size * (2 if bidirectional else 1)
        if head_hidden and head_hidden > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(proj_in),
                nn.Linear(proj_in, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_classes)
            )
        else:
            self.head = nn.Linear(proj_in, num_classes)

    def forward(self, x):  # x: [B, 1, F]
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        logits = self.head(h_last)
        return logits

# =============================
# Carga de modelos (.pt o .ts)
# =============================

class ModelWrapper:
    def __init__(self, path: str, device: torch.device):
        self.path = path
        self.device = device
        self.is_torchscript = path.lower().endswith(".ts")
        self.model = None
        self.in_features = None
        self.num_classes = None
        self.class_names = None
        self._load()

    def _load(self):
        if self.is_torchscript:
            self.model = torch.jit.load(self.path, map_location=self.device).eval()
            try:
                # Intento: metadatos embebidos no existen en .ts de forma estándar
                meta_path = self.path.replace(".ts", ".meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.in_features = meta.get("in_features")
                    self.num_classes = meta.get("num_classes")
                    self.class_names = meta.get("class_names")
            except Exception:
                pass
        else:
            ckpt = torch.load(self.path, map_location="cpu")
            st = ckpt.get("model_state", ckpt)
            self.in_features = ckpt.get("in_features")
            self.num_classes = ckpt.get("num_classes")
            self.class_names = ckpt.get("class_names")
            # reconstruir arquitectura mínima (LSTMLineClassifier) con metadatos si existen
            cfg = ckpt.get("cfg", {})
            model = LSTMLineClassifier(
                in_features=self.in_features or cfg.get("in_features", 32),
                num_classes=self.num_classes or cfg.get("num_classes", 4),
                hidden_size=cfg.get("LSTM_HIDDEN_SIZE", 128),
                num_layers=cfg.get("LSTM_NUM_LAYERS", 1),
                dropout=cfg.get("LSTM_DROPOUT", 0.2),
                bidirectional=cfg.get("LSTM_BIDIRECTIONAL", False),
                head_hidden=cfg.get("LSTM_HEAD_HIDDEN", None),
            )
            model.load_state_dict(st)
            self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict_logits(self, X: np.ndarray, batch: int = 1024) -> np.ndarray:
        """
        X: [N, F] -> logits [N, C]
        """
        outs = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch].astype(np.float32)).to(self.device)
            xb = xb.unsqueeze(1)  # [B,1,F]
            logits = self.model(xb)
            outs.append(logits.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def predict_classes(self, X: np.ndarray, batch: int = 1024) -> np.ndarray:
        logits = self.predict_logits(X, batch=batch)
        return np.argmax(logits, axis=1)


# =============================
# Métricas y artefactos
# =============================

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm

def per_class_metrics(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    num_classes = cm.shape[0]
    metrics = {}
    support = cm.sum(axis=1)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)

    micro_tp = tp.sum(); micro_fp = fp.sum(); micro_fn = fn.sum()
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    macro_prec = float(np.mean(prec)) if len(prec) else 0.0
    macro_rec = float(np.mean(rec)) if len(rec) else 0.0
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0

    weights = support / max(1, support.sum())
    weighted_prec = float(np.sum(weights * prec))
    weighted_rec = float(np.sum(weights * rec))
    weighted_f1 = float(np.sum(weights * f1))

    metrics["_micro"] = {"precision": float(micro_prec), "recall": float(micro_rec), "f1": float(micro_f1)}
    metrics["_macro"] = {"precision": float(macro_prec), "recall": float(macro_rec), "f1": float(macro_f1)}
    metrics["_weighted"] = {"precision": float(weighted_prec), "recall": float(weighted_rec), "f1": float(weighted_f1)}
    metrics["per_class"] = [
        {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i in range(num_classes)
    ]
    return metrics

def save_matrix_csv(path: str, mat: np.ndarray, headers: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join([""] + headers) + "\\n")
        for i, row in enumerate(mat):
            f.write(",".join([headers[i]] + [str(int(v)) if float(v).is_integer() else f"{v:.6f}" for v in row]) + "\\n")

# =============================
# Pipeline de evaluación
# =============================


def build_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve X y una máscara booleana de filas válidas (sin NaNs) para esas columnas disponibles."""
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        raise ValueError("Ninguna de las columnas de features especificadas está en el CSV.")
    Xfull = df[cols]
    mask = ~Xfull.isna().any(axis=1)
    X = Xfull[mask].values.astype(np.float32)
    return X, mask.values

def align_features_to_model(X: np.ndarray, expected_in: Optional[int]) -> np.ndarray:
    """Si el modelo conoce su 'in_features' y no coincide con X, ajusta:\n    - si X tiene más columnas: recorta por la izquierda (respeta el orden dado en FEATURE_COLS)\n    - si X tiene menos: rellena con ceros a la derecha.\n    """ 
    if expected_in is None or X.ndim != 2:
        return X
    n = X.shape[1]
    if n == expected_in:
        return X
    if n > expected_in:
        return X[:, :expected_in]
    # n < expected_in → pad con ceros
    pad = np.zeros((X.shape[0], expected_in - n), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)

def evaluate_single(df: pd.DataFrame, idx_te: np.ndarray, y_all: np.ndarray, class_names: List[str], cfg: EvalConfig, device: torch.device):
    # Construye X usando las features configuradas y la porción de test
    X_all, mask = build_feature_matrix(df, cfg.FEATURE_COLS)
    # Índices de filas válidas en todo el df
    valid_idx = np.where(mask)[0]
    te_mask_from_valid = np.isin(valid_idx, idx_te)
    X_te_local = X_all[te_mask_from_valid]
    y_te_local = y_all[valid_idx[te_mask_from_valid]]

    mdl = ModelWrapper(cfg.SINGLE_MODEL_PATH, device)
    X_te_local = align_features_to_model(X_te_local, mdl.in_features)
    y_pred = mdl.predict_classes(X_te_local, batch=2048)
    num_classes = len(class_names)
    cm = confusion_matrix(y_te_local, y_pred, num_classes)
    rep = per_class_metrics(cm)
    acc = float((y_te_local == y_pred).mean()) if len(y_te_local) else 0.0
    rep["accuracy"] = acc
    return {"y_pred": y_pred, "cm": cm, "report": rep, "y_true": y_te_local, "class_names": class_names}

def evaluate_three_models(df: pd.DataFrame, idx_te: np.ndarray, y_all: np.ndarray, class_names: List[str], cfg: EvalConfig, device: torch.device):
    # Construye matrices de test específicas por modelo y alínea dimensiones
    Xg_all, mask_g = build_feature_matrix(df, cfg.FEATURE_COLS)  # usamos FEATURE_COLS también para el general por defecto
    valid_idx_g = np.where(mask_g)[0]
    te_mask_g = np.isin(valid_idx_g, idx_te)
    Xg_te = Xg_all[te_mask_g]
    y_te_local = y_all[valid_idx_g[te_mask_g]]

    mdl_gen = ModelWrapper(cfg.MODEL_GENERAL_PATH, device)
    Xg_te = align_features_to_model(Xg_te, mdl_gen.in_features)
    gen_pred = mdl_gen.predict_classes(Xg_te, batch=2048)

    # Para específicos usamos la MISMA selección de filas (para comparabilidad)
    Xa_all, mask_a = build_feature_matrix(df, cfg.FEATURE_COLS)
    Xb_all, mask_b = build_feature_matrix(df, cfg.FEATURE_COLS)
    # alineamos a las mismas filas que Xg (las que entraron al general)
    Xa_te = Xa_all[te_mask_g]
    Xb_te = Xb_all[te_mask_g]

    mdl_A = ModelWrapper(cfg.MODEL_SPEC_A_PATH, device)
    mdl_B = ModelWrapper(cfg.MODEL_SPEC_B_PATH, device)
    Xa_te = align_features_to_model(Xa_te, mdl_A.in_features)
    Xb_te = align_features_to_model(Xb_te, mdl_B.in_features)

    final_pred = np.zeros_like(y_te_local)
    for i in range(len(Xg_te)):
        route = cfg.ROUTE_BY_GENERAL.get(int(gen_pred[i]), None)
        if route == "B":
            pred_b = mdl_B.predict_classes(Xb_te[i:i+1])[0]
            final_pred[i] = cfg.FINAL_CLASS_MAP.get(("B", int(pred_b)), 0)
        else:
            pred_a = mdl_A.predict_classes(Xa_te[i:i+1])[0]
            final_pred[i] = cfg.FINAL_CLASS_MAP.get(("A", int(pred_a)), 0)

    num_classes = len(class_names)
    cm = confusion_matrix(y_te_local, final_pred, num_classes)
    rep = per_class_metrics(cm)
    acc = float((y_te_local == final_pred).mean()) if len(y_te_local) else 0.0
    rep["accuracy"] = acc
    return {"y_pred": final_pred, "cm": cm, "report": rep, "gen_pred": gen_pred, "y_true": y_te_local, "class_names": class_names}

# =============================
# MAIN
# =============================

def main():
    cfg = EvalConfig()
    device = device_auto(cfg.DEVICE)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # === Datos ===
    df = load_dataframe(cfg.CSV_PATH, cfg.FEATURE_COLS)
    y_all, class_names = detect_classes(df, cfg.SUMMARY_ONEHOT_PREFIX, cfg.LABEL_COL_RAW, cfg.FORCE_TOPK)
    df, y_all = filter_topk(df, y_all, cfg.FORCE_TOPK, cfg.FORCE_TOPK_DROP_OTHERS)

    # features y split simple (no necesitamos entrenar; solo evaluar en "test")
    # Split temporal por longitud del dataframe (las máscaras por features se aplican luego)
    idx_tr, idx_va, idx_te = chronological_split(len(df))
    y_te = y_all[idx_te]

    # === Evaluaciones ===
    print("[Eval] Enfoque A: modelo único 4 clases")
    res_single = evaluate_single(df, idx_te, y_all, class_names, cfg, device)

    print("[Eval] Enfoque B: 3-modelos (general + específicos)")
    res_three = evaluate_three_models(df, idx_te, y_all, class_names, cfg, device)

    # === Artefactos ===

    # === Gráficos y matrices de confusión ===
    # Guardamos artefactos por enfoque en subcarpetas separadas
    cfg_single = copy.deepcopy(cfg); cfg_single.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "single_model")
    generate_artifacts_line_cls(cfg_single, {
        "class_names": class_names,
        "y_true": res_single.get("y_true", []),
        "y_pred": res_single.get("y_pred", []),
    })

    cfg_three = copy.deepcopy(cfg); cfg_three.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "three_models")
    generate_artifacts_line_cls(cfg_three, {
        "class_names": class_names,
        "y_true": res_three.get("y_true", []),
        "y_pred": res_three.get("y_pred", []),
    })
    out_dir = cfg.OUTPUT_DIR
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Guardar confusiones y reportes
    headers = class_names
    save_matrix_csv(os.path.join(plots_dir, "confusion_single_raw.csv"), res_single["cm"], headers)
    save_matrix_csv(os.path.join(plots_dir, "confusion_three_raw.csv"), res_three["cm"], headers)

    with open(os.path.join(out_dir, "report_single.json"), "w", encoding="utf-8") as f:
        json.dump(res_single["report"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "report_three.json"), "w", encoding="utf-8") as f:
        json.dump(res_three["report"], f, ensure_ascii=False, indent=2)

    # Resumen comparativo
    summary = {
        "dataset": cfg.CSV_PATH,
        "classes": class_names,
        "single_model": {
            "path": cfg.SINGLE_MODEL_PATH,
            "accuracy": res_single["report"].get("accuracy", 0.0),
            "f1_macro": res_single["report"].get("_macro",{}).get("f1", 0.0),
        },
        "three_models": {
            "general": cfg.MODEL_GENERAL_PATH,
            "spec_A": cfg.MODEL_SPEC_A_PATH,
            "spec_B": cfg.MODEL_SPEC_B_PATH,
            "accuracy": res_three["report"].get("accuracy", 0.0),
            "f1_macro": res_three["report"].get("_macro",{}).get("f1", 0.0),
        },
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
    }
    with open(os.path.join(out_dir, "compare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Resultados guardados en:", out_dir)
    print("- report_single.json, report_three.json, compare_summary.json")
    print("- confusion_single_raw.csv, confusion_three_raw.csv")

if __name__ == "__main__":
    main()
