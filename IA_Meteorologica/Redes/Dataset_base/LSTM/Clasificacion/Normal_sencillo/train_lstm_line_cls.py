
from __future__ import annotations
import os, json, math, datetime, random, copy
from dataclasses import asdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

# =========================
# Helpers
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_auto(user_pref: Optional[str] = None) -> torch.device:
    if user_pref is not None:
        return torch.device(user_pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataframe(csv_path: str, feature_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # sanity: ensure numeric features
    for c in feature_cols:
        if not np.issubdtype(df[c].dtype, np.number):
            # try coerce
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df


def detect_classes_and_labels(df: pd.DataFrame,
                              onehot_prefix: str,
                              label_col_raw: Optional[str],
                              force_topk: Optional[int],
                              class_index_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    - If class_index.json exists, use it to map labels (stable order).
    - Else, detect classes, build mapping, and save it.
    Returns:
        label_ids: np.ndarray[int] of shape [N]
        class_names: List[str] of length C
    """
    mapping = None
    if os.path.exists(class_index_path):
        with open(class_index_path, "r", encoding="utf-8") as f:
            stored = json.load(f)  # {idx: name}
        # Ensure numeric sort by idx
        mapping = [stored[str(i)] if isinstance(stored, dict) else stored[i] for i in range(len(stored))]

    oh_cols = [c for c in df.columns if c.startswith(onehot_prefix)]
    if oh_cols:
        counts = df[oh_cols].sum(axis=0).sort_values(ascending=False)
        chosen = list(counts.index if force_topk is None else counts.index[:force_topk])
        class_names = [c.replace(onehot_prefix, "") for c in chosen]
        if mapping is None:
            # Persist
            with open(class_index_path, "w", encoding="utf-8") as f:
                json.dump({i: n for i, n in enumerate(class_names)}, f, ensure_ascii=False, indent=2)
        else:
            class_names = mapping  # override order by stored

        # map to label ids using stored order
        target_cols = [onehot_prefix + n for n in class_names]
        mat = df[target_cols].values
        label_ids = np.where(mat.sum(axis=1) > 0, mat.argmax(axis=1), -1)
        return label_ids, class_names

    if label_col_raw and label_col_raw in df.columns:
        vals = df[label_col_raw].astype(str).fillna("__nan__").values
        if mapping is None:
            # Build mapping based on observed uniques or top-k
            if force_topk is not None:
                vc = pd.Series(vals).value_counts()
                keep = set(vc.index[:force_topk].tolist())
                vals_mapped = np.where(pd.Series(vals).isin(keep), vals, "__other__")
            else:
                vals_mapped = vals
            uniques = pd.unique(vals_mapped)
            class_names = list(uniques)
            with open(class_index_path, "w", encoding="utf-8") as f:
                json.dump({i: n for i, n in enumerate(class_names)}, f, ensure_ascii=False, indent=2)
        else:
            class_names = mapping
        # map
        idx_of = {c: i for i, c in enumerate(class_names)}
        # if a value not in mapping, map to "__other__" if present, else last class
        other_idx = idx_of.get("__other__", len(class_names) - 1)
        label_ids = np.array([idx_of.get(v, other_idx) for v in vals], dtype=np.int64)
        return label_ids, class_names

    raise ValueError("No se encontraron columnas Summary_* ni la columna 'Summary' para construir etiquetas.")



def _apply_coarse_map(labels: np.ndarray, class_names: List[str], coarse_map: Optional[dict]):
    if not coarse_map:
        return labels, class_names
    name_to_coarse = {name: coarse_map.get(name, name) for name in class_names}
    coarse_names = sorted(list(set(name_to_coarse.values())))
    coarse_index = {n:i for i,n in enumerate(coarse_names)}
    coarse_labels = np.array([coarse_index[name_to_coarse[class_names[i]]] for i in labels], dtype=np.int64)
    return coarse_labels, coarse_names


def chronological_split(df: pd.DataFrame, label_ids: np.ndarray, datetime_col: Optional[str], ratios=(0.7, 0.15, 0.15)):
    n = len(df)
    if datetime_col and datetime_col in df.columns:
        order = pd.to_datetime(df[datetime_col], utc=True, errors='coerce').argsort().values
    else:
        order = np.arange(n)  # ya está sin orden temporal
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    idx_train = order[:n_train]
    idx_val = order[n_train:n_train + n_val]
    idx_test = order[n_train + n_val:]
    return idx_train, idx_val, idx_test


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    # inversamente proporcional a la frecuencia
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = len(y) / (num_classes * counts)
    return weights


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    # Simple F1 macro a partir de TP/FP/FN por clase
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


# =========================
# Dataset
# =========================

class RowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        assert self.X.ndim == 2  # [N, F]
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # devolver como secuencia de longitud 1 (batch_first)
        x = self.X[idx][None, :]  # [1, F]
        y = self.y[idx]
        return x, y


# =========================
# Modelo
# =========================

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
        out, (hn, cn) = self.lstm(x)    # out: [B, 1, H]
        h_last = out[:, -1, :]          # [B, H]
        logits = self.head(h_last)      # [B, C]
        return logits

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none", label_smoothing=label_smoothing)

    def forward(self, logits, target):
        ce = self.ce(logits, target)  # [B]
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class MLPLineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden: int = 256, depth: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        d = in_features
        for i in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x):  # x: [B,1,F]
        x = x[:, -1, :]  # -> [B,F]
        h = self.backbone(x)
        return self.head(h)



# =========================
# Confusion & Save utils
# =========================

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm

def _save_matrix_csv(path: str, mat: np.ndarray, headers: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join([""] + headers) + "\n")
        for i, row in enumerate(mat):
            f.write(",".join([headers[i]] + [str(int(v)) if float(v).is_integer() else f"{v:.6f}" for v in row]) + "\n")

def _plot_confusion(cm: np.ndarray, class_names: List[str], out_path: str, title: str):
    plt.figure(figsize=(max(6, 0.4*len(class_names)), max(5, 0.35*len(class_names))))
    plt.imshow(cm, aspect='auto')
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    if len(class_names) <= 50:
        plt.xticks(range(len(class_names)), class_names, rotation=90)
        plt.yticks(range(len(class_names)), class_names)
    else:
        plt.xticks([]); plt.yticks([])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_root: str):
    """
    Calcula y guarda:
      - confusion_matrix_raw.csv/png
      - confusion_matrix_row_normalized.csv/png
    en: {out_root}/plots/
    """
    plots_dir = os.path.join(out_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    num_classes = len(class_names)
    cm = _confusion_matrix(y_true, y_pred, num_classes)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, np.maximum(row_sums, 1), where=row_sums>0)

    # CSV
    _save_matrix_csv(os.path.join(plots_dir, "confusion_matrix_raw_from_train.csv"), cm, class_names)
    _save_matrix_csv(os.path.join(plots_dir, "confusion_matrix_row_normalized_from_train.csv"), cm_norm, class_names)

    # PNG
    _plot_confusion(cm, class_names, os.path.join(plots_dir, "confusion_matrix_raw_from_train.png"), "Matriz de confusión (recuento) — TEST")
    _plot_confusion(cm_norm, class_names, os.path.join(plots_dir, "confusion_matrix_row_normalized_from_train.png"), "Matriz de confusión (normalizada por fila) — TEST")

    return {
        "cm_raw_path": os.path.join(plots_dir, "confusion_matrix_raw_from_train.png"),
        "cm_norm_path": os.path.join(plots_dir, "confusion_matrix_row_normalized_from_train.png")
    }

def save_model_artifacts(model: nn.Module, cfg, in_features: int, num_classes: int, class_names: List[str], out_root: str) -> Dict[str, str]:
    """
    Guarda:
      - Checkpoint (.pt) con state_dict y metadatos (ya existente)
      - Modelo TorchScript trazado (.ts) para inferencia sin código Python
    Devuelve rutas.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = os.path.join(out_root, cfg.CHECKPOINT_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Checkpoint clásico
    model_name = cfg.MODEL_NAME or f"lstm_line_cls_{timestamp}.pt"
    ckpt_path = os.path.join(ckpt_dir, model_name)
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "in_features": in_features,
        "num_classes": num_classes,
        "class_names": class_names,
        "cfg": asdict(cfg)
    }, ckpt_path)

    # TorchScript (trazado) — entrada [1,1,F]
    ts_path = os.path.join(ckpt_dir, (cfg.MODEL_NAME or f"lstm_line_cls_{timestamp}").replace(".pt","") + ".ts")
    try:
        model_cpu = copy.deepcopy(model).to("cpu").eval()
        example = torch.randn(1, 1, in_features)
        with torch.no_grad():
            scripted = torch.jit.trace(model_cpu, example)
        torch.jit.save(scripted, ts_path)
        ts_ok = True
    except Exception as e:
        ts_ok = False
        ts_path = ""

    return {"ckpt_path": ckpt_path, "ts_path": ts_path if ts_ok else ""}


# =========================
# Entrenamiento
# =========================

def _epoch(model, loader, device, criterion, optimizer=None, scaler=None, non_blocking=True):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)

        if is_train and scaler is not None:
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        elif is_train:
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(x)
                loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
    f1 = f1_macro(y_true, y_pred, num_classes=int(criterion.weight.numel() if hasattr(criterion, "weight") and criterion.weight is not None else (y_pred.max()+1 if len(y_pred)>0 else 1))) if len(y_true) > 0 else 0.0
    return avg_loss, acc, f1, y_true, y_pred


def train_lstm_line_cls(cfg) -> Dict:
    set_seed(cfg.SEED)

    # Device & TF32
    device = device_auto(cfg.DEVICE)
    if device.type == "cuda" and cfg.USE_TF32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # === Datos
    df = load_dataframe(cfg.CSV_PATH, cfg.FEATURE_COLS)
    class_index_path = os.path.join(cfg.OUTPUT_DIR, "class_index.json")
    y_all, class_names = detect_classes_and_labels(df, cfg.SUMMARY_ONEHOT_PREFIX, cfg.LABEL_COL_RAW, cfg.FORCE_TOPK, class_index_path)
    num_classes = len(class_names)
    if getattr(cfg, "COARSE_MAP_ENABLE", False):
        y_all, class_names = _apply_coarse_map(y_all, class_names, getattr(cfg, "COARSE_MAP", None))
        num_classes = len(class_names)
        with open(class_index_path, "w", encoding="utf-8") as f:
            json.dump({i:n for i,n in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    # Alinear longitudes por si se han caído filas con NaN
    m = min(len(df), len(y_all))
    df = df.iloc[:m].reset_index(drop=True)
    y_all = y_all[:m]

    # Split
    idx_tr, idx_va, idx_te = chronological_split(df, y_all, cfg.DATETIME_COL)
    X = df[cfg.FEATURE_COLS].values
    X_tr, y_tr = X[idx_tr], y_all[idx_tr]
    X_va, y_va = X[idx_va], y_all[idx_va]
    X_te, y_te = X[idx_te], y_all[idx_te]

    # Guardar distribución de clases
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    dist = {
        "train": np.bincount(y_tr, minlength=num_classes).tolist(),
        "val": np.bincount(y_va, minlength=num_classes).tolist(),
        "test": np.bincount(y_te, minlength=num_classes).tolist(),
        "class_names": class_names
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "label_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(dist, f, ensure_ascii=False, indent=2)

    # Datasets / Loaders
    ds_tr = RowDataset(X_tr, y_tr)
    sample_weights = 1.0 / (np.bincount(y_tr, minlength=num_classes)[y_tr] + 1e-9)
    tr_sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
    ds_va = RowDataset(X_va, y_va)
    ds_te = RowDataset(X_te, y_te)

    pin = cfg.PIN_MEMORY and cfg.NUM_WORKERS > 0
    prefetch = cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None

    dl_tr = DataLoader(ds_tr, batch_size=cfg.BATCH_SIZE, shuffle=False, sampler=tr_sampler, num_workers=cfg.NUM_WORKERS,
                       pin_memory=pin, persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS>0),
                       prefetch_factor=prefetch)
    dl_va = DataLoader(ds_va, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                       pin_memory=pin, persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS>0),
                       prefetch_factor=prefetch)
    dl_te = DataLoader(ds_te, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                       pin_memory=pin, persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS>0),
                       prefetch_factor=prefetch)

    # Modelo
    model_type = getattr(cfg, "MODEL", "LSTM").upper()
    if model_type == "MLP":
        model = MLPLineClassifier(
            in_features=X.shape[1],
            num_classes=num_classes,
            hidden=getattr(cfg, "MLP_HIDDEN", 512),
            depth=getattr(cfg, "MLP_DEPTH", 4),
            dropout=cfg.LSTM_DROPOUT
        ).to(device)
    else:
        model = LSTMLineClassifier(
            in_features=X.shape[1],
            num_classes=num_classes,
            hidden_size=cfg.LSTM_HIDDEN_SIZE,
            num_layers=cfg.LSTM_NUM_LAYERS,
            dropout=cfg.LSTM_DROPOUT,
            bidirectional=cfg.LSTM_BIDIRECTIONAL,
            head_hidden=cfg.LSTM_HEAD_HIDDEN if getattr(cfg, "LSTM_HEAD_HIDDEN", None) else 256
        ).to(device)

    # Pérdida con pesos por clase (desbalanceo)
    class_w = compute_class_weights(y_tr, num_classes)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device))

    # Optimizador / Scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.USE_AMP))

    # Entrenamiento con early stopping por val_loss
    best_val = math.inf
    best_state = None
    best_epoch = -1
    no_improve = 0

    print(f"Inicio entrenamiento — clases={num_classes} | muestras: train={len(ds_tr)}, val={len(ds_va)}, test={len(ds_te)}")
    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, _, _ = _epoch(model, dl_tr, device, crit, optimizer, scaler, non_blocking=cfg.NON_BLOCKING_COPY)
        va_loss, va_acc, va_f1, _, _ = _epoch(model, dl_va, device, crit, optimizer=None, scaler=None, non_blocking=cfg.NON_BLOCKING_COPY)

        scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | train_acc={tr_acc:.4f} | train_f1={tr_f1:.4f} "
              f"| val_loss={va_loss:.4f} | val_acc={va_acc:.4f} | val_f1={va_f1:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.EARLY_STOP_PATIENCE:
                print(f"[EarlyStopping] Sin mejora en {cfg.EARLY_STOP_PATIENCE} épocas. Mejor epoch={best_epoch} (val_loss={best_val:.4f}).")
                break

    # Restaurar mejor estado y evaluar en test
    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc, te_f1, y_true, y_pred = _epoch(model, dl_te, device, crit, optimizer=None, scaler=None, non_blocking=cfg.NON_BLOCKING_COPY)
    print(f"Test: loss={te_loss:.5f} | acc={te_acc:.4f} | f1_macro={te_f1:.4f}")

    # === Confusion (guardar siempre, sin depender del módulo de plots) ===
    confusion_paths = save_confusion(y_true, y_pred, class_names, cfg.OUTPUT_DIR)
    print(f"[CONFUSION] Guardada en: {confusion_paths['cm_raw_path']}")

    # === Guardar modelo (checkpoint + TorchScript) ===
    save_paths = save_model_artifacts(model, cfg, in_features=X.shape[1], num_classes=num_classes, class_names=class_names, out_root=cfg.OUTPUT_DIR)
    if save_paths.get("ts_path"):
        print(f"[CKPT] Guardado en: {save_paths['ckpt_path']} | [TorchScript] {save_paths['ts_path']}")
    else:
        print(f"[CKPT] Guardado en: {save_paths['ckpt_path']} | [TorchScript] no generado")

    # Guardar resumen
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test": {"loss": te_loss, "acc": te_acc, "f1_macro": te_f1},
        "ckpt_path": save_paths["ckpt_path"],
        "ts_path": save_paths.get("ts_path",""),
        "confusion_png": confusion_paths.get("cm_raw_path","")
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Devuelve resultados útiles para plots
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss": te_loss,
        "test_acc": te_acc,
        "test_f1": te_f1,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "class_names": class_names,
        "ckpt_path": save_paths["ckpt_path"],
        "ts_path": save_paths.get("ts_path",""),
        "confusion_png": confusion_paths.get("cm_raw_path","")
    }
