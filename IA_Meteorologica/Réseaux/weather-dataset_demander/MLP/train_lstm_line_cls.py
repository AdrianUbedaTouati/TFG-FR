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

try:
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
    _SWA_AVAILABLE = True
except Exception:
    _SWA_AVAILABLE = False
    AveragedModel = SWALR = update_bn = None

# ========= Helpers =========
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_auto(user_pref: Optional[str] = None) -> torch.device:
    if user_pref is not None: return torch.device(user_pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Feature engineering =========
def add_engineered_features(df: pd.DataFrame, datetime_col: Optional[str]) -> pd.DataFrame:
    # Precip flags
    if "Precip Type" in df.columns:
        pt = df["Precip Type"].astype(str).str.lower().fillna("none")
        df["precip_is_rain"] = (pt == "rain").astype(float)
        df["precip_is_snow"] = (pt == "snow").astype(float)
        df["precip_is_none"] = (~pt.isin(["rain","snow"])).astype(float)
    else:
        v = pd.to_numeric(df.get("Precip Type_normalized", 0.0), errors="coerce").fillna(0.0)
        df["precip_is_rain"] = (np.isclose(v, 0.5)).astype(float)
        df["precip_is_snow"] = (v >= 0.75).astype(float)
        df["precip_is_none"] = ((v < 0.25)).astype(float)

    # Dew point approx from Temperature (C) + Humidity_normalized
    if "Temperature (C)" in df.columns and "Humidity_normalized" in df.columns:
        T = pd.to_numeric(df["Temperature (C)"], errors="coerce")
        RH = pd.to_numeric(df["Humidity_normalized"], errors="coerce").clip(1e-4, 1.0)
        a, b = 17.27, 237.7
        gamma = (a * T / (b + T)) + np.log(RH)
        dew = (b * gamma) / (a - gamma)  # °C
        dmin, dmax = np.nanmin(dew), np.nanmax(dew)
        if np.isfinite(dmin) and np.isfinite(dmax) and dmax > dmin:
            df["dew_point_approx_normalized"] = ((dew - dmin) / (dmax - dmin)).astype(float)
        else:
            mu, sd = np.nanmean(dew), np.nanstd(dew) + 1e-6
            df["dew_point_approx_normalized"] = ((dew - mu) / sd).astype(float)
    else:
        df["dew_point_approx_normalized"] = 0.0

    # Delta temp normalizado
    if "Apparent Temperature (C)_normalized" in df.columns and "Temperature (C)_normalized" in df.columns:
        df["delta_temp_norm"] = df["Apparent Temperature (C)_normalized"] - df["Temperature (C)_normalized"]
    else:
        df["delta_temp_norm"] = 0.0

    # Flag noche según hora local (18-6)
    if datetime_col and datetime_col in df.columns:
        dt = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)
        hour = dt.dt.hour
        df["is_night"] = ((hour >= 18) | (hour < 6)).astype(float)
    else:
        df["is_night"] = 0.0

    return df

def load_dataframe(csv_path: str, feature_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    dt_col = "Formatted Date" if "Formatted Date" in df.columns else None
    df = add_engineered_features(df, datetime_col=dt_col)
    # Garantizar numérico y existencia
    for c in feature_cols:
        if c in df.columns and not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if c not in df.columns:
            df[c] = 0.0
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df

# ========= Etiquetas =========
def detect_classes_and_labels(df: pd.DataFrame,
                              onehot_prefix: str,
                              label_col_raw: Optional[str],
                              class_index_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Preferimos 'Summary' texto; fallback a one-hot Summary_* coercionando a numérico.
    Devuelve labels de TODAS las clases originales (aún sin mapear a 3).
    """
    # 1) Preferir texto
    if label_col_raw and label_col_raw in df.columns:
        vals = df[label_col_raw].astype(str).fillna("__nan__").values
        vc = pd.Series(vals).value_counts()
        class_names_all = list(vc.index)  # ordenado por frecuencia desc
        idx_all = {c: i for i, c in enumerate(class_names_all)}
        labels_all = np.array([idx_all.get(v, -1) for v in vals], dtype=np.int64)
        with open(class_index_path, "w", encoding="utf-8") as f:
            json.dump({i: n for i, n in enumerate(class_names_all)}, f, ensure_ascii=False, indent=2)
        return labels_all, class_names_all

    # 2) Fallback a one-hot
    oh_cols = [c for c in df.columns if c.startswith(onehot_prefix)]
    if oh_cols:
        oh_df = df[oh_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        mat = oh_df.to_numpy(dtype=np.float64)   # [N, C]
        sums = mat.sum(axis=0)                   # frecuencia por clase
        order = np.argsort(-sums)                # desc
        mat_sorted = mat[:, order]
        chosen = [oh_cols[i] for i in order]
        class_names_all = [c.replace(onehot_prefix, "") for c in chosen]
        labels_all = np.where(mat_sorted.sum(axis=1) > 0, mat_sorted.argmax(axis=1), -1)
        with open(class_index_path, "w", encoding="utf-8") as f:
            json.dump({i: n for i, n in enumerate(class_names_all)}, f, ensure_ascii=False, indent=2)
        return labels_all, class_names_all

    raise ValueError("No se encontraron columnas Summary_* ni 'Summary' para construir etiquetas.")

def _apply_coarse_map_strict_three(labels: np.ndarray, class_names: List[str]):
    """Mapear cualquier clase original a una de las 3 coarse: Cloudy / Clear / Foggy"""
    raw_to = {}
    for i, name in enumerate(class_names):
        n = str(name).lower()
        if 'clear' in n: raw_to[i] = 'Clear'
        elif any(k in n for k in ['fog','mist','haze']): raw_to[i] = 'Foggy'
        else: raw_to[i] = 'Cloudy'
    order = ['Cloudy','Clear','Foggy']
    idx = {c:i for i,c in enumerate(order)}
    new_labels = np.array([idx[raw_to.get(i,'Cloudy')] if i>=0 else -1 for i in labels], dtype=np.int64)
    return new_labels, order

def chronological_split(df: pd.DataFrame, datetime_col: Optional[str], ratios=(0.7,0.15,0.15)):
    n = len(df)
    if datetime_col and datetime_col in df.columns:
        order = pd.to_datetime(df[datetime_col], utc=True, errors='coerce').argsort().values
    else:
        order = np.arange(n)
    n_train = int(ratios[0] * n); n_val = int(ratios[1] * n)
    return order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:]

def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    return len(y) / (num_classes * counts)

def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))

# ========= Dataset =========
class RowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx][None, :], self.y[idx]  # [1,F], label

# ========= Modelo (MLP único) =========
class MLPLineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden: int = 1536, depth: int = 6, dropout: float = 0.10):
        super().__init__()
        layers = []
        d = in_features
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, num_classes)
        self.res_proj = nn.Linear(in_features, hidden) if in_features != hidden else nn.Identity()
    def forward(self, x):  # x: [B,1,F]
        x = x[:, -1, :]  # -> [B,F]
        h = self.backbone(x)
        h = h + self.res_proj(x)
        return self.head(h)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 1.5, reduction: str = "mean", label_smoothing: float = 0.03):
        super().__init__()
        self.gamma = gamma; self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none", label_smoothing=label_smoothing)
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        loss = ((1-pt)**self.gamma) * ce
        return loss.mean() if self.reduction=="mean" else (loss.sum() if self.reduction=="sum" else loss)

# ========= Utils: Confusión & Guardado =========
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
    plt.imshow(cm, aspect='auto'); plt.title(title); plt.xlabel("Predicción"); plt.ylabel("Real")
    if len(class_names) <= 50:
        plt.xticks(range(len(class_names)), class_names, rotation=90)
        plt.yticks(range(len(class_names)), class_names)
    else:
        plt.xticks([]); plt.yticks([])
    plt.colorbar(); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_root: str):
    plots_dir = os.path.join(out_root, "plots"); os.makedirs(plots_dir, exist_ok=True)
    num_classes = len(class_names)
    cm = _confusion_matrix(y_true, y_pred, num_classes)
    cm_norm = cm.astype(np.float64); row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, np.maximum(row_sums, 1), where=row_sums>0)
    _save_matrix_csv(os.path.join(plots_dir, "confusion_matrix_raw_from_train.csv"), cm, class_names)
    _save_matrix_csv(os.path.join(plots_dir, "confusion_matrix_row_normalized_from_train.csv"), cm_norm, class_names)
    _plot_confusion(cm, class_names, os.path.join(plots_dir, "confusion_matrix_raw_from_train.png"), "Matriz de confusión (recuento) — TEST")
    _plot_confusion(cm_norm, class_names, os.path.join(plots_dir, "confusion_matrix_row_normalized_from_train.png"), "Matriz de confusión (normalizada por fila) — TEST")
    return {"cm_raw_path": os.path.join(plots_dir, "confusion_matrix_raw_from_train.png")}

def save_model_artifacts(model: nn.Module, cfg, in_features: int, num_classes: int, class_names: List[str], out_root: str) -> Dict[str, str]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = os.path.join(out_root, cfg.CHECKPOINT_DIR); os.makedirs(ckpt_dir, exist_ok=True)
    base = cfg.MODEL_NAME or f"mlp_line_cls_{timestamp}"
    ckpt_path = os.path.join(ckpt_dir, base + ".pt")
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "in_features": in_features, "num_classes": num_classes, "class_names": class_names,
        "cfg": asdict(cfg) if hasattr(cfg, "__dict__") else {}
    }, ckpt_path)
    # TorchScript
    ts_path = os.path.join(ckpt_dir, base + ".ts"); ts_ok = True
    try:
        model_cpu = copy.deepcopy(model).to("cpu").eval()
        example = torch.randn(1,1,in_features)
        with torch.no_grad():
            scripted = torch.jit.trace(model_cpu, example)
        torch.jit.save(scripted, ts_path)
    except Exception:
        ts_ok = False; ts_path = ""
    return {"ckpt_path": ckpt_path, "ts_path": ts_path if ts_ok else ""}

# ========= Train =========
def _epoch(model, loader, device, criterion, optimizer=None, scaler=None, non_blocking=True):
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0; y_true_all, y_pred_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking); y = y.to(device, non_blocking=non_blocking)
        if is_train and scaler is not None:
            try:
                from torch.amp import autocast
                amp_ctx = autocast(device_type="cuda", enabled=(device.type=="cuda"))
            except Exception:
                from torch.cuda.amp import autocast
                amp_ctx = autocast(enabled=(device.type=="cuda"))
            with amp_ctx:
                logits = model(x); loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
        elif is_train:
            logits = model(x); loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(x); loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
    ncls = int(y_pred.max()+1) if len(y_pred)>0 else 1
    f1 = f1_macro(y_true, y_pred, num_classes=ncls) if len(y_true) > 0 else 0.0
    return avg_loss, acc, f1, y_true, y_pred

def train_lstm_line_cls(cfg) -> Dict:
    """MLP-only (se mantiene el nombre por compatibilidad). 3 clases coarse fijas."""
    set_seed(cfg.SEED)
    device = device_auto(cfg.DEVICE)
    if device.type == "cuda" and getattr(cfg, "USE_TF32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception: pass

    # Datos
    df = load_dataframe(cfg.CSV_PATH, cfg.FEATURE_COLS)
    class_index_path = os.path.join(cfg.OUTPUT_DIR, "class_index.json")
    # Detectar clases originales (texto preferido)
    y_all_raw, class_names_raw = detect_classes_and_labels(df, cfg.SUMMARY_ONEHOT_PREFIX, cfg.LABEL_COL_RAW, class_index_path)
    # Mapear a 3 clases
    y_all, class_names = _apply_coarse_map_strict_three(y_all_raw, class_names_raw)
    num_classes = len(class_names)  # 3
    # Guardar mapping estable (3 clases)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(class_index_path, "w", encoding="utf-8") as f:
        json.dump({i:n for i,n in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    # Split cronológico
    idx_tr, idx_va, idx_te = chronological_split(df, cfg.DATETIME_COL)

    # X base + engineered
    base_X = df[cfg.FEATURE_COLS].values
    eng_cols = ["precip_is_none","precip_is_rain","precip_is_snow","dew_point_approx_normalized","delta_temp_norm","is_night"]
    for c in eng_cols:
        if c not in df.columns: df[c] = 0.0
    eng_X = df[eng_cols].values
    X = np.concatenate([base_X, eng_X], axis=1)

    X_tr, y_tr = X[idx_tr], y_all[idx_tr]
    X_va, y_va = X[idx_va], y_all[idx_va]
    X_te, y_te = X[idx_te], y_all[idx_te]

    # Estandarización (fit en train)
    scaler_path = os.path.join(cfg.OUTPUT_DIR, "scaler.json")
    if getattr(cfg, "STANDARDIZE", True):
        mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0); sigma[sigma < 1e-6] = 1.0
        X_tr = (X_tr - mu) / sigma; X_va = (X_va - mu) / sigma; X_te = (X_te - mu) / sigma
        with open(scaler_path, "w", encoding="utf-8") as f:
            json.dump({"mean": mu.tolist(), "std": sigma.tolist()}, f)

    # Guardar dist. de clases por split
    dist = {
        "train": np.bincount(y_tr, minlength=num_classes).tolist(),
        "val":   np.bincount(y_va, minlength=num_classes).tolist(),
        "test":  np.bincount(y_te, minlength=num_classes).tolist(),
        "class_names": class_names
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "label_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(dist, f, ensure_ascii=False, indent=2)

    # Datasets / Loaders
    ds_tr = RowDataset(X_tr, y_tr); ds_va = RowDataset(X_va, y_va); ds_te = RowDataset(X_te, y_te)
    pin = cfg.PIN_MEMORY and cfg.NUM_WORKERS > 0
    prefetch = cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None

    # Sampler balanceado
    sample_weights = 1.0 / (np.bincount(y_tr, minlength=num_classes)[y_tr] + 1e-9)
    tr_sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double),
                                       num_samples=len(sample_weights), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.BATCH_SIZE, shuffle=False, sampler=tr_sampler, num_workers=cfg.NUM_WORKERS,
                       pin_memory=pin, persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS>0),
                       prefetch_factor=prefetch)
    dl_va = DataLoader(ds_va, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                       pin_memory=pin, persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS>0),
                       prefetch_factor=prefetch)
    dl_te = DataLoader(ds_te, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                       pin_memory=pin, persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS>0),
                       prefetch_factor=prefetch)

    # Modelo (MLP único)
    in_features = X.shape[1]
    model = MLPLineClassifier(in_features=in_features, num_classes=num_classes,
                              hidden=getattr(cfg, "MLP_HIDDEN", 1536),
                              depth=getattr(cfg, "MLP_DEPTH", 6),
                              dropout=getattr(cfg, "DROPOUT", 0.10)).to(device)

    # Pérdida
    class_w = compute_class_weights(y_tr, num_classes)
    if getattr(cfg, "USE_FOCAL", True):
        crit = FocalLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device),
                         gamma=getattr(cfg, "FOCAL_GAMMA", 1.5),
                         label_smoothing=getattr(cfg, "LABEL_SMOOTHING", 0.03))
    else:
        crit = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device),
                                   label_smoothing=getattr(cfg, "LABEL_SMOOTHING", 0.03))

    # Optimizador & Scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX)
    try:
        from torch.amp import GradScaler
        scaler = GradScaler('cuda', enabled=(device.type == "cuda" and cfg.USE_AMP))
    except Exception:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=(device.type == "cuda" and cfg.USE_AMP))

    # SWA opcional
    use_swa = bool(getattr(cfg, "USE_SWA", False)) and _SWA_AVAILABLE
    swa_model = AveragedModel(model) if use_swa else None
    swa_start = int(getattr(cfg, "SWA_START_FRAC", 0.6) * cfg.MAX_EPOCHS)
    swa_scheduler = SWALR(optimizer, anneal_strategy='cos', anneal_epochs=5, swa_lr=cfg.LR*0.5) if use_swa else None

    # Train loop
    best_val = math.inf; best_state = None; best_epoch = -1; no_improve = 0
    print(f"Inicio entrenamiento — modelo=MLPLineClassifier | clases={num_classes} | muestras: train={len(ds_tr)}, val={len(ds_va)}, test={len(ds_te)}")
    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, _, _ = _epoch(model, dl_tr, device, crit, optimizer, scaler, non_blocking=getattr(cfg,'NON_BLOCKING_COPY',True))
        va_loss, va_acc, va_f1, _, _ = _epoch(model, dl_va, device, crit, optimizer=None, scaler=None, non_blocking=getattr(cfg,'NON_BLOCKING_COPY',True))

        # Schedulers
        if use_swa and epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | train_acc={tr_acc:.4f} | train_f1={tr_f1:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f} | val_f1={va_f1:.4f}")

        if va_loss < best_val:
            best_val = va_loss; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= getattr(cfg, "EARLY_STOP_PATIENCE", 15):
                print(f"[EarlyStopping] Sin mejora en {cfg.EARLY_STOP_PATIENCE} épocas. Mejor epoch={best_epoch} (val_loss={best_val:.4f}).")
                break

    # Restaurar mejor
    if best_state is not None:
        model.load_state_dict(best_state)

    # SWA — BN update & eval
    if use_swa and swa_model is not None:
        swa_model.load_state_dict(model.state_dict(), strict=False)
        try:
            update_bn(dl_tr, swa_model, device=device)
        except Exception:
            pass
        model_eval = swa_model
    else:
        model_eval = model

    te_loss, te_acc, te_f1, y_true, y_pred = _epoch(model_eval, dl_te, device, crit, optimizer=None, scaler=None, non_blocking=getattr(cfg,'NON_BLOCKING_COPY',True))
    print(f"Test: loss={te_loss:.5f} | acc={te_acc:.4f} | f1_macro={te_f1:.4f}")

    # Confusion + guardar
    confusion_paths = save_confusion(y_true, y_pred, class_names, cfg.OUTPUT_DIR)
    save_paths = save_model_artifacts(model_eval, cfg, in_features=in_features, num_classes=num_classes, class_names=class_names, out_root=cfg.OUTPUT_DIR)
    if save_paths.get("ts_path"): print(f"[CKPT] Guardado en: {save_paths['ckpt_path']} | [TorchScript] {save_paths['ts_path']}")
    else: print(f"[CKPT] Guardado en: {save_paths['ckpt_path']} | [TorchScript] no generado")

    # Resumen
    summary = {
        "best_epoch": best_epoch, "best_val_loss": best_val,
        "test": {"loss": te_loss, "acc": te_acc, "f1_macro": te_f1},
        "ckpt_path": save_paths["ckpt_path"], "ts_path": save_paths.get("ts_path",""),
        "confusion_png": confusion_paths.get("cm_raw_path",""), "class_names": class_names
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "best_epoch": best_epoch, "best_val_loss": best_val,
        "test_loss": te_loss, "test_acc": te_acc, "test_f1": te_f1,
        "y_true": y_true.tolist(), "y_pred": y_pred.tolist(), "class_names": class_names,
        "ckpt_path": save_paths["ckpt_path"], "ts_path": save_paths.get("ts_path",""),
        "confusion_png": confusion_paths.get("cm_raw_path","")
    }
