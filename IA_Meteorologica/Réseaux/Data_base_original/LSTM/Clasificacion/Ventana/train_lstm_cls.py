# train_lstm_cls.py
from __future__ import annotations
import os, json, math, time, random
from dataclasses import asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp

__all__ = ["train_lstm_cls"]

# -----------------------------
# Utils
# -----------------------------

def _cfg_get(cfg, name, default=None):
    return getattr(cfg, name, default) if hasattr(cfg, name) else cfg.get(name, default) if isinstance(cfg, dict) else default

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -----------------------------
# Data
# -----------------------------

class WindowedDatasetCls(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)              # [N, H, F]
        self.y = y.astype(np.int64)               # [N, L] (labels)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def _make_loader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last, prefetch_factor, persistent_workers):
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    if num_workers and num_workers > 0:
        kwargs.update(dict(num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                           persistent_workers=persistent_workers, multiprocessing_context=mp.get_context('spawn')))
    else:
        kwargs.update(dict(num_workers=0, pin_memory=False))
    try:
        return DataLoader(dataset, **kwargs)
    except Exception as e:
        print(f"[DataLoader] Fallback to num_workers=0 (reason: {e})")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0, pin_memory=False)


def time_order_and_split(df: pd.DataFrame, datetime_col: Optional[str], train_frac: float = 0.7, val_frac: float = 0.15):
    if datetime_col is not None and datetime_col in df.columns:
        df = df.copy(); df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce', utc=True)
        df = df.sort_values(datetime_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    n = len(df); n_train = int(n*train_frac); n_val = int(n*val_frac)
    df_train = df.iloc[:n_train]; df_val = df.iloc[n_train:n_train+n_val]; df_test = df.iloc[n_train+n_val:]
    return df_train, df_val, df_test


def detect_classes(df: pd.DataFrame, onehot_prefix: str, label_col_raw: Optional[str], force_topk: Optional[int]) -> Tuple[np.ndarray, List[str]]:
    # Preferir columnas one-hot si existen
    oh_cols = [c for c in df.columns if c.startswith(onehot_prefix)]
    if oh_cols:
        # si topk, quedarnos con las más frecuentes según suma
        counts = df[oh_cols].sum(axis=0).sort_values(ascending=False)
        if force_topk is not None and force_topk < len(oh_cols):
            keep = list(counts.index[:force_topk])
        else:
            keep = list(counts.index)
        class_names = [c.replace(onehot_prefix, "") for c in keep]
        # construir labels a partir de argmax solo en columnas keep
        mat = df[keep].values
        # evitar todo-cero (si alguna fila no pertenece a ninguna de keep)
        # ponemos -1 que será máscara para filtrar
        label_ids = np.where(mat.sum(axis=1) > 0, mat.argmax(axis=1), -1)
        return label_ids, class_names
    # Si no hay one-hot, usar columna de texto
    if label_col_raw and label_col_raw in df.columns:
        vals = df[label_col_raw].astype(str).fillna("__nan__").values
        # top-k si procede
        if force_topk is not None:
            # elegir top-k por frecuencia en TODO el df (suficiente para codificación estable)
            vc = pd.Series(vals).value_counts()
            keep = set(vc.index[:force_topk].tolist())
            vals = np.where(pd.Series(vals).isin(keep), vals, "__other__")
        # mapear a ids
        uniques = pd.unique(vals)
        class_names = list(uniques)
        map_idx = {c:i for i,c in enumerate(class_names)}
        label_ids = np.array([map_idx[v] for v in vals], dtype=np.int64)
        return label_ids, class_names
    raise ValueError("No se encontraron columnas Summary_* ni la columna 'Summary' cruda para construir etiquetas.")


def build_windows_cls(df: pd.DataFrame, feature_cols: List[str], label_ids: np.ndarray, H: int, L: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    Xvals = df[feature_cols].values.astype(np.float32)
    total = len(df)
    X_list, y_list = [], []
    for t in range(H, total - L + 1, stride):
        X_list.append(Xvals[t-H:t, :])         # [H, F]
        y_list.append(label_ids[t:t+L])        # [L]
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.int64)
    return X, y

# -----------------------------
# Model
# -----------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, horizon: int, num_classes: int,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2,
                 bidirectional: bool = False, head_hidden: Optional[int] = None):
        super().__init__()
        self.horizon = horizon; self.num_classes = num_classes; self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0, bidirectional=bidirectional)
        last_size = hidden_size * (2 if bidirectional else 1)
        if head_hidden is None:
            self.head = nn.Linear(last_size, horizon * num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(last_size, head_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, horizon * num_classes)
            )
    def forward(self, x):                 # x: [B, H, F]
        out, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            fwd = h_n[-2, :, :]; bwd = h_n[-1, :, :]; last = torch.cat([fwd, bwd], dim=-1)
        else:
            last = h_n[-1, :, :]
        logits = self.head(last)          # [B, L*C]
        return logits.view(x.size(0), -1, self.num_classes)  # [B, L, C]

# -----------------------------
# Train / Eval
# -----------------------------

def _to_device(batch, device, non_blocking=False):
    x, y = batch
    return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

@torch.no_grad()
def evaluate(model, loader, criterion, device, non_blocking=False):
    model.eval(); tot_loss=0.0; tot_correct=0; tot_count=0
    for x, y in loader:
        x, y = _to_device((x,y), device, non_blocking)
        logits = model(x)                  # [B,L,C]
        B,L,C = logits.shape
        loss = criterion(logits.view(B*L, C), y.view(B*L))
        tot_loss += float(loss.item()) * B
        pred = logits.argmax(dim=-1)      # [B,L]
        tot_correct += int((pred == y).sum().item())
        tot_count += int(B*L)
    return tot_loss/max(1, len(loader.dataset)/loader.batch_size), tot_correct/max(1, tot_count)


def _infer_batches(model, loader, device, non_blocking=False):
    model.eval(); preds=[]; trues=[]
    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        with torch.no_grad(): logits = model(x).cpu().numpy()
        preds.append(logits); trues.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)  # logits [N,L,C], y [N,L]


def compute_horizon_acc(logits: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    N,L,C = logits.shape
    rows=[]
    for h in range(L):
        pred = logits[:,h,:].argmax(axis=1)
        acc = float((pred == y_true[:,h]).mean())
        rows.append({"h": h+1, "ACC": acc})
    return pd.DataFrame(rows)


def class_weights_from_train(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train.flatten(), minlength=num_classes).astype(np.float32)
    counts[counts==0] = 1.0
    weights = counts.sum()/counts
    weights = weights/weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

# -----------------------------
# Public API
# -----------------------------

def train_lstm_cls(cfg) -> Dict[str, Any]:
    # 0) Config
    seed = int(_cfg_get(cfg, 'SEED', 42)); set_seed(seed)
    dev_pref = _cfg_get(cfg, 'DEVICE', None)
    if not dev_pref: dev_pref = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_pref)
    use_amp = bool(_cfg_get(cfg, 'USE_AMP', True))
    non_blocking = bool(_cfg_get(cfg, 'NON_BLOCKING_COPY', True))
    torch.backends.cuda.matmul.allow_tf32 = bool(_cfg_get(cfg, 'USE_TF32', True))
    torch.backends.cudnn.allow_tf32 = bool(_cfg_get(cfg, 'USE_TF32', True))

    # 1) Paths
    out_dir = _ensure_dir(_cfg_get(cfg, 'OUTPUT_DIR', 'outputs/lstm_summary_cls'))
    plots_dir = _ensure_dir(os.path.join(out_dir, 'plots'))
    ckpt_dir = _ensure_dir(_cfg_get(cfg, 'CHECKPOINT_DIR', 'checkpoints'))

    # 2) Data
    csv_path = _cfg_get(cfg, 'CSV_PATH', 'data/weatherHistory_normalize.csv')
    datetime_col = _cfg_get(cfg, 'DATETIME_COL', 'Formatted Date')
    onehot_prefix = _cfg_get(cfg, 'SUMMARY_ONEHOT_PREFIX', 'Summary_')
    label_col_raw = _cfg_get(cfg, 'LABEL_COL_RAW', 'Summary')
    force_topk = _cfg_get(cfg, 'FORCE_TOPK', None)

    feature_cols = list(_cfg_get(cfg, 'FEATURE_COLS', []))
    H = int(_cfg_get(cfg, 'H', 336)); L = int(_cfg_get(cfg, 'L', 24))

    df = pd.read_csv(csv_path)

    # Detectar etiquetas y nombres de clase sobre TODO el df (codificación estable)
    label_ids_all, class_names = detect_classes(df, onehot_prefix, label_col_raw, force_topk)
    num_classes = len(class_names)

    # Filtrar filas con etiqueta válida (-1 significa fuera de top-k de one-hot)
    mask_valid = label_ids_all != -1
    if (~mask_valid).any():
        print(f"[INFO] Filas descartadas por no pertenecer a clases seleccionadas: {int((~mask_valid).sum())}")
        df = df.loc[mask_valid].reset_index(drop=True)
        label_ids_all = label_ids_all[mask_valid]

    # Chequeo de features
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    # Reajustar etiquetas si se dropearon filas
    if len(df) != len(label_ids_all):
        # reconstruir máscara basada en índices tras dropna
        # asumimos que dropna solo eliminó filas; volvemos a alinear por longitud mínima
        m = min(len(df), len(label_ids_all))
        df = df.iloc[:m].reset_index(drop=True)
        label_ids_all = label_ids_all[:m]

    # 3) Split temporal
    train_frac = float(_cfg_get(cfg, 'TRAIN_FRACTION', 0.7))
    val_frac = float(_cfg_get(cfg, 'VAL_FRACTION', 0.15))
    df_train, df_val, df_test = time_order_and_split(df, datetime_col, train_frac, val_frac)
    # Alinear arrays de etiquetas con splits
    n_tr, n_va = len(df_train), len(df_val)
    y_ids_train = label_ids_all[:n_tr]
    y_ids_val = label_ids_all[n_tr:n_tr+n_va]
    y_ids_test = label_ids_all[n_tr+n_va:]

    # 4) Ventanas
    stride_tr = int(_cfg_get(cfg, 'STRIDE_TRAIN', 1))
    stride_va = int(_cfg_get(cfg, 'STRIDE_VAL', 2))
    stride_te = int(_cfg_get(cfg, 'STRIDE_TEST', 2))
    X_train, y_train = build_windows_cls(df_train, feature_cols, y_ids_train, H, L, stride_tr)
    X_val,   y_val   = build_windows_cls(df_val,   feature_cols, y_ids_val,   H, L, stride_va)
    X_test,  y_test  = build_windows_cls(df_test,  feature_cols, y_ids_test,  H, L, stride_te)

    # 5) Dataloaders
    batch_size = int(_cfg_get(cfg, 'BATCH_SIZE', 256))
    num_workers = int(_cfg_get(cfg, 'NUM_WORKERS', 0))
    pin_memory = bool(_cfg_get(cfg, 'PIN_MEMORY', False))
    prefetch_factor = int(_cfg_get(cfg, 'PREFETCH_FACTOR', 2)) if _cfg_get(cfg, 'PREFETCH_FACTOR', None) is not None else None

    tr_ds = WindowedDatasetCls(X_train, y_train)
    va_ds = WindowedDatasetCls(X_val, y_val)
    te_ds = WindowedDatasetCls(X_test, y_test)

    tr_ld = _make_loader(tr_ds, batch_size, True,  num_workers, pin_memory, False, prefetch_factor if num_workers>0 else None, bool(_cfg_get(cfg,'PERSISTENT_WORKERS', False)))
    va_ld = _make_loader(va_ds, batch_size, False, num_workers, pin_memory, False, prefetch_factor if num_workers>0 else None, bool(_cfg_get(cfg,'PERSISTENT_WORKERS', False)))
    te_ld = _make_loader(te_ds, batch_size, False, num_workers, pin_memory, False, prefetch_factor if num_workers>0 else None, bool(_cfg_get(cfg,'PERSISTENT_WORKERS', False)))

    # 6) Modelo
    input_size = X_train.shape[-1]
    model = LSTMClassifier(
        input_size=input_size,
        horizon=L,
        num_classes=num_classes,
        hidden_size=int(_cfg_get(cfg, 'LSTM_HIDDEN_SIZE', 256)),
        num_layers=int(_cfg_get(cfg, 'LSTM_NUM_LAYERS', 2)),
        dropout=float(_cfg_get(cfg, 'LSTM_DROPOUT', 0.2)),
        bidirectional=bool(_cfg_get(cfg, 'LSTM_BIDIRECTIONAL', False)),
        head_hidden=_cfg_get(cfg, 'LSTM_HEAD_HIDDEN', None)
    ).to(device)

    # 7) Optimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(_cfg_get(cfg,'LR',2e-3)), weight_decay=float(_cfg_get(cfg,'WEIGHT_DECAY',1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(_cfg_get(cfg,'T_MAX',50)))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type=='cuda')

    # 8) Pérdida (con pesos por clase para desbalance)
    w = class_weights_from_train(y_train, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    # 9) Loop de entrenamiento
    max_epochs = int(_cfg_get(cfg, 'MAX_EPOCHS', 100))
    patience = int(_cfg_get(cfg, 'EARLY_STOP_PATIENCE', 10))
    best_val_acc = -1.0; best_state=None; best_epoch=-1
    history = {'epoch':[], 'train_loss':[], 'val_ACC':[]}

    start = time.time()
    for epoch in range(1, max_epochs+1):
        model.train(); epoch_loss=0.0; seen=0
        for x,y in tr_ld:
            x,y = _to_device((x,y), device, non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp and device.type=='cuda', dtype=torch.float16):
                logits = model(x)                        # [B,L,C]
                B,Lh,C = logits.shape
                loss = criterion(logits.view(B*Lh, C), y.view(B*Lh))
            scaler.scale(loss).backward()
            if float(_cfg_get(cfg,'GRAD_CLIP_NORM',1.0)) > 0:
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), float(_cfg_get(cfg,'GRAD_CLIP_NORM',1.0)))
            scaler.step(optimizer); scaler.update()
            epoch_loss += float(loss.item()) * x.size(0); seen += x.size(0)
        train_loss = epoch_loss / max(1, seen)

        # Validación
        val_loss, val_acc = evaluate(model, va_ld, criterion, device, non_blocking)
        scheduler.step()

        history['epoch'].append(epoch); history['train_loss'].append(train_loss); history['val_ACC'].append(val_acc)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_ACC={val_acc:.4f}")

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc; best_state={'model': model.state_dict(), 'epoch': epoch}; best_epoch = epoch
            # Guardar checkpoint
            model_name = _cfg_get(cfg, 'MODEL_NAME', None)
            if model_name is None:
                model_name = f"lstm_cls_H{H}_L{L}_C{num_classes}"
            ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
            torch.save({'state_dict': best_state['model'], 'epoch': epoch, 'config': asdict(cfg) if hasattr(cfg,'__dict__') else None,
                        'class_names': class_names}, ckpt_path)

        if epoch - best_epoch >= patience:
            print(f"Early stopping en epoch {epoch} (best {best_epoch})"); break

    train_time = time.time()-start

    # Guardar history
    hist_df = pd.DataFrame(history); hist_path = os.path.join(out_dir, 'history.csv'); hist_df.to_csv(hist_path, index=False)

    # Reload best
    if best_state is not None: model.load_state_dict(best_state['model'])

    # Test
    model.eval()
    with torch.no_grad():
        test_loss, test_acc = evaluate(model, te_ld, criterion, device, non_blocking)
    print(f"Test: loss={test_loss:.6f} | ACC={test_acc:.4f}")

    # Inference para métricas por horizonte y muestras
    logits_test, y_true_test = _infer_batches(model, te_ld, device, non_blocking)
    hm_df = compute_horizon_acc(logits_test, y_true_test)
    hm_path = os.path.join(out_dir, 'horizon_metrics.csv'); hm_df.to_csv(hm_path, index=False)

    # Samples (k)
    k_samples = int(_cfg_get(cfg, 'N_SAMPLES', 10))
    idxs = sorted(set(int(round(i)) for i in np.linspace(0, logits_test.shape[0]-1, num=min(k_samples, logits_test.shape[0]))))
    samples=[]
    for i in idxs:
        probs = torch.softmax(torch.from_numpy(logits_test[i]), dim=-1).numpy()  # [L,C]
        pred_ids = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        samples.append({
            'idx': int(i),
            'y_true': y_true_test[i].astype(np.int64),
            'y_pred': pred_ids.astype(np.int64),
            'conf': conf.astype(np.float32)
        })
    np.savez(os.path.join(out_dir, 'samples_test.npz'), samples=np.array(samples, dtype=object))

    # FI por permutación (ΔACC en validación)
    with torch.no_grad():
        logits_val, y_val_true = _infer_batches(model, va_ld, device, non_blocking)
    base_acc = float((logits_val.argmax(axis=2) == y_val_true).mean())
    X_val = va_ds.X; F = X_val.shape[-1]
    deltas=[]
    for f in range(F):
        Xp = X_val.copy(); idx = np.random.permutation(Xp.shape[0]); Xp[:,:,f] = Xp[idx,:,f]
        # eval en batches para no petar memoria
        preds=[]; i=0
        while i < Xp.shape[0]:
            xb = torch.from_numpy(Xp[i:i+batch_size]).float().to(device, non_blocking=non_blocking)
            with torch.no_grad(): lb = model(xb).cpu().numpy()
            preds.append(lb); i += batch_size
        logits_perm = np.concatenate(preds, axis=0)
        acc_perm = float((logits_perm.argmax(axis=2) == y_val_true).mean())
        deltas.append({'feature': feature_cols[f], 'delta_acc': base_acc - acc_perm})
    fi_df = pd.DataFrame(deltas).sort_values('delta_acc', ascending=False)
    fi_path = os.path.join(out_dir, 'feature_importance.csv'); fi_df.to_csv(fi_path, index=False)

    # Guardar mapping de clases
    with open(os.path.join(out_dir, 'class_index.json'), 'w', encoding='utf-8') as f:
        json.dump({i:n for i,n in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    results = {
        'best_epoch': best_epoch,
        'train_time_sec': train_time,
        'test_metrics': {'loss': test_loss, 'ACC': test_acc},
        'paths': {
            'history_csv': hist_path,
            'horizon_csv': hm_path,
            'samples_npz': os.path.join(out_dir, 'samples_test.npz'),
            'feature_importance_csv': fi_path,
            'class_index_json': os.path.join(out_dir, 'class_index.json'),
        },
        'meta': {'H': H, 'L': L, 'features': feature_cols, 'num_classes': num_classes, 'class_names': class_names,
                 'datetime_col': datetime_col}
    }
    return results
