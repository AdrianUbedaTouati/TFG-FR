from __future__ import annotations

import os
import json
import math
import time
import random
from dataclasses import asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import multiprocessing as mp

__all__ = ['train_lstm']

def _make_loader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last, prefetch_factor, persistent_workers, non_blocking=False):
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    if num_workers and num_workers > 0:
        kwargs.update(dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            multiprocessing_context=mp.get_context('spawn')
        ))
    else:
        kwargs.update(dict(num_workers=0, pin_memory=False))
    try:
        return DataLoader(dataset, **kwargs)
    except Exception as e:
        print(f"[DataLoader] Fallback to num_workers=0 (reason: {e})")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                          num_workers=0, pin_memory=False)

# -----------------------------
# Helpers
# -----------------------------

def _cfg_get(cfg, name, default=None):
    # Works with dataclass or simple objects
    return getattr(cfg, name, default) if hasattr(cfg, name) else cfg.get(name, default) if isinstance(cfg, dict) else default

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

# -----------------------------
# Data utilities (windowing)
# -----------------------------

class WindowedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)  # [N, H, F]
        self.y = y.astype(np.float32)  # [N, L]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_windows(df: pd.DataFrame, feature_cols: List[str], target_col_norm: str,
                  H: int, L: int, include_target_as_feature: bool,
                  orig_target_col: Optional[str], stride: int = 1) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """ Returns X: [N, H, F], y: [N, L], hist_target_raw: [N, H] (if orig_target_col present) """
    values_feat = df[feature_cols].values.astype(np.float32)  # [T, F]
    target_z = df[target_col_norm].values.astype(np.float32)  # [T]
    total = len(df)

    X_list, y_list, hist_raw_list = [], [], []
    use_raw_hist = orig_target_col is not None and orig_target_col in df.columns
    if use_raw_hist:
        raw = df[orig_target_col].values

    for t in range(H, total - L + 1, stride):
        X_list.append(values_feat[t - H:t, :])          # [H, F]
        y_list.append(target_z[t:t + L])                # [L]
        if use_raw_hist:
            hist_raw_list.append(raw[t - H:t])          # [H]

    X = np.stack(X_list, axis=0).astype(np.float32)        # [N, H, F]
    y = np.stack(y_list, axis=0).astype(np.float32)        # [N, L]
    hist_raw = np.stack(hist_raw_list, axis=0) if use_raw_hist else None
    return X, y, hist_raw

def time_order_and_split(df: pd.DataFrame, datetime_col: Optional[str],
                         train_frac: float = 0.7, val_frac: float = 0.15):
    if datetime_col is not None and datetime_col in df.columns:
        # Robust parse (errors='coerce'): invalids go NaT but keep order
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce', utc=True)
        df = df.sort_values(datetime_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train + n_val]
    df_test = df.iloc[n_train + n_val:]
    return df_train, df_val, df_test

# -----------------------------
# Loss: Weighted Huber across horizon
# -----------------------------

class WeightedHuberLoss(nn.Module):
    def __init__(self, horizon: int, delta: float = 1.0, w_start: float = 1.0, w_end: float = 0.6):
        super().__init__()
        self.delta = delta
        self.register_buffer('weights', torch.linspace(w_start, w_end, steps=horizon))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: [B, L]
        err = pred - target
        abs_err = err.abs()
        # huber
        quad = torch.minimum(abs_err, torch.tensor(self.delta, device=err.device))
        lin = abs_err - quad
        huber = 0.5 * quad ** 2 + self.delta * lin
        # weight per horizon (broadcast over batch)
        #w = self.weights.unsqueeze(0)  # [1, L]
        w = self.weights.to(err.device).unsqueeze(0)
        loss = (huber * w).mean()
        return loss

# -----------------------------
# Model: LSTM -> MLP head -> L outputs
# -----------------------------

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, horizon: int,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False,
                 head_hidden: Optional[int] = None):
        super().__init__()
        self.horizon = horizon
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        last_size = hidden_size * (2 if bidirectional else 1)
        if head_hidden is None:
            self.head = nn.Linear(last_size, horizon)
        else:
            self.head = nn.Sequential(
                nn.Linear(last_size, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, horizon),
            )

    def forward(self, x):  # x: [B, H, F]
        out, (h_n, c_n) = self.lstm(x)  # h_n: [num_layers*(2 if bi), B, hidden]
        if self.bidirectional:
            # take last layer's forward and backward hidden states
            fwd = h_n[-2, :, :]
            bwd = h_n[-1, :, :]
            last = torch.cat([fwd, bwd], dim=-1)
        else:
            last = h_n[-1, :, :]
        y = self.head(last)  # [B, L]
        return y

# -----------------------------
# Training & evaluation
# -----------------------------

def _to_device(batch, device, non_blocking=False):
    x, y = batch
    return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)

@torch.no_grad()
def evaluate(model, loader, device, non_blocking=False):
    model.eval()
    loss_meter = 0.0
    mae_meter = 0.0
    rmse_meter = 0.0
    count = 0
    crit = loader._criterion  # injected
    for batch in loader:
        x, y = _to_device(batch, device, non_blocking=non_blocking)
        yhat = model(x)
        loss = crit(yhat, y).item()
        loss_meter += loss * x.size(0)
        y_np = y.detach().cpu().numpy()
        yhat_np = yhat.detach().cpu().numpy()
        mae_meter += np.mean(np.abs(y_np - yhat_np)) * x.size(0)
        rmse_meter += np.sqrt(np.mean((y_np - yhat_np) ** 2)) * x.size(0)
        count += x.size(0)
    return loss_meter / count, mae_meter / count, rmse_meter / count

def _infer_batches(model, loader, device, non_blocking=False):
    model.eval()
    preds = []
    trues = []
    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        with torch.no_grad():
            yhat = model(x).cpu().numpy()
        preds.append(yhat)
        trues.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

def _compute_horizon_metrics(y_pred_z, y_true_z, mu: Optional[float], sigma: Optional[float]) -> pd.DataFrame:
    # y_*: [N, L]
    L = y_true_z.shape[1]
    rows = []
    for h in range(L):
        y_p = y_pred_z[:, h]
        y_t = y_true_z[:, h]
        m_mae_z = mae(y_p, y_t)
        m_rmse_z = rmse(y_p, y_t)
        if mu is not None and sigma is not None and sigma != 0:
            y_p_c = y_p * sigma + mu
            y_t_c = y_t * sigma + mu
            m_mae_c = mae(y_p_c, y_t_c)
            m_rmse_c = rmse(y_p_c, y_t_c)
        else:
            m_mae_c = np.nan
            m_rmse_c = np.nan
        rows.append({
            'h': h+1,
            'MAE_z': m_mae_z,
            'RMSE_z': m_rmse_z,
            'MAE_C': m_mae_c,
            'RMSE_C': m_rmse_c,
        })
    return pd.DataFrame(rows)

def _evenly_spaced_indices(n: int, k: int) -> List[int]:
    if k >= n:
        return list(range(n))
    # linspace-like integer selection
    return sorted({int(round(i)) for i in np.linspace(0, n-1, num=k)})

def _feature_importance_permutation(model, X_val, y_val, baseline_mse: float, device, batch_size: int,
                                    non_blocking: bool, feature_names: List[str]) -> pd.DataFrame:
    """Return DataFrame with columns: feature, delta_mse"""
    model.eval()
    N, H, F = X_val.shape
    deltas = []
    for f in range(F):
        Xp = X_val.copy()
        # permute across samples (keep temporal structure within each window)
        idx = np.random.permutation(N)
        Xp[:, :, f] = Xp[idx, :, f]
        # run in batches
        preds = []
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(Xp[i:i+batch_size]).float().to(device, non_blocking=non_blocking)
            with torch.no_grad():
                pb = model(xb).cpu().numpy()
            preds.append(pb)
        yhat = np.concatenate(preds, axis=0)
        mse_perm = float(np.mean((yhat - y_val) ** 2))
        deltas.append({'feature': feature_names[f], 'delta_mse': mse_perm - baseline_mse})
    return pd.DataFrame(deltas).sort_values('delta_mse', ascending=False)

# -----------------------------
# Public API: run_training(cfg)
# -----------------------------

def run_training(cfg):
    # 0) Config & seeds
    seed = _cfg_get(cfg, 'SEED', 42)
    set_seed(seed)


    dev_pref = _cfg_get(cfg, 'DEVICE', None)
    if not dev_pref:
        dev_pref = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_pref)

    use_amp = bool(_cfg_get(cfg, 'USE_AMP', True))
    non_blocking = bool(_cfg_get(cfg, 'NON_BLOCKING_COPY', True))
    use_tf32 = bool(_cfg_get(cfg, 'USE_TF32', True))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    # 1) Paths
    out_dir = _ensure_dir(_cfg_get(cfg, 'OUTPUT_DIR', 'outputs'))
    plots_dir = _ensure_dir(os.path.join(out_dir, 'plots'))
    ckpt_dir = _ensure_dir(_cfg_get(cfg, 'CHECKPOINT_DIR', 'checkpoints'))

    # 2) Data config
    csv_path = _cfg_get(cfg, 'CSV_PATH', 'data/weatherHistory_normalize.csv')
    datetime_col = _cfg_get(cfg, 'DATETIME_COL', 'Formatted Date')
    target_col_norm = _cfg_get(cfg, 'TARGET_COL_NORM', 'Temperature (C)_normalized')
    orig_target_col = _cfg_get(cfg, 'ORIG_TARGET_COL', None)
    if orig_target_col is None and target_col_norm.endswith('_normalized'):
        # Heuristic: strip suffix
        orig_target_col = target_col_norm.replace('_normalized', '')
        # Typical dataset uses 'Temperature (C)' exact name
        if orig_target_col not in ['Temperature (C)', 'Apparent Temperature (C)'] and orig_target_col not in []:
            # fallback to Temperature (C) if guessed name not present
            pass

    feature_cols = list(_cfg_get(cfg, 'FEATURE_COLS', []))
    include_target = bool(_cfg_get(cfg, 'INCLUDE_TARGET_AS_FEATURE', True))

    # Ensure target feature is in features if requested
    if include_target and target_col_norm not in feature_cols:
        feature_cols = feature_cols + [target_col_norm]

    # 3) Window sizes
    H = int(_cfg_get(cfg, 'H', 336))
    L = int(_cfg_get(cfg, 'L', 24))

    # 4) Load CSV
    df = pd.read_csv(csv_path)
    # sanity check
    missing = [c for c in feature_cols + [target_col_norm] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    # Drop rows with NANs in critical cols
    df = df.dropna(subset=feature_cols + [target_col_norm]).reset_index(drop=True)

    # 5) Split by time
    train_frac = float(_cfg_get(cfg, 'TRAIN_FRACTION', 0.7))
    val_frac = float(_cfg_get(cfg, 'VAL_FRACTION', 0.15))
    df_train, df_val, df_test = time_order_and_split(df, datetime_col, train_frac, val_frac)

    # 6) Z-denormalization stats from TRAIN raw target
    mu, sigma = None, None
    if orig_target_col is not None and orig_target_col in df_train.columns:
        mu = float(df_train[orig_target_col].mean())
        sigma = float(df_train[orig_target_col].std())
        print(f"Denormalización (TRAIN only): mean={mu:.3f}, std={sigma:.3f}")
    else:
        print("Denormalización: no se encontró columna cruda del objetivo; se reportará solo en z-score.")

    # 7) Build windows per split
    stride_tr = int(_cfg_get(cfg, 'STRIDE_TRAIN', 1))
    stride_va = int(_cfg_get(cfg, 'STRIDE_VAL', 2))
    stride_te = int(_cfg_get(cfg, 'STRIDE_TEST', 2))
    X_train, y_train, _ = build_windows(df_train, feature_cols, target_col_norm, H, L, include_target, orig_target_col, stride=stride_tr)
    X_val, y_val, _ = build_windows(df_val, feature_cols, target_col_norm, H, L, include_target, orig_target_col, stride=stride_va)
    X_test, y_test, hist_raw_test = build_windows(df_test, feature_cols, target_col_norm, H, L, include_target, orig_target_col, stride=stride_te)

    # 8) Dataloaders
    batch_size = int(_cfg_get(cfg, 'BATCH_SIZE', 256))
    num_workers = int(_cfg_get(cfg, 'NUM_WORKERS', 0))
    pin_memory = bool(_cfg_get(cfg, 'PIN_MEMORY', False))
    prefetch = int(_cfg_get(cfg, 'PREFETCH', 0))
    prefetch_factor = int(_cfg_get(cfg, 'PREFETCH_FACTOR', 2))

    train_ds = WindowedDataset(X_train, y_train)
    val_ds = WindowedDataset(X_val, y_val)
    test_ds = WindowedDataset(X_test, y_test)

    train_loader = _make_loader(train_ds, batch_size, True, num_workers, pin_memory, False, prefetch_factor if num_workers > 0 else None, bool(_cfg_get(cfg,'PERSISTENT_WORKERS', False)))
    val_loader = _make_loader(val_ds, batch_size, False, num_workers, pin_memory, False, prefetch_factor if num_workers > 0 else None, bool(_cfg_get(cfg,'PERSISTENT_WORKERS', False)))
    test_loader = _make_loader(test_ds, batch_size, False, num_workers, pin_memory, False, prefetch_factor if num_workers > 0 else None, bool(_cfg_get(cfg,'PERSISTENT_WORKERS', False)))

    # inject criterion for evaluate()
    crit = WeightedHuberLoss(horizon=L, delta=float(_cfg_get(cfg, 'HUBER_DELTA', 1.0)),
                             w_start=float(_cfg_get(cfg, 'W_H_START', 1.0)),
                             w_end=float(_cfg_get(cfg, 'W_H_END', 0.6)))
    for ld in (train_loader, val_loader, test_loader):
        ld._criterion = crit  # type: ignore

    # 9) Model
    input_size = X_train.shape[-1]
    hidden_size = int(_cfg_get(cfg, 'LSTM_HIDDEN_SIZE', 256))
    num_layers = int(_cfg_get(cfg, 'LSTM_NUM_LAYERS', 2))
    dropout = float(_cfg_get(cfg, 'LSTM_DROPOUT', 0.2))
    bidirectional = bool(_cfg_get(cfg, 'LSTM_BIDIRECTIONAL', False))
    head_hidden = _cfg_get(cfg, 'LSTM_HEAD_HIDDEN', None)
    model = LSTMForecaster(input_size, L, hidden_size, num_layers, dropout, bidirectional, head_hidden).to(device)

    # 10) Optim & sched
    lr = float(_cfg_get(cfg, 'LR', 2e-3))
    weight_decay = float(_cfg_get(cfg, 'WEIGHT_DECAY', 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    T_max = int(_cfg_get(cfg, 'T_MAX', 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == 'cuda')

    grad_clip = float(_cfg_get(cfg, 'GRAD_CLIP_NORM', 1.0))

    # 11) Train loop
    max_epochs = int(_cfg_get(cfg, 'MAX_EPOCHS', 100))
    patience = int(_cfg_get(cfg, 'EARLY_STOP_PATIENCE', 10))
    best_val_mae = float('inf')
    best_state = None
    best_epoch = -1
    history = {'epoch': [], 'train_loss': [], 'val_MAE': [], 'val_RMSE': []}

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch in train_loader:
            x, y = _to_device(batch, device, non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp and device.type == 'cuda', dtype=torch.float16):
                yhat = model(x)
                loss = crit(yhat, y)
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = epoch_loss / max(1, seen)

        # Val
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device, non_blocking)
        scheduler.step()

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_MAE'].append(val_mae)
        history['val_RMSE'].append(val_rmse)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_MAE={val_mae:.5f} | val_RMSE={val_rmse:.5f}")

        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_state = { 'model': model.state_dict(), 'epoch': epoch }
            best_epoch = epoch
            # save checkpoint
            model_name = _cfg_get(cfg, 'MODEL_NAME', f"lstm_H{H}_L{L}")
            ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
            torch.save({'state_dict': best_state['model'], 'epoch': epoch, 'config': asdict(cfg) if hasattr(cfg,'__dict__') else None}, ckpt_path)

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    train_time = time.time() - start_time

    # Save history
    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(out_dir, 'history.csv')
    hist_df.to_csv(hist_path, index=False)

    # Reload best
    if best_state is not None:
        model.load_state_dict(best_state['model'])

    # 12) Test metrics
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, device, non_blocking)
    print(f"Test: loss={test_loss:.6f} | MAE={test_mae:.4f} | RMSE={test_rmse:.4f}")

    # Predictions on splits (for horizon metrics we just need test)
    y_pred_test, y_true_test = _infer_batches(model, test_loader, device, non_blocking)

    # 13) Horizon metrics (z and °C if available)
    hm_df = _compute_horizon_metrics(y_pred_test, y_true_test, mu, sigma)
    hm_path = os.path.join(out_dir, 'horizon_metrics.csv')
    hm_df.to_csv(hm_path, index=False)

    # 14) Samples (10 evenly spaced from TEST) in °C when possible
    k_samples = int(_cfg_get(cfg, 'N_SAMPLES', 10))
    idxs = _evenly_spaced_indices(y_true_test.shape[0], k_samples)
    samples = []
    for i in idxs:
        row = {
            'idx': int(i),
            'y_true_z': y_true_test[i].astype(np.float32),
            'y_pred_z': y_pred_test[i].astype(np.float32),
        }
        if mu is not None and sigma is not None and sigma != 0:
            row['y_true_C'] = (y_true_test[i] * sigma + mu).astype(np.float32)
            row['y_pred_C'] = (y_pred_test[i] * sigma + mu).astype(np.float32)
        samples.append(row)

    # Historical tail in °C for context if available
    hist_tail = None
    if hist_raw_test is not None:
        # select the H histories aligned with selected samples
        hist_tail = hist_raw_test[idxs].astype(np.float32)

    np.savez(os.path.join(out_dir, 'samples_test.npz'),
             samples=np.array(samples, dtype=object),
             hist_tail_C=hist_tail)

    # 15) Feature importance by permutation (validation set)
    # baseline on val MSE in z
    with torch.no_grad():
        y_pred_val, y_true_val = _infer_batches(model, val_loader, device, non_blocking)
    baseline_mse = float(np.mean((y_pred_val - y_true_val) ** 2))
    fi_df = _feature_importance_permutation(
        model,
        X_val=y_val*0 + np.stack([X_val[..., j] for j in range(X_val.shape[-1])], axis=-1) if False else X_val,  # identity; kept for clarity
        y_val=y_true_val,
        baseline_mse=baseline_mse,
        device=device,
        batch_size=batch_size,
        non_blocking=non_blocking,
        feature_names=feature_cols,
    )
    fi_path = os.path.join(out_dir, 'feature_importance.csv')
    fi_df.to_csv(fi_path, index=False)

    results = {
        'best_epoch': best_epoch,
        'train_time_sec': train_time,
        'test_metrics': {'loss': test_loss, 'MAE': test_mae, 'RMSE': test_rmse},
        'paths': {
            'history_csv': hist_path,
            'horizon_csv': hm_path,
            'samples_npz': os.path.join(out_dir, 'samples_test.npz'),
            'feature_importance_csv': fi_path,
        },
        'denorm': {'mu': mu, 'sigma': sigma, 'orig_target_col': orig_target_col},
        'meta': {'H': H, 'L': L, 'features': feature_cols, 'target_norm': target_col_norm, 'datetime_col': datetime_col},
    }
    return results

# --- PUBLIC API ---
from typing import Dict, Any

def train_lstm(cfg) -> Dict[str, Any]:
    # Si tu función principal se llama run_training(cfg):
    return run_training(cfg)

