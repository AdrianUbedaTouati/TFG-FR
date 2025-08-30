# === LEAKAGE ENABLED BUILD ===
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

class WindowedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, X_future: Optional[np.ndarray] = None):
        self.X = X.astype(np.float32)           # [N, H, F]
        self.y = y.astype(np.float32)           # [N, L]
        self.X_future = None if X_future is None else X_future.astype(np.float32)  # [N, L, F_leak]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.X_future is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.X_future[idx], self.y[idx]

def build_windows(df: pd.DataFrame, feature_cols: List[str], target_col_norm: str,
                  H: int, L: int, include_target_as_feature: bool,
                  orig_target_col: Optional[str], stride: int = 1) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
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

def build_windows_with_leak(df: pd.DataFrame,
                            feature_cols: List[str],
                            leak_cols: List[str],
                            target_col_norm: str,
                            H: int, L: int,
                            include_target_as_feature: bool,
                            orig_target_col: Optional[str],
                            stride: int = 1) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    values_feat = df[feature_cols].values.astype(np.float32)     # [T, F]
    values_leak = df[leak_cols].values.astype(np.float32)        # [T, F_leak]
    target_z = df[target_col_norm].values.astype(np.float32)     # [T]
    total = len(df)

    X_list, y_list, Xf_list, hist_raw_list = [], [], [], []
    use_raw_hist = orig_target_col is not None and orig_target_col in df.columns
    if use_raw_hist:
        raw = df[orig_target_col].values

    for t in range(H, total - L + 1, stride):
        X_list.append(values_feat[t - H:t, :])           # [H, F]
        y_list.append(target_z[t:t + L])                 # [L]
        Xf_list.append(values_leak[t:t + L, :])          # [L, F_leak]
        if use_raw_hist:
            hist_raw_list.append(raw[t - H:t])           # [H]

    X = np.stack(X_list, axis=0).astype(np.float32)         # [N, H, F]
    y = np.stack(y_list, axis=0).astype(np.float32)         # [N, L]
    X_future = np.stack(Xf_list, axis=0).astype(np.float32) # [N, L, F_leak]
    hist_raw = np.stack(hist_raw_list, axis=0) if use_raw_hist else None
    return X, y, hist_raw, X_future

def time_order_and_split(df: pd.DataFrame, datetime_col: Optional[str],
                         train_frac: float = 0.7, val_frac: float = 0.15):
    if datetime_col is not None and datetime_col in df.columns:
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce', utc=True)
        df = df.sort_values(datetime_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train + n_val]
    df_test = df.iloc[n_train + n_val:]
    return df_train, df_val, df_test

class WeightedHuberLoss(nn.Module):
    def __init__(self, horizon: int, delta: float = 1.0, w_start: float = 1.0, w_end: float = 0.6):
        super().__init__()
        self.delta = delta
        self.register_buffer('weights', torch.linspace(w_start, w_end, steps=horizon))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        abs_err = err.abs()
        quad = torch.minimum(abs_err, torch.tensor(self.delta, device=err.device))
        lin = abs_err - quad
        huber = 0.5 * quad ** 2 + self.delta * lin
        w = self.weights.to(err.device).unsqueeze(0)  # [1, L]
        loss = (huber * w).mean()
        return loss

class FusionHead(nn.Module):
    """Head vectorizada que concatena el estado final con features futuras por paso."""
    def __init__(self, last_size: int, fut_size: int, horizon: int, head_hidden: Optional[int], dropout: float):
        super().__init__()
        in_dim = last_size + fut_size
        if head_hidden is None:
            self.net = nn.Linear(in_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
        self.horizon = horizon

    def forward(self, last: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        B, L, Ff = x_future.shape
        assert L == self.horizon, "x_future debe tener L pasos"
        last_tiled = last.unsqueeze(1).expand(-1, L, -1)        # [B, L, D]
        cat = torch.cat([last_tiled, x_future], dim=-1)         # [B, L, D+Ff]
        out = self.net(cat).squeeze(-1)                         # [B, L]
        return out

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, horizon: int,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False,
                 head_hidden: Optional[int] = None,
                 fut_feature_size: int = 0):
        super().__init__()
        self.horizon = horizon
        self.bidirectional = bidirectional
        self.fut_feature_size = fut_feature_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        last_size = hidden_size * (2 if bidirectional else 1)
        if fut_feature_size and fut_feature_size > 0:
            self.head = FusionHead(last_size, fut_feature_size, horizon, head_hidden, dropout)
        else:
            if head_hidden is None:
                self.head = nn.Linear(last_size, horizon)
            else:
                self.head = nn.Sequential(
                    nn.Linear(last_size, head_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, horizon),
                )

    def forward(self, x_hist, x_future: Optional[torch.Tensor] = None):
        out, (h_n, c_n) = self.lstm(x_hist)
        if self.bidirectional:
            fwd = h_n[-2, :, :]
            bwd = h_n[-1, :, :]
            last = torch.cat([fwd, bwd], dim=-1)
        else:
            last = h_n[-1, :, :]
        if self.fut_feature_size and x_future is not None:
            y = self.head(last, x_future)
        else:
            y = self.head(last)
        return y

def _to_device(batch, device, non_blocking=False):
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        x, xf, y = batch
        return (x.to(device, non_blocking=non_blocking),
                xf.to(device, non_blocking=non_blocking),
                y.to(device, non_blocking=non_blocking))
    else:
        x, y = batch
        return (x.to(device, non_blocking=non_blocking), None,
                y.to(device, non_blocking=non_blocking))

@torch.no_grad()
def evaluate(model, loader, device, non_blocking=False):
    model.eval()
    loss_meter = 0.0
    mae_meter = 0.0
    rmse_meter = 0.0
    count = 0
    crit = loader._criterion  # injected
    for batch in loader:
        x, xf, y = _to_device(batch, device, non_blocking=non_blocking)
        yhat = model(x, xf) if xf is not None else model(x)
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
    for batch in loader:
        x, xf, y = _to_device(batch, device, non_blocking=non_blocking)
        with torch.no_grad():
            yhat = model(x, xf) if xf is not None else model(x)
        preds.append(yhat.cpu().numpy())
        trues.append(y.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

def mae(a, b):
    import numpy as np
    return float(np.mean(np.abs(a - b)))

def rmse(a, b):
    import numpy as np
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _compute_horizon_metrics(y_pred_z, y_true_z, mu: Optional[float], sigma: Optional[float]):
    import numpy as np, pandas as pd
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
            m_mae_c = float('nan')
            m_rmse_c = float('nan')
        rows.append({'h': h+1, 'MAE_z': m_mae_z, 'RMSE_z': m_rmse_z, 'MAE_C': m_mae_c, 'RMSE_C': m_rmse_c})
    import pandas as pd
    return pd.DataFrame(rows)

def _evenly_spaced_indices(n: int, k: int):
    import numpy as np
    if k >= n: return list(range(n))
    return sorted({int(round(i)) for i in np.linspace(0, n-1, num=k)})

def _cfg_get(cfg, name, default=None):
    return getattr(cfg, name, default) if hasattr(cfg, name) else cfg.get(name, default) if isinstance(cfg, dict) else default

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def set_seed(seed: int = 42):
    import numpy as np, torch, random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_train_val_test(cfg):
    import pandas as pd, numpy as np
    # 2) Data config
    csv_path = _cfg_get(cfg, 'CSV_PATH', 'data/weatherHistory_normalize.csv')
    datetime_col = _cfg_get(cfg, 'DATETIME_COL', 'Formatted Date')
    target_col_norm = _cfg_get(cfg, 'TARGET_COL_NORM', 'Temperature (C)_normalized')
    orig_target_col = _cfg_get(cfg, 'ORIG_TARGET_COL', None)
    if orig_target_col is None and target_col_norm.endswith('_normalized'):
        orig_target_col = target_col_norm.replace('_normalized', '')

    feature_cols = list(_cfg_get(cfg, 'FEATURE_COLS', []))
    include_target = bool(_cfg_get(cfg, 'INCLUDE_TARGET_AS_FEATURE', True))

    leak_enabled = bool(_cfg_get(cfg, 'LEAKAGE_ENABLED', False))
    leak_cols = list(_cfg_get(cfg, 'LEAK_FEATURES', []))
    leak_strict = bool(_cfg_get(cfg, 'LEAK_STRICT', True))

    df = pd.read_csv(csv_path)
    missing = [c for c in feature_cols + [target_col_norm] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    if include_target and target_col_norm not in feature_cols:
        feature_cols = feature_cols + [target_col_norm]

    if leak_enabled:
        missing_leak = [c for c in leak_cols if c not in df.columns]
        if missing_leak:
            raise ValueError(f"Faltan columnas de leakage en el CSV: {missing_leak}")

    crit_cols = list(set(feature_cols + [target_col_norm] + (leak_cols if leak_enabled else [])))
    df = df.dropna(subset=crit_cols).reset_index(drop=True)

    # split
    df_train, df_val, df_test = time_order_and_split(df, datetime_col, float(_cfg_get(cfg,'TRAIN_FRACTION',0.7)), float(_cfg_get(cfg,'VAL_FRACTION',0.15)))

    # denorm stats
    mu = sigma = None
    if orig_target_col is not None and orig_target_col in df_train.columns:
        mu = float(df_train[orig_target_col].mean())
        sigma = float(df_train[orig_target_col].std())

    H = int(_cfg_get(cfg, 'H', 336)); L = int(_cfg_get(cfg, 'L', 24))
    stride_tr = int(_cfg_get(cfg, 'STRIDE_TRAIN', 1))
    stride_va = int(_cfg_get(cfg, 'STRIDE_VAL', 2))
    stride_te = int(_cfg_get(cfg, 'STRIDE_TEST', 2))

    if leak_enabled and len(leak_cols) > 0:
        X_tr, y_tr, _, Xf_tr = build_windows_with_leak(df_train, feature_cols, leak_cols, target_col_norm, H, L, include_target, orig_target_col, stride_tr)
        X_va, y_va, _, Xf_va = build_windows_with_leak(df_val, feature_cols, leak_cols, target_col_norm, H, L, include_target, orig_target_col, stride_va)
        X_te, y_te, hist_raw_te, Xf_te = build_windows_with_leak(df_test, feature_cols, leak_cols, target_col_norm, H, L, include_target, orig_target_col, stride_te)
    else:
        X_tr, y_tr, _ = build_windows(df_train, feature_cols, target_col_norm, H, L, include_target, orig_target_col, stride_tr)
        X_va, y_va, _ = build_windows(df_val, feature_cols, target_col_norm, H, L, include_target, orig_target_col, stride_va)
        X_te, y_te, hist_raw_te = build_windows(df_test, feature_cols, target_col_norm, H, L, include_target, orig_target_col, stride_te)
        Xf_tr = Xf_va = Xf_te = None

    return (X_tr, y_tr, Xf_tr), (X_va, y_va, Xf_va), (X_te, y_te, Xf_te, hist_raw_te), {'mu': mu, 'sigma': sigma, 'orig_target_col': orig_target_col, 'feature_cols': feature_cols}

def train_lstm(cfg) -> Dict[str, Any]:
    set_seed(int(_cfg_get(cfg, 'SEED', 42)))
    dev_pref = _cfg_get(cfg, 'DEVICE', None) or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(dev_pref)
    torch.backends.cuda.matmul.allow_tf32 = bool(_cfg_get(cfg,'USE_TF32',True))
    torch.backends.cudnn.allow_tf32 = bool(_cfg_get(cfg,'USE_TF32',True))

    out_dir = _ensure_dir(_cfg_get(cfg, 'OUTPUT_DIR', 'outputs'))
    plots_dir = _ensure_dir(os.path.join(out_dir, 'plots'))
    ckpt_dir = _ensure_dir(_cfg_get(cfg, 'CHECKPOINT_DIR', 'checkpoints'))

    (X_tr, y_tr, Xf_tr), (X_va, y_va, Xf_va), (X_te, y_te, Xf_te, hist_raw_te), den = build_train_val_test(cfg)

    batch_size = int(_cfg_get(cfg, 'BATCH_SIZE', 256))
    num_workers = int(_cfg_get(cfg, 'NUM_WORKERS', 0))
    pin_memory = bool(_cfg_get(cfg, 'PIN_MEMORY', False))
    prefetch_factor = int(_cfg_get(cfg, 'PREFETCH_FACTOR', 2))
    persistent = bool(_cfg_get(cfg, 'PERSISTENT_WORKERS', False))

    tr_ds = WindowedDataset(X_tr, y_tr, Xf_tr); va_ds = WindowedDataset(X_va, y_va, Xf_va); te_ds = WindowedDataset(X_te, y_te, Xf_te)
    tr_ld = _make_loader(tr_ds, batch_size, True, num_workers, pin_memory, False, prefetch_factor if num_workers>0 else None, persistent)
    va_ld = _make_loader(va_ds, batch_size, False, num_workers, pin_memory, False, prefetch_factor if num_workers>0 else None, persistent)
    te_ld = _make_loader(te_ds, batch_size, False, num_workers, pin_memory, False, prefetch_factor if num_workers>0 else None, persistent)

    L_hor = y_tr.shape[1]
    crit = WeightedHuberLoss(horizon=L_hor, delta=float(_cfg_get(cfg, 'HUBER_DELTA', 1.0)), w_start=float(_cfg_get(cfg, 'W_H_START', 1.0)), w_end=float(_cfg_get(cfg, 'W_H_END', 0.6)))
    for ld in (tr_ld, va_ld, te_ld): ld._criterion = crit

    fut_size = 0 if Xf_tr is None else Xf_tr.shape[-1]
    model = LSTMForecaster(
        input_size=X_tr.shape[-1], horizon=L_hor,
        hidden_size=int(_cfg_get(cfg,'LSTM_HIDDEN_SIZE',256)),
        num_layers=int(_cfg_get(cfg,'LSTM_NUM_LAYERS',2)),
        dropout=float(_cfg_get(cfg,'LSTM_DROPOUT',0.2)),
        bidirectional=bool(_cfg_get(cfg,'LSTM_BIDIRECTIONAL',False)),
        head_hidden=_cfg_get(cfg,'LSTM_HEAD_HIDDEN',None),
        fut_feature_size=fut_size
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(_cfg_get(cfg,'LR',2e-3)), weight_decay=float(_cfg_get(cfg,'WEIGHT_DECAY',1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(_cfg_get(cfg,'T_MAX',50)))
    scaler = torch.amp.GradScaler('cuda', enabled=bool(_cfg_get(cfg,'USE_AMP',True)) and device.type=='cuda')
    grad_clip = float(_cfg_get(cfg, 'GRAD_CLIP_NORM', 1.0))

    max_epochs = int(_cfg_get(cfg,'MAX_EPOCHS',100))
    patience = int(_cfg_get(cfg,'EARLY_STOP_PATIENCE',10))
    best_val_mae = float('inf'); best_state = None; best_epoch = -1
    history = {'epoch': [], 'train_loss': [], 'val_MAE': [], 'val_RMSE': []}

    for epoch in range(1, max_epochs+1):
        model.train(); epoch_loss=0.0; seen=0
        for batch in tr_ld:
            x, xf, y = (b.to(device, non_blocking=True) if b is not None else None for b in batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=bool(_cfg_get(cfg,'USE_AMP',True)) and device.type=='cuda', dtype=torch.float16):
                yhat = model(x, xf) if xf is not None else model(x)
                loss = crit(yhat, y)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip>0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item()*x.size(0); seen += x.size(0)
        train_loss = epoch_loss/max(1,seen)

        val_loss, val_mae, val_rmse = evaluate(model, va_ld, device, non_blocking=True)
        scheduler.step()
        history['epoch'].append(epoch); history['train_loss'].append(train_loss); history['val_MAE'].append(val_mae); history['val_RMSE'].append(val_rmse)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_MAE={val_mae:.5f} | val_RMSE={val_rmse:.5f}")
        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae; best_state = {'model': model.state_dict(), 'epoch': epoch}; best_epoch = epoch
            model_name = _cfg_get(cfg, 'MODEL_NAME', f"lstm_H{_cfg_get(cfg,'H',0)}_L{L_hor}")
            torch.save({'state_dict': best_state['model'], 'epoch': epoch}, os.path.join(_cfg_get(cfg,'CHECKPOINT_DIR','checkpoints'), f"{model_name}_best.pth"))
        if epoch - best_epoch >= patience: print(f"Early stopping at epoch {epoch} (best {best_epoch})"); break

    import pandas as pd, numpy as np
    hist_df = pd.DataFrame(history); hist_df.to_csv(os.path.join(out_dir, 'history.csv'), index=False)
    if best_state is not None: model.load_state_dict(best_state['model'])

    test_loss, test_mae, test_rmse = evaluate(model, te_ld, device, non_blocking=True)
    print(f"Test: loss={test_loss:.6f} | MAE={test_mae:.4f} | RMSE={test_rmse:.4f}")
    y_pred_test, y_true_test = _infer_batches(model, te_ld, device, non_blocking=True)

    hm_df = _compute_horizon_metrics(y_pred_test, y_true_test, den['mu'], den['sigma']); hm_df.to_csv(os.path.join(out_dir,'horizon_metrics.csv'), index=False)

    # Samples
    k_samples = int(_cfg_get(cfg,'N_SAMPLES',10))
    idxs = _evenly_spaced_indices(y_true_test.shape[0], k_samples)
    samples = []
    for i in idxs:
        row = {'idx': int(i), 'y_true_z': y_true_test[i].astype(np.float32), 'y_pred_z': y_pred_test[i].astype(np.float32)}
        if den['mu'] is not None and den['sigma'] is not None and den['sigma'] != 0:
            row['y_true_C'] = (y_true_test[i]*den['sigma']+den['mu']).astype(np.float32)
            row['y_pred_C'] = (y_pred_test[i]*den['sigma']+den['mu']).astype(np.float32)
        samples.append(row)
    hist_tail = None
    if 'orig_target_col' in den and den['orig_target_col'] and den['orig_target_col'] in []:
        pass
    np.savez(os.path.join(out_dir, 'samples_test.npz'), samples=np.array(samples, dtype=object), hist_tail_C=hist_tail)

    # Feature importance (permutation) on validation set
    baseline_mse = float(np.mean((y_pred_test - y_true_test) ** 2))  # quick baseline from test (ok for display)
    fi_rows = []
    # simple permutation on validation
    X_val = X_va; y_val = y_va
    F = X_val.shape[-1]
    for f in range(F):
        Xp = X_val.copy(); idx = np.random.permutation(Xp.shape[0]); Xp[:, :, f] = Xp[idx, :, f]
        preds = []
        for s in range(0, Xp.shape[0], batch_size):
            xb = torch.from_numpy(Xp[s:s+batch_size]).float().to(device)
            xfb = torch.from_numpy(Xf_va[s:s+batch_size]).float().to(device) if Xf_va is not None else None
            with torch.no_grad():
                pb = model(xb, xfb) if xfb is not None else model(xb)
            preds.append(pb.cpu().numpy())
        yhat = np.concatenate(preds, axis=0)
        mse_perm = float(np.mean((yhat - y_val) ** 2))
        fi_rows.append({'feature': str(f), 'delta_mse': mse_perm - baseline_mse})
    fi_df = pd.DataFrame(fi_rows); fi_df.to_csv(os.path.join(out_dir, 'feature_importance.csv'), index=False)

    results = {
        'best_epoch': best_epoch,
        'test_metrics': {'loss': test_loss, 'MAE': test_mae, 'RMSE': test_rmse},
        'paths': {
            'history_csv': os.path.join(out_dir, 'history.csv'),
            'horizon_csv': os.path.join(out_dir, 'horizon_metrics.csv'),
            'samples_npz': os.path.join(out_dir, 'samples_test.npz'),
            'feature_importance_csv': os.path.join(out_dir, 'feature_importance.csv'),
        },
        'denorm': den,
        'meta': {
            'H': _cfg_get(cfg,'H',None), 'L': L_hor, 'features': _cfg_get(cfg,'FEATURE_COLS',[]),
            'target_norm': _cfg_get(cfg,'TARGET_COL_NORM',''), 'datetime_col': _cfg_get(cfg,'DATETIME_COL',None),
            'leakage_enabled': bool(_cfg_get(cfg,'LEAKAGE_ENABLED',False)), 'leak_features': list(_cfg_get(cfg,'LEAK_FEATURES',[]))
        }
    }
    return results
