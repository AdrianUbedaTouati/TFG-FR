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


# -----------------------------
# Small utils
# -----------------------------

def _cfg_get(cfg, name, default=None):
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
    def __init__(self, X: np.ndarray, y: np.ndarray, future_feats: Optional[np.ndarray] = None):
        self.X = X.astype(np.float32)  # [N, H, F]
        self.y = y.astype(np.float32)  # [N, L]
        self.fut = None if future_feats is None else future_feats.astype(np.float32)  # [N, L, Ff]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.fut is None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx], self.y[idx], self.fut[idx]

def build_windows(df: pd.DataFrame, feature_cols: List[str], target_col_norm: str,
                  H: int, L: int, include_target_as_feature: bool,
                  orig_target_col: Optional[str], stride: int = 1,
                  use_time_leakage: bool = False,
                  time_leakage_features: Optional[List[str]] = None
                  ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """ 
    Returns:
      X: [N, H, F], y: [N, L],
      hist_target_raw: [N, H] (si hay orig_target_col),
      future_feats: [N, L, Ff] (si use_time_leakage=True)
    Nota: NO se crean nuevas columnas; si se activa leakage, se tomarán directamente
    del DataFrame las columnas indicadas en `time_leakage_features`.
    """
    values_feat = df[feature_cols].values.astype(np.float32)  # [T, F]
    target_z = df[target_col_norm].values.astype(np.float32)  # [T]
    total = len(df)

    future_all = None
    sel_cols: List[str] = []
    if use_time_leakage:
        # por defecto exactamente estas 6
        if time_leakage_features is None:
            time_leakage_features = ["h_sin","h_cos","dow_sin","dow_cos","doy_sin","doy_cos"]
        # comprobar que existen en el df, y usarlas en ese orden
        missing = [c for c in time_leakage_features if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Faltan columnas de leakage en el CSV: {missing}")
        sel_cols = list(time_leakage_features)
        future_all = df[sel_cols].values.astype(np.float32)  # [T, Ff]

    X_list, y_list, hist_raw_list, fut_list = [], [], [], []
    use_raw_hist = orig_target_col is not None and orig_target_col in df.columns
    if use_raw_hist:
        raw = df[orig_target_col].values.astype(np.float32)

    for start in range(0, total - (H + L) + 1, stride):
        end_hist = start + H
        end_full = end_hist + L

        X_list.append(values_feat[start:end_hist, :])  # [H, F]
        y_list.append(target_z[end_hist:end_full])     # [L]

        if use_raw_hist:
            hist_raw_list.append(raw[start:end_hist])

        if use_time_leakage:
            fut_list.append(future_all[end_hist:end_full, :])  # [L, Ff]

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, H, values_feat.shape[1]), np.float32)
    y = np.stack(y_list, axis=0) if y_list else np.empty((0, L), np.float32)
    hist_raw = np.stack(hist_raw_list, axis=0) if (use_raw_hist and hist_raw_list) else None
    future_matrix = None
    if use_time_leakage:
        future_matrix = np.stack(fut_list, axis=0) if fut_list else np.empty((0, L, len(sel_cols)), np.float32)

    return X, y, hist_raw, future_matrix

def _split_train_val_test(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    N = len(df)
    n_train = int(math.floor(N * train_frac))
    n_val = int(math.floor(N * val_frac))
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
        delta = torch.tensor(self.delta, device=err.device)
        quad = torch.minimum(abs_err, delta)
        lin = abs_err - quad
        huber = 0.5 * quad ** 2 + delta * lin
        # weight per horizon (broadcast over batch)
        w = self.weights.to(err.device).unsqueeze(0)  # [1, L]
        loss = (huber * w).mean()
        return loss

# -----------------------------
# Model: LSTM with optional per-step conditioning on future_feats
# -----------------------------

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, horizon: int,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False,
                 head_hidden: Optional[int] = None,
                 future_feat_size: int = 0,
                 head_dropout: float = 0.0):
        super().__init__()
        self.horizon = horizon
        self.bidirectional = bidirectional
        self.future_feat_size = int(future_feat_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        last_size = hidden_size * (2 if bidirectional else 1)

        if self.future_feat_size > 0:
            in_head = last_size + self.future_feat_size
            hid = head_hidden or max(32, last_size // 2)
            self.step_head = nn.Sequential(
                nn.Linear(in_head, hid),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(hid, 1)
            )
        else:
            if head_hidden is None:
                self.head = nn.Linear(last_size, horizon)
            else:
                self.head = nn.Sequential(
                    nn.Linear(last_size, head_hidden),
                    nn.ReLU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(head_hidden, horizon)
                )

    def forward(self, x: torch.Tensor, future_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, H, F]; future_feats: [B, L, Ff] or None
        out, (h_n, c_n) = self.lstm(x)
        # last hidden (use output's last time step for simplicity)
        last = out[:, -1, :]
        if self.future_feat_size > 0 and future_feats is not None:
            B, L, Ff = future_feats.shape
            last_rep = last.unsqueeze(1).expand(-1, L, -1)  # [B, L, last_size]
            step_in = torch.cat([last_rep, future_feats], dim=-1)  # [B, L, last_size+Ff]
            y = self.step_head(step_in).squeeze(-1)  # [B, L]
            return y
        else:
            return self.head(last)  # [B, L]

# -----------------------------
# Training & evaluation
# -----------------------------

def _to_device(batch, device, non_blocking=False):
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        x, y, f = batch
        return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking), f.to(device, non_blocking=non_blocking)
    else:
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
        tb = _to_device(batch, device, non_blocking=non_blocking)
        if isinstance(tb, tuple) and len(tb) == 3:
            x, y, f = tb
            yhat = model(x, future_feats=f)
        else:
            x, y = tb
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
    for batch in loader:
        tb = _to_device(batch, device, non_blocking=non_blocking)
        if isinstance(tb, tuple) and len(tb) == 3:
            x, y, f = tb
            with torch.no_grad():
                yhat = model(x, future_feats=f).cpu().numpy()
        else:
            x, y = tb
            with torch.no_grad():
                yhat = model(x).cpu().numpy()
        preds.append(yhat)
        trues.append(y.detach().cpu().numpy())
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
            'RMSE_C': m_rmse_c
        })
    return pd.DataFrame(rows)

def _evenly_spaced_indices(n: int, k: int) -> List[int]:
    if k >= n:
        return list(range(n))
    # linspace-like integer selection
    return sorted({int(round(i)) for i in np.linspace(0, n-1, num=k)})

def _feature_importance_permutation(model, X_val, y_val, baseline_mse: float, device, batch_size: int,
                                    non_blocking: bool, feature_names: List[str],
                                    future_feats_val: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Return DataFrame with columns: feature, delta_mse
    Permuta SOLO columnas de X (no toca future_feats)."""
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
            if future_feats_val is not None:
                fb = torch.from_numpy(future_feats_val[i:i+batch_size]).float().to(device, non_blocking=non_blocking)
                with torch.no_grad():
                    pb = model(xb, future_feats=fb).cpu().numpy()
            else:
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
    set_seed(int(seed) if seed is not None else 42)

    # Device
    dev_pref = _cfg_get(cfg, 'DEVICE', None)
    if dev_pref is None or str(dev_pref).lower() == 'auto':
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
    datetime_col = _cfg_get(cfg, 'DATETIME_COL', 'Formatted Date')  # kept for meta only
    target_col_norm = _cfg_get(cfg, 'TARGET_COL_NORM', 'Temperature (C)_normalized')
    orig_target_col = _cfg_get(cfg, 'ORIG_TARGET_COL', None)
    feature_cols = list(_cfg_get(cfg, 'FEATURE_COLS', []))
    include_target = bool(_cfg_get(cfg, 'INCLUDE_TARGET_AS_FEATURE', True))

    # Ensure target feature is in features if requested
    if include_target and target_col_norm not in feature_cols:
        feature_cols = feature_cols + [target_col_norm]

    # Leakage config
    use_time_leakage = bool(_cfg_get(cfg, 'USE_TIME_LEAKAGE', False))
    leakage_cols = list(_cfg_get(cfg, 'TIME_LEAKAGE_FEATURES', ['h_sin','h_cos','dow_sin','dow_cos','doy_sin','doy_cos']))

    # 3) Read CSV
    print(f"Leyendo CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    # sort by datetime if present
    if datetime_col in df.columns:
        try:
            df = df.sort_values(datetime_col).reset_index(drop=True)
        except Exception:
            pass

    # 4) Split
    train_frac = float(_cfg_get(cfg, 'TRAIN_FRACTION', 0.7))
    val_frac = float(_cfg_get(cfg, 'VAL_FRACTION', 0.15))
    df_train, df_val, df_test = _split_train_val_test(df, train_frac, val_frac)
    print(f"Split sizes: train={len(df_train)} val={len(df_val)} test={len(df_test)}")

    # 5) Denorm stats from TRAIN (if available)
    mu = sigma = None
    if orig_target_col is None and target_col_norm.endswith('_normalized'):
        # try typical raw col name
        guess = target_col_norm.replace('_normalized', '')
        if guess in df.columns:
            orig_target_col = guess
    if orig_target_col is not None and orig_target_col in df.columns:
        mu = float(df_train[orig_target_col].mean())
        sigma = float(df_train[orig_target_col].std())
        print(f"Denormalización (TRAIN): mean={mu:.3f}, std={sigma:.3f}")
    else:
        print("Denormalización: no se encontró columna cruda del objetivo; se reportará solo en z-score.")

    # 6) Window sizes & model sizes
    H = int(_cfg_get(cfg, 'H', 1440))
    L = int(_cfg_get(cfg, 'L', 120))

    # 7) Build windows per split
    stride_tr = int(_cfg_get(cfg, 'STRIDE_TRAIN', 1))
    stride_va = int(_cfg_get(cfg, 'STRIDE_VAL', 2))
    stride_te = int(_cfg_get(cfg, 'STRIDE_TEST', 2))

    X_train, y_train, _hist_tr, fut_tr = build_windows(df_train, feature_cols, target_col_norm, H, L, include_target, orig_target_col, stride=stride_tr, use_time_leakage=use_time_leakage, time_leakage_features=leakage_cols)
    X_val, y_val, _hist_va, fut_va = build_windows(df_val, feature_cols, target_col_norm, H, L, include_target, orig_target_col, stride=stride_va, use_time_leakage=use_time_leakage, time_leakage_features=leakage_cols)
    X_test, y_test, hist_raw_test, fut_te = build_windows(df_test, feature_cols, target_col_norm, H, L, include_target, orig_target_col, stride=stride_te, use_time_leakage=use_time_leakage, time_leakage_features=leakage_cols)

    # 8) Dataloaders
    batch_size = int(_cfg_get(cfg, 'BATCH_SIZE', 256))
    num_workers = int(_cfg_get(cfg, 'NUM_WORKERS', 0))
    pin_memory = bool(_cfg_get(cfg, 'PIN_MEMORY', False))
    prefetch_factor = int(_cfg_get(cfg, 'PREFETCH_FACTOR', 2))
    persistent_workers = bool(_cfg_get(cfg, 'PERSISTENT_WORKERS', False))

    def _make_loader(ds, bs, shuffle, num_workers, pin_memory, prefetch_factor, persistent_workers):
        kw = dict(batch_size=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        if num_workers and num_workers > 0:
            kw['prefetch_factor'] = prefetch_factor
            kw['persistent_workers'] = persistent_workers
        loader = DataLoader(ds, **kw)
        return loader

    train_ds = WindowedDataset(X_train, y_train, fut_tr if use_time_leakage else None)
    val_ds = WindowedDataset(X_val, y_val, fut_va if use_time_leakage else None)
    test_ds = WindowedDataset(X_test, y_test, fut_te if use_time_leakage else None)

    train_loader = _make_loader(train_ds, batch_size, True, num_workers, (pin_memory if num_workers>0 else False), prefetch_factor, persistent_workers)
    val_loader = _make_loader(val_ds, batch_size, False, num_workers, (pin_memory if num_workers>0 else False), prefetch_factor, persistent_workers)
    test_loader = _make_loader(test_ds, batch_size, False, num_workers, (pin_memory if num_workers>0 else False), prefetch_factor, persistent_workers)

    # inject criterion for evaluate()
    crit = WeightedHuberLoss(horizon=L, delta=float(_cfg_get(cfg, 'HUBER_DELTA', 1.0)),
                             w_start=float(_cfg_get(cfg, 'W_H_START', 1.0)),
                             w_end=float(_cfg_get(cfg, 'W_H_END', 0.6)))
    for ld in [val_loader, test_loader]:
        ld._criterion = crit  # type: ignore

    # 9) Model
    future_feat_size = 0
    if use_time_leakage and fut_tr is not None and fut_tr.size > 0:
        future_feat_size = fut_tr.shape[-1]

    model = LSTMForecaster(
        input_size=X_train.shape[-1],
        horizon=L,
        hidden_size=int(_cfg_get(cfg, 'LSTM_HIDDEN_SIZE', 128)),
        num_layers=int(_cfg_get(cfg, 'LSTM_NUM_LAYERS', 1)),
        dropout=float(_cfg_get(cfg, 'LSTM_DROPOUT', 0.0)),
        bidirectional=bool(_cfg_get(cfg, 'LSTM_BIDIRECTIONAL', False)),
        head_hidden=_cfg_get(cfg, 'LSTM_HEAD_HIDDEN', None),
        future_feat_size=future_feat_size,
        head_dropout=float(_cfg_get(cfg, 'LSTM_DROPOUT', 0.0)) if _cfg_get(cfg, 'LSTM_HEAD_HIDDEN', None) else 0.0
    ).to(device)

    # 10) Optim & sched
    lr = float(_cfg_get(cfg, 'LR', 7e-4))
    wd = float(_cfg_get(cfg, 'WEIGHT_DECAY', 5e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    t_max = int(_cfg_get(cfg, 'T_MAX', 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max))

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type=='cuda'))
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
            tb = _to_device(batch, device, non_blocking)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(tb, tuple) and len(tb) == 3:
                x, y, f = tb
                with torch.amp.autocast(device_type='cuda', enabled=use_amp and device.type == 'cuda', dtype=torch.float16):
                    yhat = model(x, future_feats=f)
                    loss = crit(yhat, y)
            else:
                x, y = tb
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
            torch.save({'state_dict': best_state['model'], 'epoch': best_epoch, 'cfg': asdict(cfg) if hasattr(cfg,'__dict__') else None}, ckpt_path)

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"Early stopping en epoch {epoch} (mejor val_MAE en {best_epoch})")
            break

    elapsed = time.time() - start_time
    print(f"Entrenamiento terminado en {elapsed/60.0:.1f} min. Mejor epoch: {best_epoch}")

    # 12) Eval on TEST
    if best_state is not None:
        model.load_state_dict(best_state['model'])
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

    npz_path = os.path.join(out_dir, 'samples_test.npz')
    np.savez(npz_path, samples=np.array(samples, dtype=object), hist_tail_C=hist_tail)

    # 15) Save history
    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(out_dir, 'history.csv')
    hist_df.to_csv(hist_path, index=False)

    # 16) Feature importance (permutation) on VAL
    # Baseline MSE on VAL
    y_pred_val, y_true_val = _infer_batches(model, val_loader, device, non_blocking)
    baseline_mse = float(np.mean((y_pred_val - y_true_val) ** 2))
    fi_df = _feature_importance_permutation(model, X_val, y_true_val, baseline_mse, device,
                                            batch_size, non_blocking, feature_cols,
                                            future_feats_val=(fut_va if use_time_leakage else None))
    fi_path = os.path.join(out_dir, 'feature_importance.csv')
    fi_df.to_csv(fi_path, index=False)

    results = {
        'history_csv': hist_path,
        'paths': {
            'history_csv': hist_path,
            'horizon_csv': hm_path,
            'samples_npz': npz_path,
            'feature_importance_csv': fi_path,
        },
        'test_metrics': {'loss': test_loss, 'MAE': test_mae, 'RMSE': test_rmse},
        'denorm': {'mu': mu, 'sigma': sigma, 'orig_target_col': orig_target_col},
        'meta': {'H': H, 'L': L, 'features': feature_cols, 'target_norm': target_col_norm, 'datetime_col': datetime_col},
    }
    return results

# --- PUBLIC API ---
def train_lstm(cfg) -> Dict[str, Any]:
    return run_training(cfg)
