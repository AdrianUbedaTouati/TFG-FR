
# train_nbeats.py
from __future__ import annotations
import os, math, json
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ParameterGrid, ParameterSampler

# ======================
# Utilidades generales
# ======================

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def detect_datetime_col(cols):
    candidates = [c for c in cols if str(c).lower() in {
        "formatted date", "date", "time", "datetime", "timestamp", "time_local"
    }]
    if candidates:
        return candidates[0]
    for c in cols:
        cl = str(c).lower()
        if "date" in cl or "time" in cl:
            return c
    return None

def detect_target_col(cols):
    ranked = []
    for c in cols:
        cl = str(c).lower()
        score = 0
        if "temp" in cl: score += 2
        if "_z" in cl or "norm" in cl or "normalized" in cl: score += 1
        if "temperature" in cl: score += 1
        if score > 0:
            ranked.append((score, c))
    if ranked:
        ranked.sort(reverse=True)
        return ranked[0][1]
    return None

def denorm_array(arr: np.ndarray, mean: Optional[float], std: Optional[float]) -> np.ndarray:
    if mean is None or std is None:
        return arr
    return arr * std + mean

# ======================
# Datos y ventanas
# ======================
class WindowedMultivar(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, H: int, L: int, start: int, end: int):
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.H = H; self.L = L; self.start = start; self.end = end
        starts = []
        N = len(y)
        max_t = min(end, N) - (H + L)
        t0 = max(0, start - H)
        for t in range(t0, max_t + 1):
            y_end = t + H + L
            if (t + H >= start) and (y_end <= end):
                starts.append(t)
        self.idxs = np.array(starts, dtype=np.int64)
        self.D = X.shape[1]
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        t = self.idxs[i]
        x_win = self.X[t : t + self.H, :]
        y_win = self.y[t + self.H : t + self.H + self.L]
        return x_win.reshape(-1), y_win

# ======================
# Modelo N-BEATS (genérico)
# ======================
class NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, forecast_size: int, width: int, depth: int):
        super().__init__()
        layers = []
        in_features = input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            in_features = width
        self.mlp = nn.Sequential(*layers)
        self.backcast_linear = nn.Linear(width, input_size)
        self.forecast_linear = nn.Linear(width, forecast_size)
    def forward(self, x):
        h = self.mlp(x)
        backcast = self.backcast_linear(h)
        forecast = self.forecast_linear(h)
        return backcast, forecast

class NBeats(nn.Module):
    def __init__(self, input_size: int, forecast_size: int, width: int = 256, depth_per_block: int = 2, num_blocks: int = 6):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, forecast_size, width, depth_per_block)
            for _ in range(num_blocks)
        ])
    def forward(self, x):
        residual = x
        forecast_agg = torch.zeros(x.size(0), self.blocks[0].forecast_linear.out_features, device=x.device)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_agg = forecast_agg + forecast
        return forecast_agg

# ======================
# Métricas y loops
# ======================
def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse

def train_one_epoch(model, loader, device, optimizer, criterion, grad_clip: Optional[float] = None, scaler: Optional[amp.GradScaler]=None, use_amp: bool=False):
    model.train(); total=0.0; n=0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with amp.autocast():
                y_hat = model(x)
                loss = criterion(y_hat, y)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += float(loss.detach().cpu().item()) * x.size(0); n += x.size(0)
    return total / max(1, n)

def evaluate(model, loader, device, criterion, use_amp: bool=False):
    model.eval(); total=0.0; n=0; y_all=[]; yhat_all=[]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if use_amp:
                with amp.autocast():
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
            else:
                y_hat = model(x)
                loss = criterion(y_hat, y)
            total += float(loss.detach().cpu().item()) * x.size(0); n += x.size(0)
            y_all.append(y.cpu().numpy()); yhat_all.append(y_hat.cpu().numpy())
    if n == 0: return math.nan, math.nan, math.nan
    y_all = np.concatenate(y_all, axis=0); yhat_all = np.concatenate(yhat_all, axis=0)
    mae, rmse = metrics(y_all, yhat_all)
    return total / n, mae, rmse

# ======================
# Carga de datos
# ======================
def auto_feature_cols(df: pd.DataFrame, dt_col: str, target_col: str, include_target_as_feature: bool) -> List[str]:
    num_cols = [c for c in df.columns if c != dt_col and np.issubdtype(df[c].dtype, np.number)]
    norm_like = [c for c in num_cols if ("_z" in str(c).lower()) or ("norm" in str(c).lower()) or ("normalized" in str(c).lower())]
    cols = norm_like if len(norm_like) >= 2 else num_cols
    if include_target_as_feature and target_col not in cols:
        cols = [target_col] + cols
    seen=set(); uniq=[]
    for c in cols:
        if c not in seen: uniq.append(c); seen.add(c)
    return uniq

def load_multivar(csv_path: str, datetime_col: Optional[str], target_col_norm: Optional[str], 
                  feature_cols: Optional[List[str]], include_target_as_feature: bool):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el CSV en {csv_path}")
    df = pd.read_csv(csv_path)

    dt_col = datetime_col or detect_datetime_col(df.columns)
    if dt_col is None:
        raise ValueError("No se pudo detectar la columna temporal. Especifica DATETIME_COL.")
    tgt_col = target_col_norm or detect_target_col(df.columns)
    if tgt_col is None:
        raise ValueError("No se pudo detectar la columna objetivo normalizada. Especifica TARGET_COL_NORM.")

    df[dt_col] = pd.to_datetime(df[dt_col], utc=True).dt.tz_localize(None)
    df = df.sort_values(dt_col)

    feats = feature_cols if (feature_cols is not None and len(feature_cols) > 0) else auto_feature_cols(df, dt_col, tgt_col, include_target_as_feature)
    feats = [c for c in feats if c in df.columns and c != dt_col]
    if include_target_as_feature and tgt_col not in feats:
        feats = [tgt_col] + feats
    if len(feats) < 1:
        raise ValueError("No se encontraron columnas de entrada válidas. Revisa FEATURE_COLS o el CSV.")
    for c in feats + [tgt_col]:
        if not np.issubdtype(df[c].dtype, np.number):
            raise TypeError(f"La columna '{c}' no es numérica.")
    use_cols = [dt_col] + feats + ([tgt_col] if tgt_col not in feats else [])
    use_cols = list(dict.fromkeys(use_cols))
    df = df[use_cols].dropna()
    return df, dt_col, tgt_col, feats

# ======================
# Importancia de variables por permutación
# ======================
def collect_windows(X: np.ndarray, y: np.ndarray, H: int, L: int, start: int, end: int):
    ds = WindowedMultivar(X, y, H, L, start, end)
    Xw, Yw = [], []
    for t in ds.idxs:
        x_hist = X[t:t+H, :]
        y_win = y[t+H:t+H+L]
        Xw.append(x_hist); Yw.append(y_win)
    if len(Xw) == 0: return None, None
    return np.stack(Xw, axis=0), np.stack(Yw, axis=0)

def evaluate_windows(model, Xw: np.ndarray, Yw: np.ndarray, device, batch_size: int = 1024, use_amp: bool=True) -> float:
    criterion = nn.MSELoss(reduction='mean'); model.eval(); total=0.0; n=0
    with torch.no_grad():
        for i in range(0, len(Xw), batch_size):
            xb = Xw[i:i+batch_size]; yb = Yw[i:i+batch_size]
            xbf = torch.as_tensor(xb.reshape(xb.shape[0], -1), dtype=torch.float32, device=device)
            ybt = torch.as_tensor(yb, dtype=torch.float32, device=device)
            if use_amp and device.type == 'cuda':
                with amp.autocast():
                    y_hat = model(xbf)
                    loss = criterion(y_hat, ybt)
            else:
                y_hat = model(xbf)
                loss = criterion(y_hat, ybt)
            total += float(loss.item()) * xbf.size(0); n += xbf.size(0)
    return total / max(1, n)

def permutation_importance(model, Xw: np.ndarray, Yw: np.ndarray, feature_names: List[str], H: int, device, rng: np.random.Generator) -> pd.DataFrame:
    base_loss = evaluate_windows(model, Xw, Yw, device)
    D = Xw.shape[2]
    inc_abs = []; inc_rel = []
    for j in range(D):
        Xp = Xw.copy()
        perm = rng.permutation(Xp.shape[0])
        Xp[:, :, j] = Xp[perm, :, j]
        loss_j = evaluate_windows(model, Xp, Yw, device)
        inc = loss_j - base_loss
        inc_abs.append(inc)
        inc_rel.append( (inc / base_loss * 100.0) if base_loss > 0 else np.nan )
    df_imp = pd.DataFrame({'feature': feature_names, 'delta_mse': inc_abs, 'delta_mse_%': inc_rel, 'base_mse': base_loss}).sort_values('delta_mse', ascending=False).reset_index(drop=True)
    return df_imp

# ======================
# Split helper
# ======================
@dataclass
class SplitIdx:
    train_end: int
    val_end: int
    N: int
def make_splits(N: int, train_ratio=0.70, val_ratio=0.15) -> SplitIdx:
    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))
    return SplitIdx(train_end=train_end, val_end=val_end, N=N)

# ======================
# Orquestador
# ======================
def run_training(cfg) -> Dict[str, Any]:
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ajustes de rendimiento GPU
    use_amp = bool(getattr(cfg, "USE_AMP", False) and device.type == "cuda")
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(getattr(cfg, "ENABLE_TF32", True))
            torch.backends.cudnn.allow_tf32 = bool(getattr(cfg, "ENABLE_TF32", True))
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high" if getattr(cfg, "ENABLE_TF32", True) else "medium")
        except Exception:
            pass
    scaler = amp.GradScaler(enabled=use_amp)

    # --- Carga
    df, dt_col, tgt_col, feat_cols = load_multivar(cfg.CSV_PATH, cfg.DATETIME_COL, cfg.TARGET_COL_NORM, cfg.FEATURE_COLS, cfg.INCLUDE_TARGET_AS_FEATURE)
    X_all = df[feat_cols].to_numpy().astype(np.float32)
    y_all = df[tgt_col].to_numpy().astype(np.float32)

    # Denormalización
    denorm_mean = denorm_std = None
    if cfg.DENORMALIZE_OUTPUTS:
        if (cfg.Z_MEAN is not None) and (cfg.Z_STD is not None):
            denorm_mean, denorm_std = float(cfg.Z_MEAN), float(cfg.Z_STD)
        elif cfg.ORIG_TARGET_COL and cfg.ORIG_TARGET_COL in df.columns:
            denorm_mean = float(df[cfg.ORIG_TARGET_COL].mean())
            denorm_std  = float(df[cfg.ORIG_TARGET_COL].std(ddof=0))
        print(f"Denormalización: mean={denorm_mean}, std={denorm_std}")

    N, D = X_all.shape
    if N < (cfg.H + cfg.L + 10):
        raise ValueError(f"Serie demasiado corta (N={N}) para H={cfg.H} y L={cfg.L}.")
    splits = make_splits(N, train_ratio=0.70, val_ratio=0.15)

    train_ds = WindowedMultivar(X_all, y_all, cfg.H, cfg.L, start=0,                end=splits.train_end)
    val_ds   = WindowedMultivar(X_all, y_all, cfg.H, cfg.L, start=splits.train_end, end=splits.val_end)
    test_ds  = WindowedMultivar(X_all, y_all, cfg.H, cfg.L, start=splits.val_end,   end=splits.N)

    print(f"Dataset: {cfg.CSV_PATH}")
    print(f"Tiempo: {dt_col} | Objetivo: {tgt_col}")
    print(f"Features ({D}): {feat_cols}")
    print(f"Total puntos: {N} — Train: {splits.train_end}, Val: {splits.val_end - splits.train_end}, Test: {splits.N - splits.val_end}")
    print(f"Ventana H={cfg.H}, L={cfg.L} → input_size={cfg.H*D}")
    print(f"Ventanas => Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  drop_last=False,
                               num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, prefetch_factor=cfg.PREFETCH_FACTOR,
                               persistent_workers=cfg.PERSISTENT_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=False,
                               num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, prefetch_factor=cfg.PREFETCH_FACTOR,
                               persistent_workers=cfg.PERSISTENT_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=False,
                               num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, prefetch_factor=cfg.PREFETCH_FACTOR,
                               persistent_workers=cfg.PERSISTENT_WORKERS)

    # --- HP search (opcional)
    used_hidden, used_depth, used_blocks = cfg.HIDDEN_WIDTH, cfg.DEPTH_PER_BLOCK, cfg.NUM_BLOCKS
    used_lr, used_wd = cfg.LR, cfg.WEIGHT_DECAY
    best_hp = None
    input_size = cfg.H * D
    if cfg.HP_SEARCH_ENABLED:
        print(f"Iniciando HP search: {cfg.HP_SEARCH} ...")
        mode = (cfg.HP_SEARCH or "").lower().strip()
        iterator = (list(ParameterGrid(cfg.SEARCH_PARAM_GRID))
                    if mode == "grid"
                    else list(ParameterSampler(cfg.SEARCH_PARAM_GRID, n_iter=cfg.SEARCH_MAX_ITERS, random_state=cfg.SEED)))
        results = []
        for i, p in enumerate(iterator, 1):
            width  = int(p.get("HIDDEN_WIDTH", cfg.HIDDEN_WIDTH))
            depth  = int(p.get("DEPTH_PER_BLOCK", cfg.DEPTH_PER_BLOCK))
            blocks = int(p.get("NUM_BLOCKS", cfg.NUM_BLOCKS))
            lr     = float(p.get("LR", cfg.LR))
            wd     = float(p.get("WEIGHT_DECAY", cfg.WEIGHT_DECAY))
            model = NBeats(input_size=input_size, forecast_size=cfg.L, width=width, depth_per_block=depth, num_blocks=blocks).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(2, cfg.SEARCH_PATIENCE//2))
            best_val = float('inf'); no_improve = 0
            for _ in range(1, cfg.SEARCH_EPOCHS + 1):
                _ = train_one_epoch(model, train_loader, device, optimizer, criterion, cfg.GRAD_CLIP)
                val_loss, _, _ = evaluate(model, val_loader, device, criterion)
                scheduler.step(val_loss if not math.isnan(val_loss) else 0.0)
                if val_loss < best_val: best_val, no_improve = val_loss, 0
                else:
                    no_improve += 1
                    if no_improve >= cfg.SEARCH_PATIENCE: break
            results.append({"iter": i, "HIDDEN_WIDTH": width, "DEPTH_PER_BLOCK": depth, "NUM_BLOCKS": blocks, "LR": lr, "WEIGHT_DECAY": wd, "best_val": float(best_val)})
        out_dir = cfg.OUTPUT_DIR; os.makedirs(out_dir, exist_ok=True)
        if results:
            pd.DataFrame(results).to_csv(os.path.join(out_dir, "hpsearch_results.csv"), index=False)
            with open(os.path.join(out_dir, "best_hparams.json"), "w", encoding="utf-8") as f:
                json.dump(min(results, key=lambda r: r["best_val"]), f, ensure_ascii=False, indent=2)
            best = min(results, key=lambda r: r["best_val"])
            used_hidden = int(best["HIDDEN_WIDTH"]); used_depth = int(best["DEPTH_PER_BLOCK"]); used_blocks = int(best["NUM_BLOCKS"])
            used_lr = float(best["LR"]); used_wd = float(best["WEIGHT_DECAY"]); best_hp = best
            print(f"Mejores HP → width={used_hidden}, depth={used_depth}, blocks={used_blocks}, lr={used_lr}, wd={used_wd}, val={best['best_val']:.6f}")
        else:
            print("HP search sin resultados; se usan valores por defecto.")

    # --- Modelo
    model = NBeats(input_size=input_size, forecast_size=cfg.L, width=used_hidden, depth_per_block=used_depth, num_blocks=used_blocks).to(device)
    if getattr(cfg, 'TORCH_COMPILE', False):
        try:
            model = torch.compile(model, mode='max-autotune')
        except Exception as _e:
            print(f"[WARN] torch.compile no disponible: {_e}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=used_lr, weight_decay=used_wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(2, cfg.PATIENCE//3))

    best_val = float('inf'); no_improve = 0
    ckpt_name = f"nbeats_MULTI_H{cfg.H}_L{cfg.L}_{tgt_col.replace(' ','_')}_D{D}.pth"
    best_path = os.path.join(cfg.CHECKPOINT_DIR, ckpt_name)

    train_hist, val_hist = [], []
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion, cfg.GRAD_CLIP, scaler=scaler, use_amp=use_amp)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device, criterion, use_amp=use_amp)
        scheduler.step(val_loss if not math.isnan(val_loss) else 0.0)
        train_hist.append(train_loss); val_hist.append(val_loss)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f}")
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'config': {'H': cfg.H, 'L': cfg.L, 'width': used_hidden, 'depth_per_block': used_depth, 'num_blocks': used_blocks,
                                   'target_col': tgt_col, 'feature_cols': feat_cols, 'lr': used_lr, 'weight_decay': used_wd},
                        'best_val': best_val, 'best_hp': best_hp}, best_path)
        else:
            no_improve += 1
            if no_improve >= cfg.PATIENCE:
                print(f"Early stopping (patience={cfg.PATIENCE})."); break

    # Cargar mejor checkpoint
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Cargado mejor checkpoint: {best_path}")

    # Evaluación en test
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, device, criterion)
    print(f"Test: loss={test_loss:.6f} | MAE={test_mae:.4f} | RMSE={test_rmse:.4f}")

    # Forecast operativo del último horizonte
    last_X = torch.tensor(X_all[-(cfg.H+cfg.L):-cfg.L, :].reshape(1, -1), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                last_pred = model(last_X).cpu().numpy().reshape(-1)
        else:
            last_pred = model(last_X).cpu().numpy().reshape(-1)

    # Predicciones en todo el test
    yhats = []; ys = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            if use_amp:
                with amp.autocast():
                    y_hat = model(x).cpu().numpy()
            else:
                y_hat = model(x).cpu().numpy()
            yhats.append(y_hat); ys.append(y.numpy())
    yhats = np.concatenate(yhats, axis=0); ys = np.concatenate(ys, axis=0)
    residuals = ys - yhats
    mae_per_h = np.mean(np.abs(residuals), axis=0)
    rmse_per_h = np.sqrt(np.mean(residuals**2, axis=0))

    ys_real = yhats_real = residuals_real = mae_per_h_real = rmse_per_h_real = None
    if cfg.DENORMALIZE_OUTPUTS and (denorm_mean is not None) and (denorm_std is not None):
        ys_real = denorm_array(ys, denorm_mean, denorm_std)
        yhats_real = denorm_array(yhats, denorm_mean, denorm_std)
        residuals_real = ys_real - yhats_real
        mae_per_h_real = np.mean(np.abs(residuals_real), axis=0)
        rmse_per_h_real = np.sqrt(np.mean(residuals_real**2, axis=0))

    # Importancia de variables en validación
    imp_df = None
    if cfg.COMPUTE_FEATURE_IMPORTANCE:
        Xw_val, Yw_val = collect_windows(X_all, y_all, cfg.H, cfg.L, splits.train_end, splits.val_end)
        if Xw_val is None:
            print("[WARN] No hay ventanas de validación suficientes para calcular importancia de variables.")
        else:
            rng = np.random.default_rng(cfg.SEED)
            imp_df = permutation_importance(model, Xw_val, Yw_val, feat_cols, cfg.H, device, rng)

    # Guardados no-gráficos: métricas, forecast CSV, TorchScript y config
    out_dir = cfg.OUTPUT_DIR; plots_dir = os.path.join(out_dir, "plots"); os.makedirs(out_dir, exist_ok=True); os.makedirs(plots_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_MULTI_H{cfg.H}_L{cfg.L}_{tgt_col.replace(' ','_')}.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Val best MSE: {best_val:.6f}\n")
        f.write(f"Test MSE: {test_loss:.6f}\n")
        f.write(f"Test MAE: {test_mae:.6f}\n")
        f.write(f"Test RMSE: {test_rmse:.6f}\n")

    last_fut_times  = pd.date_range(df[dt_col].iloc[-1] + pd.tseries.frequencies.to_offset('H'), periods=cfg.L, freq='H')
    forecast_cols = {dt_col: last_fut_times, f"pred_{tgt_col}": last_pred}
    if cfg.DENORMALIZE_OUTPUTS and (denorm_mean is not None) and (denorm_std is not None):
        real_name = cfg.ORIG_TARGET_COL or (tgt_col + "_real")
        forecast_cols[f"pred_{real_name}"] = denorm_array(last_pred, denorm_mean, denorm_std)
    pd.DataFrame(forecast_cols).to_csv(os.path.join(out_dir, f"forecast_last_MULTI_H{cfg.H}_L{cfg.L}_{tgt_col.replace(' ','_')}.csv"), index=False)

    try:
        example = torch.randn(1, cfg.H*D, dtype=torch.float32, device=device)
        traced = torch.jit.trace(model, example)
        torchscript_path = os.path.join(cfg.CHECKPOINT_DIR, f"nbeats_MULTI_H{cfg.H}_L{cfg.L}_{tgt_col.replace(' ','_')}_D{D}_torchscript.pt")
        traced.save(torchscript_path)
        print(f"Guardado TorchScript en: {torchscript_path}")
    except Exception as e:
        print(f"No se pudo exportar a TorchScript: {e}")

    config_dump = {
        'CSV_PATH': cfg.CSV_PATH, 'DATETIME_COL': dt_col, 'TARGET_COL_NORM': tgt_col, 'ORIG_TARGET_COL': cfg.ORIG_TARGET_COL,
        'FEATURE_COLS': feat_cols, 'DENORMALIZE_OUTPUTS': cfg.DENORMALIZE_OUTPUTS, 'Z_MEAN': cfg.Z_MEAN, 'Z_STD': cfg.Z_STD,
        'H': cfg.H, 'L': cfg.L, 'SEED': cfg.SEED, 'BATCH_SIZE': cfg.BATCH_SIZE, 'EPOCHS': cfg.EPOCHS, 'LR': cfg.LR,
        'WEIGHT_DECAY': cfg.WEIGHT_DECAY, 'PATIENCE': cfg.PATIENCE, 'HIDDEN_WIDTH': cfg.HIDDEN_WIDTH,
        'DEPTH_PER_BLOCK': cfg.DEPTH_PER_BLOCK, 'NUM_BLOCKS': cfg.NUM_BLOCKS, 'HP_SEARCH_ENABLED': cfg.HP_SEARCH_ENABLED,
        'HP_SEARCH': cfg.HP_SEARCH, 'SEARCH_PARAM_GRID': cfg.SEARCH_PARAM_GRID, 'SEARCH_MAX_ITERS': cfg.SEARCH_MAX_ITERS,
        'SEARCH_EPOCHS': cfg.SEARCH_EPOCHS, 'SEARCH_PATIENCE': cfg.SEARCH_PATIENCE, 'COMPUTE_FEATURE_IMPORTANCE': cfg.COMPUTE_FEATURE_IMPORTANCE,
        'TOPK_IMPORTANCE_PLOT': cfg.TOPK_IMPORTANCE_PLOT
    }
    with open(os.path.join(out_dir, f"run_config_MULTI_H{cfg.H}_L{cfg.L}_{tgt_col.replace(' ','_')}.json"), 'w', encoding='utf-8') as f:
        json.dump(config_dump, f, ensure_ascii=False, indent=2)

    return {
        "device": str(device), "df": df, "dt_col": dt_col, "tgt_col": tgt_col, "feat_cols": feat_cols, "splits": splits,
        "X_all": X_all, "y_all": y_all, "D": D, "N": N,
        "train_hist": train_hist, "val_hist": val_hist,
        "test_loss": test_loss, "test_mae": test_mae, "test_rmse": test_rmse,
        "yhats": yhats, "ys": ys, "residuals": residuals,
        "mae_per_h": mae_per_h, "rmse_per_h": rmse_per_h,
        "ys_real": ys_real, "yhats_real": yhats_real, "residuals_real": residuals_real,
        "mae_per_h_real": mae_per_h_real, "rmse_per_h_real": rmse_per_h_real,
        "denorm_mean": denorm_mean, "denorm_std": denorm_std,
        "last_pred": last_pred, "best_path": best_path,
        "imp_df": imp_df
    }
