
"""
N-BEATS multivariable (PyTorch) para series horarias (multi‑step L)
------------------------------------------------------------------
Versión *parcheada* para:
  • Mostrar SIEMPRE la temperatura en escala real cuando sea posible.
  • Detectar automáticamente si el objetivo ya está en escala real o es z‑score.
  • Importancia de variables mejorada: permutación con *n_repeats* + barras de error.
  • Gráficas de ventanas más legibles (zona de predicción sombreada, etiquetas y unidades).
  • CSV_PATH preparado para tu dataset.

Basado en tu script original. Requisitos:
    pip install torch pandas numpy scikit-learn matplotlib
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import ParameterGrid, ParameterSampler

# ==================================================
# CONFIG ——— cambia aquí rutas, columnas y tamaños
# ==================================================
# ← usa directamente tu CSV subido; cámbialo si lo mueves
CSV_PATH: str = "data/weatherHistory_normalize.csv"

# Si None se detecta automáticamente entre:
# ['Formatted Date','Date','time','datetime','timestamp','time_local']
DATETIME_COL: Optional[str] = "Formatted Date"

# Columna objetivo (puede ser normalizada *o* real). Si None se detecta automáticamente.
TARGET_COL: Optional[str] = "Temperature (C)_normalized"   # p.ej. "Temperature (C)_normalized" o "Temperature (C)"

# Columna objetivo en escala real (si existe). Si la TARGET_COL es z-score, se usará para denormalizar.
ORIG_TARGET_COL: Optional[str] = "Temperature (C)"

# Features de entrada (mismo *scale* que la tabla, da igual real o normalizado).
# Si None ⇒ selección automática de columnas numéricas (incluye el objetivo si INCLUDE_TARGET_AS_FEATURE=True).
FEATURE_COLS: Optional[List[str]] = ["h_sin","h_cos","dow_sin","dow_cos","doy_sin","doy_cos","Precip Type_normalized","Temperature (C)_normalized","wind_bearing_sin","wind_bearing_cos","Summary_Clear","Summary_Foggy","Summary_Mostly Cloudy","Summary_Overcast","Summary_Partly Cloudy"]

INCLUDE_TARGET_AS_FEATURE: bool = True

# Ventana y horizonte (en horas)
H: int = 336
L: int = 24

# Hiperparámetros del modelo/entrenamiento
SEED: int = 1337
BATCH_SIZE: int = 128
EPOCHS: int = 120
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-6
PATIENCE: int = 12          # early stopping
HIDDEN_WIDTH: int = 256     # neuronas por capa (reduce si tu PC es modesto)
DEPTH_PER_BLOCK: int = 2    # capas FC por bloque
NUM_BLOCKS: int = 6         # nº de bloques N-BEATS (genéricos)
GRAD_CLIP: float = 1.0
CHECKPOINT_DIR: str = "checkpoints"

# Búsqueda de hiperparámetros (opcional)
HP_SEARCH_ENABLED: bool = False
HP_SEARCH: str = "random"   # "grid" | "random"
SEARCH_PARAM_GRID = {
    "HIDDEN_WIDTH":    [128, 256],
    "DEPTH_PER_BLOCK": [2, 3],
    "NUM_BLOCKS":      [4, 6],
    "LR":              [1e-3, 5e-4],
    "WEIGHT_DECAY":    [1e-6, 1e-5],
}
SEARCH_MAX_ITERS: int = 6
SEARCH_EPOCHS: int    = 8
SEARCH_PATIENCE: int  = 4

# Importancia de variables
COMPUTE_FEATURE_IMPORTANCE: bool = True
TOPK_IMPORTANCE_PLOT: int = 15
PI_N_REPEATS: int = 8   # número de permutaciones por variable para promediar

# ======================
# Utilidades generales
# ======================

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        if "temperature" in cl: score += 1
        if "_z" in cl or "norm" in cl or "normalized" in cl or "normalizado" in cl: score += 1
        if score > 0:
            ranked.append((score, c))
    if ranked:
        ranked.sort(reverse=True)
        return ranked[0][1]
    return None


def looks_like_zscore(name: str, series: pd.Series) -> bool:
    """Heurística simple para decidir si una columna parece z-score."""
    n = series.dropna().shape[0]
    if n < 100:
        return any(t in name.lower() for t in ["_z", "zscore", "normalized", "normalizado"])
    mu = float(series.mean())
    sd = float(series.std(ddof=0))
    name_hint = any(t in name.lower() for t in ["_z", "zscore", "normalized", "normalizado"])
    # criterios suaves: media cerca de 0 y desviación alrededor de 1
    stats_hint = (abs(mu) < 0.5) and (0.5 < sd < 2.0)
    return name_hint or stats_hint


def auto_feature_cols(df: pd.DataFrame, dt_col: str, target_col: str) -> List[str]:
    num_cols = [c for c in df.columns if c != dt_col and np.issubdtype(df[c].dtype, np.number)]
    if INCLUDE_TARGET_AS_FEATURE and target_col not in num_cols and np.issubdtype(df[target_col].dtype, np.number):
        num_cols = [target_col] + num_cols
    # quitar duplicados manteniendo orden
    seen = set(); uniq = []
    for c in num_cols:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

# ======================
# Datos y ventanas
# ======================
class WindowedMultivar(Dataset):
    """Dataset de ventanas deslizantes.
    X: H×D (aplanado a 1D), y: L (objetivo univariante multi‑paso).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, H: int, L: int, start: int, end: int):
        assert X.ndim == 2, "Se espera matriz (N,D) en X."
        assert y.ndim == 1, "Se espera serie 1D para y."
        assert X.shape[0] == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.H = H
        self.L = L
        self.start = start
        self.end = end
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

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = self.idxs[i]
        x_win = self.X[t : t + self.H, :]                 # (H, D)
        y_win = self.y[t + self.H : t + self.H + self.L]  # (L,)
        x_flat = x_win.reshape(-1)                        # (H*D,)
        return x_flat, y_win

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
        # x: (B, H*D)
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
        self.input_size = input_size

    def forward(self, x):
        # x: (B, H*D)
        residual = x
        forecast_agg = torch.zeros(x.size(0), self.blocks[0].forecast_linear.out_features, device=x.device)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_agg = forecast_agg + forecast
        return forecast_agg  # (B, L)

# ======================
# Entrenamiento / evaluación
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


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def denorm_array(arr: np.ndarray, mean: Optional[float], std: Optional[float]) -> np.ndarray:
    if mean is None or std is None:
        return arr
    return arr * std + mean


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        if GRAD_CLIP is not None:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total += float(loss.detach().cpu().item()) * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def evaluate(model, loader, device, criterion):
    model.eval()
    total = 0.0
    n = 0
    y_all = []
    yhat_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            total += float(loss.detach().cpu().item()) * x.size(0)
            n += x.size(0)
            y_all.append(y.cpu().numpy())
            yhat_all.append(y_hat.cpu().numpy())
    if n == 0:
        return math.nan, math.nan, math.nan
    y_all = np.concatenate(y_all, axis=0)
    yhat_all = np.concatenate(yhat_all, axis=0)
    mae, rmse = metrics(y_all, yhat_all)
    return total / n, mae, rmse

# ======================
# Búsqueda de hiperparámetros (opcional)
# ======================

def hyperparam_search(train_loader, val_loader, device, input_size: int):
    if not HP_SEARCH_ENABLED:
        return [], None
    mode = (HP_SEARCH or "").lower().strip()
    if mode not in {"grid", "random"}:
        return [], None
    iterator = (list(ParameterGrid(SEARCH_PARAM_GRID))
                if mode == "grid"
                else list(ParameterSampler(SEARCH_PARAM_GRID, n_iter=SEARCH_MAX_ITERS, random_state=SEED)))
    results = []
    for i, p in enumerate(iterator, 1):
        width  = int(p.get("HIDDEN_WIDTH", HIDDEN_WIDTH))
        depth  = int(p.get("DEPTH_PER_BLOCK", DEPTH_PER_BLOCK))
        blocks = int(p.get("NUM_BLOCKS", NUM_BLOCKS))
        lr     = float(p.get("LR", LR))
        wd     = float(p.get("WEIGHT_DECAY", WEIGHT_DECAY))
        model = NBeats(input_size=input_size, forecast_size=L, width=width, depth_per_block=depth, num_blocks=blocks).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=max(2, SEARCH_PATIENCE//2), verbose=False
            )
        except TypeError:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=max(2, SEARCH_PATIENCE//2)
            )
        best_val = float('inf')
        no_improve = 0
        for _ in range(1, SEARCH_EPOCHS + 1):
            _ = train_one_epoch(model, train_loader, device, optimizer, criterion)
            val_loss, _, _ = evaluate(model, val_loader, device, criterion)
            scheduler.step(val_loss if not math.isnan(val_loss) else 0.0)
            if val_loss < best_val:
                best_val, no_improve = val_loss, 0
            else:
                no_improve += 1
                if no_improve >= SEARCH_PATIENCE:
                    break
        results.append({
            "iter": i, "HIDDEN_WIDTH": width, "DEPTH_PER_BLOCK": depth,
            "NUM_BLOCKS": blocks, "LR": lr, "WEIGHT_DECAY": wd, "best_val": float(best_val),
        })
    best = min(results, key=lambda r: r["best_val"]) if results else None
    return results, best

# ======================
# Carga de datos
# ======================

def load_multivar(csv_path: str,
                  datetime_col: Optional[str],
                  target_col: Optional[str],
                  feature_cols: Optional[List[str]]) -> Tuple[pd.DataFrame, str, str, List[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el CSV en {csv_path}")
    df = pd.read_csv(csv_path)

    # Detecta columnas
    dt_col = datetime_col or detect_datetime_col(df.columns)
    if dt_col is None:
        raise ValueError("No se pudo detectar la columna temporal. Especifica DATETIME_COL en la CONFIG.")
    tgt_col = target_col or detect_target_col(df.columns)
    if tgt_col is None:
        raise ValueError("No se pudo detectar la columna objetivo. Especifica TARGET_COL en la CONFIG.")

    # Parseo y orden
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=[dt_col]).sort_values(dt_col)

    # Selección de features
    feats = feature_cols if (feature_cols is not None and len(feature_cols) > 0) else auto_feature_cols(df, dt_col, tgt_col)
    feats = [c for c in feats if c in df.columns and c != dt_col]
    if INCLUDE_TARGET_AS_FEATURE and tgt_col not in feats:
        feats = [tgt_col] + feats
    if len(feats) < 1:
        raise ValueError("No se encontraron columnas de entrada válidas. Revisa FEATURE_COLS o el CSV.")
    for c in feats + [tgt_col]:
        if not np.issubdtype(df[c].dtype, np.number):
            raise TypeError(f"La columna '{c}' no es numérica. Normaliza/convierte antes de entrenar.")

    # Recorta a columnas necesarias
    use_cols = [dt_col] + list(dict.fromkeys(feats + [tgt_col]))
    df = df[use_cols].dropna()
    
    return df, dt_col, tgt_col, feats

# ======================
# Importancia de variables por permutación
# ======================

def collect_windows(X: np.ndarray, y: np.ndarray, H: int, L: int, start: int, end: int):
    ds = WindowedMultivar(X, y, H, L, start, end)
    # Materializamos todos los pares (esto es sólo para validación, suele ser pequeño)
    Xw = []
    Yw = []
    for t in ds.idxs:
        x_win = X[t:t+H, :]  # (H,D)
        y_win = y[t+H:t+H+L] # (L,)
        Xw.append(x_win)
        Yw.append(y_win)
    if len(Xw) == 0:
        return None, None
    return np.stack(Xw, axis=0), np.stack(Yw, axis=0)  # (W,H,D), (W,L)


def evaluate_windows(model, Xw: np.ndarray, Yw: np.ndarray, device, batch_size: int = 512) -> float:
    criterion = nn.MSELoss(reduction='mean')
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(Xw), batch_size):
            xb = Xw[i:i+batch_size]
            yb = Yw[i:i+batch_size]
            xbf = xb.reshape(xb.shape[0], -1)
            xbf = torch.tensor(xbf, dtype=torch.float32, device=device)
            ybt = torch.tensor(yb, dtype=torch.float32, device=device)
            y_hat = model(xbf)
            loss = criterion(y_hat, ybt)
            total_loss += float(loss.item()) * xbf.size(0)
            n += xbf.size(0)
    return total_loss / max(1, n)


def permutation_importance(model, Xw: np.ndarray, Yw: np.ndarray, feature_names: List[str], H: int, device, rng: np.random.Generator, n_repeats: int = 8) -> pd.DataFrame:
    """Permuta *por variable* (manteniendo sus H lags) a lo largo de las ventanas.
    Devuelve incremento de la pérdida (MSE) promedio y su desviación estándar tras n_repeats permutaciones.
    """
    base_loss = evaluate_windows(model, Xw, Yw, device)
    D = Xw.shape[2]
    rows = []
    for j in range(D):
        incs = []
        for _ in range(n_repeats):
            Xp = Xw.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, :, j] = Xp[perm, :, j]  # permutar sobre eje de ventanas W
            loss_j = evaluate_windows(model, Xp, Yw, device)
            incs.append(loss_j - base_loss)
        incs = np.array(incs, dtype=float)
        rows.append({
            'feature': feature_names[j],
            'delta_mse_mean': float(np.mean(incs)),
            'delta_mse_std': float(np.std(incs, ddof=0)),
            'delta_mse_%_mean': float((np.mean(incs) / base_loss * 100.0) if base_loss > 0 else np.nan),
            'base_mse': float(base_loss),
            'n_repeats': int(n_repeats),
        })
    df_imp = pd.DataFrame(rows).sort_values('delta_mse_mean', ascending=False).reset_index(drop=True)
    return df_imp

# ======================
# Main
# ======================

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Carga
    df, dt_col, tgt_col, feat_cols = load_multivar(CSV_PATH, DATETIME_COL, TARGET_COL, FEATURE_COLS)
    # Matrices numpy
    X_all = df[feat_cols].to_numpy().astype(np.float32)  # (N,D)
    y_all = df[tgt_col].to_numpy().astype(np.float32)    # (N,)

    # ¿El objetivo parece z-score?
    target_is_z = looks_like_zscore(tgt_col, df[tgt_col])
    denorm_mean = denorm_std = None
    units_label = " (z-score)" if target_is_z else ""
    y_axis_label = ORIG_TARGET_COL if (not target_is_z and ORIG_TARGET_COL) else tgt_col

    # Si el objetivo está en z-score e incluimos columna real → usamos su media/std para desnormalizar
    if target_is_z and ORIG_TARGET_COL and (ORIG_TARGET_COL in df.columns):
        denorm_mean = float(df[ORIG_TARGET_COL].mean())
        denorm_std  = float(df[ORIG_TARGET_COL].std(ddof=0))
        units_label = ""  # ahora podremos mostrar en real

    print(f"Dataset: {CSV_PATH}")
    print(f"Tiempo: {dt_col} | Objetivo: {tgt_col} | target_is_z={target_is_z}")
    print(f"Features ({X_all.shape[1]}): {feat_cols}")
    print(f"Denormalización -> mean={denorm_mean} std={denorm_std} (solo aplica si target_is_z)")

    N, D = X_all.shape
    if N < (H + L + 10):
        raise ValueError(f"Serie demasiado corta (N={N}) para H={H} y L={L}.")

    splits = make_splits(N, train_ratio=0.70, val_ratio=0.15)

    train_ds = WindowedMultivar(X_all, y_all, H, L, start=0,                   end=splits.train_end)
    val_ds   = WindowedMultivar(X_all, y_all, H, L, start=splits.train_end,    end=splits.val_end)
    test_ds  = WindowedMultivar(X_all, y_all, H, L, start=splits.val_end,      end=splits.N)

    print(f"Total puntos: {N} — Train: {splits.train_end}, Val: {splits.val_end - splits.train_end}, Test: {splits.N - splits.val_end}")
    print(f"Ventana H={H}, L={L} → input_size={H*D}")
    print(f"Ventanas => Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # --- HP search (opcional)
    used_hidden, used_depth, used_blocks = HIDDEN_WIDTH, DEPTH_PER_BLOCK, NUM_BLOCKS
    used_lr, used_wd = LR, WEIGHT_DECAY
    best_hp = None
    input_size = H * D
    if HP_SEARCH_ENABLED:
        print(f"Iniciando HP search: {HP_SEARCH} ...")
        results, best_hp = hyperparam_search(train_loader, val_loader, device, input_size)
        out_dir = "outputs"; os.makedirs(out_dir, exist_ok=True)
        if results:
            pd.DataFrame(results).to_csv(os.path.join(out_dir, "hpsearch_results.csv"), index=False)
            import json as _json
            with open(os.path.join(out_dir, "best_hparams.json"), "w", encoding="utf-8") as f:
                _json.dump(best_hp, f, ensure_ascii=False, indent=2)
        if best_hp is not None:
            used_hidden = int(best_hp["HIDDEN_WIDTH"]) 
            used_depth  = int(best_hp["DEPTH_PER_BLOCK"]) 
            used_blocks = int(best_hp["NUM_BLOCKS"]) 
            used_lr     = float(best_hp["LR"])        
            used_wd     = float(best_hp["WEIGHT_DECAY"]) 
            print(f"Mejores HP → width={used_hidden}, depth={used_depth}, blocks={used_blocks}, lr={used_lr}, wd={used_wd}, val={best_hp['best_val']:.6f}")
        else:
            print("HP search sin resultados; se usan valores por defecto.")

    # --- Modelo
    model = NBeats(input_size=input_size, forecast_size=L, width=used_hidden, depth_per_block=used_depth, num_blocks=used_blocks).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=used_lr, weight_decay=used_wd)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=max(2, PATIENCE//3), verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=max(2, PATIENCE//3)
        )

    best_val = float('inf')
    no_improve = 0
    ckpt_name = f"nbeats_MULTI_H{H}_L{L}_{tgt_col.replace(' ','_')}_D{D}.pth"
    best_path = os.path.join(CHECKPOINT_DIR, ckpt_name)

    train_hist, val_hist = [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss if not math.isnan(val_loss) else 0.0)

        train_hist.append(train_loss)
        val_hist.append(val_loss)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'H': H, 'L': L, 'width': used_hidden,
                    'depth_per_block': used_depth, 'num_blocks': used_blocks,
                    'target_col': tgt_col, 'feature_cols': feat_cols,
                    'lr': used_lr, 'weight_decay': used_wd,
                },
                'best_val': best_val,
                'best_hp': best_hp,
            }, best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping (patience={PATIENCE}).")
                break

    # Cargar mejor checkpoint
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Cargado mejor checkpoint: {best_path}")

    # Evaluación final en test
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, device, criterion)
    print(f"Test: loss={test_loss:.6f} | MAE={test_mae:.4f} | RMSE={test_rmse:.4f}{units_label}")

    # ======================
    # Predicciones y artefactos (gráficas + PDF + guardados)
    # ======================
    out_dir = "outputs"; os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots"); os.makedirs(plots_dir, exist_ok=True)

    # Forecast operativo del último horizonte
    last_X = torch.tensor(X_all[-(H+L):-L, :].reshape(1, -1), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        last_pred = model(last_X).cpu().numpy().reshape(-1)

    # Predicciones en todo el test
    yhats = []
    ys = []
    with torch.no_grad():
        for x, y in test_loader:
            y_hat = model(x.to(device)).cpu().numpy()
            yhats.append(y_hat)
            ys.append(y.numpy())
    yhats = np.concatenate(yhats, axis=0)
    ys = np.concatenate(ys, axis=0)

    residuals = ys - yhats
    mae_per_h = np.mean(np.abs(residuals), axis=0)
    rmse_per_h = np.sqrt(np.mean(residuals**2, axis=0))

    # — Escala real si sabemos deshacer z-score
    ys_real = yhats_real = None
    if target_is_z and (denorm_mean is not None):
        ys_real = denorm_array(ys, denorm_mean, denorm_std)
        yhats_real = denorm_array(yhats, denorm_mean, denorm_std)
        residuals_real = ys_real - yhats_real
        mae_per_h_real = np.mean(np.abs(residuals_real), axis=0)
        rmse_per_h_real = np.sqrt(np.mean(residuals_real**2, axis=0))

    # ======================
    # Gráficas espaciadas en test (10 ventanas)
    # ======================
    total_test_windows = len(test_ds.idxs)
    take_n = min(10, total_test_windows)
    if take_n > 0:
        sel_pos = np.linspace(0, total_test_windows - 1, take_n)
        sel_pos = np.unique(sel_pos.astype(int))
        while len(sel_pos) < take_n:
            missing = take_n - len(sel_pos)
            extras = np.arange(total_test_windows - missing, total_test_windows, dtype=int)
            sel_pos = np.unique(np.r_[sel_pos, extras])
        spaced_idxs = test_ds.idxs[sel_pos]
    else:
        spaced_idxs = np.array([], dtype=int)
    window_pngs = []

    # Índice del objetivo dentro de los features (si está incluido)
    target_in_feats = feat_cols.index(tgt_col) if tgt_col in feat_cols else None

    for k, t in enumerate(spaced_idxs):
        x_hist = X_all[t : t + H, :]        # (H, D)
        y_true = y_all[t + H : t + H + L]   # (L,)
        with torch.no_grad():
            y_pred = model(torch.tensor(x_hist.reshape(1, -1), dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)

        # Denormalización para plotting (solo si el objetivo es z-score)
        y_true_plot = denorm_array(y_true, denorm_mean, denorm_std) if target_is_z and (denorm_mean is not None) else y_true
        y_pred_plot = denorm_array(y_pred, denorm_mean, denorm_std) if target_is_z and (denorm_mean is not None) else y_pred
        if target_in_feats is not None:
            hist_target = x_hist[:, target_in_feats]
            hist_target_plot = denorm_array(hist_target, denorm_mean, denorm_std) if target_is_z and (denorm_mean is not None) else hist_target
        else:
            hist_target = y_all[t : t + H]
            hist_target_plot = denorm_array(hist_target, denorm_mean, denorm_std) if target_is_z and (denorm_mean is not None) else hist_target

        # Fechas
        hist_times = df[dt_col].iloc[t : t + H].reset_index(drop=True)
        fut_times  = df[dt_col].iloc[t + H : t + H + L].reset_index(drop=True)
        start_time = hist_times.iloc[0]
        end_time   = fut_times.iloc[-1]

        fig = plt.figure(figsize=(10, 4))
        times_concat = pd.concat([hist_times, fut_times])
        series_concat = np.concatenate([hist_target_plot, y_true_plot])
        plt.plot(times_concat, series_concat, label="Real", linewidth=1.6)
        # sombreado para el área de predicción
        last_hist_time = hist_times.iloc[-1]
        plt.axvspan(last_hist_time, fut_times.iloc[-1], alpha=0.10, label="Horizonte")
        plt.plot(fut_times, y_pred_plot, label="Predicción", linewidth=1.6)
        title_txt = f"Ventana {k+1} — {start_time} → {end_time} (H={H} h)"
        plt.title(title_txt)
        y_label = (ORIG_TARGET_COL or tgt_col) if (not target_is_z or (denorm_mean is not None)) else f"{tgt_col} (z)"
        plt.xlabel("Tiempo"); plt.ylabel(y_label)
        plt.legend(); plt.tight_layout()

        fname = f"test_spaced_MULTI_{k+1:02d}_{start_time:%Y%m%d_%H%M}_{end_time:%Y%m%d_%H%M}.png"
        fpath = os.path.join(plots_dir, fname)
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        window_pngs.append(fpath)

    # ======================
    # Importancia de variables (validación)
    # ======================
    imp_df = None
    imp_png = None
    if COMPUTE_FEATURE_IMPORTANCE:
        Xw_val, Yw_val = collect_windows(X_all, y_all, H, L, splits.train_end, splits.val_end)
        if Xw_val is None:
            print("[WARN] No hay ventanas de validación suficientes para calcular importancia de variables.")
        else:
            rng = np.random.default_rng(SEED)
            imp_df = permutation_importance(model, Xw_val, Yw_val, feat_cols, H, device, rng, n_repeats=PI_N_REPEATS)
            # Guardar CSV
            imp_csv = os.path.join(out_dir, f"feature_importance_MULTI_H{H}_L{L}.csv")
            imp_df.to_csv(imp_csv, index=False)
            # Gráfico TOP-K con barras de error
            topk = imp_df.head(TOPK_IMPORTANCE_PLOT)[::-1]  # invertimos para barh ascendente
            fig = plt.figure(figsize=(10, 5))
            plt.barh(topk['feature'], topk['delta_mse_%_mean'])
            # Añadimos barras de error en porcentaje
            if (imp_df['base_mse'] > 0).all():
                err = (topk['delta_mse_std'] / topk['base_mse']) * 100.0
                # dibujamos manualmente las "caps"
                for i, (ypos, x, e) in enumerate(zip(range(len(topk)), topk['delta_mse_%_mean'], err)):
                    plt.errorbar(x=x, y=ypos, xerr=e, fmt='none', capsize=3)
            plt.xlabel("Incremento medio de MSE al permutar (%)")
            plt.title(f"Importancia de variables — permutación (val, n_repeats={PI_N_REPEATS})")
            plt.tight_layout()
            imp_png = os.path.join(plots_dir, f"feature_importance_TOP{TOPK_IMPORTANCE_PLOT}.png")
            fig.savefig(imp_png, dpi=150)
            plt.close(fig)

    # ======================
    # PDF de análisis general
    # ======================
    pdf_path = os.path.join(out_dir, f"analysis_MULTI_H{H}_L{L}_{tgt_col.replace(' ','_')}_D{D}.pdf")
    with PdfPages(pdf_path) as pdf:
        # Página 1: resumen
        fig = plt.figure(figsize=(11, 8.5)); plt.axis('off')
        units_txt = "(real)" if (not target_is_z or (denorm_mean is not None)) else "(z-score)"
        text = (
            f"N-BEATS Multivariable — Resumen\n"
            f"Archivo: {CSV_PATH}\n"
            f"Tiempo: {dt_col} | Objetivo: {tgt_col} {units_txt}\n"
            f"Features ({D}): {feat_cols}\n"
            f"Puntos totales: {N}\n"
            f"Split → Train: {splits.train_end}  | Val: {splits.val_end - splits.train_end}  | Test: {splits.N - splits.val_end}\n"
            f"Ventana: H={H}  L={L}  input_size={H*D}\n"
            f"Métricas test → MSE: {test_loss:.6f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} {units_txt}"
        )
        plt.text(0.05, 0.95, text, va='top', fontsize=12)
        pdf.savefig(fig); plt.close(fig)

        # Página 2: curvas de aprendizaje
        fig = plt.figure(figsize=(11, 4.5))
        plt.plot(range(1, len(train_hist)+1), train_hist, label='Train Loss')
        plt.plot(range(1, len(val_hist)+1), val_hist, label='Val Loss')
        plt.xlabel('Época'); plt.ylabel('MSE'); plt.title('Curvas de aprendizaje'); plt.legend(); plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Página 3: error por paso de horizonte (escala actual)
        fig = plt.figure(figsize=(11, 4.5))
        plt.plot(mae_per_h, label='MAE por horizonte')
        plt.plot(rmse_per_h, label='RMSE por horizonte')
        plt.xlabel('Paso (horas en el futuro)'); plt.ylabel(f'Error {units_label}')
        plt.title('Error por paso del horizonte (test)'); plt.legend(); plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Página 3b: error por paso del horizonte en escala real (si procede)
        if ys_real is not None:
            fig = plt.figure(figsize=(11, 4.5))
            plt.plot(mae_per_h_real, label='MAE por horizonte (real)')
            plt.plot(rmse_per_h_real, label='RMSE por horizonte (real)')
            plt.xlabel('Paso (horas en el futuro)'); plt.ylabel('Error (real)')
            plt.title('Error por paso del horizonte (test, escala real)'); plt.legend(); plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Página 4: distribución de residuales (escala actual)
        fig = plt.figure(figsize=(11, 4.5))
        plt.hist(residuals.flatten(), bins=50)
        plt.title('Distribución de residuales (test)'); plt.xlabel(f'Residual{units_label}'); plt.ylabel('Frecuencia'); plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Página 4b: distribución de residuales en escala real
        if ys_real is not None:
            fig = plt.figure(figsize=(11, 4.5))
            plt.hist((ys_real - yhats_real).flatten(), bins=50)
            plt.title('Distribución de residuales (test, real)'); plt.xlabel('Residual (real)'); plt.ylabel('Frecuencia'); plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Página 5: promedio verdad vs. predicción por paso (escala actual)
        fig = plt.figure(figsize=(11, 4.5))
        plt.plot(np.mean(ys, axis=0), label='Real promedio')
        plt.plot(np.mean(yhats, axis=0), label='Predicción promedio')
        plt.xlabel('Paso (horas en el futuro)'); plt.ylabel(f'Valor{units_label}')
        plt.title('Promedio por paso del horizonte (test)'); plt.legend(); plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Página 5b: promedio en escala real
        if ys_real is not None:
            fig = plt.figure(figsize=(11, 4.5))
            plt.plot(np.mean(ys_real, axis=0), label='Real promedio (real)')
            plt.plot(np.mean(yhats_real, axis=0), label='Predicción promedio (real)')
            plt.xlabel('Paso (horas en el futuro)'); plt.ylabel('Valor (real)')
            plt.title('Promedio por paso del horizonte (test, real)'); plt.legend(); plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Página 6: Importancia de variables (si hay)
        if imp_df is not None:
            topk = imp_df.head(TOPK_IMPORTANCE_PLOT)
            fig = plt.figure(figsize=(11, 4.5)); plt.axis('off')
            tbl = plt.table(cellText=np.round(topk[['delta_mse_mean','delta_mse_%_mean','delta_mse_std']].values, 6),
                            colLabels=['ΔMSE (mean)','ΔMSE % (mean)','ΔMSE (std)'],
                            rowLabels=topk['feature'], loc='center')
            tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.2)
            plt.title(f'Importancia de variables (TOP-{TOPK_IMPORTANCE_PLOT}, n_repeats={PI_N_REPEATS})')
            pdf.savefig(fig); plt.close(fig)

        # Páginas siguientes: las 10 ventanas espaciadas
        for fpath in window_pngs:
            img = plt.imread(fpath)
            fig = plt.figure(figsize=(11, 4.5)); plt.imshow(img); plt.axis('off'); pdf.savefig(fig); plt.close(fig)

        # Adjuntar la barra horizontal si existe
        if imp_png is not None:
            img = plt.imread(imp_png)
            fig = plt.figure(figsize=(11, 5)); plt.imshow(img); plt.axis('off'); pdf.savefig(fig); plt.close(fig)

    # ======================
    # Guardados: métricas, forecast final, TorchScript y config
    # ======================
    # Métricas
    with open(os.path.join(out_dir, f"metrics_MULTI_H{H}_L{L}_{tgt_col.replace(' ','_')}.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Val best MSE: {best_val:.6f}\n")
        f.write(f"Test MSE: {test_loss:.6f}\n")
        f.write(f"Test MAE: {test_mae:.6f}\n")
        f.write(f"Test RMSE: {test_rmse:.6f}\n")

    # Forecast del último horizonte con fechas
    last_hist_times = df[dt_col].iloc[-(H+L):-L].reset_index(drop=True)
    last_fut_times  = pd.date_range(df[dt_col].iloc[-1] + pd.tseries.frequencies.to_offset('H'), periods=L, freq='H')
    forecast_cols = {dt_col: last_fut_times, f"pred_{tgt_col}": last_pred}
    if target_is_z and (denorm_mean is not None):
        real_name = ORIG_TARGET_COL or (tgt_col + "_real")
        forecast_cols[f"pred_{real_name}"] = denorm_array(last_pred, denorm_mean, denorm_std)
    pd.DataFrame(forecast_cols).to_csv(os.path.join(out_dir, f"forecast_last_MULTI_H{H}_L{L}_{tgt_col.replace(' ','_')}.csv"), index=False)

    # TorchScript para inferencia independiente
    try:
        example = torch.randn(1, H*D, dtype=torch.float32, device=device)
        traced = torch.jit.trace(model, example)
        torchscript_path = os.path.join(CHECKPOINT_DIR, f"nbeats_MULTI_H{H}_L{L}_{tgt_col.replace(' ','_')}_D{D}_torchscript.pt")
        traced.save(torchscript_path)
        print(f"Guardado TorchScript en: {torchscript_path}")
    except Exception as e:
        print(f"No se pudo exportar a TorchScript: {e}")

    # Config para reproducibilidad
    import json
    config = {
        'CSV_PATH': CSV_PATH,
        'DATETIME_COL': dt_col,
        'TARGET_COL': tgt_col,
        'ORIG_TARGET_COL': ORIG_TARGET_COL,
        'FEATURE_COLS': feat_cols,
        'H': H, 'L': L,
        'SEED': SEED,
        'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS,
        'LR': LR,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'PATIENCE': PATIENCE,
        'HIDDEN_WIDTH': HIDDEN_WIDTH,
        'DEPTH_PER_BLOCK': DEPTH_PER_BLOCK,
        'NUM_BLOCKS': NUM_BLOCKS,
        'HP_SEARCH_ENABLED': HP_SEARCH_ENABLED,
        'HP_SEARCH': HP_SEARCH,
        'SEARCH_PARAM_GRID': SEARCH_PARAM_GRID,
        'SEARCH_MAX_ITERS': SEARCH_MAX_ITERS,
        'SEARCH_EPOCHS': SEARCH_EPOCHS,
        'SEARCH_PATIENCE': SEARCH_PATIENCE,
        'COMPUTE_FEATURE_IMPORTANCE': COMPUTE_FEATURE_IMPORTANCE,
        'TOPK_IMPORTANCE_PLOT': TOPK_IMPORTANCE_PLOT,
        'PI_N_REPEATS': PI_N_REPEATS,
    }
    with open(os.path.join(out_dir, f"run_config_MULTI_H{H}_L{L}_{tgt_col.replace(' ','_')}.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Listo. Model checkpoint:", best_path)
    print("Resultados en: outputs/ (gráficas individuales, PDF, métricas, importancia de variables)")
    print("Consejo: si tu objetivo ya está en °C, las gráficas saldrán en °C; si es z-score y existe la columna real, se desnormaliza automáticamente.")

if __name__ == "__main__":
    main()
