"""
N-BEATS univariante (PyTorch) para series horarias
--------------------------------------------------
- Ventana de *historia* H y *horizonte* L configurables.
- Variable objetivo configurable (por defecto, temperatura ya normalizada en z‑score).
- Split temporal: 70% train, 15% valid, 15% test (sin fuga de futuro).
- Salida multi‑paso directa: el modelo predice las L horas de una vez.

Requisitos (instala una vez):
    pip install torch pandas numpy scikit-learn

Ejecutar:
    python NBEATS_univar_H24_L336.py

Ajustes rápidos: cambia las constantes en la sección CONFIG.
"""
from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import ParameterGrid, ParameterSampler

# ==================================================
# CONFIG ——— cambia aquí H, L y la variable objetivo
# ==================================================
CSV_PATH: str = "data/weatherHistory_temp_normalizacion.csv"  # Ruta al CSV
DATETIME_COL: Optional[str] = "Formatted Date"  # Si None, se intenta detectar ['Formatted Date','Date','time','datetime','timestamp']
TARGET_COL: Optional[str] = "Temperature (C)_normalized"    # Si None, se intenta detectar algo que contenga 'temp' (ej. 'temp_z', 'Temperature (C)_z')
H: int = 336   # horas de entrada (historia)
L: int = 24  # horas de salida (horizonte)

# Hiperparámetros del modelo/entrenamiento
SEED: int = 1337
BATCH_SIZE: int = 512
EPOCHS: int = 200
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-6
PATIENCE: int = 12          # early stopping
HIDDEN_WIDTH: int = 512     # neuronas por capa
DEPTH_PER_BLOCK: int = 4    # capas FC por bloque
NUM_BLOCKS: int = 8         # nº de bloques N-BEATS (genéricos)
GRAD_CLIP: float = 1.0
CHECKPOINT_DIR: str = "checkpoints"

# ======================
# Denormalización y búsqueda de hiperparámetros
# ======================
DENORMALIZE_OUTPUTS: bool = True
ORIG_TARGET_COL: Optional[str] = "Temperature (C)"
Z_MEAN: Optional[float] = None
Z_STD: Optional[float]  = None

HP_SEARCH_ENABLED: bool = True   # Activa/desactiva búsqueda
HP_SEARCH: str = "grid"        # "grid" | "random" si HP_SEARCH_ENABLED=True
SEARCH_PARAM_GRID = {
    "HIDDEN_WIDTH":    [256, 512],
    "DEPTH_PER_BLOCK": [2, 4],
    "NUM_BLOCKS":      [4, 8],
    "LR":              [1e-3, 5e-4],
    "WEIGHT_DECAY":    [1e-6, 1e-5],
}
SEARCH_MAX_ITERS: int = 6
SEARCH_EPOCHS: int    = 8
SEARCH_PATIENCE: int  = 5

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
    """Intenta adivinar la columna temporal."""
    candidates = [c for c in cols if str(c).lower() in {"formatted date", "date", "time", "datetime", "timestamp", "time_local"}]
    if candidates:
        return candidates[0]
    # fallback: el primer nombre que contenga 'date' o 'time'
    for c in cols:
        cl = str(c).lower()
        if "date" in cl or "time" in cl:
            return c
    return None


def detect_target_col(cols):
    """Intenta adivinar la columna de temperatura normalizada."""
    # Preferimos columnas que contengan 'temp' y 'z' / 'norm'
    ranked = []
    for c in cols:
        cl = str(c).lower()
        score = 0
        if "temp" in cl: score += 2
        if "z" in cl or "norm" in cl: score += 1
        if "temperature" in cl: score += 1
        if score > 0:
            ranked.append((score, c))
    if ranked:
        ranked.sort(reverse=True)
        return ranked[0][1]
    # Si no encontramos, devolvemos una columna numérica candidata
    return None


# ======================
# Datos y ventanas
# ======================
class WindowedSeries(Dataset):
    """Dataset de ventanas deslizantes X: H, y: L (multi‑step direct)."""
    def __init__(self, series: np.ndarray, H: int, L: int, start: int, end: int):
        assert series.ndim == 1, "Se espera serie univariante (1D)."
        self.series = series.astype(np.float32)
        self.H = H
        self.L = L
        self.start = start
        self.end = end
        # Generamos índices de ventana donde el objetivo y está totalmente dentro [start, end)
        # Cada ventana empieza en t, usa x = series[t:t+H] y y = series[t+H:t+H+L]
        starts = []
        N = len(series)
        max_t = min(end, N) - (H + L)
        t0 = max(0, start - H)  # permitimos que X use historia previa
        for t in range(t0, max_t + 1):
            y_end = t + H + L
            if (t + H >= start) and (y_end <= end):
                starts.append(t)
        self.idxs = np.array(starts, dtype=np.int64)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = self.idxs[i]
        x = self.series[t : t + self.H]
        y = self.series[t + self.H : t + self.H + self.L]
        return x, y


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
        # x: (B, H)
        h = self.mlp(x)
        backcast = self.backcast_linear(h)
        forecast = self.forecast_linear(h)
        return backcast, forecast


class NBeats(nn.Module):
    def __init__(self, input_size: int, forecast_size: int, width: int = 512, depth_per_block: int = 4, num_blocks: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, forecast_size, width, depth_per_block)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x: (B, H)
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
    """Deshace z-score: x = arr*std + mean. Si falta mean/std, devuelve arr sin cambios."""
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

def hyperparam_search(train_loader, val_loader, device):
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
        model = NBeats(input_size=H, forecast_size=L, width=width, depth_per_block=depth, num_blocks=blocks).to(device)
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

def load_series(csv_path: str, datetime_col: Optional[str], target_col: Optional[str]) -> Tuple[pd.DataFrame, str, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el CSV en {csv_path}")
    df = pd.read_csv(csv_path)
    # Detecta columnas si no se especifican
    dt_col = datetime_col or detect_datetime_col(df.columns)
    if dt_col is None:
        raise ValueError("No se pudo detectar la columna temporal. Especifica DATETIME_COL en la CONFIG.")
    tgt_col = target_col or detect_target_col(df.columns)
    if tgt_col is None:
        raise ValueError("No se pudo detectar la columna objetivo. Especifica TARGET_COL en la CONFIG.")

    # Parseo y orden
    # Parseo robusto: mezclas de husos -> UTC y quitamos tz para trabajar en naive
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True).dt.tz_localize(None)
    df = df.sort_values(dt_col)
    cols = [dt_col, tgt_col]
    if DENORMALIZE_OUTPUTS and ORIG_TARGET_COL and ORIG_TARGET_COL in df.columns:
        cols.append(ORIG_TARGET_COL)
    df = df[cols].dropna(subset=[tgt_col])

    # Comprobaciones básicas
    # (No normalizamos: se asume ya normalizado si corresponde)
    if not np.issubdtype(df[tgt_col].dtype, np.number):
        raise TypeError(f"La columna objetivo '{tgt_col}' no es numérica.")

    return df, dt_col, tgt_col


# ======================
# Main
# ======================

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df, dt_col, tgt_col = load_series(CSV_PATH, DATETIME_COL, TARGET_COL)
    series = df[tgt_col].to_numpy().astype(np.float32)
    # Parámetros para desnormalizar
    denorm_mean = denorm_std = None
    if DENORMALIZE_OUTPUTS:
        if (Z_MEAN is not None) and (Z_STD is not None):
            denorm_mean, denorm_std = float(Z_MEAN), float(Z_STD)
        elif ORIG_TARGET_COL and ORIG_TARGET_COL in df.columns:
            denorm_mean = float(df[ORIG_TARGET_COL].mean())
            denorm_std  = float(df[ORIG_TARGET_COL].std(ddof=0))
        if denorm_mean is not None:
            print(f"Denormalización activa — mean={denorm_mean:.4f}, std={denorm_std:.4f}")
        else:
            print("Denormalización: sin parámetros (se mostrará en z-score)")
    N = len(series)
    if N < (H + L + 10):
        raise ValueError(f"Serie demasiado corta (N={N}) para H={H} y L={L}.")

    splits = make_splits(N, train_ratio=0.70, val_ratio=0.15)
    train_ds = WindowedSeries(series, H, L, start=0, end=splits.train_end)
    val_ds   = WindowedSeries(series, H, L, start=splits.train_end, end=splits.val_end)
    test_ds  = WindowedSeries(series, H, L, start=splits.val_end, end=splits.N)

    print(f"Dataset: {CSV_PATH}")
    print(f"Tiempo: {dt_col} | Objetivo: {tgt_col}")
    print(f"Total puntos: {N} — Train: {splits.train_end}, Val: {splits.val_end - splits.train_end}, Test: {splits.N - splits.val_end}")
    print(f"Ventana H={H}, L={L}")
    print(f"Ventanas => Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Búsqueda de hiperparámetros (opcional)
    used_hidden, used_depth, used_blocks = HIDDEN_WIDTH, DEPTH_PER_BLOCK, NUM_BLOCKS
    used_lr, used_wd = LR, WEIGHT_DECAY
    best_hp = None
    if HP_SEARCH_ENABLED:
        print(f"Iniciando HP search: {HP_SEARCH} ...")
        results, best_hp = hyperparam_search(train_loader, val_loader, device)
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

    model = NBeats(input_size=H, forecast_size=L, width=used_hidden, depth_per_block=used_depth, num_blocks=used_blocks).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=used_lr, weight_decay=used_wd)
    # Compatibilidad con versiones antiguas de PyTorch que no aceptan 'verbose'
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
    best_path = os.path.join(CHECKPOINT_DIR, f"nbeats_H{H}_L{L}_{tgt_col.replace(' ','_')}.pth")

    # Historial de pérdidas para gráfico de aprendizaje
    train_hist, val_hist = [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss if not math.isnan(val_loss) else 0.0)

        # Guardamos historia
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
                    'target_col': tgt_col,
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

    # Cargamos el mejor y evaluamos en test
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Cargado mejor checkpoint: {best_path}")

    test_loss, test_mae, test_rmse = evaluate(model, test_loader, device, criterion)
    print(f"Test: loss={test_loss:.6f} | MAE={test_mae:.4f} | RMSE={test_rmse:.4f}")

    # ======================
    # Predicciones y artefactos (gráficas + PDF + guardados)
    # ======================

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # --- Predicción del último horizonte (forecast operativo)
    last_x = torch.tensor(series[-(H+L):-L], dtype=torch.float32, device=device).unsqueeze(0)  # (1, H)
    model.eval()
    with torch.no_grad():
        last_pred = model(last_x).cpu().numpy().reshape(-1)

    # --- Predicciones en todo el test para métricas agregadas
    yhats = []
    ys = []
    with torch.no_grad():
        for x, y in test_loader:
            y_hat = model(x.to(device)).cpu().numpy()
            yhats.append(y_hat)
            ys.append(y.numpy())
    yhats = np.concatenate(yhats, axis=0)
    ys = np.concatenate(ys, axis=0)

    # Errores por horizonte
    residuals = ys - yhats  # (num_windows, L)
    mae_per_h = np.mean(np.abs(residuals), axis=0)
    rmse_per_h = np.sqrt(np.mean(residuals**2, axis=0))

    # Versiones en escala real (si procede)
    ys_real = yhats_real = None
    if DENORMALIZE_OUTPUTS and (denorm_mean is not None) and (denorm_std is not None):
        ys_real = denorm_array(ys, denorm_mean, denorm_std)
        yhats_real = denorm_array(yhats, denorm_mean, denorm_std)
        residuals_real = ys_real - yhats_real
        mae_per_h_real = np.mean(np.abs(residuals_real), axis=0)
        rmse_per_h_real = np.sqrt(np.mean(residuals_real**2, axis=0))

    # ======================
    # Gráficas de las 10 últimas ventanas del test
    # ======================
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Si hay menos de 10 ventanas, usamos las que haya
    # Selección espaciada de ventanas sobre todo el test
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

    for k, t in enumerate(spaced_idxs):
        # Historia y futuro verdaderos
        x_hist = series[t : t + H]
        y_true = series[t + H : t + H + L]
        # Predicción del modelo para esta ventana
        with torch.no_grad():
            y_pred = model(torch.tensor(x_hist, dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy().reshape(-1)

        # Denormalización para plotting
        x_hist_plot = denorm_array(x_hist, denorm_mean, denorm_std) if (DENORMALIZE_OUTPUTS and denorm_mean is not None) else x_hist
        y_true_plot = denorm_array(y_true, denorm_mean, denorm_std) if (DENORMALIZE_OUTPUTS and denorm_mean is not None) else y_true
        y_pred_plot = denorm_array(y_pred, denorm_mean, denorm_std) if (DENORMALIZE_OUTPUTS and denorm_mean is not None) else y_pred

        # Fechas
        hist_times = df[dt_col].iloc[t : t + H].reset_index(drop=True)
        fut_times = df[dt_col].iloc[t + H : t + H + L].reset_index(drop=True)
        start_time = hist_times.iloc[0]
        end_time = fut_times.iloc[-1]

        # Figura
        fig = plt.figure(figsize=(10, 4))
        # Concatenamos para eje x continuo
        times_concat = pd.concat([hist_times, fut_times])
        series_concat = np.concatenate([x_hist_plot, y_true_plot])
        plt.plot(times_concat, series_concat, label="Real")
        # Línea vertical separando pasado/futuro
        plt.axvline(hist_times.iloc[-1], linestyle='--', linewidth=1)
        # Predicción solo en el futuro
        plt.plot(fut_times, y_pred_plot, label="Predicción")
        title_txt = f"Ventana {k+1} — {start_time} → {end_time}\n(Historia H={H} h mostrada)"
        plt.title(title_txt)
        plt.xlabel("Tiempo")
        y_label = ORIG_TARGET_COL if (DENORMALIZE_OUTPUTS and denorm_mean is not None and ORIG_TARGET_COL) else tgt_col
        plt.ylabel(y_label)
        plt.legend()
        plt.tight_layout()

        fname = f"test_spaced_{k+1:02d}_{start_time:%Y%m%d_%H%M}_{end_time:%Y%m%d_%H%M}.png"
        fpath = os.path.join(plots_dir, fname)
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        window_pngs.append(fpath)

    # ======================
    # PDF de análisis general
    # ======================
    pdf_path = os.path.join(out_dir, f"analysis_H{H}_L{L}_{tgt_col.replace(' ','_')}.pdf")
    with PdfPages(pdf_path) as pdf:
        # Página 1: resumen
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        text = (
            f"N-BEATS Univar — Resumen"
            f"Archivo: {CSV_PATH}"
            f"Tiempo: {dt_col} | Objetivo: {tgt_col}"
            f"Puntos totales: {N}"
            f"Split → Train: {splits.train_end}  | Val: {splits.val_end - splits.train_end}  | Test: {splits.N - splits.val_end}"
            f"Ventana: H={H}  L={L}"
            f"Métricas test → MSE: {test_loss:.6f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}"
        )
        plt.text(0.05, 0.95, text, va='top', fontsize=12)
        pdf.savefig(fig); plt.close(fig)

        # Página 2: curvas de aprendizaje
        fig = plt.figure(figsize=(11, 4.5))
        plt.plot(range(1, len(train_hist)+1), train_hist, label='Train Loss')
        plt.plot(range(1, len(val_hist)+1), val_hist, label='Val Loss')
        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.title('Curvas de aprendizaje')
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Página 3: error por paso de horizonte (z-score)
        fig = plt.figure(figsize=(11, 4.5))
        plt.plot(mae_per_h, label='MAE por horizonte (z)')
        plt.plot(rmse_per_h, label='RMSE por horizonte (z)')
        plt.xlabel('Paso (horas en el futuro)')
        plt.ylabel('Error (z)')
        plt.title('Error por paso del horizonte (test, z-score)')
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Página 3b: error por paso del horizonte en escala real
        if ys_real is not None:
            fig = plt.figure(figsize=(11, 4.5))
            plt.plot(mae_per_h_real, label='MAE por horizonte (real)')
            plt.plot(rmse_per_h_real, label='RMSE por horizonte (real)')
            plt.xlabel('Paso (horas en el futuro)')
            plt.ylabel('Error (real)')
            plt.title('Error por paso del horizonte (test, escala real)')
            plt.legend()
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Página 4: distribución de residuales (z-score, todas las horas)
        fig = plt.figure(figsize=(11, 4.5))
        plt.hist(residuals.flatten(), bins=50)
        plt.title('Distribución de residuales (test, z-score)')
        plt.xlabel('Residual (z)')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Página 4b: distribución de residuales en escala real
        if ys_real is not None:
            fig = plt.figure(figsize=(11, 4.5))
            plt.hist((ys_real - yhats_real).flatten(), bins=50)
            plt.title('Distribución de residuales (test, real)')
            plt.xlabel('Residual (real)')
            plt.ylabel('Frecuencia')
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Página 5: promedio verdad vs. predicción por paso del horizonte (z-score)
        fig = plt.figure(figsize=(11, 4.5))
        plt.plot(np.mean(ys, axis=0), label='Real promedio (z)')
        plt.plot(np.mean(yhats, axis=0), label='Predicción promedio (z)')
        plt.xlabel('Paso (horas en el futuro)')
        plt.ylabel('Valor (z)')
        plt.title('Promedio por paso del horizonte (test, z-score)')
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Página 5b: promedio en escala real
        if ys_real is not None:
            fig = plt.figure(figsize=(11, 4.5))
            plt.plot(np.mean(ys_real, axis=0), label='Real promedio (real)')
            plt.plot(np.mean(yhats_real, axis=0), label='Predicción promedio (real)')
            plt.xlabel('Paso (horas en el futuro)')
            plt.ylabel('Valor (real)')
            plt.title('Promedio por paso del horizonte (test, real)')
            plt.legend()
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Páginas siguientes: las 10 últimas ventanas
        for fpath in window_pngs:
            img = plt.imread(fpath)
            fig = plt.figure(figsize=(11, 4.5))
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig(fig); plt.close(fig)

    # ======================
    # Guardados: métricas, forecast final, checkpoint, TorchScript y config
    # ======================

    # Métricas
    with open(os.path.join(out_dir, f"metrics_H{H}_L{L}_{tgt_col.replace(' ','_')}.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Val best MSE: {best_val:.6f}")
        f.write(f"Test MSE: {test_loss:.6f}")
        f.write(f"Test MAE: {test_mae:.6f}")
        f.write(f"Test RMSE: {test_rmse:.6f}")

    # Forecast del último horizonte con fechas
    last_hist_times = df[dt_col].iloc[-(H+L):-L].reset_index(drop=True)
    last_fut_times = pd.date_range(df[dt_col].iloc[-1] + pd.tseries.frequencies.to_offset('H'), periods=L, freq='H')
    forecast_cols = {dt_col: last_fut_times, f"pred_{tgt_col}": last_pred}
    if DENORMALIZE_OUTPUTS and (denorm_mean is not None) and (denorm_std is not None):
        real_name = ORIG_TARGET_COL or (tgt_col + "_real")
        forecast_cols[f"pred_{real_name}"] = denorm_array(last_pred, denorm_mean, denorm_std)
    pd.DataFrame(forecast_cols).to_csv(os.path.join(out_dir, f"forecast_last_H{H}_L{L}_{tgt_col.replace(' ','_')}.csv"), index=False)

    # TorchScript para inferencia independiente
    try:
        example = torch.randn(1, H, dtype=torch.float32, device=device)
        traced = torch.jit.trace(model, example)
        torchscript_path = os.path.join(CHECKPOINT_DIR, f"nbeats_H{H}_L{L}_{tgt_col.replace(' ','_')}_torchscript.pt")
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
        'DENORMALIZE_OUTPUTS': DENORMALIZE_OUTPUTS,
        'Z_MEAN': Z_MEAN, 'Z_STD': Z_STD,
        'H': H,
        'L': L,
        'SEED': SEED,
        'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS,
        'LR': used_lr,
        'WEIGHT_DECAY': used_wd,
        'PATIENCE': PATIENCE,
        'HIDDEN_WIDTH': used_hidden,
        'DEPTH_PER_BLOCK': used_depth,
        'NUM_BLOCKS': used_blocks,
        'HP_SEARCH_ENABLED': HP_SEARCH_ENABLED,
        'HP_SEARCH': HP_SEARCH,
        'SEARCH_PARAM_GRID': SEARCH_PARAM_GRID,
        'SEARCH_MAX_ITERS': SEARCH_MAX_ITERS,
        'SEARCH_EPOCHS': SEARCH_EPOCHS,
        'SEARCH_PATIENCE': SEARCH_PATIENCE,
    }
    with open(os.path.join(out_dir, f"run_config_H{H}_L{L}_{tgt_col.replace(' ','_')}.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Listo. Model checkpoint: {best_path}")
    print(f"Resultados en: {out_dir}/ (gráficas individuales, PDF y métricas)")
    print("Para usar otra variable del dataset o cambiar H y L, modifica TARGET_COL / H / L en la CONFIG y vuelve a ejecutar.")


if __name__ == "__main__":
    main()
