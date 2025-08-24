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

# ==================================================
# CONFIG ——— cambia aquí H, L y la variable objetivo
# ==================================================
CSV_PATH: str = "data/weatherHistory_temp_normalizacion.csv"  # Ruta al CSV
DATETIME_COL: Optional[str] = "Formatted Date"  # Si None, se intenta detectar ['Formatted Date','Date','time','datetime','timestamp']
TARGET_COL: Optional[str] = "Temperature (C)_normalized"    # Si None, se intenta detectar algo que contenga 'temp' (ej. 'temp_z', 'Temperature (C)_z')

H: int = 24    # horas de entrada (historia)
L: int = 336   # horas de salida (horizonte)

# Hiperparámetros del modelo/entrenamiento
SEED: int = 42
BATCH_SIZE: int = 512
EPOCHS: int = 30
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-6
PATIENCE: int = 12          # early stopping
HIDDEN_WIDTH: int = 512     # neuronas por capa
DEPTH_PER_BLOCK: int = 4    # capas FC por bloque
NUM_BLOCKS: int = 8         # nº de bloques N-BEATS (genéricos)
GRAD_CLIP: float = 1.0
CHECKPOINT_DIR: str = "checkpoints"

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
    df = df[[dt_col, tgt_col]].dropna()

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

    model = NBeats(input_size=H, forecast_size=L, width=HIDDEN_WIDTH, depth_per_block=DEPTH_PER_BLOCK, num_blocks=NUM_BLOCKS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
                    'H': H, 'L': L, 'width': HIDDEN_WIDTH,
                    'depth_per_block': DEPTH_PER_BLOCK, 'num_blocks': NUM_BLOCKS,
                    'target_col': tgt_col,
                }
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

    # ======================
    # Gráficas de las 10 últimas ventanas del test
    # ======================
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Si hay menos de 10 ventanas, usamos las que haya
    take_n = min(10, len(test_ds))
    last_idxs = test_ds.idxs[-take_n:]  # índices t absolutos en la serie original
    window_pngs = []

    for k, t in enumerate(last_idxs):
        # Historia y futuro verdaderos
        x_hist = series[t : t + H]
        y_true = series[t + H : t + H + L]
        # Predicción del modelo para esta ventana
        with torch.no_grad():
            y_pred = model(torch.tensor(x_hist, dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy().reshape(-1)

        # Fechas
        hist_times = df[dt_col].iloc[t : t + H].reset_index(drop=True)
        fut_times = df[dt_col].iloc[t + H : t + H + L].reset_index(drop=True)
        start_time = hist_times.iloc[0]
        end_time = fut_times.iloc[-1]

        # Figura
        fig = plt.figure(figsize=(10, 4))
        # Concatenamos para eje x continuo
        times_concat = pd.concat([hist_times, fut_times])
        series_concat = np.concatenate([x_hist, y_true])
        plt.plot(times_concat, series_concat, label="Real")
        # Línea vertical separando pasado/futuro
        plt.axvline(hist_times.iloc[-1], linestyle='--', linewidth=1)
        # Predicción solo en el futuro
        plt.plot(fut_times, y_pred, label="Predicción")
        title_txt = f"Ventana {k+1} — {start_time} → {end_time}\n(Historia H={H} h mostrada)"
        plt.title(title_txt)
        plt.xlabel("Tiempo")
        plt.ylabel(tgt_col)
        plt.legend()
        plt.tight_layout()

        fname = f"test_last_{k+1:02d}_{start_time:%Y%m%d_%H%M}_{end_time:%Y%m%d_%H%M}.png"
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

        # Página 3: error por paso de horizonte
        fig = plt.figure(figsize=(11, 4.5))
        plt.plot(mae_per_h, label='MAE por horizonte')
        plt.plot(rmse_per_h, label='RMSE por horizonte')
        plt.xlabel('Paso (horas en el futuro)')
        plt.ylabel('Error')
        plt.title('Error por paso del horizonte (test)')
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Página 4: distribución de residuales (todas las horas del horizonte)
        fig = plt.figure(figsize=(11, 4.5))
        plt.hist(residuals.flatten(), bins=50)
        plt.title('Distribución de residuales (test)')
        plt.xlabel('Residual')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Página 5: promedio de verdad vs. predicción por paso del horizonte
        fig = plt.figure(figsize=(11, 4.5))
        plt.plot(np.mean(ys, axis=0), label='Real promedio')
        plt.plot(np.mean(yhats, axis=0), label='Predicción promedio')
        plt.xlabel('Paso (horas en el futuro)')
        plt.ylabel(tgt_col)
        plt.title('Promedio por paso del horizonte (test)')
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
    pd.DataFrame({
        dt_col: last_fut_times,
        f"pred_{tgt_col}": last_pred,
    }).to_csv(os.path.join(out_dir, f"forecast_last_H{H}_L{L}_{tgt_col.replace(' ','_')}.csv"), index=False)

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
        'H': H,
        'L': L,
        'SEED': SEED,
        'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS,
        'LR': LR,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'PATIENCE': PATIENCE,
        'HIDDEN_WIDTH': HIDDEN_WIDTH,
        'DEPTH_PER_BLOCK': DEPTH_PER_BLOCK,
        'NUM_BLOCKS': NUM_BLOCKS,
    }
    with open(os.path.join(out_dir, f"run_config_H{H}_L{L}_{tgt_col.replace(' ','_')}.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Listo. Model checkpoint: {best_path}")
    print(f"Resultados en: {out_dir}/ (gráficas individuales, PDF y métricas)")
    print("Para usar otra variable del dataset o cambiar H y L, modifica TARGET_COL / H / L en la CONFIG y vuelve a ejecutar.")


if __name__ == "__main__":
    main()
