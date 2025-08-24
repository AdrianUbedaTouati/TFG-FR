"""
Evaluador avanzado: carga el modelo entrenado y genera métricas + gráficas con fechas
===================================================================================

Uso:
- Asegúrate de haber entrenado con el script "Entrenador universal N-BEATS/N-HiTS (objetivo configurable)".
- Ajusta abajo TARGET_COL y CSV_PATH si hace falta. Si CKPT_PATH es None, se intenta deducir.
- Ejecuta este script: cargará el checkpoint, inferirá la arquitectura (W, H, stacks, blocks, etc.),
  reconstruirá la serie y las ventanas de test y generará:
  * Métricas globales y por horizonte.
  * Predicciones ensambladas en una serie temporal con timestamps (promedio cuando hay solapes).
  * Gráficas con títulos que incluyen el rango temporal (fechas y horas).
  * CSV con predicciones timestamped para el rango de test.

Requisitos: Sólo PyTorch, NumPy, Pandas y Matplotlib. Sin seaborn.
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from dataclasses import dataclass

import torch
from torch import nn

from matplotlib.dates import AutoDateLocator, AutoDateFormatter

# -----------------------------
# CONFIG
# -----------------------------
TARGET_COL = 'Temperature (C)'
CSV_PATH = 'data/General_normalized.csv'
CKPT_PATH = None   # Si None, se intenta deducir como data/nbeats_{mode}_{safe_target}.pt
METRICS_JSON_HINT = None  # Si existe, mejor pasar 'data/metrics_<mode>_<target>.json' para clasificación

# Estos tamaños se inferirán desde el checkpoint automáticamente, así evitamos mismatch
W = None
H = None

# -----------------------------
# Utilidades de datos (coinciden con el entrenador)
# -----------------------------
DT_COL_CANDIDATES = ['Formatted Date','ds','datetime','date','timestamp','time']

def detect_dt_col(df: pd.DataFrame):
    for c in DT_COL_CANDIDATES:
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    for col in df.columns:
        low = col.lower()
        if 'date' in low or 'time' in low:
            return col
    return None


def load_series(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    dt_col = detect_dt_col(df)
    if dt_col is None:
        # construye timestamps horarios sintéticos si no existen
        start = pd.Timestamp('2000-01-01 00:00:00')
        df.insert(0, 'ds', [start + timedelta(hours=i) for i in range(len(df))])
        dt_col = 'ds'
    else:
        df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce', utc=True).dt.tz_localize(None)
        df = df.dropna(subset=[dt_col])

    assert target_col in df.columns, f"No existe la columna objetivo '{target_col}'."

    s = df[[dt_col, target_col]].dropna()
    s.columns = ['ds','y']
    s = s.sort_values('ds').set_index('ds')

    # Asegurar frecuencia horaria
    try:
        inf = pd.infer_freq(s.index[:50])
    except Exception:
        inf = None
    if inf is None or (inf.upper() not in ['H','1H']):
        s = s.resample('1H').ffill().bfill()

    return s['y']  # Series con DateTimeIndex


def build_windows(x: np.ndarray, W: int, H: int, step: int = 1):
    X, Y, t0_idx = [], [], []
    for t in range(W, len(x)-H+1, step):
        X.append(x[t-W:t])
        Y.append(x[t:t+H])
        t0_idx.append(t)
    return np.asarray(X), np.asarray(Y), np.asarray(t0_idx)

# -----------------------------
# Modelo N-BEATS (idéntico al del entrenador)
# -----------------------------
class NBeatsBlock(nn.Module):
    def __init__(self, in_size, out_size, hidden, n_layers=4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(in_size if i==0 else hidden, hidden), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.theta_b = nn.Linear(hidden, in_size)   # backcast
        self.theta_f = nn.Linear(hidden, out_size)  # forecast
    def forward(self, x):
        h = self.mlp(x)
        back = self.theta_b(h)
        fore = self.theta_f(h)
        return back, fore

class NBeatsGeneric(nn.Module):
    def __init__(self, W, H, hidden=256, n_blocks=3, n_stacks=2, mode='regression', num_classes=None, emb_dim=16):
        super().__init__()
        self.mode = mode
        self.W = W
        self.H = H
        self.num_classes = num_classes
        if mode == 'regression':
            self.in_size = W
            self.out_size = H
            self.embed = None
        else:
            assert num_classes is not None and num_classes > 1
            self.embed = nn.Embedding(num_classes, emb_dim)
            self.in_size = W * emb_dim
            self.out_size = H * num_classes
        stacks = []
        for _ in range(n_stacks):
            blocks = nn.ModuleList([NBeatsBlock(self.in_size, self.out_size, hidden) for __ in range(n_blocks)])
            stacks.append(blocks)
        self.stacks = nn.ModuleList(stacks)

    def forward(self, x):
        if self.mode == 'classification':
            x = self.embed(x)              # (B, W, E)
            x = x.reshape(x.size(0), -1)   # (B, W*E)
        residual = x
        fore_total = torch.zeros(x.size(0), self.out_size, device=x.device)
        for blocks in self.stacks:
            for blk in blocks:
                back, fore = blk(residual)
                residual = residual - back
                fore_total = fore_total + fore
        if self.mode == 'regression':
            return fore_total  # (B, H)
        else:
            return fore_total.view(-1, self.H, self.num_classes)  # (B, H, K)

# -----------------------------
# Carga checkpoint e inferencia de arquitectura
# -----------------------------

def infer_arch_from_ckpt(ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')
    sd = state['model'] if isinstance(state, dict) and 'model' in state else state

    # Infer n_stacks y n_blocks
    stack_ids = set()
    block_ids_per_stack = {}
    for k in sd.keys():
        if k.startswith('stacks.'):
            parts = k.split('.')
            si = int(parts[1]); bi = int(parts[2])
            stack_ids.add(si)
            block_ids_per_stack.setdefault(si, set()).add(bi)
    n_stacks = (max(stack_ids) + 1) if stack_ids else 1
    n_blocks = max(len(v) for v in block_ids_per_stack.values()) if block_ids_per_stack else 1

    # Hidden size e in/out sizes a partir de un bloque
    any_block = None
    for k, v in sd.items():
        if k.endswith('mlp.0.weight'):
            any_block = v
            break
    assert any_block is not None, 'No se pudo inferir hidden/in_size.'
    hidden = any_block.shape[0]
    in_size = any_block.shape[1]

    # out_size desde theta_f
    any_theta_f = None
    for k, v in sd.items():
        if k.endswith('theta_f.weight'):
            any_theta_f = v
            break
    assert any_theta_f is not None, 'No se pudo inferir out_size.'
    out_size = any_theta_f.shape[0]

    # ¿clasificación o regresión?
    is_classif = any('embed.weight' in k for k in sd.keys())
    if is_classif:
        # emb dims
        emb_w = sd['embed.weight']
        num_classes, emb_dim = emb_w.shape
        # H a partir de out_size = H * K
        assert out_size % num_classes == 0, 'out_size no divisible por num_classes'
        H = out_size // num_classes
        # W de in_size = W * emb_dim
        assert in_size % emb_dim == 0, 'in_size no divisible por emb_dim'
        W = in_size // emb_dim
        mode = 'classification'
    else:
        mode = 'regression'
        H = out_size
        W = in_size
        num_classes = None
        emb_dim = None

    return {
        'mode': mode,
        'W': W,
        'H': H,
        'hidden': hidden,
        'n_blocks': n_blocks,
        'n_stacks': n_stacks,
        'num_classes': num_classes,
        'emb_dim': emb_dim,
        'state': sd
    }

# -----------------------------
# Preparación de datos (incluye mapeo para clasificación)
# -----------------------------

def prepare_windows_with_labels(series: pd.Series, W: int, H: int, mode: str, labelmap=None):
    y = series.copy()
    if mode == 'classification':
        if labelmap is None:
            raise RuntimeError("Para clasificación necesito 'labelmap' consistente con el entrenamiento. Proporciona METRICS_JSON_HINT del entrenador.")
        # aplicar mapeo consistente
        y_idx = np.array([labelmap.get(str(v), 0) for v in y.astype(str).values], dtype=np.int64)
        y = pd.Series(y_idx, index=y.index)
    else:
        y = y.astype(float)
    X, Y, t0_idx = build_windows(y.values, W=W, H=H, step=1)
    return X, Y, t0_idx

# -----------------------------
# Ensamblado de predicciones en timeline
# -----------------------------

def stitch_predictions(timestamps, t0_idx, preds, H):
    """Devuelve un DataFrame con columnas: ds, y_true, y_pred (promedio en solapes)
    Requiere que 'timestamps' sea el índice DateTimeIndex de la serie completa.
    't0_idx' son los índices donde empieza cada forecast.
    'preds' es (N,H). Para clasificación, usar argmax antes de pasar aquí si quieres clase predicha.
    """
    start = int(t0_idx.min())
    end = int(t0_idx.max() + H)
    idx = np.arange(start, end)
    ds = timestamps[idx]

    yhat_sum = np.zeros(len(idx), dtype=float)
    yhat_cnt = np.zeros(len(idx), dtype=float)

    for i, t0 in enumerate(t0_idx):
        for h in range(H):
            pos = (t0 - start) + h
            if 0 <= pos < len(idx):
                yhat_sum[pos] += preds[i, h]
                yhat_cnt[pos] += 1.0

    yhat_avg = np.divide(yhat_sum, np.maximum(1.0, yhat_cnt))
    out = pd.DataFrame({'ds': ds, 'y_pred': yhat_avg})
    return out

# -----------------------------
# Plot helpers
# -----------------------------

def fmt_date_ax(ax):
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(AutoDateFormatter(ax.xaxis.get_major_locator()))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha('center')

# -----------------------------
# MAIN
# -----------------------------

def main():
    # 1) Deducir CKPT si no se da
    safe_target = TARGET_COL.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')

    # Intentaremos ambos modos si CKPT_PATH es None
    ckpt_guess = None
    if CKPT_PATH is None:
        for mode in ['regression','classification']:
            guess = f"data/nbeats_{mode}_{safe_target}.pt"
            if os.path.exists(guess):
                ckpt_guess = guess; break
    ckpt = CKPT_PATH or ckpt_guess
    assert ckpt is not None and os.path.exists(ckpt), f"No encuentro el checkpoint. Prueba con CKPT_PATH explícito. Busqué: {ckpt_guess}"

    # 2) Inferir arquitectura del ckpt
    arch = infer_arch_from_ckpt(ckpt)
    mode = arch['mode']
    W_ckpt, H_ckpt = arch['W'], arch['H']

    # 3) Cargar serie original y preparar mapeo si clasificación
    series = load_series(CSV_PATH, TARGET_COL)

    # Intentar cargar labelmap desde metrics JSON si clasificación
    labelmap = None
    if mode == 'classification':
        # Detectar metrics json automáticamente si no se indicó
        if METRICS_JSON_HINT is None:
            m_guess = f"data/metrics_{mode}_{safe_target}.json"
            if os.path.exists(m_guess):
                METRICS_JSON_HINT = m_guess
        if METRICS_JSON_HINT and os.path.exists(METRICS_JSON_HINT):
            with open(METRICS_JSON_HINT, 'r') as f:
                met = json.load(f)
                labelmap = met.get('labelmap', None)
        if labelmap is None:
            raise RuntimeError("No encontré 'labelmap' para clasificación. Pasa METRICS_JSON_HINT o re-ejecuta el entrenador para generar el JSON de métricas.")

    # 4) Construir ventanas y splits como en el entrenador
    X_all, Y_all, t0_idx_all = build_windows(series.values if mode=='regression' else series.astype(str).values, W=W_ckpt, H=H_ckpt, step=1)
    # Aplicar codificación si clasificación
    if mode == 'classification':
        vals = series.astype(str).values
        idx_encoded = np.array([labelmap.get(v, 0) for v in vals], dtype=np.int64)
        X_all, Y_all, t0_idx_all = build_windows(idx_encoded, W=W_ckpt, H=H_ckpt, step=1)

    n = len(series)
    test_hours = 24*30
    val_hours = 24*14
    test_start = max(W_ckpt, n - test_hours - H_ckpt)
    val_start = max(W_ckpt, n - test_hours - val_hours - H_ckpt)

    split = np.full(len(t0_idx_all), 'train', dtype=object)
    split[(t0_idx_all >= val_start) & (t0_idx_all < (n - test_hours))] = 'val'
    split[(t0_idx_all >= (n - test_hours))] = 'test'

    X_test = X_all[split=='test']
    Y_test = Y_all[split=='test']
    t0_test = t0_idx_all[split=='test']

    print(f"Test windows: {len(X_test)} | W={W_ckpt} | H={H_ckpt}")
    print(f"Rango test: {series.index[int(t0_test.min())]} — {series.index[int(t0_test.max()+H_ckpt-1)]}")

    # 5) Reconstruir modelo y cargar pesos
    if mode == 'regression':
        model = NBeatsGeneric(W_ckpt, H_ckpt, hidden=arch['hidden'], n_blocks=arch['n_blocks'], n_stacks=arch['n_stacks'], mode=mode)
    else:
        model = NBeatsGeneric(W_ckpt, H_ckpt, hidden=arch['hidden'], n_blocks=arch['n_blocks'], n_stacks=arch['n_stacks'], mode=mode, num_classes=arch['num_classes'], emb_dim=arch['emb_dim'])
    model.load_state_dict(arch['state'])
    model.eval()

    # 6) Inferir en test
    with torch.no_grad():
        xb = torch.from_numpy(X_test)
        if mode == 'regression':
            xb = xb.float()
            preds = model(xb).numpy()       # (N,H)
        else:
            xb = xb.long()
            logits = model(xb).numpy()      # (N,H,K)
            preds = logits.argmax(-1)       # (N,H)

    # 7) Métricas
    out_dir = 'data'
    os.makedirs(out_dir, exist_ok=True)

    if mode == 'regression':
        y_true = Y_test.astype(float)
        mae = float(np.mean(np.abs(preds - y_true)))
        rmse = float(np.sqrt(np.mean((preds - y_true)**2)))
        mae_h = np.mean(np.abs(preds - y_true), axis=0)
        rmse_h = np.sqrt(np.mean((preds - y_true)**2, axis=0))
        print(f"TEST → MAE={mae:.4f} | RMSE={rmse:.4f}")

        # Ensamblar timeline y guardar CSV
        stitched = stitch_predictions(series.index, t0_test, preds, H_ckpt)
        # Añadir verdad real
        start_ts = series.index[int(t0_test.min())]
        end_ts   = series.index[int(t0_test.max()+H_ckpt-1)]
        truth = series.loc[start_ts:end_ts].rename('y_true').reset_index().rename(columns={'ds':'ds'})
        truth.columns = ['ds','y_true']
        merged = pd.merge(truth, stitched, on='ds', how='left')
        csv_path = os.path.join(out_dir, f"test_forecasts_{TARGET_COL.replace(' ','_')}.csv")
        merged.to_csv(csv_path, index=False)

        # Plot 1: serie real vs predicción ensamblada
        plt.figure(figsize=(11,4))
        plt.plot(merged['ds'], merged['y_true'], label='verdad')
        plt.plot(merged['ds'], merged['y_pred'], label='predicción')
        plt.title(f"{TARGET_COL} – Test ensamblado\n{start_ts:%Y-%m-%d %H:%M} — {end_ts:%Y-%m-%d %H:%M}")
        plt.xlabel('Fecha/hora'); plt.ylabel(TARGET_COL)
        plt.legend(); fmt_date_ax(plt.gca()); plt.tight_layout()
        p1 = os.path.join(out_dir, f"timeline_{TARGET_COL.replace(' ','_')}.png")
        plt.savefig(p1); plt.close()

        # Plot 2: MAE por horizonte (con rango en título)
        plt.figure(figsize=(8,4))
        plt.plot(np.arange(1, H_ckpt+1), mae_h)
        plt.title(f"MAE por horizonte – {TARGET_COL}\nTest: {start_ts:%Y-%m-%d %H:%M} — {end_ts:%Y-%m-%d %H:%M}")
        plt.xlabel('Paso (hora)'); plt.ylabel('MAE'); plt.tight_layout()
        p2 = os.path.join(out_dir, f"mae_h_{TARGET_COL.replace(' ','_')}_eval.png")
        plt.savefig(p2); plt.close()

        # Plot 3: Histograma de residuos (todas las horas)
        resid = (preds - y_true).ravel()
        plt.figure(figsize=(8,4))
        plt.hist(resid, bins=50)
        plt.title(f"Histograma de residuos – {TARGET_COL}\nTest: {start_ts:%Y-%m-%d %H:%M} — {end_ts:%Y-%m-%d %H:%M}")
        plt.xlabel('Error'); plt.ylabel('Frecuencia'); plt.tight_layout()
        p3 = os.path.join(out_dir, f"residuals_hist_{TARGET_COL.replace(' ','_')}.png")
        plt.savefig(p3); plt.close()

        results = {
            'mode': mode,
            'target': TARGET_COL,
            'ckpt': ckpt,
            'test_range': {'start': str(start_ts), 'end': str(end_ts)},
            'metrics': {'MAE': mae, 'RMSE': rmse, 'MAE_h': mae_h.tolist(), 'RMSE_h': rmse_h.tolist()},
            'artifacts': {'csv': csv_path, 'timeline_plot': p1, 'mae_h_plot': p2, 'residual_hist': p3}
        }

    else:
        y_true = Y_test.astype(int)
        acc = float((preds == y_true).mean())
        acc_h = (preds == y_true).mean(axis=0)
        start_ts = series.index[int(t0_test.min())]
        end_ts   = series.index[int(t0_test.max()+H_ckpt-1)]

        # Ensamblado de clases predichas (opcional; aquí guardamos clase argmax)
        stitched = stitch_predictions(series.index, t0_test, preds.astype(float), H_ckpt)
        truth = series.loc[start_ts:end_ts].astype(str).rename('y_true').reset_index()
        truth.columns = ['ds','y_true']
        stitched['y_pred'] = stitched['y_pred'].round().astype(int)
        # Intentar invertir a etiquetas (si el metrics json trae labelmap)
        labelmap = None
        if METRICS_JSON_HINT and os.path.exists(METRICS_JSON_HINT):
            with open(METRICS_JSON_HINT, 'r') as f:
                met = json.load(f)
                labelmap = met.get('labelmap', None)
        if labelmap:
            inv_map = {int(v): k for k, v in {k:int(v) for k,v in labelmap.items()}.items()}
            stitched['y_pred_label'] = stitched['y_pred'].map(inv_map)
            truth['y_true_label'] = truth['y_true']
        merged = pd.merge(truth, stitched[['ds','y_pred'] + ([ 'y_pred_label'] if 'y_pred_label' in stitched.columns else [])], on='ds', how='left')
        csv_path = os.path.join(out_dir, f"test_forecasts_{TARGET_COL.replace(' ','_')}_classes.csv")
        merged.to_csv(csv_path, index=False)

        # Plot 1: Accuracy por horizonte con rango
        plt.figure(figsize=(8,4))
        plt.plot(np.arange(1, H_ckpt+1), acc_h)
        plt.title(f"Accuracy por horizonte – {TARGET_COL}\nTest: {start_ts:%Y-%m-%d %H:%M} — {end_ts:%Y-%m-%d %H:%M}")
        plt.xlabel('Paso (hora)'); plt.ylabel('Accuracy'); plt.ylim(0,1); plt.tight_layout()
        p1 = os.path.join(out_dir, f"acc_h_{TARGET_COL.replace(' ','_')}_eval.png")
        plt.savefig(p1); plt.close()

        # Plot 2: Línea temporal de clases (entera) – si mapeo disponible, opcional
        # (Para categorías, la visualización con líneas puede ser menos informativa; aquí omitimos)

        results = {
            'mode': mode,
            'target': TARGET_COL,
            'ckpt': ckpt,
            'test_range': {'start': str(start_ts), 'end': str(end_ts)},
            'metrics': {'accuracy': acc, 'accuracy_h': acc_h.tolist()},
            'artifacts': {'csv': csv_path, 'acc_h_plot': p1}
        }

    # Guardar JSON de evaluación
    out_json = os.path.join(out_dir, f"evaluation_{mode}_{TARGET_COL.replace(' ','_')}.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print("\nRESUMEN EVALUACIÓN:\n", json.dumps(results, indent=2))
    print(f"\nGuardado en: {out_json}")

if __name__ == '__main__':
    main()
