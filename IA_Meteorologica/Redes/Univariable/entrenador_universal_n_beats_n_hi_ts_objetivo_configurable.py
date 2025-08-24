"""
Entrenador universal N-BEATS / N-HiTS (objetivo configurable)
=============================================================

Cambia solo TARGET_COL (y opcionalmente TARGET_MODE) y este script:
- Carga data/General_normalized.csv
- Detecta timestamps (Formatted Date) y asegura frecuencia horaria
- Construye ventanas W -> H con lags de la MISMA serie objetivo (univariante)
- Entrena un modelo estilo N-BEATS minimal (PyTorch), con early stopping
- Evalúa y guarda métricas + gráficas en data

Soporta:
- REGRESIÓN: variables numéricas (p.ej., 'Temperature (C)', 'Wind Speed (km/h)', 'Pressure (millibars)')
- CLASIFICACIÓN: variables categóricas (p.ej., 'Summary')* → softmax por horizonte

*Nota: Predecir categorías a varios pasos vista es más ruidoso; considera reducir H o agrupar categorías raras.
"""

import os
import json
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import timedelta

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# CONFIG ÚNICA QUE SUELEN CAMBIAR
# -----------------------------
TARGET_COL = 'Temperature (C)'   # <— cambia aquí ('Wind Speed (km/h)', 'Humidity', 'Summary', ...)
TARGET_MODE = 'auto'             # 'auto' | 'regression' | 'classification'
CSV_PATH = 'data/General_normalized.csv'
DT_COL_CANDIDATES = ['Formatted Date','ds','datetime','date','timestamp','time']
W = 168        # ventana de entrada (horas)
H = 24         # horizonte de salida (horas)
BATCH = 128
LR = 1e-3
MAX_EPOCHS = 50
PATIENCE = 5
HIDDEN = 256
N_BLOCKS = 3
N_STACKS = 2
CLASS_EMB_DIM = 16   # sólo para clasificación (embedding de categorías)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(7)

# -----------------------------
# Carga y preparación de serie objetivo
# -----------------------------

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


def load_target_series(csv_path: str, target_col: str):
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
    if inf is None or inf.upper() not in ['H','1H']:
        s = s.resample('1H').ffill().bfill()

    # Detectar tipo de objetivo
    y = s['y']
    mode = TARGET_MODE
    if mode == 'auto':
        if pd.api.types.is_numeric_dtype(y):
            # ¿parece categórico codificado entero con pocas clases?
            uniq = pd.unique(y.values)
            if len(uniq) <= 50 and np.all(np.isfinite(uniq)) and np.allclose(uniq, np.round(uniq)):
                mode = 'classification'
            else:
                mode = 'regression'
        else:
            mode = 'classification'

    # Codificación si es clasificación
    labelmap = None
    if mode == 'classification':
        if not pd.api.types.is_integer_dtype(y):
            # crear mapeo categoría→índice
            cats = s['y'].astype(str).fillna('UNK').values
            # ordenar por frecuencia para estabilidad
            vals, counts = np.unique(cats, return_counts=True)
            order = np.argsort(-counts)
            ordered = vals[order]
            labelmap = {c: i for i, c in enumerate(ordered.tolist())}
            y_idx = np.array([labelmap.get(str(v), 0) for v in cats], dtype=np.int64)
        else:
            # ya es entero; normalizamos a 0..K-1
            uniq = np.unique(y.values.astype(int))
            labelmap = {int(v): i for i, v in enumerate(sorted(uniq.tolist()))}
            y_idx = np.array([labelmap[int(v)] for v in y.values.astype(int)], dtype=np.int64)
        s['y'] = y_idx
    else:
        s['y'] = y.astype(float).values

    return s['y'], mode, labelmap


def build_windows(x: np.ndarray, W: int, H: int, step: int = 1):
    X, Y, t0_idx = [], [], []
    for t in range(W, len(x)-H+1, step):
        X.append(x[t-W:t])
        Y.append(x[t:t+H])
        t0_idx.append(t)
    return np.asarray(X), np.asarray(Y), np.asarray(t0_idx)

# -----------------------------
# Datasets
# -----------------------------

class WindowsDS(Dataset):
    def __init__(self, X, Y, mode: str):
        # X: (N, W) floats (reg) o ints (clf)
        # Y: (N, H) floats o ints
        self.mode = mode
        if mode == 'regression':
            self.X = torch.from_numpy(X).float()
            self.Y = torch.from_numpy(Y).float()
        else:
            self.X = torch.from_numpy(X).long()
            self.Y = torch.from_numpy(Y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# -----------------------------
# Modelo N-BEATS minimal (genérico)
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
        # x: (B, W) float (reg) o long (clf)
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

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    def forward(self, pred, target):
        err = pred - target
        abs_err = err.abs()
        quad = torch.minimum(abs_err, torch.tensor(self.delta, device=err.device))
        lin = abs_err - quad
        return torch.mean(0.5 * quad**2 + self.delta * lin)

# -----------------------------
# Entrenamiento + Evaluación
# -----------------------------

def train_and_eval():
    # 1) Datos
    series, mode, labelmap = load_target_series(CSV_PATH, TARGET_COL)
    x = series.values
    X, Y, t0_idx = build_windows(x, W=W, H=H, step=1)

    # splits
    n = len(series)
    test_hours = 24*30
    val_hours = 24*14
    test_start = max(W, n - test_hours - H)
    val_start = max(W, n - test_hours - val_hours - H)

    split = np.full(len(t0_idx), 'train', dtype=object)
    split[(t0_idx >= val_start) & (t0_idx < (n - test_hours))] = 'val'
    split[(t0_idx >= (n - test_hours))] = 'test'

    train_ds = WindowsDS(X[split=='train'], Y[split=='train'], mode)
    val_ds   = WindowsDS(X[split=='val'],   Y[split=='val'],   mode)
    test_ds  = WindowsDS(X[split=='test'],  Y[split=='test'],  mode)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    # 2) Modelo
    if mode == 'regression':
        model = NBeatsGeneric(W, H, hidden=HIDDEN, n_blocks=N_BLOCKS, n_stacks=N_STACKS, mode=mode).to(DEVICE)
        criterion = HuberLoss(1.0)
    else:
        num_classes = int(series.max()) + 1 if labelmap is None else len(labelmap)
        model = NBeatsGeneric(W, H, hidden=HIDDEN, n_blocks=N_BLOCKS, n_stacks=N_STACKS,
                              mode=mode, num_classes=num_classes, emb_dim=CLASS_EMB_DIM).to(DEVICE)
        criterion = nn.CrossEntropyLoss()  # aplica sobre (B*H, K)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # 3) Entrenar con early stopping
    best_val = float('inf')
    patience = PATIENCE
    ckpt = f"data/nbeats_{mode}_{TARGET_COL.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')}.pt"

    for epoch in range(1, MAX_EPOCHS+1):
        model.train(); tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            yhat = model(xb)
            if mode == 'regression':
                loss = criterion(yhat, yb)
            else:
                # yhat: (B, H, K), yb: (B, H)
                loss = criterion(yhat.view(-1, yhat.size(-1)), yb.view(-1))
            loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= max(1, len(train_ds))

        # validación
        model.eval(); val_metric = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                yhat = model(xb)
                if mode == 'regression':
                    val_metric += nn.L1Loss()(yhat, yb).item() * xb.size(0)  # MAE
                else:
                    preds = yhat.argmax(-1)
                    acc = (preds == yb).float().mean().item()
                    val_metric += acc * xb.size(0)
        val_metric /= max(1, len(val_ds))

        if mode == 'regression':
            print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_MAE {val_metric:.4f}")
            improved = val_metric < best_val - 1e-4
        else:
            print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_acc {val_metric:.4f}")
            # maximizamos accuracy → invertimos criterio para early stopping
            improved = (1.0 - val_metric) < best_val - 1e-4

        if improved:
            best_val = (val_metric if mode=='regression' else (1.0 - val_metric))
            torch.save({'model': model.state_dict()}, ckpt)
            patience = PATIENCE
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping.')
                break

    # cargar mejor
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state['model'])

    # 4) Evaluación en test + gráficas
    metrics = {}
    if mode == 'regression':
        preds, trues = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in test_dl:
                yhat = model(xb.to(DEVICE)).cpu().numpy()
                preds.append(yhat)
                trues.append(yb.numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae = float(np.mean(np.abs(preds - trues)))
        rmse = float(np.sqrt(np.mean((preds - trues)**2)))
        mae_h = np.mean(np.abs(preds - trues), axis=0)
        rmse_h = np.sqrt(np.mean((preds - trues)**2, axis=0))
        metrics.update({'MAE': mae, 'RMSE': rmse, 'MAE_h': mae_h.tolist(), 'RMSE_h': rmse_h.tolist()})

        # gráficas
        os.makedirs('data', exist_ok=True)
        # 1) Curva de error por horizonte
        plt.figure(figsize=(8,4))
        plt.plot(np.arange(1, H+1), mae_h)
        plt.title(f"MAE por horizonte – {TARGET_COL}")
        plt.xlabel('Paso (hora)'); plt.ylabel('MAE'); plt.tight_layout()
        plot1 = f"data/mae_h_{TARGET_COL.replace(' ','_')}.png"; plt.savefig(plot1); plt.close()

        # 2) Ejemplo de predicción vs verdad (último sample de test)
        plt.figure(figsize=(8,4))
        plt.plot(trues[-1], label='verdad')
        plt.plot(preds[-1], label='predicción')
        plt.title(f"Forecast H={H} – {TARGET_COL}")
        plt.xlabel('Paso (hora)'); plt.ylabel(TARGET_COL); plt.legend(); plt.tight_layout()
        plot2 = f"data/forecast_example_{TARGET_COL.replace(' ','_')}.png"; plt.savefig(plot2); plt.close()

        out = {'mode': mode, 'target': TARGET_COL, 'ckpt': ckpt, 'metrics': metrics,
               'plots': {'mae_per_h': plot1, 'forecast_example': plot2}}

    else:
        # clasificación
        model.eval(); total=0; correct=0
        per_h_correct = np.zeros(H, dtype=np.int64)
        per_h_total   = np.zeros(H, dtype=np.int64)
        with torch.no_grad():
            for xb, yb in test_dl:
                logits = model(xb.to(DEVICE)).cpu()  # (B,H,K)
                pred = logits.argmax(-1)             # (B,H)
                total += yb.numel()
                correct += (pred == yb).sum().item()
                eq = (pred == yb).numpy()
                per_h_correct += eq.sum(axis=0)
                per_h_total   += eq.shape[0]
        acc = float(correct / total)
        acc_h = (per_h_correct / np.maximum(1, per_h_total)).astype(float)
        metrics.update({'accuracy': acc, 'accuracy_h': acc_h.tolist(), 'num_classes': model.num_classes})

        # gráficas
        plt.figure(figsize=(8,4))
        plt.plot(np.arange(1, H+1), acc_h)
        plt.title(f"Accuracy por horizonte – {TARGET_COL}")
        plt.xlabel('Paso (hora)'); plt.ylabel('Accuracy'); plt.ylim(0,1); plt.tight_layout()
        plot1 = f"data/acc_h_{TARGET_COL.replace(' ','_')}.png"; plt.savefig(plot1); plt.close()

        out = {'mode': mode, 'target': TARGET_COL, 'ckpt': ckpt, 'metrics': metrics,
               'plots': {'accuracy_per_h': plot1}, 'labelmap': labelmap}

    # Guardar métricas
    MET_PATH = f"data/metrics_{mode}_{TARGET_COL.replace(' ','_')}.json"
    with open(MET_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    print("\nResumen:")
    print(json.dumps(out, indent=2))
    print(f"\nMétricas guardadas en: {MET_PATH}")
    return out

if __name__ == '__main__':
    train_and_eval()
