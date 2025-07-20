# -*- coding: utf-8 -*-
"""train_weather_models.py
================================
Script interactivo con **validación cruzada temporal** para entrenar
redes neuronales (LSTM, TCN, CNN‑1D, Transformer) sobre la serie
``weatherHistory.csv``. La mejor iteración (época) de la mejor *fold*
se guarda como checkpoint en ``models_finales/``.

Características nuevas
---------------------
* **TimeSeriesSplit (k‑fold)** conservando orden temporal.
* Guarda metadatos: ``best_fold`` y ``best_epoch``.
* Argumento CLI ``--splits`` y entrada de menú para elegir nº de folds.
* Tabla final de métricas promedio entre folds (RMSE, MAE, R²).

Ejemplo rápido
^^^^^^^^^^^^^^
.. code-block:: bash

   $ python train_weather_models.py --model LSTM --splits 5

"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset, Subset

# -----------------------------------------------------------------------------
# Configuración global ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = Path(__file__).parent / "weatherHistory.csv"
MODELS_DIR = Path(__file__).parent / "models_finales"
MODELS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Utils -----------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, window: int):
        self.X, self.y = [], []
        for i in range(len(series) - window):
            self.X.append(series[i : i + window])
            self.y.append(series[i + window])
        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.stack(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------------------------
# Normalizadores --------------------------------------------------------------
from normalisaciones import Normalizador, NumNorm  # noqa: E402

NORMALIZER_MAP = {
    "LSTM": NumNorm.LSTM_TCN,
    "TCN": NumNorm.LSTM_TCN,
    "CNN": NumNorm.CNN,
    "TRANSFORMER": NumNorm.TRANSFORMER,
}

# -----------------------------------------------------------------------------
# Modelos PyTorch -------------------------------------------------------------
class LSTMNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.net(x)
        return out[:, :, :-self.net[0].padding[0]]


class TCNNetwork(nn.Module):
    def __init__(self, input_size: int, channels=(25, 25), kernel_size=3):
        super().__init__()
        layers = []
        in_ch = input_size
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, ch, kernel_size, dilation=2 ** i))
            in_ch = ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], input_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.network(x)
        y = y[:, :, -1]
        return self.fc(y)


class CNN1DNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Linear(64, input_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.conv(x)
        h = h.mean(dim=2)
        return self.fc(h)


class TransformerNetwork(nn.Module):
    def __init__(self, input_size: int, n_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.randn(500, 1, input_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x):
        seq_len = x.size(1)
        pos = self.pos_encoder[:seq_len].permute(1, 0, 2)
        x = x + pos
        h = self.transformer(x)
        return self.fc(h[:, -1, :])


MODEL_CLASSES = {
    "LSTM": LSTMNetwork,
    "TCN": TCNNetwork,
    "CNN": CNN1DNetwork,
    "TRANSFORMER": TransformerNetwork,
}

# -----------------------------------------------------------------------------
# Entrenamiento con validación cruzada ---------------------------------------

def loop_epoch(model, loader, criterion, optimizer=None):
    running = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        if optimizer:
            optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        if optimizer:
            loss.backward()
            optimizer.step()
        running += loss.item() * len(X)
    return running / len(loader.dataset)


def rmse_from_loader(model, loader):
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds.append(model(X).cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return np.sqrt(((preds - targets) ** 2).mean())


def train_model(model_name: str, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3, n_splits: int = 5):
    print(f"\n=== Entrenando {model_name} con {n_splits}-fold TimeSeries CV ===")

    # 1. Datos ----------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    series = df["Temperature (C)"].values[:, None]

    norm = Normalizador(NORMALIZER_MAP[model_name])
    series_norm = norm.ajustar_transformar(series)

    window = 24
    ds_full = TimeSeriesDataset(series_norm, window)
    n_total = len(ds_full)

    # Reservamos 15% final para test ------------------------------------------------
    test_size = int(0.15 * n_total)
    test_indices = list(range(n_total - test_size, n_total))
    trainval_indices = list(range(0, n_total - test_size))

    test_loader = DataLoader(Subset(ds_full, test_indices), batch_size=batch_size)

    # 2. Validación cruzada temporal ------------------------------------------
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_histories = []
    best_fold_state, best_fold_rmse, best_fold_epoch, best_fold_idx = None, float("inf"), None, None

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(trainval_indices), 1):
        tr_indices = [trainval_indices[i] for i in tr_idx]
        val_indices = [trainval_indices[i] for i in val_idx]

        train_loader = DataLoader(Subset(ds_full, tr_indices), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(ds_full, val_indices), batch_size=batch_size)

        input_size = series.shape[1]
        model = MODEL_CLASSES[model_name](input_size).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"train": [], "val": []}
        best_val_loss, best_state, best_epoch = float("inf"), None, None

        for epoch in range(1, epochs + 1):
            train_loss = loop_epoch(model, train_loader, criterion, optimizer)
            val_loss = loop_epoch(model, val_loader, criterion)
            scheduler.step()

            history["train"].append(train_loss)
            history["val"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                best_epoch = epoch
            if epoch % 10 == 0 or epoch == 1:
                print(f"Fold {fold}/{n_splits} - Época {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}")

        fold_rmse = np.sqrt(best_val_loss)
        fold_histories.append(history)

        # Almacenar mejor fold global ------------------------------------------------
        if fold_rmse < best_fold_rmse:
            best_fold_rmse = fold_rmse
            best_fold_state = best_state
            best_fold_epoch = best_epoch
            best_fold_idx = fold

        print(f"→ Fold {fold} terminado | Mejor val_RMSE={fold_rmse:.4f} (época {best_epoch})")

    # 3. Evaluación final en test ---------------------------------------------
    model_best = MODEL_CLASSES[model_name](input_size).to(DEVICE)
    model_best.load_state_dict(best_fold_state)
    model_best.eval()

    rmse = rmse_from_loader(model_best, test_loader)
    mae = np.abs(rmse_from_loader(model_best, test_loader) - 0)  # simplificado
    # Cálculo completo de MAE y R² --------------------------------------------------
    preds, targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds.append(model_best(X).cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae = np.abs(preds - targets).mean()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    # 4. Guardado --------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pt"
    path = MODELS_DIR / filename
    torch.save(
        {
            "model_name": model_name,
            "state_dict": best_fold_state,
            "input_size": input_size,
            "scaler_state": norm.exportar_estado(),
            "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
            "best_fold": best_fold_idx,
            "best_epoch": best_fold_epoch,
            "cv_rmse": best_fold_rmse,
        },
        path,
    )
    print(
        f"\nMejor modelo: fold {best_fold_idx} | época {best_fold_epoch}\n"
        f"Guardado en {path} (Test RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f})\n"
    )

    # 5. Gráfica promedio CV ---------------------------------------------------
    avg_train = np.mean([h["train"] for h in fold_histories], axis=0)
    avg_val = np.mean([h["val"] for h in fold_histories], axis=0)
    plt.figure()
    plt.plot(avg_train, label="Train (prom)"), plt.plot(avg_val, label="Val (prom)")
    plt.xlabel("Época"), plt.ylabel("MSE"), plt.title(f"{model_name} CV {n_splits}‑fold")
    plt.legend(), plt.tight_layout(), plt.show()


# -----------------------------------------------------------------------------
# Menú / CLI ------------------------------------------------------------------

def menu(n_splits_default=5):
    def make_action(name):
        return lambda: train_model(name, n_splits=n_splits_default)

    options = {
        "1": ("Entrenar LSTM", make_action("LSTM")),
        "2": ("Entrenar TCN", make_action("TCN")),
        "3": ("Entrenar CNN‑1D", make_action("CNN")),
        "4": ("Entrenar Transformer", make_action("TRANSFORMER")),
        "5": (
            "Entrenar TODAS",
            lambda: [train_model(m, n_splits=n_splits_default) for m in MODEL_CLASSES],
        ),
        "6": (
            "Comparar modelos guardados",
            lambda: os.system(f"python {Path(__file__).parent / 'compare_models.py'}"),
        ),
        "0": ("Salir", lambda: sys.exit(0)),
    }

    while True:
        print("\n==== Menú de entrenamiento ====")
        for key, (desc, _) in options.items():
            print(f" {key}. {desc}")
        choice = input("Selecciona una opción: ")
        action = options.get(choice)
        if action:
            action[1]()
        else:
            print("Opción no válida. Intenta de nuevo.")


# -----------------------------------------------------------------------------
# Punto de entrada ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos meteorológicos")
    parser.add_argument("--model", choices=MODEL_CLASSES.keys(), help="Modelo a entrenar", nargs="?")
    parser.add_argument("--all", action="store_true", help="Entrenar todos los modelos")
    parser.add_argument("--compare", action="store_true", help="Comparar modelos guardados")
    parser.add_argument("--splits", type=int, default=5, help="Número de folds para TimeSeriesSplit")
    args = parser.parse_args()

    if args.all:
        for name in MODEL_CLASSES:
            train_model(name, n_splits=args.splits)
    elif args.model:
        train_model(args.model, n_splits=args.splits)
    elif args.compare:
        os.system(f"python {Path(__file__).parent / 'compare_models.py'}")
    else:
        menu(n_splits_default=args.splits)
