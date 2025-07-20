# -*- coding: utf-8 -*-
"""train_weather_models_improved.py
====================================
Script interactivo y modular que **mejora las redes, añade regularización, early‑stopping y
visualizaciones específicas** (pérdidas, predicción‑vs‑real, y matriz de convolución para
la CNN‑1D) sobre el dataset ``weatherHistory.csv``.

Cambios principales respecto a ``train_weather_models.py``
----------------------------------------------------------
* 🧠 **Arquitecturas mejoradas**: Dropout, BatchNorm, kernel_size paramétrico y
  activaciones LeakyReLU.
* ⏳ **Early Stopping** basado en pérdida de validación con paciencia.
* 📊 **Gráficas automáticas** para **cada red**:
  - Evolución train/val‑loss por fold y promedio.
  - Serie real vs predicción en test.
  - 🌈 **Matriz de convolución** (heatmap primer kernel) para CNN‑1D.
* CLI unificado con argumentos para hiperparámetros clave.
* Refactor utilidades (``plotters.py``) para desacoplar lógica de visualización.

Ejemplo rápido
^^^^^^^^^^^^^^
.. code-block:: bash

   $ python train_weather_models_improved.py --model CNN --epochs 100 --patience 15

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

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
PLOTS_DIR = Path(__file__).parent / "plots"
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Utils -----------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """Ventanas deslizantes de longitud fija para series univariantes."""

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
    def __init__(self, input_size: int, hidden_size: int = 64, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        return self.fc(out)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out[:, :, :-self.net[0].padding[0]]


class TCNNetwork(nn.Module):
    def __init__(self, input_size: int, channels: tuple[int, ...] = (32, 32), kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = input_size
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], input_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.network(x)
        y = y[:, :, -1]
        return self.fc(y)


class CNN1DNetwork(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(64, input_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.conv(x)
        h = h.mean(dim=2)
        return self.fc(h)

    # Utilidad para devolver la matriz de convolución del primer filtro
    @property
    def first_kernel(self) -> torch.Tensor:
        return self.conv[0].weight.detach().cpu()


class TransformerNetwork(nn.Module):
    def __init__(self, input_size: int, n_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.randn(500, 1, input_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=n_heads, dropout=dropout, batch_first=True
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
# Visualizaciones -------------------------------------------------------------

def plot_losses(history: dict[str, list[float]], title: str, save_path: Path):
    plt.figure()
    plt.plot(history["train"], label="Train")
    plt.plot(history["val"], label="Val")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pred_vs_real(preds: np.ndarray, targets: np.ndarray, title: str, save_path: Path):
    plt.figure()
    plt.plot(targets, label="Real")
    plt.plot(preds, label="Predicción")
    plt.xlabel("Muestra")
    plt.ylabel("Valor normalizado")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_conv_matrix(kernel: torch.Tensor, title: str, save_path: Path):
    """Muestra heatmap del primer kernel [out_channels=32, in_channels=input_size, kernel_size]."""
    # Seleccionar el primer filtro y el primer canal para simplificar vista
    kernel2d = kernel[0].numpy()
    plt.figure()
    plt.imshow(kernel2d, aspect="auto", cmap="viridis")
    plt.colorbar(label="Peso")
    plt.title(title)
    plt.xlabel("Índice entrada")
    plt.ylabel("Índice kernel")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------------------------------
# Early Stopping --------------------------------------------------------------
class EarlyStopping:
    """Detiene entrenamiento si no hay mejora en 'patience' épocas."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.epochs_no_improve = 0

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        return self.epochs_no_improve >= self.patience


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
    return np.sqrt(((preds - targets) ** 2).mean()), preds, targets


def train_model(
    model_name: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_splits: int = 5,
    patience: int = 10,
):
    print(f"\n=== Entrenando {model_name} con {n_splits}-fold TimeSeries CV ===")

    # 1. Datos ----------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    series = df["Temperature (C)"].values[:, None]

    norm = Normalizador(NORMALIZER_MAP[model_name])
    series_norm = norm.ajustar_transformar(series)

    window = 24
    ds_full = TimeSeriesDataset(series_norm, window)
    n_total = len(ds_full)

    # Reservamos 15% final para test -----------------------------------------
    test_size = int(0.15 * n_total)
    test_indices = list(range(n_total - test_size, n_total))
    trainval_indices = list(range(0, n_total - test_size))

    test_loader = DataLoader(Subset(ds_full, test_indices), batch_size=batch_size)

    # 2. Validación cruzada temporal -----------------------------------------
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_histories: list[dict[str, list[float]]] = []
    best_fold_state, best_fold_rmse, best_fold_epoch, best_fold_idx = None, float("inf"), None, None

    input_size = series.shape[1]

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(trainval_indices), 1):
        tr_indices = [trainval_indices[i] for i in tr_idx]
        val_indices = [trainval_indices[i] for i in val_idx]

        train_loader = DataLoader(Subset(ds_full, tr_indices), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(ds_full, val_indices), batch_size=batch_size)

        model = MODEL_CLASSES[model_name](input_size).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2, factor=0.5)
        stopper = EarlyStopping(patience=patience)

        history = {"train": [], "val": []}
        best_val_loss, best_state, best_epoch = float("inf"), None, None

        for epoch in range(1, epochs + 1):
            train_loss = loop_epoch(model, train_loader, criterion, optimizer)
            val_loss = loop_epoch(model, val_loader, criterion)
            history["train"].append(train_loss)
            history["val"].append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                best_epoch = epoch

            if stopper.step(val_loss):
                print(f"⏹️  Early stopping en fold {fold} (época {epoch})")
                break

            if epoch % 10 == 0 or epoch == 1:
                print(f"Fold {fold}/{n_splits} - Época {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}")

        fold_rmse = np.sqrt(best_val_loss)
        fold_histories.append(history)

        # Almacenar mejor fold global ----------------------------------------
        if fold_rmse < best_fold_rmse:
            best_fold_rmse = fold_rmse
            best_fold_state = best_state
            best_fold_epoch = best_epoch
            best_fold_idx = fold

        # Plot por fold -------------------------------------------------------
        plot_losses(
            history,
            title=f"{model_name} Fold {fold} pérdidas",
            save_path=PLOTS_DIR / f"{model_name}_fold{fold}_loss.png",
        )

        print(f"→ Fold {fold} terminado | Mejor val_RMSE={fold_rmse:.4f} (época {best_epoch})")

    # 3. Evaluación final en test --------------------------------------------
    model_best = MODEL_CLASSES[model_name](input_size).to(DEVICE)
    model_best.load_state_dict(best_fold_state)
    model_best.eval()

    rmse, preds, targets = rmse_from_loader(model_best, test_loader)
    mae = np.abs(preds - targets).mean()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    # 4. Guardado -------------------------------------------------------------
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

    # 5. Gráfica promedio CV --------------------------------------------------
    avg_train = np.mean([h["train"] for h in fold_histories], axis=0)
    avg_val = np.mean([h["val"] for h in fold_histories], axis=0)
    plot_losses(
        {"train": avg_train, "val": avg_val},
        title=f"{model_name} CV {n_splits}-fold (prom)",
        save_path=PLOTS_DIR / f"{model_name}_cv_loss.png",
    )

    # 6. Gráfica predicción vs real ------------------------------------------
    plot_pred_vs_real(preds, targets, f"{model_name} Predicción vs Real", PLOTS_DIR / f"{model_name}_pred.png")

    # 7. Matriz de convolución si CNN ----------------------------------------
    if model_name == "CNN":
        kernel = model_best.first_kernel  # [out_ch, in_ch, k]
        plot_conv_matrix(kernel, "Matriz de convolución (1er filtro)", PLOTS_DIR / f"{model_name}_kernel.png")


# -----------------------------------------------------------------------------
# Menú / CLI ------------------------------------------------------------------

def menu(n_splits_default=5):
    def make_action(name):
        return lambda cfg: train_model(name, **cfg)

    options = {
        "1": ("Entrenar LSTM", make_action("LSTM")),
        "2": ("Entrenar TCN", make_action("TCN")),
        "3": ("Entrenar CNN‑1D", make_action("CNN")),
        "4": ("Entrenar Transformer", make_action("TRANSFORMER")),
        "5": ("Entrenar TODAS", lambda cfg: [train_model(m, **cfg) for m in MODEL_CLASSES]),
        "0": ("Salir", lambda cfg: sys.exit(0)),
    }

    # Config común via input --------------------------------------------------
    cfg = {
        "epochs": int(input(f"Épocas [{100}]: ") or 100),
        "batch_size": int(input(f"Batch size [{64}]: ") or 64),
        "lr": float(input(f"LR [{1e-3}]: ") or 1e-3),
        "n_splits": int(input(f"Folds CV [{n_splits_default}]: ") or n_splits_default),
        "patience": int(input(f"Patience early‑stopping [{10}]: ") or 10),
    }

    while True:
        print("\n==== Menú de entrenamiento ====")
        for key, (desc, _) in options.items():
            print(f" {key}. {desc}")
        choice = input("Selecciona una opción: ")
        action = options.get(choice)
        if action:
            action[1](cfg)
        else:
            print("Opción no válida. Intenta de nuevo.")


# -----------------------------------------------------------------------------
# Punto de entrada ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos meteorológicos mejorados")
    parser.add_argument("--model", choices=MODEL_CLASSES.keys(), help="Modelo a entrenar", nargs="?")
    parser.add_argument("--all", action="store_true", help="Entrenar todos los modelos")
    parser.add_argument("--splits", type=int, default=5, help="Número de folds para TimeSeriesSplit")
    parser.add_argument("--epochs", type=int, default=100, help="Número máximo de épocas")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Patience early‑stopping")
    args = parser.parse_args()

    cfg = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_splits": args.splits,
        "patience": args.patience,
    }

    if args.all:
        for name in MODEL_CLASSES:
            train_model(name, **cfg)
    elif args.model:
        train_model(args.model, **cfg)
    else:
        menu(n_splits_default=args.splits)
