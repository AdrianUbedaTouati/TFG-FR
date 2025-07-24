# -*- coding: utf-8 -*-
"""train_weather_models_interactivo.py
=====================================
Entrenamiento (con CV temporal, early-stopping, gráficas, etc.) de distintas
arquitecturas sobre *weatherHistory.csv*, **permitiendo escoger interactivamente
la normalización numérica y de texto** para cada red.

Uso rápido (CLI)
----------------
$ python redNeuronal.py             # Lanza menú interactivo
$ python redNeuronal.py LSTM        # Default normals (recomendados)
$ python redNeuronal.py LSTM --num_norm Z_SCORE --text_norm STRIP
$ python redNeuronal.py --all       # Entrena todas con defaults
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset, Subset

# ──────────────────────────────────────────────────────────────────────────────
# Configuración global
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = Path(__file__).parent / "databases/weatherHistory.csv"
MODELS_DIR = Path(__file__).parent / "models_finales"
PLOTS_DIR = Path(__file__).parent / "plots"
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset helper
# ──────────────────────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    """Ventanas deslizantes univariantes"""

    def __init__(self, series: np.ndarray, window: int):
        self.X, self.y = [], []
        for i in range(len(series) - window):
            self.X.append(series[i : i + window])
            self.y.append(series[i + window])
        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.stack(self.y), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ──────────────────────────────────────────────────────────────────────────────
# Normalizaciones disponibles
# ──────────────────────────────────────────────────────────────────────────────
from normalisaciones import Normalizador, NumNorm, TextNorm

# Mapeo “recomendado” por arquitectura
NORMALIZER_MAP: Dict[str, NumNorm] = {
    "LSTM": NumNorm.LSTM_TCN,
    "TCN": NumNorm.LSTM_TCN,
    "CNN": NumNorm.CNN,
    "TRANSFORMER": NumNorm.TRANSFORMER,
}

RECOMMENDED_TEXT = TextNorm.LOWER

# ──────────────────────────────────────────────────────────────────────────────
# Modelos PyTorch (LSTM, TCN, CNN-1D, Transformer)
# ──────────────────────────────────────────────────────────────────────────────
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
        padding = (kernel_size - 1)
        layers: List[nn.Module] = [
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
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.net(x) + self.downsample(x)
        return torch.relu(out)


class TCNNetwork(nn.Module):
    def __init__(self, input_size: int, num_channels: List[int] = [32, 32], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation_size, dropout)
            )
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], input_size)

    def forward(self, x):
        # PyTorch TCN espera (batch, channels, seq_len)
        y1 = self.tcn(x.transpose(1, 2))
        o = self.fc(y1[:, :, -1])
        return o


class CNN1DNetwork(nn.Module):
    def __init__(self, input_size: int, channels: List[int] = [16, 32], kernel_size: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = 1
        for ch in channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, ch, kernel_size, padding=kernel_size - 1),
                    nn.BatchNorm1d(ch),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(2),
                ]
            )
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        conv_out_size = channels[-1]
        self.fc = nn.Linear(conv_out_size, input_size)

    def forward(self, x):
        # Para CNN usamos (batch, channels=1, seq_len)
        y = self.conv(x.transpose(1, 2))
        y = y.mean(dim=-1)
        return self.fc(y)


class TransformerNetwork(nn.Module):
    def __init__(self, input_size: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, input_size)

    def forward(self, x):
        y = self.input_proj(x)
        y = self.encoder(y)
        return self.fc(y[:, -1, :])


MODEL_CLASSES: Dict[str, Callable[[int], nn.Module]] = {
    "LSTM": LSTMNetwork,
    "TCN": TCNNetwork,
    "CNN": CNN1DNetwork,
    "TRANSFORMER": TransformerNetwork,
}

# ──────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares de entrenamiento
# ──────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


@torch.no_grad()
def loop_epoch(model, loader, criterion, optimizer: torch.optim.Optimizer | None = None):
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        if is_train:
            optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        if is_train:
            loss.backward()
            optimizer.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def rmse_from_loader(model, loader):
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        preds.append(model(X).cpu().numpy())
        targets.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    mse = ((preds - targets) ** 2).mean()
    rmse = np.sqrt(mse)
    return rmse, preds.flatten(), targets.flatten()


def plot_losses(history: Dict[str, List[float]], title: str, save_path: Path):
    plt.figure()
    plt.plot(history["train"], label="Train")
    plt.plot(history["val"], label="Val")
    plt.xlabel("Época")
    plt.ylabel("Pérdida (MSE)")
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

# ──────────────────────────────────────────────────────────────────────────────
# Entrenamiento principal
# ──────────────────────────────────────────────────────────────────────────────
def train_model(
    model_name: str,
    num_norm: NumNorm | None = None,
    text_norm: TextNorm | None = None,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_splits: int = 5,
    patience: int = 10,
):
    """Entrena un modelo concreto con las normalizaciones seleccionadas."""

    # Asignar valores por defecto si el usuario no eligió
    if num_norm is None:
        num_norm = NORMALIZER_MAP[model_name]
    if text_norm is None:
        text_norm = RECOMMENDED_TEXT

    print(
        f"\n=== Entrenando {model_name} | NumNorm={num_norm.name} | "
        f"TextNorm={text_norm.name} | CV={n_splits} ==="
    )

    # 1. Carga y normalización de datos
    df = pd.read_csv(DATA_PATH)
    norm = Normalizador(
        df=df[["Temperature (C)"]],
        metodo_numerico=num_norm,
        metodo_texto=text_norm,
    )
    series_norm = norm.df_normalizado.values

    window = 24
    ds_full = TimeSeriesDataset(series_norm, window)
    n_total = len(ds_full)

    test_size = int(0.15 * n_total)
    test_indices = list(range(n_total - test_size, n_total))
    trainval_indices = list(range(0, n_total - test_size))

    test_loader = DataLoader(Subset(ds_full, test_indices), batch_size=batch_size)

    # 2. CV temporal
    tscv = TimeSeriesSplit(n_splits=n_splits)
    input_size = 1
    best_fold_state, best_fold_rmse, best_fold_epoch, best_fold_idx = None, float("inf"), None, None

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
        if fold_rmse < best_fold_rmse:
            best_fold_rmse = fold_rmse
            best_fold_state = best_state
            best_fold_epoch = best_epoch
            best_fold_idx = fold

        plot_losses(
            history,
            title=f"{model_name} Fold {fold} pérdidas",
            save_path=PLOTS_DIR / f"{model_name}_{num_norm.name}_{fold}_loss.png",
        )
        print(f"→ Fold {fold} terminado | Mejor val_RMSE={fold_rmse:.4f} (época {best_epoch})")

    # 3. Evaluación final en test
    model_best = MODEL_CLASSES[model_name](input_size).to(DEVICE)
    model_best.load_state_dict(best_fold_state)
    rmse, preds, targets = rmse_from_loader(model_best, test_loader)
    mae = np.abs(preds - targets).mean()
    r2 = 1 - ((targets - preds) ** 2).sum() / ((targets - targets.mean()) ** 2).sum()

    # 4. Guardado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{num_norm.name}_{text_norm.name}_{timestamp}.pt"
    path = MODELS_DIR / filename
    torch.save(
        {
            "model_name": model_name,
            "state_dict": best_fold_state,
            "input_size": input_size,
            "normalizacion": {
                "numerica": num_norm.name,
                "texto": text_norm.name,
            },
            "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
            "best_fold": best_fold_idx,
            "best_epoch": best_fold_epoch,
            "cv_rmse": best_fold_rmse,
        },
        path,
    )
    print(
        f"\n✔️  Modelo guardado en {path}\n"
        f"Test RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}\n"
    )

    # Gráfica final real vs predicción
    plot_pred_vs_real(
        preds,
        targets,
        title=f"{model_name}-{num_norm.name}-test",
        save_path=PLOTS_DIR / f"{model_name}_{num_norm.name}_pred_vs_real.png",
    )

# ──────────────────────────────────────────────────────────────────────────────
# Menú / CLI
# ──────────────────────────────────────────────────────────────────────────────
def choose_enum(title: str, enum_cls, recommended) -> Enum:
    """Muestra un menú de opciones para un Enum y devuelve la elección."""
    options = list(enum_cls)
    print(f"\n{title}:")
    for i, opt in enumerate(options, 1):
        rec = " (recomendado)" if opt == recommended else ""
        print(f" {i}. {opt.name}{rec}")
    while True:
        sel = input("Selecciona una opción (Enter = recomendado): ").strip()
        if sel == "":
            return recommended
        if sel.isdigit() and 1 <= int(sel) <= len(options):
            return options[int(sel) - 1]
        print("⚠️  Entrada no válida, intenta de nuevo.")


def menu(n_splits_default: int = 5):
    MODEL_KEYS = {"1": "LSTM", "2": "TCN", "3": "CNN", "4": "TRANSFORMER"}
    print("=== Configuración global ===")
    cfg = {
        "epochs": int(input(f"Épocas [100]: ") or 100),
        "batch_size": int(input(f"Batch size [64]: ") or 64),
        "lr": float(input(f"LR [1e-3]: ") or 1e-3),
        "n_splits": int(input(f"Folds CV [{n_splits_default}]: ") or n_splits_default),
        "patience": int(input(f"Patience early-stopping [10]: ") or 10),
    }

    while True:
        print("\n==== Menú de modelos ====")
        for key, name in MODEL_KEYS.items():
            print(f" {key}. Entrenar {name}")
        print(" 5. Entrenar TODOS (defaults)")
        print(" 0. Salir")

        choice = input("Selecciona una opción: ").strip()
        if choice == "0":
            sys.exit(0)
        elif choice == "5":
            for model in MODEL_CLASSES:
                train_model(
                    model,
                    NORMALIZER_MAP[model],
                    RECOMMENDED_TEXT,
                    **cfg,
                )
        elif choice in MODEL_KEYS:
            model_name = MODEL_KEYS[choice]
            num_norm = choose_enum(
                f"Normalización numérica para {model_name}",
                NumNorm,
                NORMALIZER_MAP[model_name],
            )
            text_norm = choose_enum(
                "Normalización de texto",
                TextNorm,
                RECOMMENDED_TEXT,
            )
            train_model(model_name, num_norm, text_norm, **cfg)
        else:
            print("⚠️  Opción no válida.")

# ──────────────────────────────────────────────────────────────────────────────
# Entrada principal
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos meteorológicos")
    parser.add_argument("model", nargs="?", choices=MODEL_CLASSES.keys(), help="Modelo a entrenar")
    parser.add_argument("--all", action="store_true", help="Entrenar todos los modelos")
    parser.add_argument("--num_norm", choices=[e.name for e in NumNorm], help="Normalización numérica")
    parser.add_argument("--text_norm", choices=[e.name for e in TextNorm], help="Normalización de texto")
    parser.add_argument("--splits", type=int, default=5, help="Número de folds para TimeSeriesSplit")
    parser.add_argument("--epochs", type=int, default=100, help="Número máximo de épocas")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Patience early-stopping")
    args = parser.parse_args()

    cfg_common = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_splits": args.splits,
        "patience": args.patience,
    }

    # Parse posibles Enum via CLI
    num_norm_cli = NumNorm[args.num_norm] if args.num_norm else None
    text_norm_cli = TextNorm[args.text_norm] if args.text_norm else None

    if args.all:
        for model in MODEL_CLASSES:
            train_model(
                model,
                num_norm_cli or NORMALIZER_MAP[model],
                text_norm_cli or RECOMMENDED_TEXT,
                **cfg_common,
            )
    elif args.model:
        train_model(
            args.model,
            num_norm_cli,
            text_norm_cli,
            **cfg_common,
        )
    else:
        menu(n_splits_default=args.splits)
