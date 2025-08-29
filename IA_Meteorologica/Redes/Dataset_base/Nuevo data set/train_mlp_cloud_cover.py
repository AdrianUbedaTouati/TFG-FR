
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mlp_cloud_cover.py

Entrena un MLP para clasificar `Cloud Cover` con solo columnas del dataset o derivadas.
Requiere los archivos generados por `normalize_cloud_cover.py`:
- cloud_cover_normalized.csv
- stats.json (para lista de features)
- label_mapping.json

Ejemplo:
    python train_mlp_cloud_cover.py \
        --data-dir /mnt/data/cloud_cover_prepared \
        --outdir /mnt/data/cloud_cover_results \
        --epochs 60 --batch-size 256 --hidden "256,128" --dropout 0.2
"""

import argparse, os, json, math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt

# ============== Utilidades generales ==============

def set_seed(seed: int = 42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinismo razonable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_prepared(data_dir: str):
    data_dir = Path(data_dir)
    csv_path = data_dir / "cloud_cover_normalized.csv"
    stats_path = data_dir / "stats.json"
    map_path = data_dir / "label_mapping.json"
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe {csv_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"No existe {stats_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"No existe {map_path}")
    df = pd.read_csv(csv_path)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    feature_cols = stats["feature_columns"]
    # Validar que existen
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de features en el CSV preparado: {missing}")
    return df, feature_cols, mapping

class TabularDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_cols, label_col="label_idx"):
        self.X = frame[feature_cols].astype(np.float32).values
        self.y = frame[label_col].astype(np.int64).values
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list, num_classes: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            last = h
        layers.append(nn.Linear(last, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def compute_class_weights(labels: np.ndarray, num_classes: int):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    # Inversa de la frecuencia (suavizada)
    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_y.append(yb.cpu())
    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0).numpy()
    y_prob = torch.softmax(logits, dim=1).numpy()
    y_pred = np.argmax(y_prob, axis=1)
    loss = None  # opcional: calcular con loss_fn si se pasa
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "balanced_accuracy": float(bal_acc),
    }

def plot_confusion(cm, class_names, out_path: Path, title="Confusion Matrix"):
    fig = plt.figure()
    ax = fig.gca()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

# ============== Entrenamiento ==============

def train(args):
    set_seed(args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    df, feature_cols, mapping = load_prepared(args.data_dir)
    label_to_index = mapping["label_to_index"]
    index_to_label = {int(k): v for k, v in mapping["index_to_label"].items()}
    class_names = [index_to_label[i] for i in range(len(index_to_label))]

    # Splits
    tr_df = df[df["split"] == "train"].reset_index(drop=True)
    va_df = df[df["split"] == "val"].reset_index(drop=True)
    te_df = df[df["split"] == "test"].reset_index(drop=True)

    n_features = len(feature_cols)
    n_classes = len(class_names)
    print(f"Features: {n_features} | Clases: {n_classes}")
    print(f"Tamaño splits → train={len(tr_df)} val={len(va_df)} test={len(te_df)}")

    # Datasets y loaders
    train_ds = TabularDataset(tr_df, feature_cols)
    val_ds = TabularDataset(va_df, feature_cols)
    test_ds = TabularDataset(te_df, feature_cols)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Modelo
    hidden = [int(h) for h in args.hidden.split(",") if h.strip()]
    model = MLP(in_dim=n_features, hidden=hidden, num_classes=n_classes, dropout=args.dropout).to(device)

    # Pesos de clase desde TRAIN
    class_weights = compute_class_weights(tr_df["label_idx"].values, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(2, args.patience//2), verbose=True)

    best_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    # Entrenamiento
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(1, n_batches)

        # Evaluación
        train_eval = evaluate(model, train_loader, device)
        val_eval = evaluate(model, val_loader, device)

        scheduler.step(val_eval["f1_macro"])

        print(f"Epoch {epoch:03d} | "
              f"train_loss={train_loss:.4f} | "
              f"train_f1={train_eval['f1_macro']:.4f} | "
              f"val_f1={val_eval['f1_macro']:.4f} | "
              f"val_balacc={val_eval['balanced_accuracy']:.4f}")

        # Early stopping por F1 macro
        if val_eval["f1_macro"] > best_f1 + 1e-6:
            best_f1 = val_eval["f1_macro"]
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "feature_cols": feature_cols,
                "class_names": class_names,
                "args": vars(args),
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping en epoch {epoch} (sin mejora en {args.patience} epochs).")
                break

    # Preparar salida
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Restaurar mejor estado
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    torch.save(best_state, outdir / "checkpoint.pt")

    # Evaluaciones finales
    tr_eval = evaluate(model, train_loader, device)
    va_eval = evaluate(model, val_loader, device)
    te_eval = evaluate(model, test_loader, device)

    # Reportes
    y_true = te_eval["y_true"]
    y_pred = te_eval["y_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    report = classification_report(y_true, y_pred, labels=list(range(n_classes)), target_names=class_names, output_dict=True, zero_division=0)

    # Graficar CM
    plot_confusion(cm, class_names, outdir / "confusion_matrix_test.png", title="Confusion Matrix (Test)")

    # Guardar métricas
    metrics = {
        "train": {k: v for k, v in tr_eval.items() if k in ["accuracy", "f1_macro", "balanced_accuracy"]},
        "val": {k: v for k, v in va_eval.items() if k in ["accuracy", "f1_macro", "balanced_accuracy"]},
        "test": {k: v for k, v in te_eval.items() if k in ["accuracy", "f1_macro", "balanced_accuracy"]},
        "best_val_f1_macro": float(best_f1),
        "class_names": class_names,
        "label_to_index": mapping["label_to_index"] if 'mapping' in locals() else None
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(outdir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Guardar columnas y mapping para inferencia
    with open(outdir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    with open(outdir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # Predicciones (test)
    pred_df = pd.DataFrame({
        "y_true_idx": y_true,
        "y_pred_idx": y_pred,
        "y_true_str": [class_names[i] for i in y_true],
        "y_pred_str": [class_names[i] for i in y_pred],
    })
    # Probabilidades por clase
    probs = pd.DataFrame(te_eval["y_prob"], columns=[f"prob_{c}" for c in class_names])
    pred_df = pd.concat([pred_df, probs], axis=1)
    pred_df.to_csv(outdir / "predictions_test.csv", index=False)

    print(f"[OK] Guardado checkpoint en {outdir / 'checkpoint.pt'}")
    print(f"[OK] Métricas → {outdir / 'metrics.json'}")
    print(f"[OK] Matriz de confusión → {outdir / 'confusion_matrix_test.png'}")
    print(f"[OK] Reporte por clase → {outdir / 'classification_report.json'}")
    print(f"[OK] Predicciones test → {outdir / 'predictions_test.csv'}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="/mnt/data/cloud_cover_prepared")
    ap.add_argument("--outdir", type=str, default="/mnt/data/cloud_cover_results")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="Forzar CPU")
    args = ap.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
