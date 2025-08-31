#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mlp_cloud_cover_cv.py

Robustez:
- Stratified K-Fold (por defecto K=5) sobre el split train+val; test se mantiene aparte.
- Ensamble: media de probabilidades de los K modelos sobre TEST.
- Mixup opcional para tabular (soft targets).
- WeightedRandomSampler (oversampling de clases minoritarias).
- FocalLoss/CE + grad clipping + ReduceLROnPlateau (compat sin 'verbose').
- Guarda métricas por fold y finales, y una CM final del ensamble.

Uso:
  python train_mlp_cloud_cover_cv.py \
    --data-dir cloud_cover_prepared \
    --outdir cloud_cover_results_cv \
    --epochs 80 --batch-size 256 \
    --hidden "512,256,128" --dropout 0.15 \
    --loss focal --gamma 2.0 \
    --mixup-alpha 0.4 \
    --sampler weighted --sampler-alpha 1.0 \
    --kfolds 5 --patience 10 --lr 3e-4
"""

import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, classification_report
import matplotlib.pyplot as plt

# ---------- Utils ----------
def set_seed(seed: int = 42):
    import random, os
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_prepared(data_dir: str):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "cloud_cover_normalized.csv")
    with open(data_dir / "stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    with open(data_dir / "label_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    feature_cols = stats["feature_columns"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de features en el CSV preparado: {missing}")
    # Mantenemos TEST fijo
    df_trainval = df[df["split"] != "test"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    return df_trainval, df_test, feature_cols, mapping

class TabDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_cols, label_col="label_idx"):
        self.X = frame[feature_cols].astype(np.float32).values
        self.y = frame[label_col].astype(np.int64).values
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout=0.1):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, num_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def invfreq_weights(labels, n_classes):
    cnt = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = cnt.sum() / (cnt + 1e-6); w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__(); self.gamma=gamma; self.weight=weight; self.reduction=reduction
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = F.softmax(logits, dim=1).gather(1, target.view(-1,1)).squeeze(1).clamp_min(1e-6)
        loss = (1-pt).pow(self.gamma)*ce
        return loss.mean() if self.reduction=="mean" else loss.sum() if self.reduction=="sum" else loss

class SoftTargetCrossEntropy(nn.Module):
    """CE para targets suaves (p.ej., mixup). Permite ponderar por clase usando pesos esperados."""
    def __init__(self, class_weight=None, reduction="mean"):
        super().__init__(); self.w = class_weight; self.reduction=reduction
    def forward(self, logits, soft_targets):
        logp = F.log_softmax(logits, dim=1)
        if self.w is not None:
            # peso esperado = sum_c w[c] * q[c]
            w_exp = (soft_targets * self.w.view(1, -1)).sum(dim=1, keepdim=True)
            loss = -(soft_targets * logp).sum(dim=1) * w_exp.squeeze(1)
        else:
            loss = -(soft_targets * logp).sum(dim=1)
        return loss.mean() if self.reduction=="mean" else loss.sum() if self.reduction=="sum" else loss

def mixup(x, y, num_classes, alpha=0.4):
    if alpha <= 0: return x, F.one_hot(y, num_classes=num_classes).float(), 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    y2 = y_onehot[idx]
    xm = lam * x + (1 - lam) * x2
    ym = lam * y_onehot + (1 - lam) * y2
    return xm, ym, lam

def evaluate(model, loader, device):
    model.eval(); logits_all=[]; y_all=[]
    with torch.no_grad():
        for xb, yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            logits = model(xb)
            logits_all.append(logits.cpu()); y_all.append(yb.cpu())
    logits = torch.cat(logits_all); y = torch.cat(y_all).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    pred = probs.argmax(1)
    return dict(
        y_true=y, y_pred=pred, y_prob=probs,
        accuracy=float(accuracy_score(y, pred)),
        f1_macro=float(f1_score(y, pred, average="macro")),
        balanced_accuracy=float(balanced_accuracy_score(y, pred))
    )

def plot_cm_counts_and_pct(cm, class_names, out_png, title="Confusion Matrix"):
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(pct, interpolation='nearest')
    ax.set_title(title)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}\n{pct[i,j]*100:.1f}%", ha="center", va="center")
    ax.set_ylabel("Real"); ax.set_xlabel("Predicho")
    fig.tight_layout(); fig.savefig(out_png, dpi=180, bbox_inches="tight"); plt.close(fig)

# ---------- Train one fold ----------
def train_one_fold(model, train_loader, val_loader, device, args, n_classes, class_w):
    if args.loss == "focal":
        criterion_hard = FocalLoss(gamma=args.gamma, weight=class_w.to(device))
        criterion_soft = None  # no focal con soft targets
    else:
        # CE con pesos + CE para soft targets si hay mixup
        try:
            criterion_hard = nn.CrossEntropyLoss(weight=class_w.to(device), label_smoothing=args.label_smoothing if args.label_smoothing>0 else 0.0)
        except TypeError:
            criterion_hard = nn.CrossEntropyLoss(weight=class_w.to(device))
        criterion_soft = SoftTargetCrossEntropy(class_weight=class_w.to(device))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    try:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=max(2, args.patience//2), verbose=True)
    except TypeError:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=max(2, args.patience//2))

    best_f1, best_state, wait = -1.0, None, 0
    for epoch in range(1, args.epochs+1):
        model.train(); run_loss=0.0; nb=0
        for xb, yb in train_loader:
            xb=xb.to(device); yb=yb.to(device)
            # mixup si procede
            if args.mixup_alpha>0 and criterion_soft is not None:
                xm, ym, _ = mixup(xb, yb, n_classes, alpha=args.mixup_alpha)
                logits = model(xm)
                loss = criterion_soft(logits, ym)
            else:
                logits = model(xb)
                loss = criterion_hard(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.clip_norm>0: nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            opt.step()
            run_loss += loss.item(); nb += 1
        tr_loss = run_loss / max(1, nb)

        # eval
        tr_eval = evaluate(model, train_loader, device)
        va_eval = evaluate(model, val_loader, device)
        sched.step(va_eval["f1_macro"])
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | train_f1={tr_eval['f1_macro']:.4f} | val_f1={va_eval['f1_macro']:.4f} | val_balacc={va_eval['balanced_accuracy']:.4f}")

        if va_eval["f1_macro"] > best_f1 + 1e-6:
            best_f1 = va_eval["f1_macro"]; wait = 0
            best_state = { "epoch": epoch, "model_state": {k:v.cpu() for k,v in model.state_dict().items()} }
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping ({args.patience} sin mejora)."); break

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    return model, best_f1

# ---------- Main ----------
def main(args):
    set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    df_trv, df_te, feat_cols, mapping = load_prepared(args.data_dir)
    idx2lab = {int(k): v for k, v in mapping["index_to_label"].items()}
    class_names = [idx2lab[i] for i in range(len(idx2lab))]
    n_classes = len(class_names); in_dim = len(feat_cols)

    Xy = df_trv[feat_cols + ["label_idx"]].copy()
    y_all = Xy["label_idx"].values

    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.random_state)
    test_ds = TabDataset(df_te, feat_cols); test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fold_metrics = []; test_prob_aggr = None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all), start=1):
        print(f"\n===== Fold {fold}/{args.kfolds} =====")
        tr_df = df_trv.iloc[tr_idx].reset_index(drop=True)
        va_df = df_trv.iloc[va_idx].reset_index(drop=True)

        # datasets/loaders
        train_ds = TabDataset(tr_df, feat_cols); val_ds = TabDataset(va_df, feat_cols)
        sampler = None
        if args.sampler == "weighted":
            counts = np.bincount(tr_df["label_idx"].values, minlength=n_classes).astype(np.float64)
            cls_w = np.power(1.0/(counts+1e-6), args.sampler_alpha)
            sample_w = cls_w[tr_df["label_idx"].values]
            sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double), num_samples=len(tr_df), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        # modelo + pesos
        hidden = [int(h) for h in args.hidden.split(",") if h.strip()]
        model = MLP(in_dim=in_dim, hidden=hidden, num_classes=n_classes, dropout=args.dropout).to(device)
        class_w = invfreq_weights(tr_df["label_idx"].values, n_classes)

        # entrenar fold
        model, best_f1 = train_one_fold(model, train_loader, val_loader, device, args, n_classes, class_w)
        fold_metrics.append({"fold": fold, "best_val_f1_macro": float(best_f1)})

        # guardar checkpoint por fold
        ckpt = {
            "model_state": model.state_dict(),
            "feature_cols": feat_cols,
            "class_names": class_names,
            "args": vars(args),
            "fold": fold
        }
        torch.save(ckpt, outdir / f"checkpoint_fold{fold}.pt")

        # predicciones test para este fold
        te_eval = evaluate(model, test_loader, device)
        probs_fold = te_eval["y_prob"]
        test_prob_aggr = probs_fold if test_prob_aggr is None else (test_prob_aggr + probs_fold)

    # promedio de probabilidades
    test_prob_aggr /= args.kfolds
    y_true = df_te["label_idx"].values
    y_pred = test_prob_aggr.argmax(1)

    # métricas finales
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    balacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    rpt = classification_report(y_true, y_pred, labels=list(range(n_classes)), target_names=class_names, output_dict=True, zero_division=0)

    # guardar
    metrics = {
        "cv_folds": fold_metrics,
        "test": {"accuracy": float(acc), "f1_macro": float(f1m), "balanced_accuracy": float(balacc)},
        "class_names": class_names
    }
    with open(outdir / "metrics_cv.json", "w", encoding="utf-8") as f: json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(outdir / "classification_report_cv.json", "w", encoding="utf-8") as f: json.dump(rpt, f, ensure_ascii=False, indent=2)

    # predicciones test finales (ensamble)
    te_pred_df = pd.DataFrame({
        "y_true_idx": y_true, "y_pred_idx": y_pred,
        "y_true_str": [class_names[i] for i in y_true],
        "y_pred_str": [class_names[i] for i in y_pred],
    })
    prob_cols = [f"prob_{c}" for c in class_names]
    te_pred_df = pd.concat([te_pred_df, pd.DataFrame(test_prob_aggr, columns=prob_cols)], axis=1)
    te_pred_df.to_csv(outdir / "predictions_test_cv.csv", index=False)

    # matriz de confusión bonita
    title = f"Matriz de confusión CV (Acc={acc:.3f}, F1m={f1m:.3f})"
    plot_cm_counts_and_pct(cm, class_names, outdir / "confusion_matrix_test_cv.png", title=title)

    print("\n[OK] Ensamble CV guardado en:", outdir)
    print("[OK] Métricas →", outdir / "metrics_cv.json")
    print("[OK] CM →", outdir / "confusion_matrix_test_cv.png")
    print("[OK] Predicciones →", outdir / "predictions_test_cv.csv")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="cloud_cover_prepared")
    ap.add_argument("--outdir", type=str, default="cloud_cover_results_cv")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--hidden", type=str, default="512,256,128")
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")

    # Robustez extra
    ap.add_argument("--loss", type=str, default="focal", choices=["ce", "focal"])
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--sampler", type=str, default="weighted", choices=["none", "weighted"])
    ap.add_argument("--sampler-alpha", type=float, default=1.0)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--mixup-alpha", type=float, default=0.4)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
