
from __future__ import annotations
import os, json
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    rows = []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        supp = int((y_true == c).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        rows.append({
            "class_id": c, "support": supp,
            "precision": float(prec), "recall": float(rec), "f1": float(f1)
        })
    return rows

def _plot_per_class_f1(df_rep: pd.DataFrame, class_names: List[str], out_path: str):
    order = np.argsort(df_rep["f1"].values)[::-1]
    vals  = df_rep.loc[order, "f1"].values
    labels = [class_names[i] for i in df_rep.loc[order, "class_id"].values]
    plt.figure(figsize=(max(6, 0.8*len(labels)), 4.5))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("F1 por clase"); plt.ylim(0, 1.0)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def _plot_distributions(dist_json_path: str, out_path: str):
    if not os.path.exists(dist_json_path):
        return False
    with open(dist_json_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    class_names = info.get("class_names", [f"C{i}" for i in range(len(info.get("train", [])))])
    train = np.array(info.get("train", []), dtype=float)
    val   = np.array(info.get("val", []), dtype=float)
    test  = np.array(info.get("test", []), dtype=float)
    totals = np.array([train.sum(), val.sum(), test.sum()])
    # Convertir a porcentajes por split
    train_p = (train / train.sum()) if train.sum() > 0 else np.zeros_like(train)
    val_p   = (val   / val.sum())   if val.sum()   > 0 else np.zeros_like(val)
    test_p  = (test  / test.sum())  if test.sum()  > 0 else np.zeros_like(test)

    x = np.arange(len(class_names))
    w = 0.25
    plt.figure(figsize=(max(6, 1.1*len(class_names)), 4.8))
    plt.bar(x - w, train_p, width=w, label="Train")
    plt.bar(x,       val_p, width=w, label="Val")
    plt.bar(x + w, test_p, width=w, label="Test")
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("Distribución por split (%)")
    plt.ylim(0, 1.0); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    return True

def generate_artifacts_line_cls(cfg, results: Dict):
    out_root = getattr(cfg, "OUTPUT_DIR", "outputs/mlp_summary_coarse")
    plots_dir = os.path.join(out_root, "plots"); os.makedirs(plots_dir, exist_ok=True)

    # 1) Reporte por clase (prec/rec/f1 + macro)
    y_true = np.array(results.get("y_true", []), dtype=int)
    y_pred = np.array(results.get("y_pred", []), dtype=int)
    class_names = results.get("class_names", None)
    if class_names is None:
        ncls = int(max(y_true.max() if y_true.size else 2, y_pred.max() if y_pred.size else 2) + 1)
        class_names = [f"C{i}" for i in range(ncls)]
    num_classes = len(class_names)

    rows = _per_class_metrics(y_true, y_pred, num_classes)
    df_rep = pd.DataFrame(rows)
    df_rep["class_name"] = df_rep["class_id"].map({i:n for i,n in enumerate(class_names)})
    macro = {
        "class_id": -1, "class_name": "macro_avg",
        "support": int(df_rep["support"].sum()),
        "precision": float(df_rep["precision"].mean()),
        "recall": float(df_rep["recall"].mean()),
        "f1": float(df_rep["f1"].mean())
    }
    df_full = pd.concat([df_rep, pd.DataFrame([macro])], ignore_index=True)
    rep_csv = os.path.join(plots_dir, "classification_report.csv")
    df_full.to_csv(rep_csv, index=False)

    # 2) Gráfico F1 por clase
    f1_png = os.path.join(plots_dir, "per_class_f1.png")
    _plot_per_class_f1(df_rep, class_names, f1_png)

    # 3) Distribución de clases por split (si existe el JSON)
    dist_json = os.path.join(out_root, "label_distribution.json")
    _plot_distributions(dist_json, os.path.join(plots_dir, "label_distribution.png"))

    # 4) Guardar un resumen rápido en JSON
    summary = {
        "test_acc": float(results.get("test_acc", 0.0)),
        "test_f1_macro": float(results.get("test_f1", 0.0)),
        "report_csv": rep_csv,
        "per_class_f1_png": f1_png,
        "confusion_png": results.get("confusion_png", ""),
    }
    with open(os.path.join(out_root, "plots_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[PLOTS] Artefactos guardados en:", plots_dir)
    return summary
