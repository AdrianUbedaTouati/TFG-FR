
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_cloud_cover.py

Genera una **buena** matriz de confusión para `Cloud Cover` con:
- Valores absolutos (conteos) y porcentajes por fila (recall por clase).
- Título con precisión global (accuracy) y F1 macro.
- Archivos auxiliares:
    - confusion_matrix_pretty.png
    - confusion_matrix_counts.csv
    - confusion_matrix_row_percent.csv

Entrada esperada (producida por train_mlp_cloud_cover.py):
- results_dir/predictions_test.csv
- results_dir/metrics.json
- results_dir/label_mapping.json  (opcional; si no, se infiere de predictions_test.csv)

Uso:
    python plots_cloud_cover.py --results-dir /mnt/data/cloud_cover_results
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def load_artifacts(results_dir: Path):
    pred_path = results_dir / "predictions_test.csv"
    metrics_path = results_dir / "metrics.json"
    mapping_path = results_dir / "label_mapping.json"  # opcional

    if not pred_path.exists():
        raise FileNotFoundError(f"No se encontró {pred_path}. Ejecuta el entrenamiento primero.")

    preds = pd.read_csv(pred_path)
    if not metrics_path.exists():
        metrics = {}
    else:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        idx2lab = {int(k): v for k, v in mapping.get("index_to_label", {}).items()}
        # Orden por índice
        class_names = [idx2lab[i] for i in range(len(idx2lab))]
    else:
        # Fallback: inferir por orden de aparición de índices
        if "y_true_idx" in preds.columns:
            uniq = sorted(pd.unique(preds["y_true_idx"].astype(int)))
            # si hay nombres string:
            if "y_true_str" in preds.columns:
                map_from_true = preds.groupby("y_true_idx")["y_true_str"].agg(lambda s: s.iloc[0])
                class_names = [str(map_from_true.get(i, f"class_{i}")) for i in uniq]
            else:
                class_names = [f"class_{i}" for i in uniq]
        else:
            raise ValueError("predictions_test.csv no contiene y_true_idx para inferir las clases.")
    return preds, metrics, class_names


def plot_confusion_matrix(cm_counts: np.ndarray, class_names: list, title_suffix: str, out_png: Path):
    # Normalización por filas (recall por clase verdadera)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_row_pct = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums!=0)

    # Plot
    fig = plt.figure()
    ax = fig.gca()
    im = ax.imshow(cm_row_pct, interpolation='nearest')
    ax.set_title(f"Matriz de confusión {title_suffix}")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Anotar: conteo y %
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            count = int(cm_counts[i, j])
            pct = 100.0 * cm_row_pct[i, j]
            text = f"{count}\n{pct:.1f}%"
            ax.text(j, i, text, ha="center", va="center")

    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return cm_row_pct


def main(args):
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    preds, metrics, class_names = load_artifacts(results_dir)

    if "y_true_idx" not in preds.columns or "y_pred_idx" not in preds.columns:
        raise ValueError("predictions_test.csv debe contener columnas y_true_idx y y_pred_idx.")

    y_true = preds["y_true_idx"].astype(int).values
    y_pred = preds["y_pred_idx"].astype(int).values
    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Título con métricas globales
    acc = None
    f1m = None
    if isinstance(metrics, dict):
        acc = metrics.get("test", {}).get("accuracy", None)
        f1m = metrics.get("test", {}).get("f1_macro", None)
    suffix_parts = []
    if acc is not None:
        suffix_parts.append(f"Acc={acc:.3f}")
    if f1m is not None:
        suffix_parts.append(f"F1macro={f1m:.3f}")
    title_suffix = f"({', '.join(suffix_parts)})" if suffix_parts else ""

    # Plot y guardado
    out_png = results_dir / "confusion_matrix_pretty.png"
    cm_row_pct = plot_confusion_matrix(cm, class_names, title_suffix, out_png)

    # CSVs auxiliares
    counts_df = pd.DataFrame(cm, index=[f"real_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names])
    counts_df.to_csv(results_dir / "confusion_matrix_counts.csv")
    pct_df = pd.DataFrame(cm_row_pct, index=[f"real_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names])
    pct_df.to_csv(results_dir / "confusion_matrix_row_percent.csv")

    print(f"[OK] Guardado: {out_png}")
    print(f"[OK] Guardado: {results_dir / 'confusion_matrix_counts.csv'}")
    print(f"[OK] Guardado: {results_dir / 'confusion_matrix_row_percent.csv'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="/mnt/data/cloud_cover_results")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
