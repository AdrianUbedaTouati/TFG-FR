
from __future__ import annotations
import os, json
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm

def _per_class_metrics(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    # cm[i, j] = count of true i predicted j
    num_classes = cm.shape[0]
    metrics = {}
    support = cm.sum(axis=1)  # true counts per class
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)

    metrics["_micro"] = {}  # filled below
    metrics["_macro"] = {}
    metrics["_weighted"] = {}

    per_class = []
    for i in range(num_classes):
        per_class.append((prec[i], rec[i], f1[i], support[i]))

    # aggregates
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    macro_prec = float(np.mean(prec)) if len(prec) else 0.0
    macro_rec = float(np.mean(rec)) if len(rec) else 0.0
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0

    weights = support / max(1, support.sum())
    weighted_prec = float(np.sum(weights * prec))
    weighted_rec = float(np.sum(weights * rec))
    weighted_f1 = float(np.sum(weights * f1))

    metrics["_micro"] = {"precision": float(micro_prec), "recall": float(micro_rec), "f1": float(micro_f1)}
    metrics["_macro"] = {"precision": float(macro_prec), "recall": float(macro_rec), "f1": float(macro_f1)}
    metrics["_weighted"] = {"precision": float(weighted_prec), "recall": float(weighted_rec), "f1": float(weighted_f1)}

    metrics["per_class"] = [
        {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i in range(num_classes)
    ]
    return metrics

def _save_matrix_csv(path: str, mat: np.ndarray, headers: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join([""] + headers) + "\\n")
        for i, row in enumerate(mat):
            f.write(",".join([headers[i]] + [str(int(v)) if float(v).is_integer() else f"{v:.6f}" for v in row]) + "\\n")

def _plot_confusion(cm: np.ndarray, class_names: List[str], out_path: str, title: str):
    """
    Dibuja la matriz de confusión con anotaciones (recuentos y/o %).
    - Si 'cm' es entera (recuentos), anota "n\npp%".
    - Si 'cm' es flotante (normalizada por fila), anota "pp%".
    El color del texto se elige según el contraste real con el color del píxel
    para que siempre sea legible (negro sobre celdas claras, blanco sobre oscuras).
    """
    fig_w = max(6, 0.4*len(class_names))
    fig_h = max(5, 0.35*len(class_names))
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(cm, aspect='auto')
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    n_classes = len(class_names)
    annotate = n_classes <= 50

    # ticks
    if n_classes <= 50:
        plt.xticks(range(n_classes), class_names, rotation=90)
        plt.yticks(range(n_classes), class_names)
    else:
        plt.xticks([]); plt.yticks([])

    plt.colorbar(im)

    if annotate:
        is_int = np.issubdtype(cm.dtype, np.integer)
        row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
        # Función para decidir color por luminancia real del colormap
        def pick_text_color(val):
            # normaliza val a [0,1] con la misma norma del imshow
            try:
                v = float(im.norm(val))
            except Exception:
                v = 0.0
            r, g, b, _ = im.cmap(v)
            # luminancia relativa (sRGB)
            L = 0.2126*r + 0.7152*g + 0.0722*b
            return "white" if L < 0.5 else "black"

        for i in range(n_classes):
            rs = row_sums[i, 0] if row_sums[i, 0] > 0 else 1.0
            for j in range(n_classes):
                val = float(cm[i, j])
                perc = (val / rs) * 100.0 if rs > 0 else 0.0
                if is_int:
                    text = f"{int(round(val))}\n{perc:.1f}%"
                else:
                    text = f"{(val * 100.0 if rs <= 1.0000001 else perc):.1f}%"
                color = pick_text_color(val)
                plt.text(j, i, text, ha="center", va="center", fontsize=10, fontweight="bold", color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _plot_bar(values: List[float], labels: List[str], out_path: str, title: str, ylabel: str):
    idx = np.arange(len(values))
    plt.figure(figsize=(max(6, 0.2*len(values)), 5))
    plt.bar(idx, values)
    plt.xticks(idx, labels, rotation=90 if len(labels) > 12 else 0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _top_confusions(cm: np.ndarray, class_names: List[str], k: int = 20) -> List[Tuple[str, str, int]]:
    # Return top-k (true!=pred) confusions by count
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]

def generate_artifacts_line_cls(cfg, results: Dict):
    """
    Genera artefactos de evaluación para la LSTM de clasificación por fila:
    - Matriz de confusión (cruda y normalizada por filas)
    - Métricas por clase y agregadas (JSON + CSV)
    - Barras de F1 por clase
    - Distribución de clases (train/val/test) si está disponible
    - Top-20 confusiones más frecuentes (CSV)
    """
    out_dir = _ensure_dir(os.path.join(cfg.OUTPUT_DIR, "plots"))
    class_names: List[str] = results.get("class_names", [])
    y_true = np.array(results.get("y_true", []), dtype=np.int64)
    y_pred = np.array(results.get("y_pred", []), dtype=np.int64)
    num_classes = len(class_names)

    # === Confusion matrices ===
    cm = _confusion_matrix(y_true, y_pred, num_classes)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, np.maximum(row_sums, 1), where=row_sums>0)

    # Save CSVs
    _save_matrix_csv(os.path.join(out_dir, "confusion_matrix_raw.csv"), cm, class_names)
    _save_matrix_csv(os.path.join(out_dir, "confusion_matrix_row_normalized.csv"), cm_norm, class_names)

    # Plots
    _plot_confusion(cm, class_names, os.path.join(out_dir, "confusion_matrix_raw.png"), "Matriz de confusión (recuento)")
    _plot_confusion(cm_norm, class_names, os.path.join(out_dir, "confusion_matrix_row_normalized.png"), "Matriz de confusión (normalizada por fila)")

    # === Metrics ===
    metrics = _per_class_metrics(cm)
    # add accuracy
    acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
    metrics["accuracy"] = acc

    # Save JSON
    with open(os.path.join(out_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save per-class CSV
    with open(os.path.join(out_dir, "classification_report_per_class.csv"), "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1,support\n")
        for i, name in enumerate(class_names):
            m = metrics["per_class"][i]
            f.write(f"{name},{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['support']}\n")

    # F1 bar chart
    f1_vals = [m["f1"] for m in metrics["per_class"]]
    _plot_bar(f1_vals, class_names, os.path.join(out_dir, "f1_per_class.png"), "F1 por clase", "F1")

    # Precision bar chart (resumen de precisión por clase)
    prec_vals = [m["precision"] for m in metrics["per_class"]]
    _plot_bar(prec_vals, class_names, os.path.join(out_dir, "precision_per_class.png"), "Precisión por clase", "Precisión")

    # === Label distribution (train/val/test) if available ===
    dist_path = os.path.join(cfg.OUTPUT_DIR, "label_distribution.json")
    if os.path.exists(dist_path):
        with open(dist_path, "r", encoding="utf-8") as f:
            dist = json.load(f)
        for split in ["train", "val", "test"]:
            counts = dist.get(split)
            if counts is None: 
                continue
            total = max(1, sum(counts))
            frac = [c/total for c in counts]
            _plot_bar(counts, class_names, os.path.join(out_dir, f"class_count_{split}.png"), f"Distribución de clases — {split}", "recuento")
            _plot_bar(frac, class_names, os.path.join(out_dir, f"class_fraction_{split}.png"), f"Distribución (fracción) — {split}", "fracción")

    # === Top confusions ===
    topk = _top_confusions(cm, class_names, k=20)
    with open(os.path.join(out_dir, "top_confusions.csv"), "w", encoding="utf-8") as f:
        f.write("true,pred,count\n")
        for t, p, c in topk:
            f.write(f"{t},{p},{c}\n")

    # README
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Artefactos generados:\n"
            "- confusion_matrix_raw.png / confusion_matrix_raw.csv\n"
            "- confusion_matrix_row_normalized.png / confusion_matrix_row_normalized.csv\n"
            "- classification_report.json / classification_report_per_class.csv\n"
            "- f1_per_class.png\n"
            "- class_count_*.png / class_fraction_*.png (si hay label_distribution.json)\n"
            "- top_confusions.csv\n"
        )
    print(f"[PLOTS] Artefactos guardados en: {out_dir}")
