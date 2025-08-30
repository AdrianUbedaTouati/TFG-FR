from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _get_paths(cfg, results: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    out_dir = getattr(cfg, "OUTPUT_DIR", "outputs")
    plots_dir = getattr(cfg, "_PLOTS_DIR", os.path.join(out_dir, "plots"))
    _ensure_dir(plots_dir)

    # Preferir los paths devueltos por el entrenamiento si existen
    if results and "paths" in results:
        hist_csv = results["paths"].get("history_csv", os.path.join(out_dir, "history.csv"))
        hor_csv = results["paths"].get("horizon_csv", os.path.join(out_dir, "horizon_metrics.csv"))
        samp_npz = results["paths"].get("samples_npz", os.path.join(out_dir, "samples_test.npz"))
        fi_csv = results["paths"].get("feature_importance_csv", os.path.join(out_dir, "feature_importance.csv"))
    else:
        hist_csv = os.path.join(out_dir, "history.csv")
        hor_csv = os.path.join(out_dir, "horizon_metrics.csv")
        samp_npz = os.path.join(out_dir, "samples_test.npz")
        fi_csv = os.path.join(out_dir, "feature_importance.csv")

    report_pdf = os.path.join(out_dir, "report_lstm.pdf")

    return {
        "OUT_DIR": out_dir,
        "PLOTS_DIR": plots_dir,
        "HISTORY_CSV": hist_csv,
        "HORIZON_CSV": hor_csv,
        "SAMPLES_NPZ": samp_npz,
        "FI_CSV": fi_csv,
        "REPORT_PDF": report_pdf,
    }


def _save_fig(fig, path_png: str, pdf: Optional[PdfPages] = None):
    fig.tight_layout()
    fig.savefig(path_png, dpi=140, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_history(history_csv: str, plots_dir: str, pdf: Optional[PdfPages]):
    df = pd.read_csv(history_csv)
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    if "train_loss" in df.columns:
        ax.plot(df["epoch"], df["train_loss"], label="train_loss")
    if "val_MAE" in df.columns:
        ax.plot(df["epoch"], df["val_MAE"], label="val_MAE")
    if "val_RMSE" in df.columns:
        ax.plot(df["epoch"], df["val_RMSE"], label="val_RMSE")
    ax.set_xlabel("Epoch")
    ax.set_title("Historial de entrenamiento")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_fig(fig, os.path.join(plots_dir, "01_history.png"), pdf)


def _plot_horizon_metrics(hor_csv: str, plots_dir: str, pdf: Optional[PdfPages]):
    df = pd.read_csv(hor_csv)
    # Z-score siempre está
    fig1 = plt.figure(figsize=(8, 4.8))
    ax1 = fig1.add_subplot(111)
    ax1.plot(df["h"], df["MAE_z"], marker="o", label="MAE (z)")
    ax1.plot(df["h"], df["RMSE_z"], marker="o", label="RMSE (z)")
    ax1.set_xlabel("Horizonte h")
    ax1.set_title("Métricas por horizonte (z-score)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    _save_fig(fig1, os.path.join(plots_dir, "02_horizon_z.png"), pdf)

    # Si hay columnas en °C y no son NaN, también las pintamos
    has_C = "MAE_C" in df.columns and "RMSE_C" in df.columns and df["MAE_C"].notna().any()
    if has_C:
        fig2 = plt.figure(figsize=(8, 4.8))
        ax2 = fig2.add_subplot(111)
        ax2.plot(df["h"], df["MAE_C"], marker="o", label="MAE (°C)")
        ax2.plot(df["h"], df["RMSE_C"], marker="o", label="RMSE (°C)")
        ax2.set_xlabel("Horizonte h")
        ax2.set_title("Métricas por horizonte (unidades reales)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        _save_fig(fig2, os.path.join(plots_dir, "03_horizon_real.png"), pdf)


def _plot_feature_importance(fi_csv: str, plots_dir: str, pdf: Optional[PdfPages], top_k: int = 20):
    if not os.path.exists(fi_csv):
        return
    df = pd.read_csv(fi_csv)
    df = df.sort_values("delta_mse", ascending=False).head(top_k)
    fig = plt.figure(figsize=(8, 5.2))
    ax = fig.add_subplot(111)
    ax.barh(df["feature"][::-1], df["delta_mse"][::-1])
    ax.set_title("Importancia por permutación (ΔMSE en VAL)")
    ax.set_xlabel("ΔMSE (mayor = más importante)")
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "04_feature_importance.png"), pdf)


def _load_samples(samples_npz: str) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    if not os.path.exists(samples_npz):
        return [], None
    data = np.load(samples_npz, allow_pickle=True)
    samples_arr = data["samples"]
    samples: List[Dict[str, Any]] = []
    for item in samples_arr:
        # cada item es un dict con y_true_z, y_pred_z, y_true_C (opcional), y_pred_C (opcional), idx
        samples.append(dict(item))
    hist_tail = None
    if "hist_tail_C" in data.files:
        hist_tail = data["hist_tail_C"]
        # puede ser None guardado como object
        if isinstance(hist_tail, np.ndarray) and hist_tail.dtype == object and hist_tail.size == 1:
            if hist_tail.item() is None:
                hist_tail = None
    return samples, hist_tail


def _plot_samples(samples_npz: str, plots_dir: str, pdf: Optional[PdfPages], max_plots: int = 4):
    samples, hist_tail = _load_samples(samples_npz)
    if not samples:
        return
    k = min(max_plots, len(samples))
    for i in range(k):
        s = samples[i]
        ytz = np.array(s["y_true_z"], dtype=float)
        ypz = np.array(s["y_pred_z"], dtype=float)

        # ¿tenemos unidades reales?
        have_C = ("y_true_C" in s) and ("y_pred_C" in s) and s["y_true_C"] is not None

        if have_C:
            ytC = np.array(s["y_true_C"], dtype=float)
            ypC = np.array(s["y_pred_C"], dtype=float)

        # Figura 1: z-score
        fig1 = plt.figure(figsize=(8, 4.8))
        ax1 = fig1.add_subplot(111)
        ax1.plot(ytz, label="true (z)")
        ax1.plot(ypz, label="pred (z)")
        ax1.set_title(f"Muestra #{s.get('idx', i)} (z-score)")
        ax1.set_xlabel("Paso en horizonte")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        _save_fig(fig1, os.path.join(plots_dir, f"10_sample_{i:02d}_z.png"), pdf)

        # Figura 2: unidades reales, si existen
        if have_C:
            fig2 = plt.figure(figsize=(8, 4.8))
            ax2 = fig2.add_subplot(111)
            ax2.plot(ytC, label="true (real)")
            ax2.plot(ypC, label="pred (real)")
            ax2.set_title(f"Muestra #{s.get('idx', i)} (unidades reales)")
            ax2.set_xlabel("Paso en horizonte")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            _save_fig(fig2, os.path.join(plots_dir, f"11_sample_{i:02d}_real.png"), pdf)

        # Figura 3 (opcional): histórico previo en reales, si se guardó
        if hist_tail is not None:
            # hist_tail: [K, H] alineado con las muestras seleccionadas
            if i < hist_tail.shape[0]:
                fig3 = plt.figure(figsize=(8, 3.0))
                ax3 = fig3.add_subplot(111)
                ax3.plot(hist_tail[i], lw=1.0)
                ax3.set_title(f"Histórico previo (C) — muestra #{s.get('idx', i)}")
                ax3.set_xlabel("Paso histórico")
                ax3.grid(True, alpha=0.3)
                _save_fig(fig3, os.path.join(plots_dir, f"12_sample_{i:02d}_history.png"), pdf)


def generate_artifacts(cfg, results: Optional[Dict[str, Any]] = None):
    paths = _get_paths(cfg, results)

    print("[PLOTS] Cargando artefactos desde:", json.dumps(paths, indent=2))

    pdf_path = paths["REPORT_PDF"]
    with PdfPages(pdf_path) as pdf:
        # 1) Historial entrenamiento
        if os.path.exists(paths["HISTORY_CSV"]):
            _plot_history(paths["HISTORY_CSV"], paths["PLOTS_DIR"], pdf)
        else:
            print(f"[PLOTS] No existe {paths['HISTORY_CSV']}")

        # 2) Métricas por horizonte
        if os.path.exists(paths["HORIZON_CSV"]):
            _plot_horizon_metrics(paths["HORIZON_CSV"], paths["PLOTS_DIR"], pdf)
        else:
            print(f"[PLOTS] No existe {paths['HORIZON_CSV']}")

        # 3) Importancia de features
        if os.path.exists(paths["FI_CSV"]):
            _plot_feature_importance(paths["FI_CSV"], paths["PLOTS_DIR"], pdf)
        else:
            print(f"[PLOTS] No existe {paths['FI_CSV']}")

        # 4) Muestras de test
        if os.path.exists(paths["SAMPLES_NPZ"]):
            _plot_samples(paths["SAMPLES_NPZ"], paths["PLOTS_DIR"], pdf, max_plots=4)
        else:
            print(f"[PLOTS] No existe {paths['SAMPLES_NPZ']}")

    print(f"[PLOTS] Reporte PDF guardado en: {pdf_path}")
    print(f"[PLOTS] PNGs en: {paths['PLOTS_DIR']}")
