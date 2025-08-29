# plots_nhits.py
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from main import Config

def generate_artifacts(cfg: Config, results: dict):
    """
    Crea un PDF resumen + ficheros auxiliares a partir de los paths de `results`.
    Además, guarda:
      - PNGs individuales de las 10 muestras en outputs/plots/
      - PNG de la página de métricas por horizonte en outputs/plots/horizon_metrics.png
    """
    out_dir = cfg.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    pdf_path = os.path.join(out_dir, "summary_nhits.pdf")
    with PdfPages(pdf_path) as pdf:
        # Página 1 — resumen
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_subplot(111)
        ax.axis('off')
        lines = [
            "N-HiTS Multivariable — Resumen",
            "",
            f"CSV: {cfg.CSV_PATH}",
            f"Target (z): {cfg.TARGET_COL_NORM}  → real: {cfg.ORIG_TARGET_COL}  (train μ={results['denorm_mean']:.3f}, σ={results['denorm_std']:.3f})",
            f"H={cfg.H}, L={cfg.L}",
            f"POOL_SIZES={cfg.POOL_SIZES}",
            f"WIDTH={cfg.HIDDEN_WIDTH}, DEPTH/BLOCK={cfg.DEPTH_PER_BLOCK}, BLOCKS/SCALE={cfg.BLOCKS_PER_SCALE}",
            f"EPOCHS={cfg.EPOCHS}, BATCH={cfg.BATCH_SIZE}, LR={cfg.LR}",
            f"AMP={cfg.USE_AMP}, TF32={cfg.ENABLE_TF32}, Patience={cfg.PATIENCE}",
            f"Features (D={len(results.get('feature_cols', []))}):",
            ", ".join(results.get('feature_cols', [])[:40]) + (" ..." if len(results.get('feature_cols', []))>40 else "")
        ]
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
        pdf.savefig(fig); plt.close(fig)

        # Página 2 — curvas de entrenamiento
        hist_df = pd.read_csv(results["history_csv"])
        fig = plt.figure(figsize=(8.3, 11.7))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
        ax1.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
        ax1.set_title("Loss (Huber ponderado)"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(hist_df["epoch"], hist_df["val_mae"], label="val_mae"); ax2.set_title("Val MAE (z)"); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax3.plot(hist_df["epoch"], hist_df["val_rmse"], label="val_rmse"); ax3.set_title("Val RMSE (z)"); ax3.legend(); ax3.grid(True, alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Página 3 — métricas por horizonte (doble línea: °C y z)
        if results.get("horizon_csv") and os.path.exists(results["horizon_csv"]):
            hdf = pd.read_csv(results["horizon_csv"])
            fig = plt.figure(figsize=(8.3, 11.7))
            a1 = fig.add_subplot(211)
            a2 = fig.add_subplot(212)

            # MAE: °C (primero, azul por defecto) + z (segundo, naranja por defecto)
            a1.plot(hdf["h"], hdf["mae_C"], label="MAE (°C)")
            a1.plot(hdf["h"], hdf["mae_z"], label="MAE (z)")
            a1.set_title("MAE por horizonte"); a1.set_xlabel("h (0..L-1)"); a1.grid(True, alpha=0.3); a1.legend()

            # RMSE: °C + z
            a2.plot(hdf["h"], hdf["rmse_C"], label="RMSE (°C)")
            a2.plot(hdf["h"], hdf["rmse_z"], label="RMSE (z)")
            a2.set_title("RMSE por horizonte"); a2.set_xlabel("h (0..L-1)"); a2.grid(True, alpha=0.3); a2.legend()

            fig.tight_layout()
            # Guardar como PNG además de añadir al PDF
            horizon_png_path = os.path.join(plots_dir, "horizon_metrics.png")
            fig.savefig(horizon_png_path, dpi=150)
            pdf.savefig(fig); plt.close(fig)

        # Página extra — Importancia de variables Top-K (si existe)
        if results.get("fi_csv") and os.path.exists(results["fi_csv"]):
            fidf = pd.read_csv(results["fi_csv"]).sort_values("delta_mse_%", ascending=False)
            topk = int(getattr(cfg, "TOPK_IMPORTANCE_PLOT", 5) or 5)
            fidf_top = fidf.head(topk)
            fig = plt.figure(figsize=(8.3, 5.5))
            ax = fig.add_subplot(111)
            ax.barh(list(fidf_top["feature"][::-1]), list(fidf_top["delta_mse_%"][::-1]))
            ax.set_xlabel("Δ MSE vs. base (%)")
            ax.set_title(f"Importancia por permutación — Top-{topk}")
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Páginas siguientes — muestras de test con histórico H (si existe)
        data = np.load(results["samples_npz"])
        preds = data["preds"]
        trues = data["trues"]
        hist = data["hist"] if "hist" in data.files else None
        K, L = preds.shape
        for k in range(K):
            fig = plt.figure(figsize=(8.3, 5.0))
            ax = fig.add_subplot(111)
            if hist is not None:
                Hh = hist.shape[1]
                ax.plot(range(-Hh, 0), hist[k], label=f"Histórico {Hh}h (°C)")
            ax.plot(range(0, L), trues[k], label="Real (°C)")
            ax.plot(range(0, L), preds[k], label="Pred (°C)")
            ax.axvline(0, ls="--", alpha=0.5)
            ax.set_title(f"Test sample {k+1}/{K} — Ventana H + Horizonte {L}h")
            ax.set_xlabel("h (horas; negativo=pasado, 0=frontera)")
            ax.set_ylabel("Temperature (°C)")
            ax.grid(True, alpha=0.3); ax.legend()
            plt.tight_layout()
            # Guardar PNG individual
            png_path = os.path.join(plots_dir, f"sample_{k+1:02d}.png")
            fig.savefig(png_path, dpi=150)
            pdf.savefig(fig); plt.close(fig)

    # summary.json (atajos)
    summary_json = os.path.join(out_dir, "summary.json")
    last = pd.read_csv(results["history_csv"]).iloc[-1].to_dict()
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": os.path.relpath(results["best_model_path"], out_dir),
            "history_csv": os.path.relpath(results["history_csv"], out_dir),
            "samples_npz": os.path.relpath(results["samples_npz"], out_dir),
            "horizon_csv": os.path.relpath(results["horizon_csv"], out_dir) if results.get("horizon_csv") else None,
            "feature_importance_csv": os.path.relpath(results["fi_csv"], out_dir) if results.get("fi_csv") else None,
            "last_epoch": last.get("epoch", None),
        }, f, indent=2, ensure_ascii=False)
