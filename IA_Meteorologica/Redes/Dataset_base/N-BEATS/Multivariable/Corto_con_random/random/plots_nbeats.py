
# plots_nbeats.py
from __future__ import annotations
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def generate_artifacts(cfg, R: Dict[str, Any]):
    """Crea gráficas individuales (ventanas espaciadas), importancia de variables, y un PDF resumen."""
    out_dir = cfg.OUTPUT_DIR; plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(plots_dir, exist_ok=True)

    df = R["df"]; dt_col = R["dt_col"]; tgt_col = R["tgt_col"]; feat_cols = R["feat_cols"]
    H, L = cfg.H, cfg.L
    X_all = R["X_all"]; y_all = R["y_all"]
    denorm_mean, denorm_std = R["denorm_mean"], R["denorm_std"]
    ys, yhats = R["ys"], R["yhats"]
    residuals = R["residuals"]
    mae_per_h, rmse_per_h = R["mae_per_h"], R["rmse_per_h"]
    ys_real, yhats_real = R["ys_real"], R["yhats_real"]
    residuals_real = R["residuals_real"]
    mae_per_h_real, rmse_per_h_real = R["mae_per_h_real"], R["rmse_per_h_real"]
    test_loss, test_mae, test_rmse = R["test_loss"], R["test_mae"], R["test_rmse"]
    D, N = R["D"], R["N"]
    splits = R["splits"]

    # Índice del objetivo dentro de los features
    target_in_feats = feat_cols.index(tgt_col) if tgt_col in feat_cols else None

    # Ventanas espaciadas del test (10)
    # Reconstruimos índices de test de forma determinista
    from train_nbeats import WindowedMultivar
    test_ds = WindowedMultivar(X_all, y_all, H, L, start=splits.val_end, end=splits.N)
    total_test_windows = len(test_ds.idxs)
    take_n = min(10, total_test_windows)
    if take_n > 0:
        sel_pos = np.linspace(0, total_test_windows - 1, take_n)
        sel_pos = np.unique(sel_pos.astype(int))
        while len(sel_pos) < take_n:
            missing = take_n - len(sel_pos)
            extras = np.arange(total_test_windows - missing, total_test_windows, dtype=int)
            sel_pos = np.unique(np.r_[sel_pos, extras])
        spaced_idxs = test_ds.idxs[sel_pos]
    else:
        spaced_idxs = np.array([], dtype=int)

    window_pngs = []
    for k, t in enumerate(spaced_idxs):
        x_hist = X_all[t : t + H, :]
        y_true = y_all[t + H : t + H + L]

        # Para graficar histórico del objetivo
        if target_in_feats is not None:
            hist_target = x_hist[:, target_in_feats]
        else:
            hist_target = y_all[t : t + H]

        # Denormalizaciones
        def d(arr):
            if cfg.DENORMALIZE_OUTPUTS and (denorm_mean is not None and denorm_std is not None):
                return arr * denorm_std + denorm_mean
            return arr

        # Fechas
        hist_times = df[dt_col].iloc[t : t + H].reset_index(drop=True)
        fut_times  = df[dt_col].iloc[t + H : t + H + L].reset_index(drop=True)
        start_time = hist_times.iloc[0]; end_time = fut_times.iloc[-1]

        # Para la pred, usamos el primero de yhats que le corresponde a esa ventana. (test_loader ordenado)
        # El índice relativo en test es la posición dentro de test_ds.idxs:
        rel_idx = np.where(test_ds.idxs == t)[0][0]
        y_pred = yhats[rel_idx]

        fig = plt.figure(figsize=(10, 4))
        times_concat = pd.concat([hist_times, fut_times])
        series_concat = np.concatenate([d(hist_target), d(y_true)])
        plt.plot(times_concat, series_concat, label="Real")
        plt.axvline(hist_times.iloc[-1], linestyle='--', linewidth=1)
        plt.plot(fut_times, d(y_pred), label="Predicción")
        title_txt = f"Ventana {k+1} — {start_time} → {end_time} (H={H} h)"
        plt.title(title_txt)
        y_label = cfg.ORIG_TARGET_COL if (cfg.DENORMALIZE_OUTPUTS and denorm_mean is not None and cfg.ORIG_TARGET_COL) else tgt_col
        plt.xlabel("Tiempo"); plt.ylabel(y_label); plt.legend(); plt.tight_layout()
        fname = f"test_spaced_MULTI_{k+1:02d}_{start_time:%Y%m%d_%H%M}_{end_time:%Y%m%d_%H%M}.png"
        fpath = os.path.join(plots_dir, fname)
        fig.savefig(fpath, dpi=150); plt.close(fig)
        window_pngs.append(fpath)

    # Importancia de variables
    imp_df = R.get("imp_df", None); imp_png = None
    if imp_df is not None and len(imp_df) > 0:
        # Guardar CSV
        imp_csv = os.path.join(out_dir, f"feature_importance_MULTI_H{H}_L{L}.csv")
        imp_df.to_csv(imp_csv, index=False)
        # Gráfico TOP-K
        topk = imp_df.head(cfg.TOPK_IMPORTANCE_PLOT)
        fig = plt.figure(figsize=(10, 5))
        plt.barh(topk['feature'][::-1], topk['delta_mse_%'][::-1])
        plt.xlabel("Incremento de MSE al permutar (%)")
        plt.title("Importancia de variables (validación, permutación)")
        plt.tight_layout()
        imp_png = os.path.join(plots_dir, f"feature_importance_TOP{cfg.TOPK_IMPORTANCE_PLOT}.png")
        fig.savefig(imp_png, dpi=150); plt.close(fig)

    # PDF de análisis
    pdf_path = os.path.join(out_dir, f"analysis_MULTI_H{H}_L{L}_{tgt_col.replace(' ','_')}_D{D}.pdf")
    with PdfPages(pdf_path) as pdf:
        # Resumen
        fig = plt.figure(figsize=(11, 8.5)); plt.axis('off')
        text = (
            f"N-BEATS Multivariable — Resumen\n"
            f"Archivo: {cfg.CSV_PATH}\n"
            f"Tiempo: {dt_col} | Objetivo: {tgt_col}\n"
            f"Features ({D}): {feat_cols}\n"
            f"Puntos totales: {N}\n"
            f"Split → Train: {splits.train_end}  | Val: {splits.val_end - splits.train_end}  | Test: {splits.N - splits.val_end}\n"
            f"Ventana: H={H}  L={L}  input_size={H*D}\n"
            f"Métricas test → MSE: {test_loss:.6f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}"
        )
        plt.text(0.05, 0.95, text, va='top', fontsize=12)
        pdf.savefig(fig); plt.close(fig)

        # Curvas de aprendizaje
        fig = plt.figure(figsize=(11, 4.5))
        plt.plot(range(1, len(R["train_hist"])+1), R["train_hist"], label='Train Loss')
        plt.plot(range(1, len(R["val_hist"])+1), R["val_hist"], label='Val Loss')
        plt.xlabel('Época'); plt.ylabel('MSE'); plt.title('Curvas de aprendizaje'); plt.legend(); plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Error por horizonte (z)
        fig = plt.figure(figsize=(11, 4.5))
        plt.plot(mae_per_h, label='MAE por horizonte (z)')
        plt.plot(rmse_per_h, label='RMSE por horizonte (z)')
        plt.xlabel('Paso (horas en el futuro)'); plt.ylabel('Error (z)')
        plt.title('Error por paso del horizonte (test, z-score)'); plt.legend(); plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Error por horizonte (real)
        if ys_real is not None:
            fig = plt.figure(figsize=(11, 4.5))
            plt.plot(mae_per_h_real, label='MAE por horizonte (real)')
            plt.plot(rmse_per_h_real, label='RMSE por horizonte (real)')
            plt.xlabel('Paso (horas en el futuro)'); plt.ylabel('Error (real)')
            plt.title('Error por paso del horizonte (test, escala real)'); plt.legend(); plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Distribución residuales (z)
        fig = plt.figure(figsize=(11, 4.5))
        plt.hist(residuals.flatten(), bins=50)
        plt.title('Distribución de residuales (test, z-score)'); plt.xlabel('Residual (z)'); plt.ylabel('Frecuencia'); plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Distribución residuales (real)
        if residuals_real is not None:
            fig = plt.figure(figsize=(11, 4.5))
            plt.hist(residuals_real.flatten(), bins=50)
            plt.title('Distribución de residuales (test, real)'); plt.xlabel('Residual (real)'); plt.ylabel('Frecuencia'); plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Promedio por paso (z)
        fig = plt.figure(figsize=(11, 4.5))
        plt.plot(np.mean(ys, axis=0), label='Real promedio (z)')
        plt.plot(np.mean(yhats, axis=0), label='Predicción promedio (z)')
        plt.xlabel('Paso (horas en el futuro)'); plt.ylabel('Valor (z)')
        plt.title('Promedio por paso del horizonte (test, z-score)'); plt.legend(); plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Promedio por paso (real)
        if ys_real is not None and yhats_real is not None:
            fig = plt.figure(figsize=(11, 4.5))
            plt.plot(np.mean(ys_real, axis=0), label='Real promedio (real)')
            plt.plot(np.mean(yhats_real, axis=0), label='Predicción promedio (real)')
            plt.xlabel('Paso (horas en el futuro)'); plt.ylabel('Valor (real)')
            plt.title('Promedio por paso del horizonte (test, real)'); plt.legend(); plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Importancia de variables tabla + barra
        if imp_df is not None and len(imp_df) > 0:
            topk = imp_df.head(cfg.TOPK_IMPORTANCE_PLOT)
            fig = plt.figure(figsize=(11, 4.5)); plt.axis('off')
            tbl = plt.table(cellText=np.round(topk[['delta_mse','delta_mse_%']].values, 6),
                            colLabels=['ΔMSE','ΔMSE %'],
                            rowLabels=topk['feature'], loc='center')
            tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.2)
            plt.title('Importancia de variables (TOP-K)')
            pdf.savefig(fig); plt.close(fig)

        # Ventanas espaciadas
        for fpath in window_pngs:
            img = plt.imread(fpath)
            fig = plt.figure(figsize=(11, 4.5)); plt.imshow(img); plt.axis('off'); pdf.savefig(fig); plt.close(fig)

        # Barra horizontal si existe
        if imp_png is not None:
            img = plt.imread(imp_png)
            fig = plt.figure(figsize=(11, 5)); plt.imshow(img); plt.axis('off'); pdf.savefig(fig); plt.close(fig)

    print("PDF guardado en:", pdf_path)
