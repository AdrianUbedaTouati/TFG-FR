
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional

def _cfg_get(cfg, name, default=None):
    return getattr(cfg, name, default) if hasattr(cfg, name) else cfg.get(name, default) if isinstance(cfg, dict) else default

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _plot_history(ax, hist_df: pd.DataFrame):
    ax.plot(hist_df['epoch'], hist_df['train_loss'], label='train_loss')
    ax.plot(hist_df['epoch'], hist_df['val_MAE'], label='val_MAE')
    ax.plot(hist_df['epoch'], hist_df['val_RMSE'], label='val_RMSE')
    ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('Historial de entrenamiento')

def _plot_horizon(ax, hm_df: pd.DataFrame):
    if 'MAE_C' in hm_df.columns and 'RMSE_C' in hm_df.columns:
        ax.plot(hm_df['h'], hm_df['MAE_C'], label='MAE (°C)')
        ax.plot(hm_df['h'], hm_df['RMSE_C'], label='RMSE (°C)')
        ax.set_ylabel('Error (°C)')
    else:
        ax.plot(hm_df['h'], hm_df['MAE_z'], label='MAE (z)')
        ax.plot(hm_df['h'], hm_df['RMSE_z'], label='RMSE (z)')
        ax.set_ylabel('Error (z)')
    ax.set_xlabel('Horizonte (h)')
    ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('Métricas por horizonte')

def _plot_feature_importance(ax, fi_df: pd.DataFrame, top_k: int = 15):
    fi = fi_df.sort_values('delta_mse', ascending=False).head(top_k)
    ax.barh(fi['feature'][::-1], fi['delta_mse'][::-1])
    ax.set_xlabel('Δ MSE vs baseline'); ax.set_title('Importancia por permutación'); ax.grid(True, axis='x', alpha=0.2)

def _plot_sample(ax, sample, hist_tail_C: Optional[np.ndarray], H: int, L: int, idx: int):
    if 'y_true_C' in sample and sample['y_true_C'] is not None:
        y_t = np.array(sample['y_true_C']); y_p = np.array(sample['y_pred_C']); unit = '°C'
    else:
        y_t = np.array(sample['y_true_z']); y_p = np.array(sample['y_pred_z']); unit = 'z'
    t_hist = np.arange(-H, 0); t_fut = np.arange(0, L)
    if hist_tail_C is not None and idx < len(hist_tail_C) and hist_tail_C[idx] is not None:
        ax.plot(t_hist, hist_tail_C[idx], alpha=0.8, label=f'Histórico ({unit})')
    ax.plot(t_fut, y_t, label=f'Real ({unit})')
    ax.plot(t_fut, y_p, label=f'Pred ({unit})', linestyle='--')
    ax.set_title(f'Muestra test #{idx}'); ax.set_xlabel('Horas desde t0')
    ax.grid(True, alpha=0.3); ax.legend()

def _coerce_samples(arr):
    try:
        seq = list(arr)
    except Exception:
        seq = arr
    out = []
    for s in seq:
        if isinstance(s, dict):
            out.append(s)
        elif hasattr(s, 'item'):
            obj = s.item()
            out.append(obj if isinstance(obj, dict) else obj)
        else:
            out.append(s)
    return out

def generate_artifacts(cfg, results):
    out_dir = _ensure_dir(_cfg_get(cfg, 'OUTPUT_DIR', 'outputs'))
    plots_dir = _ensure_dir(os.path.join(out_dir, 'plots'))

    hist_df = pd.read_csv(results['paths']['history_csv'])
    hm_df = pd.read_csv(results['paths']['horizon_csv'])
    samples_npz = np.load(results['paths']['samples_npz'], allow_pickle=True)
    samples = _coerce_samples(samples_npz['samples'])
    hist_tail = samples_npz['hist_tail_C'] if 'hist_tail_C' in samples_npz else None
    den = results.get('denorm', {}); meta = results.get('meta', {})
    test_m = results.get('test_metrics', {})

    pdf_path = os.path.join(out_dir, 'report_lstm.pdf')
    with PdfPages(pdf_path) as pdf:
        # Portada
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
        txt = (
            "Resumen — LSTM Forecasting (mejorado)\n\n"
            f"Dataset: {os.path.basename(_cfg_get(cfg, 'CSV_PATH', 'data.csv'))}\n"
            f"Tiempo: {meta.get('datetime_col', None)} | Objetivo: {meta.get('target_norm','')}\n"
            f"Features ({len(meta.get('features', []))}): {meta.get('features', [])}\n\n"
            f"Denormalización (TRAIN): mean={den.get('mu', None)} std={den.get('sigma', None)}\n"
            f"Ventana H={meta.get('H', '?')}, L={meta.get('L','?')}\n"
            f"Mejor epoch: {results.get('best_epoch', '?')}\n"
            f"Test (z): MAE={test_m.get('MAE', None):.4f} | RMSE={test_m.get('RMSE', None):.4f}\n"
        )
        if test_m.get('MAE_C', None) is not None and test_m.get('RMSE_C', None) is not None:
            txt += f"Test (°C): MAE={test_m['MAE_C']:.3f} | RMSE={test_m['RMSE_C']:.3f}\n"
        plt.text(0.05, 0.95, txt, va='top', fontsize=10, family='monospace')
        pdf.savefig(fig); plt.close(fig)

        # History
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 6))
        _plot_history(ax, hist_df); pdf.savefig(fig); plt.savefig(os.path.join(plots_dir, 'history_lstm.png')); plt.close(fig)

        # Horizon
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 6))
        _plot_horizon(ax, hm_df); pdf.savefig(fig); plt.savefig(os.path.join(plots_dir, 'horizon_metrics_lstm.png')); plt.close(fig)

        # Feature importance
        fi_path = results['paths'].get('feature_importance_csv', None)
        if fi_path and os.path.exists(fi_path):
            fi_df = pd.read_csv(fi_path)
            fig, ax = plt.subplots(1, 1, figsize=(11.69, 7))
            _plot_feature_importance(ax, fi_df, top_k=min(15, len(fi_df)))
            pdf.savefig(fig); plt.savefig(os.path.join(plots_dir, 'feature_importance_lstm.png')); plt.close(fig)

        # Muestras
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        idxs = np.linspace(0, len(samples)-1, num=4, dtype=int)
        for ax, i in zip(axes.flatten(), idxs):
            _plot_sample(ax, samples[i], hist_tail, meta.get('H', 336), meta.get('L', 24), i)
        pdf.savefig(fig); plt.close(fig)

    summary = {
        'best_epoch': results.get('best_epoch'),
        'test_metrics': results.get('test_metrics'),
        'artifacts': {
            'pdf': os.path.relpath(pdf_path, out_dir),
            'history': os.path.relpath(results['paths']['history_csv'], out_dir),
            'horizon': os.path.relpath(results['paths']['horizon_csv'], out_dir),
            'samples': os.path.relpath(results['paths']['samples_npz'], out_dir),
            'feature_importance': os.path.relpath(results['paths'].get('feature_importance_csv',''), out_dir),
            'plots_dir': os.path.relpath(plots_dir, out_dir),
        }
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {'pdf_path': pdf_path, 'summary_json': os.path.join(out_dir, 'summary.json')}
