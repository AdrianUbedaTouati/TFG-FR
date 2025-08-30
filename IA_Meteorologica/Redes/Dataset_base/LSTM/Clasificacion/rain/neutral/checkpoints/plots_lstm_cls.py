# plots_lstm_cls.py
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional


def _cfg_get(cfg, name, default=None):
    return getattr(cfg, name, default) if hasattr(cfg, name) else cfg.get(name, default) if isinstance(cfg, dict) else default
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p


def _plot_history(ax, hist_df: pd.DataFrame):
    ax.plot(hist_df['epoch'], hist_df['train_loss'], label='train_loss')
    if 'val_ACC' in hist_df.columns:
        ax.plot(hist_df['epoch'], hist_df['val_ACC'], label='val_ACC')
    ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('Historial de entrenamiento (clasificación)')


def _plot_horizon(ax, hm_df: pd.DataFrame):
    ax.plot(hm_df['h'], hm_df['ACC'], label='Accuracy')
    ax.set_xlabel('Horizonte (h)'); ax.set_ylabel('ACC'); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('Accuracy por horizonte')


def _plot_feature_importance(ax, fi_df: pd.DataFrame, top_k: int = 15):
    fi = fi_df.sort_values('delta_acc', ascending=False).head(top_k)
    ax.barh(fi['feature'][::-1], fi['delta_acc'][::-1])
    ax.set_xlabel('Δ ACC vs baseline'); ax.set_title('Importancia por permutación'); ax.grid(True, axis='x', alpha=0.2)


def _decode_seq(ids, class_names):
    return [class_names[i] if 0 <= int(i) < len(class_names) else str(int(i)) for i in ids]


def _plot_sample(ax, sample, class_names, H: int, L: int, idx: int):
    # sample: {'y_true': [L], 'y_pred':[L], 'conf':[L]}
    y_t = _decode_seq(sample['y_true'], class_names)
    y_p = _decode_seq(sample['y_pred'], class_names)
    conf = [float(c) for c in sample.get('conf', [np.nan]*len(y_p))]
    lines = [f"h+{i+1:02d}: true={t:>18} | pred={p:>18} | conf={c:5.2f}" for i,(t,p,c) in enumerate(zip(y_t, y_p, conf))]
    txt = f"Muestra test #{idx}\n" + "\n".join(lines)
    ax.axis('off'); ax.text(0.01, 0.98, txt, va='top', family='monospace')


def generate_artifacts_cls(cfg, results):
    out_dir = _ensure_dir(_cfg_get(cfg, 'OUTPUT_DIR', 'outputs/lstm_summary_cls'))
    plots_dir = _ensure_dir(os.path.join(out_dir, 'plots'))

    hist_df = pd.read_csv(results['paths']['history_csv'])
    hm_df = pd.read_csv(results['paths']['horizon_csv'])
    samples_npz = np.load(results['paths']['samples_npz'], allow_pickle=True)
    samples = list(samples_npz['samples'])

    meta = results.get('meta', {})
    class_names = meta.get('class_names', [])
    best_epoch = results.get('best_epoch', '?')
    test_m = results.get('test_metrics', {})

    pdf_path = os.path.join(out_dir, 'report_lstm_summary_cls.pdf')
    with PdfPages(pdf_path) as pdf:
        # Portada
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
        txt = (
            "Resumen — LSTM Clasificación de Summary\n\n"
            f"Dataset: {os.path.basename(_cfg_get(cfg, 'CSV_PATH', 'data.csv'))}\n"
            f"Tiempo: {meta.get('datetime_col', None)} | Clases: {meta.get('num_classes', '?')}\n"
            f"Features ({len(meta.get('features', []))}): {meta.get('features', [])}\n\n"
            f"Ventana H={meta.get('H', '?')}, L={meta.get('L','?')}\n"
            f"Mejor epoch: {best_epoch}\n"
            f"Test: ACC={test_m.get('ACC', None):.4f}\n"
        )
        plt.text(0.05, 0.95, txt, va='top', fontsize=10, family='monospace')
        pdf.savefig(fig); plt.close(fig)

        # History
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 6))
        _plot_history(ax, hist_df); pdf.savefig(fig); plt.savefig(os.path.join(plots_dir, 'history_cls.png')); plt.close(fig)

        # Horizon
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 6))
        _plot_horizon(ax, hm_df); pdf.savefig(fig); plt.savefig(os.path.join(plots_dir, 'horizon_metrics_cls.png')); plt.close(fig)

        # Feature importance
        fi_path = results['paths'].get('feature_importance_csv', None)
        if fi_path and os.path.exists(fi_path):
            fi_df = pd.read_csv(fi_path)
            fig, ax = plt.subplots(1, 1, figsize=(11.69, 7))
            _plot_feature_importance(ax, fi_df, top_k=min(15, len(fi_df)))
            pdf.savefig(fig); plt.savefig(os.path.join(plots_dir, 'feature_importance_cls.png')); plt.close(fig)

        # Muestras
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        idxs = np.linspace(0, len(samples)-1, num=4, dtype=int)
        for ax, i in zip(axes.flatten(), idxs):
            _plot_sample(ax, samples[i].item() if hasattr(samples[i],'item') else samples[i], class_names, meta.get('H',336), meta.get('L',24), i)
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
            'class_index': os.path.relpath(results['paths'].get('class_index_json',''), out_dir),
            'plots_dir': os.path.relpath(plots_dir, out_dir),
        }
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {'pdf_path': pdf_path, 'summary_json': os.path.join(out_dir, 'summary.json')}
