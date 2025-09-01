# train_nhits.py
from __future__ import annotations
import os, math
from dataclasses import dataclass
from contextlib import nullcontext
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from main import Config


# ======================
# Dataset de ventanas
# ======================
class WindowedMultivar(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, H: int, L: int, start: int, end: int):
        assert X.shape[0] == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.H, self.L = H, L
        self.start, self.end = start, end
        # i s.t. [i-H,i) ⊂ [start,end) y [i,i+L) ⊂ [start,end)
        self.idxs = list(range(start + H, end - L + 1))

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, k: int):
        i = self.idxs[k]
        Xw = self.X[i-self.H:i, :]   # (H, D)
        yL = self.y[i:i+self.L]      # (L,)
        return Xw, yL, i

# ======================
# Modelo N-HiTS
# ======================
class NHITSBlock(nn.Module):
    def __init__(self, H: int, D: int, L: int, pool: int, width: int, depth: int):
        super().__init__()
        self.H, self.D, self.L, self.pool = H, D, L, max(1, int(pool))
        Lc = math.ceil(L / self.pool)
        in_features = (H // self.pool) * D
        layers, f = [], in_features
        for _ in range(depth):
            layers += [nn.Linear(f, width), nn.GELU(), nn.Dropout(p=0.1)]
            f = width
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(width, Lc)

    def forward(self, x):  # x: (B,H,D)
        B, H, D = x.shape
        if self.pool > 1:
            xp = F.avg_pool1d(x.transpose(1,2), kernel_size=self.pool, stride=self.pool).transpose(1,2)
        else:
            xp = x
        z = xp.reshape(B, -1)
        h = self.mlp(z)
        coarse = self.head(h).unsqueeze(1)  # (B,1,Lc)
        up = F.interpolate(coarse, size=self.L, mode="linear", align_corners=False)  # (B,1,L)
        return up.squeeze(1)  # (B,L)

class NHiTS(nn.Module):
    def __init__(self, H: int, D: int, L: int, pool_sizes: List[int], width: int=256, depth_per_block: int=2, blocks_per_scale: int=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for p in pool_sizes:
            for _ in range(blocks_per_scale):
                self.blocks.append(NHITSBlock(H, D, L, p, width, depth_per_block))
    def forward(self, x):  # x: (B,H,D)
        yhat = 0.0
        for b in self.blocks:
            yhat = yhat + b(x)
        return yhat

# ======================
# Loss y métricas
# ======================
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0, horizon_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.delta = delta
        self.w = horizon_weights
    def forward(self, y_hat, y_true):
        err = y_hat - y_true
        abs_e = torch.abs(err)
        quad = torch.clamp(abs_e, max=self.delta)
        lin  = abs_e - quad
        huber = 0.5 * quad**2 + self.delta * lin
        if self.w is not None:
            huber = huber * self.w.to(huber.device)
        return huber.mean()

@torch.no_grad()
def compute_metrics(y_hat, y_true):
    e = y_hat - y_true
    mse = (e**2).mean().item()
    mae = torch.abs(e).mean().item()
    rmse = math.sqrt(mse)
    return mse, mae, rmse

def evaluate(model, loader, device, criterion, use_amp=False):
    model.eval()
    total_loss, n_batches = 0.0, 0
    yhat_all, y_all = [], []
    ctx = torch.amp.autocast('cuda', enabled=(use_amp and device.type=='cuda')) if device.type=='cuda' else nullcontext()
    with ctx:
        for Xw, yL, _ in loader:
            Xw = Xw.to(device); yL = yL.to(device)
            yhat = model(Xw)
            loss = criterion(yhat, yL)
            total_loss += loss.item(); n_batches += 1
            yhat_all.append(yhat.detach()); y_all.append(yL.detach())
    if yhat_all:
        yh = torch.cat(yhat_all, dim=0); yt = torch.cat(y_all, dim=0)
        mse, mae, rmse = compute_metrics(yh, yt)
    else:
        mse = mae = rmse = float('nan')
    return total_loss / max(1, n_batches), mae, rmse

# ======================
# Data prep (usa FEATURE_COLS del cfg)
# ======================
def build_Xy_from_cfg(cfg: Config) -> Dict[str, Any]:
    df = pd.read_csv(cfg.CSV_PATH)
    if cfg.DATETIME_COL and cfg.DATETIME_COL in df.columns:
        df = df.sort_values(cfg.DATETIME_COL).reset_index(drop=True)

    # y en z-score desde el CSV
    if not cfg.TARGET_COL_NORM or cfg.TARGET_COL_NORM not in df.columns:
        raise ValueError(f"No encuentro TARGET_COL_NORM='{cfg.TARGET_COL_NORM}' en el CSV")
    if not cfg.ORIG_TARGET_COL or cfg.ORIG_TARGET_COL not in df.columns:
        raise ValueError(f"No encuentro ORIG_TARGET_COL='{cfg.ORIG_TARGET_COL}' en el CSV")
    y_z = df[cfg.TARGET_COL_NORM].astype(np.float32).to_numpy()

    # Selección de features
    feature_cols = list(cfg.FEATURE_COLS) if cfg.FEATURE_COLS else None
    if not feature_cols:
        # fallback: todas las numéricas salvo el target crudo
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in num_cols if c != cfg.ORIG_TARGET_COL]

    # Excluir el target-z si el usuario no lo quiere como feature
    if not cfg.INCLUDE_TARGET_AS_FEATURE and cfg.TARGET_COL_NORM in feature_cols:
        feature_cols = [c for c in feature_cols if c != cfg.TARGET_COL_NORM]

    # Validación de columnas
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Las siguientes columnas de FEATURE_COLS no existen en el CSV: {missing}")

    X = df[feature_cols].astype(np.float32).to_numpy()

    # Splits 70/15/15
    T = len(df); train_end = int(0.70*T); val_end = int(0.85*T)

    # Estadísticos para desnormalizar a °C (solo TRAIN)
    z_mean = float(df.loc[:train_end-1, cfg.ORIG_TARGET_COL].mean()) if cfg.Z_MEAN is None else float(cfg.Z_MEAN)
    z_std  = float(df.loc[:train_end-1, cfg.ORIG_TARGET_COL].std(ddof=0) or 1.0) if cfg.Z_STD is None else float(cfg.Z_STD)

    print(f"Denormalización (TRAIN only): mean={z_mean:.3f}, std={z_std:.3f}")
    print(f"Dataset: {cfg.CSV_PATH}")
    print(f"Tiempo: {cfg.DATETIME_COL} | Objetivo: {cfg.TARGET_COL_NORM}")
    print(f"Features ({len(feature_cols)}): {feature_cols[:50]}{' ...' if len(feature_cols)>50 else ''}")

    return dict(
        df=df, X=X, y_z=y_z, feature_cols=feature_cols,
        T=T, train_end=train_end, val_end=val_end, z_mean=z_mean, z_std=z_std
    )

# ======================
# Entrenamiento principal
# ======================
@dataclass
class TrainResult:
    history_csv: str
    best_model_path: str
    samples_npz: str
    fi_csv: Optional[str]
    horizon_csv: Optional[str]
    device: str
    feature_cols: List[str]
    denorm_mean: float
    denorm_std: float

def run_training(cfg: Config) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.ENABLE_TF32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Semillas
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED); torch.cuda.manual_seed_all(cfg.SEED)

    # Data
    data = build_Xy_from_cfg(cfg)
    df = data["df"]
    X, y_z = data["X"], data["y_z"]
    feat_cols = data["feature_cols"]
    H, L = cfg.H, cfg.L
    T, D = X.shape
    train_end, val_end = data["train_end"], data["val_end"]
    den_mean, den_std = data["z_mean"], data["z_std"]

    # Datasets y loaders
    train_ds = WindowedMultivar(X, y_z, H, L, start=0,         end=train_end)
    val_ds   = WindowedMultivar(X, y_z, H, L, start=train_end, end=val_end)
    test_ds  = WindowedMultivar(X, y_z, H, L, start=val_end,   end=T)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                              drop_last=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False,
                              drop_last=False, num_workers=0, pin_memory=False)

    # Info
    print(f"Total puntos: {T} — Train: {train_end}, Val: {val_end-train_end}, Test: {T-val_end}", flush=True)
    print(f"Ventanas => Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}", flush=True)
    gpu_name = torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'
    print(f"[INFO] device={device.type} | gpu={gpu_name} | amp={cfg.USE_AMP}", flush=True)

    # Modelo
    model = NHiTS(H=H, D=D, L=L,
                  pool_sizes=cfg.POOL_SIZES,
                  width=cfg.HIDDEN_WIDTH,
                  depth_per_block=cfg.DEPTH_PER_BLOCK,
                  blocks_per_scale=cfg.BLOCKS_PER_SCALE).to(device)

    # --- Resumen del modelo (antes de entrenar) ---
    def count_params(m): 
        return sum(p.numel() for p in m.parameters())
    total_params = count_params(model)
    n_blocks = len(cfg.POOL_SIZES) * cfg.BLOCKS_PER_SCALE
    print("[RESUMEN MODELO]",
          f"H={H}, L={L}, D={D} | bloques={n_blocks} | width={cfg.HIDDEN_WIDTH} | depth/block={cfg.DEPTH_PER_BLOCK}",
          f"| pool_sizes={cfg.POOL_SIZES}",
          f"| params_totales={total_params:,}", sep=" ")

    # Optimización
    w = torch.linspace(1.0, 0.6, L).float()
    criterion = WeightedHuberLoss(delta=1.0, horizon_weights=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.EPOCHS, 20), eta_min=cfg.LR*0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.USE_AMP and device.type=='cuda'))

    # Entreno
    out_dir = cfg.OUTPUT_DIR; os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(cfg.CHECKPOINT_DIR, f"nhits_MULTI_H{H}_L{L}_{cfg.TARGET_COL_NORM}_D{D}.pth")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    best_val_mae = float("inf")
    history = []; no_improve = 0
    for epoch in range(1, cfg.EPOCHS+1):
        model.train()
        tr_loss, nb = 0.0, 0
        for Xw, yL, _ in train_loader:
            Xw = Xw.to(device); yL = yL.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(cfg.USE_AMP and device.type=='cuda')):
                yhat = model(Xw); loss = criterion(yhat, yL)
            scaler.scale(loss).backward()
            if cfg.GRAD_CLIP and cfg.GRAD_CLIP>0:
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            tr_loss += loss.item(); nb += 1
        scheduler.step()
        tr_loss /= max(1, nb)

        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device, criterion, use_amp=cfg.USE_AMP)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, "val_mae": val_mae, "val_rmse": val_rmse, "lr": float(optimizer.param_groups[0]["lr"])})
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | val_loss={val_loss:.6f} | val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

        if val_mae < best_val_mae - 1e-7:
            best_val_mae = val_mae
            torch.save({"state_dict": model.state_dict(), "feature_cols": feat_cols, "cfg": cfg.__dict__}, best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.PATIENCE:
                break

    # Cargar mejor checkpoint
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # === Evaluación en TEST (después de cargar el mejor checkpoint) ===
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, device, criterion, use_amp=cfg.USE_AMP)
    print(f"Test: loss={test_loss:.6f} | MAE={test_mae:.4f} | RMSE={test_rmse:.4f}", flush=True)

    # Guardar history
    hist_csv = os.path.join(out_dir, "history.csv")
    pd.DataFrame(history).to_csv(hist_csv, index=False)

    # Muestras test (en °C) + histórico H (°C)
    K = min(10, len(test_ds))
    step = max(1, len(test_ds)//max(1, K))
    sample_idx = [i for i in range(0, len(test_ds), step)][:K]
    preds_C, trues_C, hist_C = [], [], []
    model.eval()
    with torch.no_grad():
        for k in sample_idx:
            Xw, yL, i_abs = test_ds[k]                        # numpy + índice absoluto
            x = torch.from_numpy(Xw).unsqueeze(0).to(device)
            yhat_z = model(x).squeeze(0).cpu().numpy()
            ytrue_z = yL
            preds_C.append(yhat_z * den_std + den_mean)
            trues_C.append(ytrue_z * den_std + den_mean)
            # histórico H horas en °C desde la columna cruda
            hist_series = df[cfg.ORIG_TARGET_COL].to_numpy()[i_abs-cfg.H:i_abs]
            hist_C.append(hist_series.astype("float32"))
    samples_npz = os.path.join(out_dir, "samples_test.npz")
    np.savez_compressed(samples_npz,
                        preds=np.array(preds_C, dtype=np.float32),
                        trues=np.array(trues_C, dtype=np.float32),
                        hist=np.array(hist_C, dtype=np.float32),
                        horizon=np.arange(L))

    # Métricas por horizonte (z y °C)
    yhat_all_z, y_all_z = [], []
    with torch.no_grad():
        for Xw, yL, _ in test_loader:
            yh = model(Xw.to(device)).cpu().numpy()
            y  = yL.numpy()
            yhat_all_z.append(yh); y_all_z.append(y)
    if yhat_all_z:
        YH = np.concatenate(yhat_all_z, axis=0); YT = np.concatenate(y_all_z, axis=0)
        R  = YT - YH
        mae_per_h_z  = np.mean(np.abs(R), axis=0)
        rmse_per_h_z = np.sqrt(np.mean(R**2, axis=0))
        mae_per_h_C  = mae_per_h_z * den_std
        rmse_per_h_C = rmse_per_h_z * den_std
        horizon_csv = os.path.join(out_dir, "horizon_metrics.csv")
        pd.DataFrame({
            "h": np.arange(L, dtype=int),
            "mae_z": mae_per_h_z, "rmse_z": rmse_per_h_z,
            "mae_C": mae_per_h_C, "rmse_C": rmse_per_h_C
        }).to_csv(horizon_csv, index=False)
    else:
        horizon_csv = None

    
    # Importancia por permutación (val)
    fi_csv = None
    if cfg.COMPUTE_FEATURE_IMPORTANCE:
        base_loss, _, _ = evaluate(model, val_loader, device, nn.MSELoss(), use_amp=cfg.USE_AMP)
        rng = np.random.default_rng(cfg.SEED)
        deltas = []
        # NOTA: en Windows, crear DataLoaders con num_workers>0 dentro de un bucle
        # dispara re-spawns que re-importan módulos y ralentizan/ensucian la salida.
        # Por eso aquí forzamos num_workers=0 para FI.
        for j, col in enumerate(feat_cols):
            Xp = X.copy()
            idx = np.arange(train_end, val_end); rng.shuffle(idx)
            Xp[train_end:val_end, j] = Xp[idx, j]
            val_ds_p = WindowedMultivar(Xp, y_z, H, L, start=train_end, end=val_end)
            val_loader_p = DataLoader(val_ds_p, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                      drop_last=False, num_workers=0, pin_memory=False)
            loss_p, _, _ = evaluate(model, val_loader_p, device, nn.MSELoss(), use_amp=False)
            deltas.append({"feature": col, "delta_mse": float(loss_p - base_loss)})
        fi_df = pd.DataFrame(deltas).sort_values("delta_mse", ascending=False)
        fi_df["delta_mse_%"] = 100.0 * fi_df["delta_mse"] / max(1e-12, base_loss)
        fi_csv = os.path.join(out_dir, "feature_importance.csv")
        fi_df.to_csv(fi_csv, index=False)


    results = dict(test_loss=float(test_loss), test_mae=float(test_mae), test_rmse=float(test_rmse), 
        history_csv=hist_csv,
        best_model_path=best_path,
        samples_npz=samples_npz,
        fi_csv=fi_csv,
        horizon_csv=horizon_csv,
        device=str(device),
        feature_cols=feat_cols,
        denorm_mean=den_mean,
        denorm_std=den_std
    )
    return results
