#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_cloud_cover.py (v2)
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd_list):
    print("[CMD]", " ".join(cmd_list))
    res = subprocess.run(cmd_list, stdout=sys.stdout, stderr=sys.stderr, check=True)
    return res.returncode

def main(args):
    here = Path(__file__).resolve().parent
    norm_script = args.normalize_script or str(here / "normalize_cloud_cover.py")
    train_script = args.train_script or str(here / "train_mlp_cloud_cover.py")

    # 1) NORMALIZACIÓN
    if not args.skip_normalize:
        cmd_norm = [
            sys.executable, norm_script,
            "--input", args.input,
            "--outdir", args.prep_dir,
            "--target", args.target,
            "--random-state", str(args.random_state),
        ]
        if args.no_derived: cmd_norm.append("--no-derived")
        run_cmd(cmd_norm)
    else:
        print("[INFO] Saltando normalización por --skip-normalize")

    # 2) ENTRENAMIENTO
    if not args.only_normalize:
        cmd_train = [
            sys.executable, train_script,
            "--data-dir", args.prep_dir,
            "--outdir", args.results_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--hidden", args.hidden,
            "--dropout", str(args.dropout),
            "--patience", str(args.patience),
            "--random-state", str(args.random_state),
            "--loss", args.loss,
            "--gamma", str(args.gamma),
            "--cb-beta", str(args.cb_beta),
            "--sampler", args.sampler,
            "--sampler-alpha", str(args.sampler_alpha),
            "--clip-norm", str(args.clip_norm),
            "--label-smoothing", str(args.label_smoothing),
        ]
        if args.cpu: cmd_train.append("--cpu")
        run_cmd(cmd_train)
    else:
        print("[INFO] Modo --only-normalize: entrenamiento omitido.")

def parse_args():
    ap = argparse.ArgumentParser()
    # Normalización
    ap.add_argument("--input", type=str, default="/mnt/data/weather_classification_data.csv")
    ap.add_argument("--prep-dir", type=str, default="/mnt/data/cloud_cover_prepared")
    ap.add_argument("--target", type=str, default="Cloud Cover")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--no-derived", action="store_true")
    ap.add_argument("--skip-normalize", action="store_true")
    ap.add_argument("--only-normalize", action="store_true")
    ap.add_argument("--normalize-script", type=str, default=None)
    # Entrenamiento
    ap.add_argument("--results-dir", type=str, default="/mnt/data/cloud_cover_results")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--train-script", type=str, default=None)
    # NUEVOS (passthrough)
    ap.add_argument("--loss", type=str, default="focal", choices=["ce", "focal", "cb_ce", "cb_focal"])
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--cb-beta", type=float, default=0.999)
    ap.add_argument("--sampler", type=str, default="weighted", choices=["none", "weighted"])
    ap.add_argument("--sampler-alpha", type=float, default=1.0)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
