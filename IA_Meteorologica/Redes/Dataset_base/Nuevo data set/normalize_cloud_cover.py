
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_cloud_cover.py

Prepara el dataset para clasificar `Cloud Cover` usando solo columnas presentes
en el CSV o derivadas directas de ellas. Genera:
- cloud_cover_normalized.csv   (dataset listo para entrenar)
- stats.json                   (estadísticas + configuración del pipeline)
- label_mapping.json           (mapeo etiqueta ↔ índice)

Uso (por defecto lee el archivo subido en /mnt/data):
    python normalize_cloud_cover.py \
        --input /mnt/data/weather_classification_data.csv \
        --outdir /mnt/data/cloud_cover_prepared \
        --target "Cloud Cover" \
        --random-state 42
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

# --------- Utilidades ----------

def minmax_normalize(series: pd.Series):
    s = series.astype(float)
    s_min = float(np.nanmin(s.values))
    s_max = float(np.nanmax(s.values))
    if np.isfinite(s_min) and np.isfinite(s_max) and s_max > s_min:
        norm = (s - s_min) / (s_max - s_min)
    else:
        # Columna constante o vacía → 0
        norm = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return norm, {"min": s_min, "max": s_max, "mean": float(np.nanmean(s.values)), "std": float(np.nanstd(s.values))}

def stratified_split(labels, train=0.8, val=0.1, test=0.1, seed=42):
    assert abs((train + val + test) - 1.0) < 1e-6, "Las proporciones deben sumar 1"
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)
    idx = np.arange(n)
    splits = np.empty(n, dtype=object)

    unique, inv = np.unique(labels, return_inverse=True)
    for cls_id, cls in enumerate(unique):
        cls_idx = idx[inv == cls_id]
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)
        n_train = int(round(train * n_cls))
        n_val = int(round(val * n_cls))
        # Ajuste para que todo sume n_cls
        n_train = min(n_train, n_cls)
        n_val = min(n_val, n_cls - n_train)
        n_test = n_cls - n_train - n_val
        splits[cls_idx[:n_train]] = "train"
        splits[cls_idx[n_train:n_train+n_val]] = "val"
        splits[cls_idx[n_train+n_val:]] = "test"
    return splits

def sanitize_colname(name: str) -> str:
    # Nombre de columna seguro para features derivadas
    return name.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct").replace("/", "_").replace("-", "_")

# --------- Pipeline ----------

def run(input_path: str,
        outdir: str,
        target_col: str = "Cloud Cover",
        random_state: int = 42,
        include_derived: bool = True):

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    orig_rows, orig_cols = df.shape

    # Validación etiqueta
    if target_col not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target_col}'. Columnas disponibles: {list(df.columns)}")

    # Columnas a excluir explícitamente como features
    EXCLUDE = {target_col, "Weather Type"}  # Evitamos usar otra etiqueta como feature

    # Detección de numéricas/categóricas
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE]
    # Columnas que frecuentemente son numéricas aunque vengan como object (por comas/puntos)
    likely_numeric = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)", "Atmospheric Pressure", "UV Index", "Visibility (km)"]
    for c in likely_numeric:
        if c in df.columns and c not in numeric_cols and c not in EXCLUDE:
            # Intento de conversión
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].dtype.kind in "fcui":
                numeric_cols.append(c)

    # Categóricas candidatas (definidas y/o inferidas)
    known_cats = [c for c in ["Season", "Location"] if c in df.columns]
    other_object = [c for c in df.select_dtypes(include=["object"]).columns if c not in EXCLUDE and c not in known_cats]
    # Por prudencia no incluimos otras object automáticas; sólo Season/Location
    cat_cols = known_cats

    # Limpieza básica (solo datos del dataset)
    if "Humidity" in df.columns:
        df["Humidity"] = df["Humidity"].clip(lower=0, upper=100)
    if "Precipitation (%)" in df.columns:
        df["Precipitation (%)"] = df["Precipitation (%)"].clip(lower=0, upper=100)

    # Derivadas (opcionales, todas basadas en columnas del dataset)
    derived_defs = {}
    if include_derived:
        if ("Temperature" in df.columns) and ("Humidity" in df.columns):
            # Aprox de punto de rocío (simple, sólo dataset)
            df["dew_point_approx"] = df["Temperature"] - (100 - df["Humidity"]) / 5.0
            derived_defs["dew_point_approx"] = {"formula": "Temperature - (100 - Humidity)/5.0"}

        if ("Humidity" in df.columns) and ("Precipitation (%)" in df.columns):
            # Interacción normalizada (se normaliza más abajo)
            # Guardamos cruda y normalizamos luego
            df["humid_x_precip_raw"] = df["Humidity"] * df["Precipitation (%)"]
            derived_defs["humid_x_precip_raw"] = {"formula": "Humidity * Precipitation (%)"}

    # Eliminar filas con NA en target
    df = df[~df[target_col].isna()].copy()
    df.reset_index(drop=True, inplace=True)

    # Normalización min-max de numéricas + derivadas
    to_normalize = list(numeric_cols)
    if include_derived:
        for c in ["dew_point_approx", "humid_x_precip_raw"]:
            if c in df.columns:
                to_normalize.append(c)

    norm_stats = {}
    norm_cols = []
    for c in to_normalize:
        norm, stats = minmax_normalize(df[c])
        norm_name = f"{sanitize_colname(c)}_normalized"
        df[norm_name] = norm
        norm_stats[c] = stats
        norm_cols.append(norm_name)

    # One-hot de categóricas (solo Season/Location si existen)
    one_hot_cols = []
    for c in cat_cols:
        # Limpiar texto
        df[c] = df[c].astype(str).str.strip()
        dummies = pd.get_dummies(df[c], prefix=sanitize_colname(c))
        for dc in dummies.columns:
            df[dc] = dummies[dc].astype("int8")
        one_hot_cols.extend(list(dummies.columns))

    # Label mapping
    labels = df[target_col].astype(str).str.strip().str.lower()
    unique_labels = sorted(labels.unique())
    label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
    index_to_label = {i: lab for lab, i in label_to_index.items()}
    df["label_str"] = labels
    df["label_idx"] = labels.map(label_to_index).astype("int64")

    # Split estratificado 80/10/10
    splits = stratified_split(df["label_idx"].values, train=0.8, val=0.1, test=0.1, seed=random_state)
    df["split"] = splits

    # Lista final de features sugeridas para el modelo
    feature_cols = list(norm_cols) + list(one_hot_cols)

    # Estadísticas y metadatos
    class_counts = Counter(df["label_str"])
    stats = {
        "input_path": str(Path(input_path).resolve()),
        "rows_in": int(orig_rows),
        "cols_in": int(orig_cols),
        "rows_out": int(len(df)),
        "target_col": target_col,
        "unique_labels": unique_labels,
        "class_distribution": dict(sorted(class_counts.items(), key=lambda kv: kv[0])),
        "splits": {k:int(v) for k,v in Counter(df["split"]).items()},
        "numeric_columns_used": numeric_cols,
        "categorical_columns_used": cat_cols,
        "one_hot_columns": one_hot_cols,
        "derived_features": list(derived_defs.keys()),
        "normalization": "minmax",
        "random_state": int(random_state),
        "normalization_stats_per_column": norm_stats,
        "feature_columns": feature_cols
    }

    # Guardar archivos
    out_csv = Path(outdir) / "cloud_cover_normalized.csv"
    out_stats = Path(outdir) / "stats.json"
    out_map = Path(outdir) / "label_mapping.json"

    df.to_csv(out_csv, index=False)
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    with open(out_map, "w", encoding="utf-8") as f:
        json.dump({"label_to_index": label_to_index, "index_to_label": index_to_label}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Guardado: {out_csv}")
    print(f"[OK] Guardado: {out_stats}")
    print(f"[OK] Guardado: {out_map}")
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="/mnt/data/weather_classification_data.csv")
    ap.add_argument("--outdir", type=str, default="/mnt/data/cloud_cover_prepared")
    ap.add_argument("--target", type=str, default="Cloud Cover")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--no-derived", action="store_true", help="Desactiva features derivadas")
    args = ap.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        outdir=args.outdir,
        target_col=args.target,
        random_state=args.random_state,
        include_derived=not args.no_derived
    )
