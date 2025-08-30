
"""
normalize_weather.py
Normaliza el dataset original de Kaggle y genera features útiles para clasificar Summary
en 3 clases "coarse": Cloudy / Clear / Foggy.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

def minmax_norm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    smin, smax = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(smin) or pd.isna(smax) or smax <= smin:
        # fallback a zscore
        mu, sd = s.mean(skipna=True), s.std(skipna=True)
        if pd.isna(mu) or pd.isna(sd) or sd == 0:
            return s.fillna(0.0).astype(float)
        return ((s - mu) / (sd + 1e-9)).fillna(0.0).astype(float)
    return ((s - smin) / (smax - smin)).fillna(0.0).astype(float)

def cyclical_enc(angle_deg: pd.Series):
    rad = np.deg2rad(pd.to_numeric(angle_deg, errors="coerce").fillna(0.0))
    return np.sin(rad).astype(float), np.cos(rad).astype(float)

def time_features(dt: pd.Series):
    d = pd.to_datetime(dt, errors="coerce", utc=True)
    hour = d.dt.hour.fillna(0).astype(int)
    dow  = d.dt.dayofweek.fillna(0).astype(int)  # 0=lunes
    doy  = d.dt.dayofyear.fillna(1).astype(int)  # 1..366
    h_sin = np.sin(2*np.pi*hour/24.0)
    h_cos = np.cos(2*np.pi*hour/24.0)
    dow_sin = np.sin(2*np.pi*dow/7.0)
    dow_cos = np.cos(2*np.pi*dow/7.0)
    doy_sin = np.sin(2*np.pi*doy/366.0)
    doy_cos = np.cos(2*np.pi*doy/366.0)
    return h_sin, h_cos, dow_sin, dow_cos, doy_sin, doy_cos

def summary_to_coarse(s: str) -> str:
    if not isinstance(s, str):
        return "Cloudy"
    sl = s.lower()
    if "clear" in sl:
        return "Clear"
    if any(k in sl for k in ["fog", "mist", "haze"]):
        return "Foggy"
    return "Cloudy"

def build_onehot_from_text(series: pd.Series, prefix="Summary_") -> pd.DataFrame:
    vals = series.fillna("__nan__").astype(str)
    cats = pd.Index(sorted(vals.unique()))
    mat = pd.get_dummies(vals)
    mat = mat.reindex(columns=cats, fill_value=0)
    mat.columns = [prefix + c for c in mat.columns]
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_csv", type=str, default="weatherHistory_original.csv")
    ap.add_argument("--out", dest="output_csv", type=str, default="weatherHistory_normalize.csv")
    args = ap.parse_args()

    inp = Path(args.input_csv)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    # Time features
    if "Formatted Date" not in df.columns:
        raise ValueError("El CSV debe incluir la columna 'Formatted Date'.")
    h_sin, h_cos, dow_sin, dow_cos, doy_sin, doy_cos = time_features(df["Formatted Date"])
    df["h_sin"] = h_sin; df["h_cos"] = h_cos
    df["dow_sin"] = dow_sin; df["dow_cos"] = dow_cos
    df["doy_sin"] = doy_sin; df["doy_cos"] = doy_cos

    # Wind bearing sin/cos
    if "Wind Bearing (degrees)" in df.columns:
        s, c = cyclical_enc(df["Wind Bearing (degrees)"])
    elif "Wind Bearing (degrees)_normalized" in df.columns:
        deg = 360.0 * pd.to_numeric(df["Wind Bearing (degrees)_normalized"], errors="coerce").fillna(0.0)
        s, c = cyclical_enc(deg)
    else:
        s = pd.Series(0.0, index=df.index); c = pd.Series(1.0, index=df.index)
    df["wind_bearing_sin"] = s; df["wind_bearing_cos"] = c

    # Humidity
    if "Humidity" in df.columns:
        df["Humidity_normalized"] = pd.to_numeric(df["Humidity"], errors="coerce").clip(0,1).fillna(0.0)
    else:
        df["Humidity_normalized"] = df.get("Humidity_normalized", 0.0)

    # Precip Type -> numeric
    pt = df.get("Precip Type", pd.Series(["none"] * len(df))).astype(str).str.lower()
    pt_num = pt.replace({"rain": 0.5, "snow": 1.0}).where(pt.isin(["rain","snow"]), 0.0).astype(float)
    df["Precip Type_normalized"] = pt_num.fillna(0.0)

    # Wind Speed
    col_ws = "Wind Speed (km/h)" if "Wind Speed (km/h)" in df.columns else "Wind Speed (km/h)_normalized"
    df["Wind Speed (km/h)_normalized"] = minmax_norm(df[col_ws])

    # Visibility
    col_v = "Visibility (km)" if "Visibility (km)" in df.columns else "Visibility (km)_normalized"
    df["Visibility (km)_normalized"] = minmax_norm(df[col_v])

    # Pressure
    col_p = "Pressure (millibars)" if "Pressure (millibars)" in df.columns else "Pressure (millibars)_normalized"
    df["Pressure (millibars)_normalized"] = minmax_norm(df[col_p])

    # Temperature
    col_t = "Temperature (C)" if "Temperature (C)" in df.columns else "Temperature (C)_normalized"
    col_at = "Apparent Temperature (C)" if "Apparent Temperature (C)" in df.columns else "Apparent Temperature (C)_normalized"
    df["Temperature (C)_normalized"] = minmax_norm(df[col_t])
    df["Apparent Temperature (C)_normalized"] = minmax_norm(df[col_at])

    # Trend (rolling 24 sobre temp ordenada cronológicamente)
    order = pd.to_datetime(df["Formatted Date"], errors="coerce", utc=True).argsort().values
    temp_series = pd.to_numeric(df.loc[order, col_t], errors="coerce")
    trend = temp_series.rolling(window=24, min_periods=1).mean()
    trend_norm = minmax_norm(trend)
    trend_full = pd.Series(index=df.index, dtype=float)
    trend_full.loc[order] = trend_norm.values
    df["trend_normalized"] = trend_full.ffill().bfill().fillna(0.5)

    # Summary y coarse
    if "Summary" not in df.columns and any(c.startswith("Summary_") for c in df.columns):
        oh = df[[c for c in df.columns if c.startswith("Summary_")]]
        df["Summary"] = oh.idxmax(axis=1).str.replace("Summary_","", regex=False)
    df["Summary"] = df["Summary"].astype(str)
    df["Summary_coarse"] = df["Summary"].apply(summary_to_coarse)

    # One-hot opcional
    oh = build_onehot_from_text(df["Summary"])
    df = pd.concat([df, oh], axis=1)

    # Salida
    base_cols = ["Formatted Date","Summary","Summary_coarse","Precip Type","Daily Summary"]
    feat_norm_cols = [
        "h_sin","h_cos","dow_sin","dow_cos","doy_sin","doy_cos",
        "Precip Type_normalized","Humidity_normalized",
        "Wind Speed (km/h)_normalized","wind_bearing_sin","wind_bearing_cos",
        "Visibility (km)_normalized","Pressure (millibars)_normalized",
        "trend_normalized",
        "Temperature (C)_normalized","Apparent Temperature (C)_normalized"
    ]
    raw_keep = [c for c in ["Temperature (C)","Apparent Temperature (C)","Humidity",
                            "Wind Speed (km/h)","Wind Bearing (degrees)","Visibility (km)","Pressure (millibars)"]
                if c in df.columns]
    oh_cols = [c for c in df.columns if c.startswith("Summary_")]
    out_cols = [c for c in base_cols if c in df.columns] + raw_keep + feat_norm_cols + oh_cols
    df_out = df[out_cols].copy()

    # Evitar columnas duplicadas por nombres repetidos
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]

    # Stats
    counts = df["Summary_coarse"].astype(str).value_counts(dropna=False).to_dict()
    total = int(df.shape[0])
    pct = {k: float(v) * 100.0 / total for k, v in counts.items()}

    stats = {
        "n_rows": total,
        "class_counts": counts,
        "class_percent": pct,
        "normalization_info": {
            "minmax_on_full_dataset": True,
            "trend_window": 24,
            "humidity_clipped_0_1": True,
            "precip_mapping": {"none": 0.0, "rain": 0.5, "snow": 1.0}
        }
    }

    df_out.to_csv(out, index=False)
    with open(out.with_suffix(".class_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] Guardado CSV normalizado en: {out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
