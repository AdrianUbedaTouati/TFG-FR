"""
Meteostat – Observación horaria últimos 36 meses (tz-safe)
=========================================================

Arregla el error "Cannot compare tz-naive and tz-aware" convirtiendo las fechas
que se pasan a `meteostat` en **naive** (sin tz). La librería `meteostat` espera
`datetime` sin tz para `Stations.inventory()` y `Hourly()`.

Uso
---
1) pip install meteostat pandas
2) python 01_fetch_meteostat_alicante_v3.py --months-back 36 --lat 38.345 --lon -0.481 \
       --out "observations_meteostat_alicante_36m.csv"
   (o fija estación):
   python 01_fetch_meteostat_alicante_v3.py --months-back 36 --station 08359 --out "...csv"

Salida
------
CSV con columnas: time (UTC tz-aware), time_local (Europe/Madrid), temp, dwpt, rhum, prcp, snow,
                  wdir, wspd, wpgt, pres, tsun, coco
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from meteostat import Hourly, Stations, Point

LOCAL_TZ = "Europe/Madrid"
EXPECTED_COLS = [
    "temp","dwpt","rhum","prcp","snow","wdir","wspd","wpgt","pres","tsun","coco"
]


def compute_range_naive(months_back: int):
    # Hora actual UTC (tz-aware) redondeada a la hora
    end_aware = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Restar meses manualmente
    y = end_aware.year
    m = end_aware.month - months_back
    while m <= 0:
        m += 12
        y -= 1
    d = min(end_aware.day, [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
                             31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m-1])
    start_aware = end_aware.replace(year=y, month=m, day=d)
    # Devolver **naive** para meteostat + versiones aware sólo para impresión
    return start_aware.replace(tzinfo=None), end_aware.replace(tzinfo=None), start_aware, end_aware


def pick_station(lat: Optional[float], lon: Optional[float], station_id: Optional[str], start_naive, end_naive) -> str:
    if station_id:
        return station_id
    if lat is None or lon is None:
        raise SystemExit("Debes pasar --station o --lat y --lon")
    st = Stations().nearby(lat, lon)
    st = st.inventory('hourly', (start_naive, end_naive))
    df = st.fetch(10)
    if df.empty:
        raise SystemExit("No se encontraron estaciones con horario en el rango cerca del punto dado")
    chosen = df.index[0]
    print(f"Estación elegida: {chosen} – {df.loc[chosen, 'name']} ({df.loc[chosen, 'country']})")
    return chosen


def fetch_hourly_df(station_or_point, start_naive, end_naive, use_model: bool) -> pd.DataFrame:
    h = Hourly(station_or_point, start_naive, end_naive, model=use_model)
    df = h.fetch()
    if df.empty:
        raise SystemExit("Meteostat devolvió DataFrame vacío")
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    # índice → columnas, asegurar UTC tz-aware
    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    out = df.reset_index().rename(columns={'time': 'time'})
    if 'time' not in out.columns:
        out.rename(columns={out.columns[0]: 'time'}, inplace=True)
    out['time_local'] = out['time'].dt.tz_convert(LOCAL_TZ)
    cols = ['time','time_local'] + EXPECTED_COLS
    out = out[cols]
    out = out.drop_duplicates(subset=['time']).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Meteostat tz-safe – últimos N meses (Alicante/punto)")
    ap.add_argument('--months-back', type=int, default=36)
    ap.add_argument('--station', type=str, default=None, help='ID Meteostat (08359 para Alicante)')
    ap.add_argument('--lat', type=float, default=None)
    ap.add_argument('--lon', type=float, default=None)
    ap.add_argument('--out', type=str, default='observations_meteostat_alicante_36m.csv')
    ap.add_argument('--model', type=str, default='false', help='true/false – usar model fill para huecos')
    args = ap.parse_args()

    use_model = str(args.model).lower() in ('1','true','yes','y')

    start_naive, end_naive, start_aware, end_aware = compute_range_naive(args.months_back)
    print(f"Rango UTC: {start_aware.isoformat()} — {end_aware.isoformat()}")

    if args.station:
        station_id = args.station
        src = station_id
    else:
        station_id = pick_station(args.lat, args.lon, None, start_naive, end_naive)
        src = f"nearby({args.lat},{args.lon})→{station_id}"
    print(f"Fuente Meteostat: {src}")

    station_or_point = station_id if args.station else Point(args.lat, args.lon)
    df = fetch_hourly_df(station_or_point, start_naive, end_naive, use_model)

    df.to_csv(args.out, index=False)
    print(f"Guardado {len(df):,} filas → {args.out}")
    print(f"Primer ts: {df['time'].iloc[0]} | Último ts: {df['time'].iloc[-1]}")

if __name__ == '__main__':
    main()
