# -*- coding: utf-8 -*-
"""compare_models.py
====================
Compara los checkpoints almacenados en la carpeta ``models_finales/`` y
muestra qué red ofrece mayor precisión según la métrica elegida
(RMSE por defecto).

Uso
---
.. code-block:: bash

    # Comparar usando RMSE (menor es mejor)
    python compare_models.py --metric rmse

    # Comparar por R² (mayor es mejor)
    python compare_models.py --metric r2 --plot

    # Guardar la tabla en CSV
    python compare_models.py --csv resultados.csv

El script puede llamarse desde ``train_weather_models.py`` mediante la
opción «Comparar modelos guardados» del menú.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch

# -----------------------------------------------------------------------------
# Constantes y utilidades ------------------------------------------------------
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models_finales"

VALID_METRICS = {"rmse": "menor", "mae": "menor", "r2": "mayor"}


def load_checkpoints() -> List[dict]:
    """Carga todos los archivos ``.pt`` de *models_finales/*."""
    if not MODELS_DIR.exists():
        print("⚠️  La carpeta 'models_finales' no existe o está vacía.")
        sys.exit(1)

    ckpts = []
    for file in sorted(MODELS_DIR.glob("*.pt")):
        try:
            ckpt = torch.load(file, map_location="cpu")
            ckpt["_filename"] = file.name
            ckpts.append(ckpt)
        except Exception as exc:
            print(f"✖️  No se pudo cargar {file.name}: {exc}")
    if not ckpts:
        print("⚠️  No se encontraron checkpoints válidos en 'models_finales/'.")
        sys.exit(1)
    return ckpts


def build_dataframe(ckpts: List[dict]) -> pd.DataFrame:
    """Convierte la lista de checkpoints en un DataFrame ordenable."""
    records = []
    for ck in ckpts:
        m = ck.get("metrics", {})
        records.append(
            {
                "archivo": ck["_filename"],
                "modelo": ck.get("model_name"),
                "rmse": m.get("rmse"),
                "mae": m.get("mae"),
                "r2": m.get("r2"),
                "fold": ck.get("best_fold"),
                "época": ck.get("best_epoch"),
                "timestamp": ck["_filename"].split("_")[-1].replace(".pt", ""),
            }
        )
    df = pd.DataFrame(records)
    return df


def sort_dataframe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    asc = VALID_METRICS[metric] == "menor"
    return df.sort_values(metric, ascending=asc).reset_index(drop=True)


def print_table(df: pd.DataFrame, metric: str):
    """Imprime tabla bonita en consola."""
    header = f"\n===== Ranking por {metric.upper()} ({'↓' if VALID_METRICS[metric]=='menor' else '↑'}) ====="
    print(header)
    print(df.to_string(index=False, formatters={"rmse": "{:.4f}".format, "mae": "{:.4f}".format, "r2": "{:.4f}".format}))


def plot_bar(df: pd.DataFrame, metric: str):
    """Muestra gráfico de barras con la métrica escogida."""
    plt.figure(figsize=(8, 4))
    df.plot.bar(x="modelo", y=metric, legend=False)
    plt.ylabel(metric.upper())
    plt.title(f"Comparativa de modelos ({metric.upper()})")
    if VALID_METRICS[metric] == "menor":
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser("Comparación de modelos guardados")
    parser.add_argument("--metric", choices=VALID_METRICS.keys(), default="rmse", help="Métrica para ordenar (default: rmse)")
    parser.add_argument("--csv", metavar="FICHERO", help="Guardar resultados en CSV")
    parser.add_argument("--plot", action="store_true", help="Mostrar gráfico de barras de la métrica escogida")
    args = parser.parse_args(argv)

    ckpts = load_checkpoints()
    df = build_dataframe(ckpts)
    df_sorted = sort_dataframe(df, args.metric)

    print_table(df_sorted, args.metric)

    if args.csv:
        csv_path = Path(args.csv)
        df_sorted.to_csv(csv_path, index=False)
        print(f"\n✅ Resultados guardados en {csv_path}")

    if args.plot:
        plot_bar(df_sorted, args.metric)


if __name__ == "__main__":
    main()
