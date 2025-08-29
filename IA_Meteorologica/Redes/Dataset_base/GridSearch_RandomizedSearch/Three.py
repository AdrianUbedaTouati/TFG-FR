# Three.py (versión ligera y rápida)
# ------------------------------------------------------------
# Objetivo: predecir Temperature_C en weatherHistory.csv
# con RandomizedSearchCV "barato" y validación temporal.
# ------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

# ================== AJUSTES RÁPIDOS ==================
CSV_PATH = "data/weatherHistory.csv"   # cambia si hace falta
MAX_ROWS = 30000                  # pon None para usar todo
N_SPLITS = 3                      # nº de folds (más pequeño = más rápido)
N_ITER_RS = 10                    # iteraciones en RandomizedSearch (10-20 razonable)
RANDOM_STATE = 42
# =====================================================

# 1) Carga
df = pd.read_csv(CSV_PATH)

# 2) Normaliza nombres de columnas
df.columns = (
    df.columns.str.strip()
              .str.replace(r"\s+", "_", regex=True)
              .str.replace(r"[()]", "", regex=True)
              .str.replace("/", "_", regex=True)
)

# 3) Parseo robusto de fecha -> UTC -> naive
date_col = "Formatted_Date"
df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
df[date_col] = df[date_col].dt.tz_convert(None)
df = df.dropna(subset=[date_col]).sort_values(date_col)

# 4) (Opcional) Submuestreo para ir rápido
if isinstance(MAX_ROWS, int) and MAX_ROWS > 0 and len(df) > MAX_ROWS:
    df = df.iloc[:MAX_ROWS].copy()

# 5) Extrae features temporales baratas
df["year"]  = df[date_col].dt.year.astype("int16")
df["month"] = df[date_col].dt.month.astype("int8")
df["day"]   = df[date_col].dt.day.astype("int8")
df["hour"]  = df[date_col].dt.hour.astype("int8")

# 6) Target y features
target = "Temperature_C"
if target not in df.columns:
    raise ValueError(f"No se encontró la columna '{target}'. Columnas: {df.columns.tolist()}")

# Quitamos columnas muy pesadas en cardinalidad
drop_cols = ["Formatted_Date"]
if "Daily_Summary" in df.columns:
    drop_cols.append("Daily_Summary")
if "Summary" in df.columns:
    drop_cols.append("Summary")

X = df.drop(columns=[target] + drop_cols)
y = df[target]

# Downcast numérico para ahorrar RAM
for c in X.select_dtypes(include=[np.number]).columns:
    if pd.api.types.is_float_dtype(X[c]):
        X[c] = X[c].astype("float32")
    elif pd.api.types.is_integer_dtype(X[c]):
        # ya casteamos arriba year/month/day/hour; para el resto:
        X[c] = pd.to_numeric(X[c], downcast="integer")

# Detectar tipos
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# 7) Preprocesamiento sencillo
# OneHot con reducción de rarezas: min_frequency agrupa categorías raras
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01) if len(cat_cols) > 0 else "drop"

pre = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", ohe, cat_cols) if len(cat_cols) > 0 else ("drop", "drop", [])
    ],
    remainder="drop"
)

# 8) Modelo base (moderado)
rf = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    n_estimators=150,   # menos árboles -> más rápido
    max_depth=12        # limita la profundidad para acelerar
)

pipe = Pipeline(steps=[("pre", pre), ("model", rf)])

# 9) Validación cruzada temporal ligera
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# 10) RandomizedSearch 'barato'
param_dist = {
    "model__n_estimators": randint(80, 220),     # rangos moderados
    "model__max_depth": [8, 10, 12, 14, None],
    "model__min_samples_split": randint(2, 12),
    "model__min_samples_leaf": randint(1, 6),
    "model__max_features": ["sqrt", "log2", None]
}

rnd = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=N_ITER_RS,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# 11) Entrena búsqueda
rnd.fit(X, y)
print("RANDOM best params:", rnd.best_params_)
print("RANDOM best MAE (CV):", -rnd.best_score_)

# 12) Evaluación hold-out final (último 20% en el tiempo)
split_ix = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_ix], X.iloc[split_ix:]
y_train, y_test = y.iloc[:split_ix], y.iloc[split_ix:]

best_model = rnd.best_estimator_
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)

# Calcula métricas
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)   # en tu sklearn sin 'squared'
rmse = float(np.sqrt(mse))

print(f"Hold-out MAE:  {mae:.3f}")
print(f"Hold-out MSE:  {mse:.3f}")
print(f"Hold-out RMSE: {rmse:.3f}")
