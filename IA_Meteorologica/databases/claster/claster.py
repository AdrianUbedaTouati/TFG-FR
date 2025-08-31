# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from caas_jupyter_tools import display_dataframe_to_user

# 1) Cargar datos
csv_path = "/mnt/data/weather_classification_data.csv"
df = pd.read_csv(csv_path)

# 2) Detectar la columna de Cloud Cover (asumimos que hay exactamente una)
cloud_cols = [c for c in df.columns if "cloud" in c.lower() and "cover" in c.lower()]
if not cloud_cols:
    # Fallback: cualquier columna que contenga "cloud"
    cloud_cols = [c for c in df.columns if "cloud" in c.lower()]

if not cloud_cols:
    raise RuntimeError("No se encontró una columna de 'Cloud Cover' en el archivo.")
cloud_col = cloud_cols[0]

# 3) Preparar variables predictoras: todas las columnas numéricas excepto la de Cloud Cover
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in num_cols if c != cloud_col]

if not feature_cols:
    raise RuntimeError("No hay columnas numéricas (aparte de Cloud Cover) para caracterizar los estados.")

# Limpieza: imputar NaN con la media y eliminar columnas con varianza cero
X = df[feature_cols].copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(X.mean(numeric_only=True))

variances = X.var(axis=0)
feature_cols = variances[variances > 0].index.tolist()
X = X[feature_cols]

# Estandarizar
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# 4) Estados de Cloud Cover (categorías)
states_raw = df[cloud_col].astype(str).fillna("NA")
states = states_raw.unique().tolist()

# Confirmar que hay exactamente 4 estados (como indicó el usuario). Si hay más/menos, seguimos de todos modos.
# 5) Vector representante (centroide) por estado, usando TODAS las filas de ese estado
centroids = []
state_labels = []
counts = []
for s in states:
    idx = states_raw == s
    if idx.sum() == 0:
        continue
    centroids.append(Xs[idx, :].mean(axis=0))
    state_labels.append(s)
    counts.append(int(idx.sum()))
centroids = np.vstack(centroids)

# 6) Distancias entre estados (euclídea sobre variables estandarizadas)
D_condensed = pdist(centroids, metric="euclidean")
D_square = squareform(D_condensed)

# 7) Clustering jerárquico de estados
Z = linkage(D_condensed, method="average")

# 8) Selección automática del número de clusters (k) en {2..min(3, n_estados-1)}
n_states = len(state_labels)
k_candidates = list(range(2, min(3, n_states - 1) + 1)) if n_states > 2 else [2]
best_k = None
best_db = np.inf
best_sil = -np.inf
for k in k_candidates:
    labels = fcluster(Z, k, criterion="maxclust")
    # Métrica 1: Davies-Bouldin (menor es mejor). Usa los propios centroides como "muestras".
    try:
        db = davies_bouldin_score(centroids, labels)
    except Exception:
        db = np.inf
    # Métrica 2: Silhouette sobre distancias precomputadas (mayor es mejor)
    try:
        sil = silhouette_score(D_square, labels, metric="precomputed")
    except Exception:
        sil = -np.inf
    # Selección: primero por menor DB, empate por mayor silhouette
    if (db < best_db) or (np.isclose(db, best_db) and sil > best_sil):
        best_db = db
        best_sil = sil
        best_k = k

if best_k is None:
    best_k = 2

final_labels = fcluster(Z, int(best_k), criterion="maxclust")

# 9) Graficar Dendrograma de ESTADOS con etiquetas = nombres de cada estado
#    Color threshold cerca del corte que produce best_k
if best_k >= 2:
    idx = -(best_k - 1)
    if abs(idx) <= Z.shape[0]:
        color_threshold = Z[idx, 2] - 1e-8
    else:
        color_threshold = None
else:
    color_threshold = None

plt.figure(figsize=(10, 5))
dendrogram(
    Z,
    labels=state_labels,
    leaf_rotation=0,
    leaf_font_size=10,
    color_threshold=color_threshold if color_threshold is not None else 0
)
plt.title(f"Clustering de estados de Cloud Cover (k recomendado = {best_k}, DB={best_db:.3f}, Sil={best_sil:.3f})")
plt.tight_layout()
fig_dendro_path = "/mnt/data/cloud_states_dendrogram.png"
plt.savefig(fig_dendro_path, dpi=150, bbox_inches="tight")
plt.show()

# 10) Graficar mapa de calor de distancias entre estados
plt.figure(figsize=(6, 5))
plt.imshow(D_square)
plt.xticks(range(n_states), state_labels, rotation=45, ha="right")
plt.yticks(range(n_states), state_labels)
plt.title("Distancias euclídeas entre estados (sobre variables estandarizadas)")
plt.colorbar()
plt.tight_layout()
fig_heatmap_path = "/mnt/data/cloud_states_distance_heatmap.png"
plt.savefig(fig_heatmap_path, dpi=150, bbox_inches="tight")
plt.show()

# 11) Proyección 2D (PCA) de los estados para visualizar cercanía
#     Si hay 1 sola variable tras limpieza, la segunda componente será 0 y el gráfico seguirá siendo válido.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(centroids)

plt.figure(figsize=(6, 5))
for i, label in enumerate(state_labels):
    plt.scatter(coords_2d[i,0], coords_2d[i,1])
    plt.text(coords_2d[i,0], coords_2d[i,1], f" {label}", va="center", ha="left")
plt.title("Estados de Cloud Cover en 2D (PCA sobre variables estandarizadas)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
fig_scatter_path = "/mnt/data/cloud_states_pca_scatter.png"
plt.savefig(fig_scatter_path, dpi=150, bbox_inches="tight")
plt.show()

# 12) Tablas para el usuario
# 12a) Asignación de cluster por estado
assign_df = pd.DataFrame({
    "Estado_CloudCover": state_labels,
    "Cluster": final_labels,
    "N_filas": counts
}).sort_values(["Cluster", "Estado_CloudCover"]).reset_index(drop=True)

display_dataframe_to_user("Clusters recomendados por estado de Cloud Cover", assign_df)

# 12b) Matriz de distancias
dist_df = pd.DataFrame(D_square, index=state_labels, columns=state_labels)
display_dataframe_to_user("Matriz de distancias entre estados de Cloud Cover", dist_df)

print(f"Dendrograma: {fig_dendro_path}")
print(f"Heatmap distancias: {fig_heatmap_path}")
print(f"Dispersión PCA: {fig_scatter_path}")