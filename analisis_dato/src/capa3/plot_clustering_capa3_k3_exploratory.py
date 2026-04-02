import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =====================================================
# Configuración
# =====================================================
USE_BALANCED = True

if USE_BALANCED:
    INPUT_PATH = "analisis_dato/data/analytic/capa3/capa3_clustering_balanced_generation.csv"
    suffix = "balanced"
    title_suffix = "BALANCED"
else:
    INPUT_PATH = "analisis_dato/data/input/capa3/capa3_clustering_ready.csv"
    suffix = "main"
    title_suffix = "MAIN"

OUTPUT_DIR = "analisis_dato/data/analytic/capa3"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =====================================================
# Carga
# =====================================================
df = pd.read_csv(INPUT_PATH)

drop_cols = ["id_respuesta"]
X = df.drop(columns=drop_cols).copy()

categorical_features = ["grupo_edad", "sexo", "gasto_mensual_moda"]
numeric_features = [col for col in X.columns if col not in categorical_features]

# =====================================================
# Preprocesado
# =====================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)
X_dense = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed

# =====================================================
# PCA a 2D
# =====================================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_dense)
explained_var = pca.explained_variance_ratio_

# =====================================================
# KMeans exploratorio con k=3 en PCA 2D
# =====================================================
kmeans_k3 = KMeans(n_clusters=3, random_state=42, n_init=20)
labels_k3 = kmeans_k3.fit_predict(X_pca)
centroids_k3 = kmeans_k3.cluster_centers_

# =====================================================
# Guardar asignaciones
# =====================================================
df_k3 = df.copy()
df_k3["cluster_k3"] = labels_k3
df_k3.to_csv(
    os.path.join(REPORTS_DIR, f"capa3_clustering_assignments_k3_exploratory_{suffix}.csv"),
    index=False
)

# Perfil numérico
profile_k3 = df_k3.groupby("cluster_k3")[numeric_features].mean().round(3)
profile_k3.to_csv(
    os.path.join(REPORTS_DIR, f"capa3_clustering_profile_numeric_k3_exploratory_{suffix}.csv")
)

# Tamaños
cluster_sizes = df_k3["cluster_k3"].value_counts().sort_index()
cluster_sizes_df = cluster_sizes.reset_index()
cluster_sizes_df.columns = ["cluster_k3", "n_obs"]
cluster_sizes_df["pct"] = cluster_sizes_df["n_obs"] / len(df_k3)
cluster_sizes_df.to_csv(
    os.path.join(REPORTS_DIR, f"capa3_clustering_sizes_k3_exploratory_{suffix}.csv"),
    index=False
)

print("\n--- TAMAÑO CLÚSTERES K=3 EXPLORATORIO ---")
print(cluster_sizes_df)

print("\n--- PERFIL NUMÉRICO MEDIO K=3 EXPLORATORIO ---")
print(profile_k3)

# =====================================================
# Mallado para fronteras de decisión
# =====================================================
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = kmeans_k3.predict(grid_points)
Z = Z.reshape(xx.shape)

# =====================================================
# Gráfico con fronteras
# =====================================================
plt.figure(figsize=(10, 7))

plt.contourf(xx, yy, Z, alpha=0.18)
plt.contour(xx, yy, Z, colors="black", linewidths=1)

for cluster_id in sorted(np.unique(labels_k3)):
    mask = labels_k3 == cluster_id
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cluster_id}",
        alpha=0.75
    )

plt.scatter(
    centroids_k3[:, 0],
    centroids_k3[:, 1],
    marker="X",
    s=250,
    linewidths=1.5,
    label="Centroides"
)

plt.title(
    f"KMeans exploratorio con k=3 en PCA 2D - {title_suffix}\n"
    f"Varianza explicada: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}"
)
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, f"capa3_clusters_pca_2d_k3_exploratory_{suffix}.png"),
    dpi=300
)
plt.close()

print(
    f"\nGráfico k=3 exploratorio guardado en figures/"
    f"capa3_clusters_pca_2d_k3_exploratory_{suffix}.png"
)