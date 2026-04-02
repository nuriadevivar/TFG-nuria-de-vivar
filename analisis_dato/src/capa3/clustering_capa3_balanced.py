import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================================
# Rutas
# =====================================================
INPUT_PATH = "analisis_dato/data/analytic/capa3/capa3_clustering_balanced_generation.csv"
OUTPUT_DIR = "analisis_dato/data/analytic/capa3"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =====================================================
# Carga de datos
# =====================================================
df = pd.read_csv(INPUT_PATH)

print("\n--- INFO GENERAL ---")
print(df.info())

print("\n--- PRIMERAS FILAS ---")
print(df.head())

print("\n--- NULOS POR COLUMNA ---")
print(df.isnull().sum())

# =====================================================
# Preparación
# =====================================================
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

# =====================================================
# Búsqueda de k
# =====================================================
k_values = range(2, 7)
results = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_processed)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_processed, labels)

    results.append({
        "k": k,
        "inertia": inertia,
        "silhouette": silhouette
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(REPORTS_DIR, "capa3_clustering_k_selection_balanced.csv"), index=False)

with open(os.path.join(REPORTS_DIR, "capa3_clustering_k_selection_balanced.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("\n--- SELECCIÓN DE K BALANCED ---")
print(results_df)

# =====================================================
# Elbow
# =====================================================
plt.figure(figsize=(8, 5))
plt.plot(results_df["k"], results_df["inertia"], marker="o")
plt.title("Método del codo - KMeans capa 3 BALANCED")
plt.xlabel("Número de clústeres (k)")
plt.ylabel("Inercia")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa3_kmeans_elbow_balanced.png"), dpi=300)
plt.close()

# =====================================================
# Silhouette
# =====================================================
plt.figure(figsize=(8, 5))
plt.plot(results_df["k"], results_df["silhouette"], marker="o")
plt.title("Silhouette score - KMeans capa 3 BALANCED")
plt.xlabel("Número de clústeres (k)")
plt.ylabel("Silhouette")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa3_kmeans_silhouette_balanced.png"), dpi=300)
plt.close()

# =====================================================
# Mejor k
# =====================================================
best_k = int(results_df.sort_values(by="silhouette", ascending=False).iloc[0]["k"])
print(f"\n--- MEJOR K BALANCED SEGÚN SILHOUETTE ---\n{best_k}")

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = kmeans_final.fit_predict(X_processed)

df_clusters = df.copy()
df_clusters["cluster"] = cluster_labels

# =====================================================
# Guardar asignaciones
# =====================================================
df_clusters.to_csv(os.path.join(REPORTS_DIR, "capa3_clustering_assignments_balanced.csv"), index=False)

# =====================================================
# Perfil numérico
# =====================================================
profile_numeric = df_clusters.groupby("cluster")[numeric_features].mean().round(3)
profile_numeric.to_csv(os.path.join(REPORTS_DIR, "capa3_clustering_profile_numeric_balanced.csv"))

cluster_sizes = df_clusters["cluster"].value_counts().sort_index()
cluster_sizes_df = cluster_sizes.reset_index()
cluster_sizes_df.columns = ["cluster", "n_obs"]
cluster_sizes_df["pct"] = cluster_sizes_df["n_obs"] / len(df_clusters)
cluster_sizes_df.to_csv(os.path.join(REPORTS_DIR, "capa3_clustering_sizes_balanced.csv"), index=False)

print("\n--- TAMAÑO DE CLÚSTERES BALANCED ---")
print(cluster_sizes_df)

print("\n--- PERFIL NUMÉRICO MEDIO POR CLÚSTER BALANCED ---")
print(profile_numeric)

# =====================================================
# Gráfico tamaños
# =====================================================
plt.figure(figsize=(8, 5))
plt.bar(cluster_sizes_df["cluster"].astype(str), cluster_sizes_df["n_obs"])
plt.title("Tamaño de clústeres - Capa 3 BALANCED")
plt.xlabel("Cluster")
plt.ylabel("Número de observaciones")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa3_cluster_sizes_balanced.png"), dpi=300)
plt.close()

print("\nScript de clustering capa 3 BALANCED completado.")