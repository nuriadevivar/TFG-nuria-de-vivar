import os
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
    INPUT_PATH = "data/analytic/capa3/capa3_clustering_balanced_generation.csv"
    OUTPUT_NAME = "capa3_clusters_pca_2d_balanced.png"
    TITLE_SUFFIX = "BALANCED"
else:
    INPUT_PATH = "data/input/capa3/capa3_clustering_ready.csv"
    OUTPUT_NAME = "capa3_clusters_pca_2d_main.png"
    TITLE_SUFFIX = "MAIN"

INPUT_PATH = "data/analytic/capa3/reports/capa3_cluster_assignments.csv"

OUTPUT_DIR = "data/analytic/capa3"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def relabel_clusters_by_influence(df_base: pd.DataFrame, cluster_labels, influence_col: str = "indice_influencia_rrss"):
    df_tmp = df_base.copy()
    df_tmp["cluster_tmp"] = cluster_labels

    ordered_labels = (
        df_tmp.groupby("cluster_tmp")[influence_col]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    label_map = {old_label: new_label for new_label, old_label in enumerate(ordered_labels)}
    relabeled = pd.Series(cluster_labels).map(label_map).values
    return relabeled, label_map

# =====================================================
# Carga
# =====================================================
df = pd.read_csv(INPUT_PATH)

drop_cols = ["id_respuesta"]
X = df.drop(columns=drop_cols).copy()

categorical_features = ["grupo_edad", "sexo", "gasto_mensual_moda"]
numeric_features = [col for col in X.columns if col not in categorical_features]

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

kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
labels_raw = kmeans.fit_predict(X_dense)

labels, label_map = relabel_clusters_by_influence(
    df_base=df,
    cluster_labels=labels_raw,
    influence_col="indice_influencia_rrss",
)

df["cluster"] = labels

old_labels_in_new_order = [
    old_label for old_label, new_label in sorted(label_map.items(), key=lambda x: x[1])
]
centers_reordered = kmeans.cluster_centers_[old_labels_in_new_order]

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_dense)
centroids_pca = pca.transform(centers_reordered)

explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(10, 7))

for cluster_id in sorted(set(labels)):
    mask = labels == cluster_id
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cluster_id}",
        alpha=0.75
    )

plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    marker="X",
    s=250,
    linewidths=1.5,
    label="Centroides"
)

plt.title(
    f"Visualización 2D de clústeres con PCA - {TITLE_SUFFIX}\n"
    f"Varianza explicada: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}"
)
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, OUTPUT_NAME), dpi=300)
plt.close()

print(f"Gráfico PCA guardado en figures/{OUTPUT_NAME}")