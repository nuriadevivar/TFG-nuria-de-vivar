import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Configuración
# =====================================================
USE_BALANCED = True

INPUT_PATH = "data/analytic/capa3/reports/capa3_cluster_assignments.csv"

OUTPUT_DIR = "data/analytic/capa3"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

if USE_BALANCED:
    INPUT_PATH = os.path.join(REPORTS_DIR, "capa3_clustering_assignments_balanced.csv")
    suffix = "balanced"
    title_suffix = "BALANCED"
else:
    INPUT_PATH = os.path.join(REPORTS_DIR, "capa3_clustering_assignments_main.csv")
    suffix = "main"
    title_suffix = "MAIN"

# =====================================================
# Carga
# =====================================================
df = pd.read_csv(INPUT_PATH)

print("\n--- COLUMNAS ---")
print(df.columns.tolist())

categorical_features = ["grupo_edad", "sexo", "gasto_mensual_moda"]
numeric_features = [
    col for col in df.columns
    if col not in categorical_features + ["id_respuesta", "cluster"]
]

# =====================================================
# Perfil numérico
# =====================================================
profile_numeric = df.groupby("cluster")[numeric_features].mean().round(3)
profile_numeric.to_csv(os.path.join(REPORTS_DIR, f"capa3_clustering_profile_numeric_full_{suffix}.csv"))

print("\n--- PERFIL NUMÉRICO COMPLETO ---")
print(profile_numeric)

if len(profile_numeric) == 2:
    diff = (profile_numeric.loc[0] - profile_numeric.loc[1]).sort_values(key=np.abs, ascending=False)
    diff_df = diff.reset_index()
    diff_df.columns = ["feature", "difference_cluster0_minus_cluster1"]
    diff_df.to_csv(os.path.join(REPORTS_DIR, f"capa3_clustering_numeric_differences_{suffix}.csv"), index=False)

    print("\n--- VARIABLES QUE MÁS SEPARAN LOS CLÚSTERES ---")
    print(diff_df.head(15))

for col in categorical_features:
    ctab = pd.crosstab(df["cluster"], df[col], normalize="index").round(3)
    ctab.to_csv(os.path.join(REPORTS_DIR, f"capa3_cluster_profile_{col}_{suffix}.csv"))
    print(f"\n--- PERFIL CATEGÓRICO: {col} ---")
    print(ctab)

if len(profile_numeric) == 2:
    top_features = diff_df.head(8)["feature"].tolist()

    plot_df = profile_numeric[top_features].T
    plot_df.plot(kind="bar", figsize=(12, 6))
    plt.title(f"Comparación de variables clave entre clústeres - {title_suffix}")
    plt.xlabel("Variable")
    plt.ylabel("Valor medio")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"capa3_cluster_key_numeric_comparison_{suffix}.png"), dpi=300)
    plt.close()

zscore_df = df[numeric_features].copy()
zscore_df = (zscore_df - zscore_df.mean()) / zscore_df.std()

zscore_df["cluster"] = df["cluster"]
z_profile = zscore_df.groupby("cluster").mean().round(3)
z_profile.to_csv(os.path.join(REPORTS_DIR, f"capa3_clustering_profile_zscores_{suffix}.csv"))

if len(z_profile) == 2:
    top_z_features = diff_df.head(8)["feature"].tolist()
    z_plot_df = z_profile[top_z_features].T
    z_plot_df.plot(kind="bar", figsize=(12, 6))
    plt.title(f"Comparación estandarizada entre clústeres (z-score) - {title_suffix}")
    plt.xlabel("Variable")
    plt.ylabel("Media estandarizada")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"capa3_cluster_zscore_comparison_{suffix}.png"), dpi=300)
    plt.close()

print(f"\nPerfilado de clústeres {title_suffix} completado.")