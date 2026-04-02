import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Configuración
# =====================================================
USE_BALANCED = True

OUTPUT_DIR = "analisis_dato/data/analytic/capa3"
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

# =====================================================
# Tabla 1: distribución de generaciones dentro de cada clúster
# =====================================================
tab_cluster_gen = pd.crosstab(df["cluster"], df["grupo_edad"], normalize="index").round(3)
tab_cluster_gen.to_csv(os.path.join(REPORTS_DIR, f"capa3_cluster_by_generation_{suffix}.csv"))

print("\n--- GENERACIONES DENTRO DE CADA CLÚSTER ---")
print(tab_cluster_gen)

# =====================================================
# Tabla 2: distribución de clústeres dentro de cada generación
# =====================================================
tab_gen_cluster = pd.crosstab(df["grupo_edad"], df["cluster"], normalize="index").round(3)
tab_gen_cluster.to_csv(os.path.join(REPORTS_DIR, f"capa3_generation_by_cluster_{suffix}.csv"))

print("\n--- CLÚSTERES DENTRO DE CADA GENERACIÓN ---")
print(tab_gen_cluster)

# =====================================================
# Gráfico 1: generaciones dentro de cada clúster
# =====================================================
tab_cluster_gen.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title(f"Distribución generacional dentro de cada clúster - {title_suffix}")
plt.xlabel("Cluster")
plt.ylabel("Proporción")
plt.legend(title="Grupo de edad", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, f"capa3_cluster_by_generation_{suffix}.png"), dpi=300)
plt.close()

# =====================================================
# Gráfico 2: clústeres dentro de cada generación
# =====================================================
tab_gen_cluster.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title(f"Distribución de clústeres dentro de cada generación - {title_suffix}")
plt.xlabel("Grupo de edad")
plt.ylabel("Proporción")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, f"capa3_generation_by_cluster_{suffix}.png"), dpi=300)
plt.close()

print(f"\nCruce generación × clúster completado ({title_suffix}).")