import os
import pandas as pd

# =====================================================
# Rutas
# =====================================================
INPUT_CLUSTER_PATH = "data/input/capa3/capa3_clustering_ready.csv"
INPUT_SUP_PATH = "data/input/capa3/capa3_supervised_ready.csv"

OUTPUT_DIR = "data/analytic/capa3"
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

OUTPUT_CLUSTER_BALANCED = os.path.join(OUTPUT_DIR, "capa3_clustering_balanced_generation.csv")
OUTPUT_SUP_BALANCED = os.path.join(OUTPUT_DIR, "capa3_supervised_balanced_generation.csv")
OUTPUT_REPORT = os.path.join(REPORTS_DIR, "capa3_generation_balancing_report.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =====================================================
# Función robusta de balanceo por generación
# =====================================================
def balance_by_generation(df, group_col="grupo_edad", random_state=42):
    if group_col not in df.columns:
        raise ValueError(
            f"La columna '{group_col}' no existe en el dataset. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    counts = df[group_col].value_counts()
    min_count = counts.min()

    balanced_parts = []
    for group_value in counts.index:
        subset = df[df[group_col] == group_value].sample(
            n=min_count,
            random_state=random_state
        )
        balanced_parts.append(subset)

    balanced_df = (
        pd.concat(balanced_parts, axis=0)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    return balanced_df, counts, min_count

# =====================================================
# Supervisado
# =====================================================
df_sup = pd.read_csv(INPUT_SUP_PATH)
print("\n=== COLUMNAS SUPERVISADO ===")
print(df_sup.columns.tolist())

balanced_sup, counts_sup, min_count_sup = balance_by_generation(
    df_sup,
    group_col="grupo_edad"
)

sup_output_path = os.path.join(
    OUTPUT_DIR,
    "capa3_supervised_balanced_generation.csv"
)
balanced_sup.to_csv(sup_output_path, index=False)

# =====================================================
# Clustering
# =====================================================
df_cluster = pd.read_csv(INPUT_CLUSTER_PATH)
print("\n=== COLUMNAS CLUSTERING ===")
print(df_cluster.columns.tolist())

balanced_cluster, counts_cluster, min_count_cluster = balance_by_generation(
    df_cluster,
    group_col="grupo_edad"
)

cluster_output_path = os.path.join(
    OUTPUT_DIR,
    "capa3_clustering_balanced_generation.csv"
)
balanced_cluster.to_csv(cluster_output_path, index=False)

# =====================================================
# Reporte
# =====================================================
report_lines = []

report_lines.append("=== DISTRIBUCIÓN ORIGINAL SUPERVISADO ===")
report_lines.append(counts_sup.to_string())
report_lines.append(f"\nTamaño mínimo por generación: {min_count_sup}\n")

report_lines.append("=== DISTRIBUCIÓN BALANCEADA SUPERVISADO ===")
report_lines.append(balanced_sup["grupo_edad"].value_counts().to_string())
report_lines.append("\n")

report_lines.append("=== DISTRIBUCIÓN ORIGINAL CLUSTERING ===")
report_lines.append(counts_cluster.to_string())
report_lines.append(f"\nTamaño mínimo por generación: {min_count_cluster}\n")

report_lines.append("=== DISTRIBUCIÓN BALANCEADA CLUSTERING ===")
report_lines.append(balanced_cluster["grupo_edad"].value_counts().to_string())
report_lines.append("\n")

report_path = os.path.join(REPORTS_DIR, "capa3_generation_balancing_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# =====================================================
# Consola
# =====================================================
print("\n=== SUPERVISADO ORIGINAL ===")
print(counts_sup)
print(f"\nTamaño mínimo: {min_count_sup}")

print("\n=== SUPERVISADO BALANCEADO ===")
print(balanced_sup["grupo_edad"].value_counts())

print("\n=== CLUSTERING ORIGINAL ===")
print(counts_cluster)
print(f"\nTamaño mínimo: {min_count_cluster}")

print("\n=== CLUSTERING BALANCEADO ===")
print(balanced_cluster["grupo_edad"].value_counts())

print("\nDatasets balanceados creados correctamente.")
print(f"\nCSV supervisado balanceado: {sup_output_path}")
print(f"CSV clustering balanceado:  {cluster_output_path}")
print(f"Reporte generado en:       {report_path}")