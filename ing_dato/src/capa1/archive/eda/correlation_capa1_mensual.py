import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_PROCESSED, OUTPUT_TABLES, OUTPUT_FIGURES

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

file_path = DATA_PROCESSED / "capa1_master_mensual_analysis.csv"
df = pd.read_csv(file_path)

corr_vars = [
    "indice_retail_moda",
    "indice_retail_total",
    "ratio_moda_vs_total",
    "dif_moda_vs_total",
]

corr_df = df[corr_vars].corr(method="pearson")
corr_df.to_csv(OUTPUT_TABLES / "capa1_mensual_correlation.csv")

plt.figure(figsize=(8, 6))
plt.imshow(corr_df, aspect="auto")
plt.colorbar(label="Correlación")
plt.xticks(range(len(corr_vars)), corr_vars, rotation=45, ha="right")
plt.yticks(range(len(corr_vars)), corr_vars)
plt.title("Matriz de correlación - capa 1 mensual")
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "capa1_mensual_correlation_heatmap.png")
plt.close()

print("Correlación calculada.")
print(corr_df)
print("")
print("Archivos guardados en:")
print(OUTPUT_TABLES / "capa1_mensual_correlation.csv")
print(OUTPUT_FIGURES / "capa1_mensual_correlation_heatmap.png")