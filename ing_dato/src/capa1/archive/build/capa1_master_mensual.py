import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_PROCESSED, OUTPUT_FIGURES, OUTPUT_TABLES

# Cargar datos
file_path = DATA_PROCESSED / "capa1_master_mensual_analysis.csv"
df = pd.read_csv(file_path)

# Convertir fecha
df["fecha"] = pd.to_datetime(df["fecha"])

# Crear carpetas de salida
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

print("Dimensiones:", df.shape)
print("")
print("Preview:")
print(df.head(12))
print("")
print("Descriptivos:")
print(df.describe(include="all"))

# Guardar descriptivos numéricos
desc_path = OUTPUT_TABLES / "capa1_mensual_descriptivos.csv"
df.describe().to_csv(desc_path)

# =========================
# GRÁFICO 1: Moda vs total
# =========================
plt.figure(figsize=(12, 6))
plt.plot(df["fecha"], df["indice_retail_moda"], label="Retail moda")
plt.plot(df["fecha"], df["indice_retail_total"], label="Retail total")
plt.title("Evolución mensual del retail de moda frente al retail total")
plt.xlabel("Fecha")
plt.ylabel("Índice base 2015=100")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "capa1_mensual_moda_vs_total.png")
plt.close()

# =========================
# GRÁFICO 2: Ratio moda/total
# =========================
plt.figure(figsize=(12, 6))
plt.plot(df["fecha"], df["ratio_moda_vs_total"])
plt.title("Ratio entre retail de moda y retail total")
plt.xlabel("Fecha")
plt.ylabel("Ratio")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "capa1_mensual_ratio_moda_vs_total.png")
plt.close()

# =========================
# GRÁFICO 3: Diferencia moda-total
# =========================
plt.figure(figsize=(12, 6))
plt.plot(df["fecha"], df["dif_moda_vs_total"])
plt.axhline(0, linestyle="--")
plt.title("Diferencia entre retail de moda y retail total")
plt.xlabel("Fecha")
plt.ylabel("Diferencia de índices")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "capa1_mensual_dif_moda_vs_total.png")
plt.close()

# =========================
# GRÁFICO 4: Medias anuales
# =========================
annual_summary = (
    df.groupby("anio")[["indice_retail_moda", "indice_retail_total"]]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))
plt.plot(annual_summary["anio"], annual_summary["indice_retail_moda"], marker="o", label="Retail moda")
plt.plot(annual_summary["anio"], annual_summary["indice_retail_total"], marker="o", label="Retail total")
plt.title("Media anual del retail de moda frente al retail total")
plt.xlabel("Año")
plt.ylabel("Índice medio anual")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "capa1_mensual_media_anual_moda_vs_total.png")
plt.close()

# =========================
# GRÁFICO 5: Heatmap por año/mes
# =========================
pivot_moda = df.pivot_table(index="anio", columns="mes", values="indice_retail_moda", aggfunc="mean")
plt.figure(figsize=(12, 6))
plt.imshow(pivot_moda, aspect="auto")
plt.colorbar(label="Índice retail moda")
plt.title("Heatmap mensual del retail de moda")
plt.xlabel("Mes")
plt.ylabel("Año")
plt.xticks(range(len(pivot_moda.columns)), pivot_moda.columns)
plt.yticks(range(len(pivot_moda.index)), pivot_moda.index)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "capa1_mensual_heatmap_moda.png")
plt.close()

# =========================
# GRÁFICO 6: Heatmap ratio moda/total
# =========================
pivot_ratio = df.pivot_table(index="anio", columns="mes", values="ratio_moda_vs_total", aggfunc="mean")
plt.figure(figsize=(12, 6))
plt.imshow(pivot_ratio, aspect="auto")
plt.colorbar(label="Ratio moda / total")
plt.title("Heatmap del ratio retail moda frente a retail total")
plt.xlabel("Mes")
plt.ylabel("Año")
plt.xticks(range(len(pivot_ratio.columns)), pivot_ratio.columns)
plt.yticks(range(len(pivot_ratio.index)), pivot_ratio.index)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "capa1_mensual_heatmap_ratio.png")
plt.close()

# Guardar resumen anual
annual_summary_path = OUTPUT_TABLES / "capa1_mensual_media_anual.csv"
annual_summary.to_csv(annual_summary_path, index=False)

print("")
print("EDA mensual completado.")
print("Descriptivos guardados en:", desc_path)
print("Resumen anual guardado en:", annual_summary_path)
print("Gráficos guardados en:", OUTPUT_FIGURES)