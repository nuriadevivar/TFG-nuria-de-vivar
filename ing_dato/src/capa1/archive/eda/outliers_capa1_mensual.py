import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_PROCESSED, OUTPUT_TABLES, OUTPUT_FIGURES

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

file_path = DATA_PROCESSED / "capa1_master_mensual_analysis.csv"
df = pd.read_csv(file_path)
df["fecha"] = pd.to_datetime(df["fecha"])

variables = [
    "indice_retail_moda",
    "indice_retail_total",
    "ratio_moda_vs_total",
    "dif_moda_vs_total",
]

outlier_rows = []

for var in variables:
    s = pd.to_numeric(df[var], errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (s < lower) | (s > upper)
    df_out = df.loc[mask, ["fecha", var]].copy()
    df_out["variable"] = var
    df_out["lower_bound"] = lower
    df_out["upper_bound"] = upper
    df_out["is_outlier"] = True

    outlier_rows.append(df_out)

    # boxplot
    plt.figure(figsize=(8, 5))
    plt.boxplot(s.dropna())
    plt.title(f"Boxplot - {var}")
    plt.ylabel(var)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / f"boxplot_{var}.png")
    plt.close()

final_outliers = pd.concat(outlier_rows, ignore_index=True)
final_outliers.to_csv(OUTPUT_TABLES / "capa1_mensual_outliers_iqr.csv", index=False)

print("Outliers detectados.")
print("Archivo guardado en:")
print(OUTPUT_TABLES / "capa1_mensual_outliers_iqr.csv")
print("")
print(final_outliers.head(20))