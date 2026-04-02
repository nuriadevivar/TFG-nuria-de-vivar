import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# =====================================================
# Rutas
# =====================================================
INPUT_PATH = "analisis_dato/data/input/capa2/trends_marcas_clean.csv"
OUTPUT_DIR = "analisis_dato/data/analytic/capa2"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =====================================================
# Carga
# =====================================================
df = pd.read_csv(INPUT_PATH)

print("\n--- INFO GENERAL ---")
print(df.info())

print("\n--- PRIMERAS FILAS ---")
print(df.head())

print("\n--- NULOS POR COLUMNA ---")
print(df.isnull().sum())

print("\n--- NULOS POR MARCA ---")
print(df.groupby("termino")["valor_trends"].apply(lambda x: x.isnull().sum()))

# Parse fecha
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df = df.sort_values(["termino", "fecha"]).reset_index(drop=True)

print("\n--- FECHAS ---")
print(f"Inicio: {df['fecha'].min()}")
print(f"Fin:    {df['fecha'].max()}")

print("\n--- OBSERVACIONES POR MARCA ---")
print(df["termino"].value_counts())

# =====================================================
# Resumen descriptivo por marca
# =====================================================
summary_by_brand = df.groupby("termino")["valor_trends"].describe()
summary_by_brand.to_csv(os.path.join(REPORTS_DIR, "trends_summary_by_brand.csv"))

# =====================================================
# Gráfico conjunto
# =====================================================
plt.figure(figsize=(13, 6))
for brand in sorted(df["termino"].unique()):
    temp = df[df["termino"] == brand]
    plt.plot(temp["fecha"], temp["valor_trends"], label=brand)

plt.title("Evolución temporal comparada de Google Trends por marca")
plt.xlabel("Fecha")
plt.ylabel("Valor Google Trends")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "trends_series_comparadas.png"), dpi=300)
plt.close()

# =====================================================
# Series individuales con media móvil
# =====================================================
for brand in sorted(df["termino"].unique()):
    temp = df[df["termino"] == brand].copy()
    temp = temp.sort_values("fecha")
    temp["media_movil_12"] = temp["valor_trends"].rolling(window=12).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(temp["fecha"], temp["valor_trends"], label="Serie original")
    plt.plot(temp["fecha"], temp["media_movil_12"], label="Media móvil 12 meses")
    plt.title(f"Google Trends - {brand}")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"trends_{brand}_media_movil.png"), dpi=300)
    plt.close()

# =====================================================
# Boxplot por mes (estacionalidad visual)
# =====================================================
for brand in sorted(df["termino"].unique()):
    temp = df[df["termino"] == brand].copy()
    temp["mes_num"] = temp["fecha"].dt.month

    plt.figure(figsize=(10, 5))
    temp.boxplot(column="valor_trends", by="mes_num", grid=False)
    plt.title(f"Distribución mensual Google Trends - {brand}")
    plt.suptitle("")
    plt.xlabel("Mes")
    plt.ylabel("Valor")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"trends_{brand}_boxplot_mes.png"), dpi=300)
    plt.close()

# =====================================================
# Test ADF para Zara y Mango
# =====================================================
def adf_test(series, series_name):
    series = series.dropna()
    result = adfuller(series, autolag="AIC")
    return {
        "serie": series_name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_value_1%": result[4]["1%"],
        "critical_value_5%": result[4]["5%"],
        "critical_value_10%": result[4]["10%"],
    }

adf_results = []

for brand in ["zara", "mango"]:
    temp = df[df["termino"] == brand].copy().sort_values("fecha")
    series = temp["valor_trends"]

    adf_results.append(adf_test(series, f"{brand}_original"))
    adf_results.append(adf_test(series.diff(), f"{brand}_diff1"))

adf_df = pd.DataFrame(adf_results)
adf_df.to_csv(os.path.join(REPORTS_DIR, "trends_adf_zara_mango.csv"), index=False)

print("\n--- RESULTADOS ADF ZARA / MANGO ---")
print(adf_df)

print("\nScript de EDA Trends completado.")