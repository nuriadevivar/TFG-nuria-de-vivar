import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# =====================================================
# Rutas
# =====================================================
INPUT_PATH = "data/input/capa1/capa1_master_mensual_analysis.csv"
OUTPUT_DIR = "data/analytic/capa1"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =====================================================
# Carga de datos
# =====================================================
df = pd.read_csv(INPUT_PATH)

# =====================================================
# Paso 1. Inspección y validación
# =====================================================
print("\n--- INFO GENERAL ---")
print(df.info())

print("\n--- PRIMERAS FILAS ---")
print(df.head())

print("\n--- RESUMEN DESCRIPTIVO ---")
print(df.describe())

# Parseo de fecha
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

# Ordenar por fecha
df = df.sort_values("fecha").reset_index(drop=True)

# Validación de nulos
print("\n--- NULOS POR COLUMNA ---")
print(df.isnull().sum())

# Validación de duplicados
print("\n--- DUPLICADOS ---")
print(f"Duplicados exactos: {df.duplicated().sum()}")

# Validación de fechas duplicadas
print("\n--- FECHAS DUPLICADAS ---")
print(f"Fechas duplicadas: {df['fecha'].duplicated().sum()}")

# Establecer fecha como índice
df = df.set_index("fecha")

# Comprobar frecuencia mensual
freq_detected = pd.infer_freq(df.index)
print("\n--- FRECUENCIA TEMPORAL ---")
print(f"Frecuencia detectada: {freq_detected}")

# Si no se detecta correctamente, forzamos mensual de inicio de mes
df = df.asfreq("MS")

print("\n--- RANGO TEMPORAL ---")
print(f"Inicio: {df.index.min()}")
print(f"Fin:    {df.index.max()}")
print(f"Número de observaciones: {len(df)}")

# =====================================================
# Paso 2. EDA temporal
# =====================================================

# Variables principales
series_cols = [
    "indice_retail_moda",
    "indice_retail_total",
    "ratio_moda_vs_total"
]

# Gráfico de evolución de las tres series
plt.figure(figsize=(12, 6))
for col in series_cols:
    plt.plot(df.index, df[col], label=col)
plt.title("Evolución temporal de las series principales - Capa 1")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "capa1_series_principales.png"), dpi=300)
plt.close()

# Gráficos individuales con media móvil de 12 meses
for col in series_cols:
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df[col], label=col)
    plt.plot(df.index, df[col].rolling(window=12).mean(), label="Media móvil 12 meses")
    plt.title(f"Evolución y media móvil - {col}")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{col}_media_movil.png"), dpi=300)
    plt.close()

# Boxplot por mes para explorar estacionalidad visual
for col in ["indice_retail_moda", "indice_retail_total", "ratio_moda_vs_total"]:
    temp = df.copy()
    temp["mes_num"] = temp.index.month

    plt.figure(figsize=(10, 5))
    temp.boxplot(column=col, by="mes_num", grid=False)
    plt.title(f"Distribución mensual - {col}")
    plt.suptitle("")
    plt.xlabel("Mes")
    plt.ylabel("Valor")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{col}_boxplot_mes.png"), dpi=300)
    plt.close()

# Guardar resumen descriptivo
df.describe().to_csv(os.path.join(REPORTS_DIR, "capa1_resumen_descriptivo.csv"))

# Descomposición temporal para serie principal
target_col = "indice_retail_moda"

decomposition = seasonal_decompose(df[target_col], model="additive", period=12)

fig = decomposition.plot()
fig.set_size_inches(12, 8)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, f"{target_col}_decomposition.png"), dpi=300)
plt.close(fig)

# =====================================================
# Paso 3. Test de estacionariedad
# =====================================================
def adf_test(series, series_name):
    """
    Ejecuta test ADF y devuelve resultados en dict.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    output = {
        "serie": series_name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_value_1%": result[4]["1%"],
        "critical_value_5%": result[4]["5%"],
        "critical_value_10%": result[4]["10%"],
    }
    return output

adf_results = []

for col in ["indice_retail_moda", "ratio_moda_vs_total"]:
    # serie original
    adf_results.append(adf_test(df[col], f"{col}_original"))

    # primera diferencia
    df[f"{col}_diff1"] = df[col].diff()
    adf_results.append(adf_test(df[f"{col}_diff1"], f"{col}_diff1"))

adf_df = pd.DataFrame(adf_results)
adf_df.to_csv(os.path.join(REPORTS_DIR, "capa1_adf_results.csv"), index=False)

print("\n--- RESULTADOS TEST ADF ---")
print(adf_df)

print("\nScript de EDA temporal y estacionariedad completado.")