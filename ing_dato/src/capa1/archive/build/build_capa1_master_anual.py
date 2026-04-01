import pandas as pd
from config import DATA_PROCESSED

# Rutas
moda_path = DATA_PROCESSED / "eurostat" / "eurostat_moda_mensual_clean.csv"
retail_path = DATA_PROCESSED / "eurostat" / "eurostat_retail_total_mensual_clean.csv"

# Cargar
moda = pd.read_csv(moda_path)
retail = pd.read_csv(retail_path)

# Convertir fecha
moda["fecha"] = pd.to_datetime(moda["fecha"])
retail["fecha"] = pd.to_datetime(retail["fecha"])

# Renombrar valores
moda = moda.rename(columns={"valor_indice": "indice_retail_moda"})
retail = retail.rename(columns={"valor_indice": "indice_retail_total"})

# Quedarnos con columnas útiles
moda = moda[["fecha", "indice_retail_moda"]].copy()
retail = retail[["fecha", "indice_retail_total"]].copy()

# Merge mensual
master_mensual = moda.merge(retail, on="fecha", how="inner")

# Variables derivadas
master_mensual["anio"] = master_mensual["fecha"].dt.year
master_mensual["mes"] = master_mensual["fecha"].dt.month
master_mensual["ratio_moda_vs_total"] = (
    master_mensual["indice_retail_moda"] / master_mensual["indice_retail_total"]
)

# Diferencia absoluta
master_mensual["dif_moda_vs_total"] = (
    master_mensual["indice_retail_moda"] - master_mensual["indice_retail_total"]
)

# Ordenar
master_mensual = master_mensual.sort_values("fecha").reset_index(drop=True)

# Guardar versión completa
output_full = DATA_PROCESSED / "capa1_master_mensual_full.csv"
master_mensual.to_csv(output_full, index=False)

# Guardar versión de análisis
master_analysis = master_mensual[
    master_mensual["anio"].between(2015, 2023)
].copy().reset_index(drop=True)

output_analysis = DATA_PROCESSED / "capa1_master_mensual_analysis.csv"
master_analysis.to_csv(output_analysis, index=False)

print("Master mensual full guardado en:")
print(output_full)
print("")
print("Master mensual analysis guardado en:")
print(output_analysis)
print("")
print("Preview analysis:")
print(master_analysis.head(12))
print("")
print("Columnas:")
print(master_analysis.columns.tolist())
print("")
print("Rango temporal:")
print(master_analysis["fecha"].min(), "->", master_analysis["fecha"].max())
print("")
print("Número de filas:", len(master_analysis))