import pandas as pd
from config import DATA_PROCESSED

file_path = DATA_PROCESSED / "comercio_electronico_clean.csv"

df = pd.read_csv(file_path)

print("Dimensiones:", df.shape)
print("")
print("Años disponibles:", sorted(df["anio"].dropna().unique().tolist()))
print("")
print("Tamaños de empresa:", df["tamano_empresa"].dropna().unique().tolist())
print("")
print("Número de indicadores únicos:", df["indicador"].nunique())
print("")
print("Primeros 50 indicadores únicos:")
print(df["indicador"].dropna().unique()[:50])

# Guardar listado completo
indicadores = pd.DataFrame({"indicador": sorted(df["indicador"].dropna().unique())})
output_path = DATA_PROCESSED / "comercio_electronico_indicadores_unicos.csv"
indicadores.to_csv(output_path, index=False)

print("")
print("Listado de indicadores guardado en:")
print(output_path)