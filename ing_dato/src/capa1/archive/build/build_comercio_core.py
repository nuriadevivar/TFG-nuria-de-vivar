import pandas as pd
from config import DATA_PROCESSED

file_path = DATA_PROCESSED / "comercio_electronico_clean.csv"
df = pd.read_csv(file_path)

patterns = [
    "empresas que han realizado ventas por comercio electrónico",
    "ventas mediante comercio electrónico (miles de euros)",
    "% ventas mediante comercio electrónico sobre el total de ventas",
    "empresas que han realizado ventas mediante páginas web o apps",
]

# Filtrar indicadores clave
mask = df["indicador"].str.contains("|".join(patterns), case=False, na=False)
df_core = df[mask].copy()

# Quitar algunos que ahora no queremos
exclude_patterns = [
    ">= 1%",
    ">= 2%",
    ">= 5%",
    ">= 10%",
    ">= 25%",
    ">= 50%",
    "límites",
    "sin ventas",
    "mensajes tipo EDI"  # quítalo si de momento no quieres esta parte
]

exclude_mask = df_core["indicador"].str.contains("|".join(exclude_patterns), case=False, na=False)
df_core = df_core[~exclude_mask].copy()

output_path = DATA_PROCESSED / "comercio_electronico_core.csv"
df_core.to_csv(output_path, index=False)

print("Dimensiones core:", df_core.shape)
print("")
print("Indicadores incluidos:")
print(df_core["indicador"].drop_duplicates().tolist())
print("")
print("Archivo guardado en:")
print(output_path)