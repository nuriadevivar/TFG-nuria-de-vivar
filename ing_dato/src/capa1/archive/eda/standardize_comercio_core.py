import pandas as pd
from config import DATA_PROCESSED

file_path = DATA_PROCESSED / "comercio_electronico_core.csv"
df = pd.read_csv(file_path)

def map_indicator(indicador: str) -> str:
    text = str(indicador).lower()

    if "empresas que han realizado ventas por comercio electrónico" in text and ".1.1" not in text:
        return "pct_empresas_venden_ecommerce"

    if "% ventas mediante comercio electrónico sobre el total de ventas de las empresas que venden por comercio electrónico" in text:
        return "pct_ventas_ecommerce_sobre_total_empresas_que_venden"

    if "% ventas mediante comercio electrónico sobre el total de ventas" in text:
        return "pct_ventas_ecommerce_sobre_total"

    if "empresas que han realizado ventas mediante páginas web o apps" in text:
        return "pct_empresas_venden_web_apps"

    return "otro"

df["indicador_std"] = df["indicador"].apply(map_indicator)

# Conservamos solo los 4 que nos interesan
df_std = df[df["indicador_std"] != "otro"].copy()

output_path = DATA_PROCESSED / "comercio_electronico_core_std.csv"
df_std.to_csv(output_path, index=False)

print("Dimensiones:", df_std.shape)
print("")
print("Indicadores estándar detectados:")
print(df_std["indicador_std"].value_counts())
print("")
print("Archivo guardado en:")
print(output_path)