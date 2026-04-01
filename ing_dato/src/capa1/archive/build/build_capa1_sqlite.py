import sqlite3
import pandas as pd
from config import DATA_PROCESSED, BASE_DIR

db_path = BASE_DIR / "data" / "capa1.db"
conn = sqlite3.connect(db_path)

tables = {
    "contexto_digitalizacion_clean": DATA_PROCESSED / "contexto" / "contexto_digitalizacion_clean.csv",
    "contexto_digitalizacion_extended": DATA_PROCESSED / "contexto" / "contexto_digitalizacion_extended.csv",
    "eurostat_moda_mensual_clean": DATA_PROCESSED / "eurostat" / "eurostat_moda_mensual_clean.csv",
    "eurostat_retail_total_mensual_clean": DATA_PROCESSED / "eurostat" / "eurostat_retail_total_mensual_clean.csv",
    "eurostat_online_empresas_clean": DATA_PROCESSED / "eurostat" / "eurostat_online_empresas_clean.csv",
    "comercio_electronico_core_std": DATA_PROCESSED / "comercio_electronico" / "comercio_electronico_core_std.csv",
    "capa1_inventory": DATA_PROCESSED / "capa1_inventory.csv",
    "capa1_master_anual_analysis": DATA_PROCESSED / "capa1_master_anual_analysis.csv",
    "capa1_master_mensual_analysis": DATA_PROCESSED / "capa1_master_mensual_analysis.csv",
}

for table_name, path in tables.items():
    df = pd.read_csv(path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Tabla cargada: {table_name}")

conn.close()

print("")
print("Base SQLite creada en:")
print(db_path)