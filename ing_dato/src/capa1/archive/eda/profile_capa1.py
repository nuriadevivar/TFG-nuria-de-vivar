import pandas as pd
from pathlib import Path
from config import DATA_PROCESSED, OUTPUT_TABLES

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "contexto_digitalizacion_clean": DATA_PROCESSED / "contexto" / "contexto_digitalizacion_clean.csv",
    "contexto_digitalizacion_extended": DATA_PROCESSED / "contexto" / "contexto_digitalizacion_extended.csv",
    "eurostat_moda_mensual_clean": DATA_PROCESSED / "eurostat" / "eurostat_moda_mensual_clean.csv",
    "eurostat_retail_total_mensual_clean": DATA_PROCESSED / "eurostat" / "eurostat_retail_total_mensual_clean.csv",
    "eurostat_online_empresas_clean": DATA_PROCESSED / "eurostat" / "eurostat_online_empresas_clean.csv",
    "comercio_electronico_core_std": DATA_PROCESSED / "comercio_electronico" / "comercio_electronico_core_std.csv",
    "capa1_master_anual_analysis": DATA_PROCESSED / "capa1_master_anual_analysis.csv",
    "capa1_master_mensual_analysis": DATA_PROCESSED / "capa1_master_mensual_analysis.csv",
}

summary_rows = []
null_rows = []
numeric_rows = []

for name, path in DATASETS.items():
    df = pd.read_csv(path)

    # Dimensiones y duplicados
    n_rows, n_cols = df.shape
    duplicated_rows = int(df.duplicated().sum())

    # Rango temporal
    start_period = None
    end_period = None
    if "fecha" in df.columns:
        fechas = pd.to_datetime(df["fecha"], errors="coerce")
        if fechas.notna().any():
            start_period = str(fechas.min().date())
            end_period = str(fechas.max().date())
    elif "anio" in df.columns:
        anios = pd.to_numeric(df["anio"], errors="coerce")
        if anios.notna().any():
            start_period = str(int(anios.min()))
            end_period = str(int(anios.max()))

    summary_rows.append({
        "dataset": name,
        "rows": n_rows,
        "cols": n_cols,
        "duplicated_rows": duplicated_rows,
        "start_period": start_period,
        "end_period": end_period,
    })

    # Nulos por columna
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        pct_null = (n_null / len(df) * 100) if len(df) > 0 else 0
        null_rows.append({
            "dataset": name,
            "column": col,
            "null_count": n_null,
            "null_pct": round(pct_null, 2),
            "dtype": str(df[col].dtype),
        })

    # Resumen numérico
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        numeric_rows.append({
            "dataset": name,
            "column": col,
            "count": int(series.count()),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "q25": series.quantile(0.25),
            "median": series.median(),
            "q75": series.quantile(0.75),
            "max": series.max(),
        })

summary_df = pd.DataFrame(summary_rows)
nulls_df = pd.DataFrame(null_rows)
numeric_df = pd.DataFrame(numeric_rows)

summary_df.to_csv(OUTPUT_TABLES / "capa1_profile_summary.csv", index=False)
nulls_df.to_csv(OUTPUT_TABLES / "capa1_profile_nulls.csv", index=False)
numeric_df.to_csv(OUTPUT_TABLES / "capa1_profile_numeric.csv", index=False)

print("Profiling completado.")
print("Archivos generados:")
print(OUTPUT_TABLES / "capa1_profile_summary.csv")
print(OUTPUT_TABLES / "capa1_profile_nulls.csv")
print(OUTPUT_TABLES / "capa1_profile_numeric.csv")
print("")
print(summary_df)