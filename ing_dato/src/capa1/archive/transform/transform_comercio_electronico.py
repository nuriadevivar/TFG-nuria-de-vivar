import pandas as pd
from pathlib import Path
from config import RAW_COMERCIO_DIR, DATA_PROCESSED
from utils import clean_numeric

def get_first_sheet_name(file_path: Path) -> str:
    xls = pd.ExcelFile(file_path)
    if not xls.sheet_names:
        raise ValueError(f"No se encontraron hojas en el archivo {file_path.name}")
    return xls.sheet_names[0]

def transform_single_comercio_file(file_path: Path) -> pd.DataFrame:
    year_str = "".join(filter(str.isdigit, file_path.stem))
    anio = int(year_str)

    print(f"\nProcesando: {file_path.name} | Año detectado: {anio}")

    sheet_name = get_first_sheet_name(file_path)
    print(f"Hoja detectada: {sheet_name}")

    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    print(f"Dimensiones raw: {raw.shape}")

    header_row_idx = 6
    df = raw.iloc[header_row_idx + 1:].copy()

    # Protección mínima
    if df.shape[1] < 5:
        print(f"[WARNING] Año {anio}: menos de 5 columnas tras la cabecera.")
        return pd.DataFrame()

    df = df.iloc[:, :5].copy()
    df.columns = ["indicador", "total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"]

    # Limpiar columna de indicador
    df["indicador"] = df["indicador"].astype(str).str.strip()

    print("Primeros indicadores antes de filtrar:")
    print(df["indicador"].head(10).tolist())

    # Filtros
    df = df[df["indicador"].notna()].copy()
    df = df[~df["indicador"].isin(["nan", "Total Empresas", ""])]
    df = df[df["indicador"].str.match(r"^[A-Z]\.\d", na=False)].copy()

    print(f"Filas tras filtrar indicadores codificados: {len(df)}")

    if df.empty:
        print(f"[WARNING] Año {anio}: dataframe vacío tras filtros.")
        return pd.DataFrame()

    value_cols = ["total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"]
    for col in value_cols:
        df[col] = df[col].apply(clean_numeric)

    df["anio"] = anio

    df_long = df.melt(
        id_vars=["anio", "indicador"],
        value_vars=value_cols,
        var_name="tamano_empresa",
        value_name="valor"
    )

    df_long = df_long[["anio", "indicador", "tamano_empresa", "valor"]].reset_index(drop=True)

    print(f"Filas finales año {anio}: {len(df_long)}")
    return df_long

def transform_all_comercio():
    files = sorted(RAW_COMERCIO_DIR.glob("comercio_electronico*.xlsx"))

    if not files:
        raise FileNotFoundError(f"No se encontraron archivos en {RAW_COMERCIO_DIR}")

    all_dfs = []

    for file_path in files:
        df_year = transform_single_comercio_file(file_path)

        if df_year.empty:
            print(f"[WARNING] Se omite {file_path.name} porque quedó vacío.")
            continue

        all_dfs.append(df_year)

        year = "".join(filter(str.isdigit, file_path.stem))
        output_year = DATA_PROCESSED / f"comercio_electronico_{year}_clean.csv"
        df_year.to_csv(output_year, index=False)

    if not all_dfs:
        raise ValueError("Todos los dataframes anuales quedaron vacíos.")

    final_df = pd.concat(all_dfs, ignore_index=True)

    output_all = DATA_PROCESSED / "comercio_electronico_clean.csv"
    final_df.to_csv(output_all, index=False)

    print("\nArchivo consolidado guardado en:")
    print(output_all)
    print("\nAños incluidos:", sorted(final_df["anio"].unique().tolist()))
    print("Número de filas finales:", len(final_df))

    return final_df

if __name__ == "__main__":
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_clean = transform_all_comercio()