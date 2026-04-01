import pandas as pd
from pathlib import Path
from config import RAW_COMERCIO_DIR, DATA_PROCESSED
from utils import clean_numeric

def transform_comercio_2015():
    file_path = RAW_COMERCIO_DIR / "comercio_electronico2015.xlsx"

    print("Ruta del archivo:")
    print(file_path)
    print("")

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {file_path}")

    raw = pd.read_excel(file_path, sheet_name="tabla-0", header=None)

    print("Dimensiones raw:", raw.shape)

    # La cabecera real parece estar en la fila 6
    header_row_idx = 6
    header = raw.iloc[header_row_idx].tolist()

    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = ["indicador", "total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"]

    print("Preview tras asignar cabecera:")
    print(df.head(10))
    print("")

    # Eliminar filas vacías o filas que no son indicadores reales
    df["indicador"] = df["indicador"].astype(str).str.strip()
    df = df[df["indicador"].notna()].copy()
    df = df[~df["indicador"].isin(["nan", "Total Empresas", ""])]
    df = df[df["indicador"].str.startswith("K.", na=False)].copy()

    # Limpiar valores numéricos
    value_cols = ["total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"]
    for col in value_cols:
        df[col] = df[col].apply(clean_numeric)

    # Añadir año
    df["anio"] = 2015

    # Pasar a formato largo
    df_long = df.melt(
        id_vars=["anio", "indicador"],
        value_vars=value_cols,
        var_name="tamano_empresa",
        value_name="valor"
    )

    # Orden de columnas
    df_long = df_long[["anio", "indicador", "tamano_empresa", "valor"]].reset_index(drop=True)

    # Guardar
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "comercio_electronico_2015_clean.csv"
    df_long.to_csv(output_path, index=False)

    print("Archivo guardado en:")
    print(output_path)
    print("")
    print("Preview del dataset limpio:")
    print(df_long.head(12))
    print("")
    print("Número de filas finales:", len(df_long))

    return df_long

if __name__ == "__main__":
    df_clean = transform_comercio_2015()