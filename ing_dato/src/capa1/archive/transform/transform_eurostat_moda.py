import pandas as pd
from config import RAW_EUROSTAT_MODA, DATA_PROCESSED
from utils import clean_numeric

def transform_eurostat_moda():
    print("Ruta esperada del archivo:")
    print(RAW_EUROSTAT_MODA)
    print("")

    if not RAW_EUROSTAT_MODA.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {RAW_EUROSTAT_MODA}")

    # Leer sin cabecera fija
    raw = pd.read_excel(RAW_EUROSTAT_MODA, header=None)

    print("Dimensiones raw:", raw.shape)

    # Buscar la fila donde aparece TIME
    time_row_idx = None
    for i in range(len(raw)):
        row_values = raw.iloc[i].astype(str).tolist()
        if any("TIME" in str(v) for v in row_values):
            time_row_idx = i
            break

    if time_row_idx is None:
        raise ValueError("No se encontró la fila de TIME en el archivo Eurostat moda.")

    print("Fila donde empieza TIME:", time_row_idx)

    # Usar esa fila como cabecera
    header = raw.iloc[time_row_idx].tolist()
    df = raw.iloc[time_row_idx + 1:].copy()
    df.columns = header

    print("Columnas detectadas:")
    print(df.columns.tolist()[:10])
    print("...")

    # Renombrar primera columna
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "geo"})

    # Filtrar España
    df_spain = df[df["geo"].astype(str).str.contains("Spain", case=False, na=False)].copy()

    if df_spain.empty:
        raise ValueError("No se encontró fila de Spain.")

    print("Filas encontradas para Spain:", len(df_spain))

    # Nos quedamos con la primera fila de España
    row = df_spain.iloc[0]

    # Detectar columnas que sean fechas YYYY-MM
    date_cols = [c for c in df.columns if isinstance(c, str) and len(c) == 7 and c[4] == "-"]

    print("Número de columnas fecha detectadas:", len(date_cols))

    long_rows = []
    for col in date_cols:
        value = row[col]
        long_rows.append({
            "fecha": col,
            "valor_indice": clean_numeric(value)
        })

    result = pd.DataFrame(long_rows)

    # Convertir fecha
    result["fecha"] = pd.to_datetime(result["fecha"], format="%Y-%m", errors="coerce")

    # Filtrar periodo útil
    result = result[result["fecha"].dt.year.between(2015, 2023)].copy()

    # Resetear índice
    result = result.reset_index(drop=True)

    # Añadir metadatos útiles
    result["fuente"] = "Eurostat"
    result["indicador"] = "retail_moda_volumen_ventas_indice_base2015"

    # Guardar
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "eurostat_moda_mensual_clean.csv"
    result.to_csv(output_path, index=False)

    print("Archivo guardado en:")
    print(output_path)
    print("")
    print("Preview del dataset limpio:")
    print(result.head())
    print(result.tail())

    return result

if __name__ == "__main__":
    df_clean = transform_eurostat_moda()