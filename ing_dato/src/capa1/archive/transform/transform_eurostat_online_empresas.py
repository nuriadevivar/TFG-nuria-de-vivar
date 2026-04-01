import pandas as pd
from config import RAW_EUROSTAT_ONLINE_EMPRESAS, DATA_PROCESSED
from utils import clean_numeric

def transform_eurostat_online_empresas():
    print("Ruta esperada del archivo:")
    print(RAW_EUROSTAT_ONLINE_EMPRESAS)
    print("")

    if not RAW_EUROSTAT_ONLINE_EMPRESAS.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {RAW_EUROSTAT_ONLINE_EMPRESAS}")

    raw = pd.read_excel(RAW_EUROSTAT_ONLINE_EMPRESAS, sheet_name="Sheet 1", header=None)

    print("Dimensiones raw:", raw.shape)
    print("")
    print("Primeras 15 filas:")
    print(raw.head(15))
    print("")

    # Buscar filas clave
    time_row_idx = None
    geo_row_idx = None
    spain_row_idx = None

    for i in range(len(raw)):
        row_values = [str(v).strip() for v in raw.iloc[i].tolist()]
        joined = " | ".join(row_values)

        if "TIME" in joined and time_row_idx is None:
            time_row_idx = i
        if "GEO (Labels)" in joined and geo_row_idx is None:
            geo_row_idx = i
        if "Spain" in joined and spain_row_idx is None:
            spain_row_idx = i

    if time_row_idx is None:
        raise ValueError("No se encontró la fila TIME.")
    if geo_row_idx is None:
        raise ValueError("No se encontró la fila GEO (Labels).")
    if spain_row_idx is None:
        raise ValueError("No se encontró la fila Spain.")

    print("Fila TIME:", time_row_idx)
    print("Fila GEO:", geo_row_idx)
    print("Fila Spain:", spain_row_idx)

    time_row = raw.iloc[time_row_idx].tolist()
    spain_row = raw.iloc[spain_row_idx].tolist()

    # Buscar todas las columnas con año + valor válido en Spain
    detected_rows = []

    for idx, (time_val, spain_val) in enumerate(zip(time_row, spain_row)):
        if pd.isna(time_val):
            continue

        try:
            year_num = float(str(time_val).strip())
            if 2000 <= year_num <= 2100:
                value_num = clean_numeric(spain_val)
                if pd.notna(value_num):
                    detected_rows.append({
                        "anio": int(year_num),
                        "geo": "Spain",
                        "valor_pct": value_num
                    })
        except:
            continue

    if not detected_rows:
        raise ValueError("No se encontraron columnas con año y valor válido para Spain.")

    result = pd.DataFrame(detected_rows)

    # Si hubiera duplicados de año, nos quedamos con el último
    result = result.sort_values("anio").drop_duplicates(subset=["anio"], keep="last").reset_index(drop=True)

    result["fuente"] = "Eurostat"
    result["indicador"] = "participacion_empresas_ventas_online_pct"
    result = result[result["anio"].between(2015, 2023)].copy().reset_index(drop=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "eurostat_online_empresas_clean.csv"
    result.to_csv(output_path, index=False)

    print("")
    print("Años detectados:")
    print(result["anio"].tolist())
    print("")
    print("Archivo guardado en:")
    print(output_path)
    print("")
    print("Preview del dataset limpio:")
    print(result)

    return result

if __name__ == "__main__":
    df_clean = transform_eurostat_online_empresas()