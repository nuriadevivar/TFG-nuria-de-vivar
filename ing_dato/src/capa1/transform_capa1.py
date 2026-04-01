import re
import warnings
from pathlib import Path

import pandas as pd

from src.common.config import PROCESSED_CAPA1, RAW_CAPA1
from src.common.utils import clean_numeric

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    (PROCESSED_CAPA1 / "contexto_digitalizacion").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA1 / "eurostat").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA1 / "comercio_electronico").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA1 / "integrated").mkdir(parents=True, exist_ok=True)


def _find_first_sheet(file_path: Path) -> str:
    xls = pd.ExcelFile(file_path)
    return xls.sheet_names[0]


def _find_time_row(df_raw: pd.DataFrame) -> int:
    for i, row in df_raw.iterrows():
        row_values = row.astype(str).str.strip().tolist()
        if any(v == "TIME" for v in row_values):
            return i
    raise ValueError("No se encontró la fila TIME.")


def _extract_eurostat_monthly_series(
    file_path: Path,
    indicator_name: str,
) -> pd.DataFrame:
    raw = pd.read_excel(file_path, header=None)

    time_row_idx = _find_time_row(raw)
    header = raw.iloc[time_row_idx].tolist()
    df = raw.iloc[time_row_idx + 1 :].copy()
    df.columns = header

    geo_col = "TIME"
    if geo_col not in df.columns:
        raise ValueError("No se encontró la columna TIME tras asignar cabecera.")

    spain_row = df[df[geo_col].astype(str).str.strip().eq("Spain")]
    if spain_row.empty:
        raise ValueError("No se encontró la fila Spain en el archivo.")

    spain_row = spain_row.iloc[0]

    date_cols = [c for c in df.columns if isinstance(c, str) and re.match(r"^\d{4}-\d{2}$", c)]

    data = []
    for col in date_cols:
        value = clean_numeric(spain_row[col])
        if pd.notna(value):
            data.append(
                {
                    "fecha": pd.to_datetime(f"{col}-01"),
                    "valor_indice": value,
                    "fuente": "Eurostat",
                    "indicador": indicator_name,
                }
            )

    out = pd.DataFrame(data).sort_values("fecha").reset_index(drop=True)
    return out


# =========================
# 1. CONTEXTO
# =========================

def transform_contexto() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    file_path = RAW_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion.xlsx"
    df = pd.read_excel(file_path)

    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "pct_usuarios_rrss": "pct_usuarios_rrss",
        "pct_personas_compra_online": "pct_personas_compra_online",
        "pct_personas_compra_ropa_online": "pct_personas_compra_ropa_online",
        "comentarios_hitos": "comentarios_hitos",
        "fuente_usuarios_redes": "fuente_usuarios_redes",
        "fuente_compra_online": "fuente_compra_online",
        "fuente_compra_ropa_online": "fuente_compra_ropa_online",
        "anio": "anio",
    }
    df = df.rename(columns=rename_map)

    numeric_cols = [
        "pct_usuarios_rrss",
        "pct_personas_compra_online",
        "pct_personas_compra_ropa_online",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    # Homogeneizar escala de porcentajes:
    # si vienen como proporción 0-1, convertir a 0-100
    for col in numeric_cols:
        if col in df.columns:
            max_val = df[col].max(skipna=True)
            if pd.notna(max_val) and max_val <= 1.0:
                df[col] = df[col] * 100

    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")

    df_extended = df.copy()

    analytic_cols = [
        "anio",
        "pct_usuarios_rrss",
        "pct_personas_compra_online",
    ]
    df_clean = df[analytic_cols].copy()

    documented_cols = [
        "anio",
        "pct_usuarios_rrss",
        "pct_personas_compra_online",
        "pct_personas_compra_ropa_online",
        "comentarios_hitos",
        "fuente_usuarios_redes",
        "fuente_compra_online",
        "fuente_compra_ropa_online",
    ]
    df_documented = df[documented_cols].copy()

    out_dir = PROCESSED_CAPA1 / "contexto_digitalizacion"
    df_clean.to_csv(out_dir / "contexto_digitalizacion_clean.csv", index=False)
    df_documented.to_csv(out_dir / "contexto_digitalizacion_documentado.csv", index=False)
    df_extended.to_csv(out_dir / "contexto_digitalizacion_extended.csv", index=False)

    print("Contexto transformado.")
    return df_clean, df_documented, df_extended


# =========================
# 2. EUROSTAT MODA
# =========================

def transform_eurostat_moda() -> pd.DataFrame:
    _ensure_dirs()

    file_path = RAW_CAPA1 / "eurostat" / "eurostat_moda_base2021.xlsx"
    df = _extract_eurostat_monthly_series(
        file_path=file_path,
        indicator_name="retail_moda_volumen_ventas_indice_base2015",
    )

    out_path = PROCESSED_CAPA1 / "eurostat" / "eurostat_moda_mensual_clean.csv"
    df.to_csv(out_path, index=False)

    print("Eurostat moda transformado.")
    return df


# =========================
# 3. EUROSTAT RETAIL TOTAL
# =========================

def transform_eurostat_retail() -> pd.DataFrame:
    _ensure_dirs()

    file_path = RAW_CAPA1 / "eurostat" / "eurostat_retail_total_base2021.xlsx"
    df = _extract_eurostat_monthly_series(
        file_path=file_path,
        indicator_name="retail_total_volumen_ventas_indice_base2015",
    )

    out_path = PROCESSED_CAPA1 / "eurostat" / "eurostat_retail_total_mensual_clean.csv"
    df.to_csv(out_path, index=False)

    print("Eurostat retail total transformado.")
    return df


# =========================
# 4. EUROSTAT ONLINE EMPRESAS
# =========================

def transform_eurostat_online_empresas() -> pd.DataFrame:
    _ensure_dirs()

    file_path = RAW_CAPA1 / "eurostat" / "eurostat_participacion_ventas_online_empresas.xlsx"

    raw = pd.read_excel(file_path, sheet_name="Sheet 1", header=None)

    time_row = None
    geo_row = None
    spain_row = None

    for i, row in raw.iterrows():
        values = row.astype(str).str.strip().tolist()
        if "TIME" in values:
            time_row = i
        if "GEO (Labels)" in values:
            geo_row = i
        if "Spain" in values:
            spain_row = i

    if time_row is None or spain_row is None:
        raise ValueError("No se encontraron filas TIME o Spain en el archivo online empresas.")

    header_row = raw.iloc[time_row]
    spain_values = raw.iloc[spain_row]

    years = []
    for idx, val in enumerate(header_row):
        val_str = str(val).strip()
        if re.match(r"^\d{4}(\.0+)?$", val_str):
            years.append((idx, int(float(val_str))))

    data = []
    for idx, year in years:
        value = clean_numeric(spain_values.iloc[idx])
        if pd.notna(value):
            data.append(
                {
                    "anio": year,
                    "geo": "Spain",
                    "valor_pct": value,
                    "fuente": "Eurostat",
                    "indicador": "participacion_empresas_ventas_online_pct",
                }
            )

    df = pd.DataFrame(data).sort_values("anio").reset_index(drop=True)

    # Si quieres restringir a 2015-2023, como hiciste antes:
    df = df[df["anio"].between(2015, 2023)].reset_index(drop=True)

    out_path = PROCESSED_CAPA1 / "eurostat" / "eurostat_online_empresas_clean.csv"
    df.to_csv(out_path, index=False)

    print("Eurostat online empresas transformado.")
    return df


# =========================
# 5. COMERCIO ELECTRONICO
# =========================

def transform_single_comercio_file(file_path: Path) -> pd.DataFrame:
    year_match = re.search(r"(\d{4})", file_path.stem)
    if not year_match:
        raise ValueError(f"No se pudo detectar el año en {file_path.name}")
    year = int(year_match.group(1))

    sheet_name = _find_first_sheet(file_path)
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    header_row_idx = 6
    header = raw.iloc[header_row_idx].tolist()

    df = raw.iloc[header_row_idx + 1 :].copy()
    df.columns = ["indicador", "total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"]

    df["indicador"] = df["indicador"].astype(str).str.strip()

    # Filtrar indicadores codificados tipo K.1 / M.1 / L.1 / J.1 / I.1...
    pattern = r"^[A-Z]\.\d"
    df = df[df["indicador"].str.match(pattern, na=False)].copy()

    for col in ["total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"]:
        df[col] = df[col].apply(clean_numeric)

    df_long = df.melt(
        id_vars="indicador",
        value_vars=["total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"],
        var_name="tamano_empresa",
        value_name="valor",
    )

    df_long["anio"] = year
    df_long = df_long[["anio", "indicador", "tamano_empresa", "valor"]].reset_index(drop=True)

    return df_long


def transform_comercio_electronico() -> pd.DataFrame:
    _ensure_dirs()

    comercio_dir = RAW_CAPA1 / "comercio_electronico"
    files = sorted(comercio_dir.glob("comercio_electronico*.xlsx"))

    all_dfs = []
    for file_path in files:
        print(f"Procesando {file_path.name}...")
        try:
            df_year = transform_single_comercio_file(file_path)
            if not df_year.empty:
                all_dfs.append(df_year)
        except Exception as e:
            print(f"[WARNING] Error en {file_path.name}: {e}")

    if not all_dfs:
        raise ValueError("No se pudo transformar ningún archivo de comercio electrónico.")

    final_df = pd.concat(all_dfs, ignore_index=True)

    out_path = PROCESSED_CAPA1 / "comercio_electronico" / "comercio_electronico_clean.csv"
    final_df.to_csv(out_path, index=False)

    print("Comercio electrónico transformado.")
    return final_df


# =========================
# RUN ALL
# =========================

def run_all_transforms() -> None:
    transform_contexto()
    transform_eurostat_moda()
    transform_eurostat_retail()
    transform_eurostat_online_empresas()
    transform_comercio_electronico()
    print("Todas las transformaciones de capa 1 completadas.")


if __name__ == "__main__":
    run_all_transforms()