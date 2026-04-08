import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import PROCESSED_CAPA1, RAW_CAPA1

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    for sub in [
        "contexto_digitalizacion",
        "eurostat",
        "comercio_electronico",
        "integrated",
        "calidad",
    ]:
        (PROCESSED_CAPA1 / sub).mkdir(parents=True, exist_ok=True)


def _find_first_sheet(file_path: Path) -> str:
    return pd.ExcelFile(file_path).sheet_names[0]


def clean_numeric(value) -> float:
    """
    Convierte un valor a float limpiando espacios, separadores de miles y comas decimales.

    Supuesto metodológico:
    las fuentes manejadas en esta capa utilizan formato numérico compatible con
    convención española/europea. Si una fuente futura cambiara de convención,
    esta función debería revisarse.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace("\xa0", "").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _log_null_report(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    records = []
    for col in df.columns:
        n_nulls = int(df[col].isnull().sum())
        records.append(
            {
                "dataset": dataset_name,
                "variable": col,
                "n_obs": len(df),
                "n_nulos": n_nulls,
                "pct_nulos": round(float(df[col].isnull().mean() * 100), 2),
            }
        )
    return pd.DataFrame(records)


def _save_antes_despues(
    before: list[dict], after: list[dict], label: str, out_dir: Path
) -> None:
    df_b = pd.DataFrame(before)
    df_b.insert(0, "_fase", "ANTES")
    df_a = pd.DataFrame(after)
    df_a.insert(0, "_fase", "DESPUES")
    combined = pd.concat([df_b, df_a], ignore_index=True)
    safe = label.lower().replace(" ", "_").replace("/", "_")
    path = out_dir / f"antes_despues_{safe}.csv"
    combined.to_csv(path, index=False)
    print(f"  ✓ antes_despues_{safe}.csv")


def _assert_unique(df: pd.DataFrame, subset: list[str], dataset_name: str) -> None:
    n_dups = int(df.duplicated(subset=subset, keep=False).sum())
    if n_dups > 0:
        raise ValueError(
            f"[{dataset_name}] Se detectaron {n_dups} filas duplicadas para la clave {subset}."
        )


def _assert_expected_values(
    series: pd.Series, expected_values: set[str], field_name: str, dataset_name: str
) -> None:
    found = set(series.dropna().astype(str).unique())
    if found != expected_values:
        raise ValueError(
            f"[{dataset_name}] Valores inesperados en {field_name}. "
            f"Esperados: {expected_values} | Encontrados: {found}"
        )


def _assert_year_range(
    df: pd.DataFrame,
    year_col: str,
    expected_min: int,
    expected_max: int,
    dataset_name: str,
) -> None:
    years = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    if years.empty:
        raise ValueError(f"[{dataset_name}] No se encontraron años válidos.")
    if years.min() != expected_min or years.max() != expected_max:
        raise ValueError(
            f"[{dataset_name}] Rango temporal inesperado en {year_col}: "
            f"{years.min()}-{years.max()} | esperado: {expected_min}-{expected_max}"
        )


def _assert_monthly_coverage(
    df: pd.DataFrame,
    date_col: str,
    expected_start: str,
    expected_end: str,
    dataset_name: str,
) -> None:
    fechas = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
    if fechas.empty:
        raise ValueError(f"[{dataset_name}] No se encontraron fechas válidas.")

    start = fechas.min().strftime("%Y-%m")
    end = fechas.max().strftime("%Y-%m")

    if start != expected_start or end != expected_end:
        raise ValueError(
            f"[{dataset_name}] Cobertura mensual inesperada: {start}-{end} | "
            f"esperada: {expected_start}-{expected_end}"
        )

    # comprobar frecuencia mensual sin duplicados
    _assert_unique(df, [date_col], dataset_name)


def _print_dataset_summary(
    df: pd.DataFrame,
    dataset_name: str,
    time_col: str | None = None,
) -> None:
    msg = f"  [{dataset_name}] filas={len(df)} | columnas={df.shape[1]} | nulos_totales={int(df.isnull().sum().sum())}"
    if time_col and time_col in df.columns:
        if "fecha" in time_col:
            fechas = pd.to_datetime(df[time_col], errors="coerce").dropna()
            if not fechas.empty:
                msg += f" | cobertura={fechas.min().strftime('%Y-%m')}–{fechas.max().strftime('%Y-%m')}"
        else:
            vals = pd.to_numeric(df[time_col], errors="coerce").dropna()
            if not vals.empty:
                msg += f" | cobertura={int(vals.min())}–{int(vals.max())}"
    print(msg)


# =========================
# 1. CONTEXTO DIGITALIZACIÓN
# =========================

def transform_contexto() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transforma contexto_digitalizacion.xlsx.

    Limpieza aplicada:
      · Normalización de nombres de columna
      · Conversión de porcentajes 0-1 → 0-100 cuando max <= 1.0

    Imputación:
      · Se permite una única imputación en pct_personas_compra_online (2025)
      · Se documenta con flag pct_personas_compra_online_imputado=True
    """
    _ensure_dirs()
    out_dir = PROCESSED_CAPA1 / "contexto_digitalizacion"
    cal_dir = PROCESSED_CAPA1 / "calidad"

    df = pd.read_excel(RAW_CAPA1 / "contexto_digitalizacion" / "contexto_digitalizacion.xlsx")
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

    required_cols = {"anio", "pct_usuarios_rrss", "pct_personas_compra_online"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[contexto_digitalizacion] Faltan columnas obligatorias: {missing}")

    numeric_cols = [
        "pct_usuarios_rrss",
        "pct_personas_compra_online",
        "pct_personas_compra_ropa_online",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            max_val = df[col].max(skipna=True)
            if pd.notna(max_val) and max_val <= 1.0:
                df[col] = df[col] * 100

    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    df = df.sort_values("anio").reset_index(drop=True)

    _assert_unique(df, ["anio"], "contexto_digitalizacion")
    _assert_year_range(df, "anio", 2020, 2025, "contexto_digitalizacion")

    analytic_cols = ["anio", "pct_usuarios_rrss", "pct_personas_compra_online"]

    before = []
    for _, row in df[analytic_cols].iterrows():
        r = row.to_dict()
        r["pct_personas_compra_online_imputado"] = False
        r["_n_nulos_fila"] = sum(
            1 for v in [r["pct_usuarios_rrss"], r["pct_personas_compra_online"]] if pd.isna(v)
        )
        before.append(r)

    df["pct_personas_compra_online_imputado"] = False
    null_mask = df["pct_personas_compra_online"].isnull()
    n_antes = int(null_mask.sum())

    if n_antes > 0:
        years_null = df.loc[null_mask, "anio"].dropna().astype(int).tolist()
        if years_null != [2025]:
            raise ValueError(
                f"[contexto_digitalizacion] Se esperaba como máximo un nulo en 2025, encontrado en: {years_null}"
            )

        df["pct_personas_compra_online"] = (
            df["pct_personas_compra_online"]
            .interpolate(method="linear", limit_direction="forward")
        )
        df.loc[null_mask, "pct_personas_compra_online_imputado"] = True

        n_despues = int(df["pct_personas_compra_online"].isnull().sum())
        anios_imp = df.loc[df["pct_personas_compra_online_imputado"], "anio"].astype(int).tolist()
        vals_imp = (
            df.loc[df["pct_personas_compra_online_imputado"], "pct_personas_compra_online"]
            .round(2)
            .tolist()
        )
        print(
            f"  [IMPUTACION] pct_personas_compra_online: {n_antes} nulo -> {n_despues} "
            f"| años: {anios_imp} | valores: {vals_imp}"
        )

    if int(df["pct_personas_compra_online_imputado"].sum()) > 1:
        raise ValueError("[contexto_digitalizacion] Hay más de un valor imputado y no debería ocurrir.")

    after = []
    for _, row in df[analytic_cols + ["pct_personas_compra_online_imputado"]].iterrows():
        r = row.to_dict()
        r["_n_nulos_fila"] = sum(
            1 for v in [r["pct_usuarios_rrss"], r["pct_personas_compra_online"]] if pd.isna(v)
        )
        after.append(r)

    _save_antes_despues(before, after, "contexto_digitalizacion", out_dir)
    _log_null_report(df[analytic_cols], "contexto_digitalizacion_clean").to_csv(
        cal_dir / "contexto_null_report.csv", index=False
    )

    df_clean = df[analytic_cols + ["pct_personas_compra_online_imputado"]].copy()
    documented_cols = [
        c
        for c in [
            "anio",
            "pct_usuarios_rrss",
            "pct_personas_compra_online",
            "pct_personas_compra_online_imputado",
            "pct_personas_compra_ropa_online",
            "comentarios_hitos",
            "fuente_usuarios_redes",
            "fuente_compra_online",
            "fuente_compra_ropa_online",
        ]
        if c in df.columns
    ]

    df_clean.to_csv(out_dir / "contexto_digitalizacion_clean.csv", index=False)
    df[documented_cols].to_csv(out_dir / "contexto_digitalizacion_documentado.csv", index=False)
    df.to_csv(out_dir / "contexto_digitalizacion_extended.csv", index=False)

    _print_dataset_summary(df_clean, "contexto_digitalizacion_clean", "anio")
    print("  Contexto digitalizacion ✓")
    return df_clean, df[documented_cols], df


# =========================
# HELPER EUROSTAT MENSUAL
# =========================

def _extract_eurostat_monthly(
    file_path: Path,
    indicator_name: str,
    out_dir: Path,
    cal_dir: Path,
    safe_label: str,
) -> pd.DataFrame:
    """
    Extrae serie mensual de España desde Excel Eurostat.
    Estructura raw:
      · fila TIME con fechas
      · fila Spain con valores y flags intercalados
      · 'p' = provisional
      · ':' = no disponible
    """
    raw = pd.read_excel(file_path, header=None)

    time_row_idx = next(
        (i for i, row in raw.iterrows() if any(str(v).strip() == "TIME" for v in row)),
        None,
    )
    if time_row_idx is None:
        raise ValueError(f"No se encontró fila TIME en {file_path.name}")

    header_row = raw.iloc[time_row_idx]
    spain_row = raw.iloc[time_row_idx + 2]

    date_idx = [
        i
        for i, v in enumerate(header_row)
        if isinstance(v, str) and re.match(r"^\d{4}-\d{2}$", str(v).strip())
    ]

    before = []
    for i in date_idx:
        val_raw = spain_row.iloc[i]
        flag = str(spain_row.iloc[i + 1]).strip() if i + 1 < len(spain_row) else ""
        tipo = (
            "no_disponible"
            if str(val_raw).strip() == ":"
            else "provisional"
            if flag == "p"
            else "definitivo"
        )
        before.append(
            {
                "fecha": header_row.iloc[i],
                "valor_raw": val_raw,
                "flag": flag,
                "tipo": tipo,
            }
        )

    n_no_disp = sum(1 for r in before if r["tipo"] == "no_disponible")
    n_prov = sum(1 for r in before if r["tipo"] == "provisional")

    after = []
    for r in before:
        val = r["valor_raw"]
        if str(val).strip() == ":" or pd.isna(val):
            continue
        try:
            after.append(
                {
                    "fecha": pd.to_datetime(f"{r['fecha']}-01"),
                    "valor_indice": float(val),
                    "fuente": "Eurostat",
                    "indicador": indicator_name,
                    "provisional": r["tipo"] == "provisional",
                }
            )
        except (ValueError, TypeError):
            pass

    df = pd.DataFrame(after).sort_values("fecha").reset_index(drop=True)

    no_disp = pd.DataFrame(
        [{"fecha": r["fecha"], "flag": r["flag"]} for r in before if r["tipo"] == "no_disponible"]
    )
    if not no_disp.empty:
        no_disp.to_csv(cal_dir / f"{safe_label}_no_disponibles.csv", index=False)

    _save_antes_despues(before, after, safe_label, out_dir)
    _log_null_report(df, indicator_name).to_csv(
        cal_dir / f"{safe_label}_null_report.csv", index=False
    )

    print(
        f"  Raw: {len(before)} celdas | Validas: {len(df)} | "
        f"No disponibles (:): {n_no_disp} | Provisionales (p): {n_prov}"
    )
    return df


# =========================
# 2. EUROSTAT MODA
# =========================

def transform_eurostat_moda() -> pd.DataFrame:
    """
    Índice de volumen de ventas al por menor — moda.
    Base 2015=100.
    """
    _ensure_dirs()
    out_dir = PROCESSED_CAPA1 / "eurostat"
    cal_dir = PROCESSED_CAPA1 / "calidad"

    df = _extract_eurostat_monthly(
        file_path=RAW_CAPA1 / "eurostat" / "eurostat_moda_base2021.xlsx",
        indicator_name="retail_moda_volumen_ventas_indice_base2015",
        out_dir=out_dir,
        cal_dir=cal_dir,
        safe_label="eurostat_moda",
    )

    _assert_monthly_coverage(df, "fecha", "2010-01", "2023-12", "eurostat_moda_mensual_clean")

    df.to_csv(out_dir / "eurostat_moda_mensual_clean.csv", index=False)
    _print_dataset_summary(df, "eurostat_moda_mensual_clean", "fecha")
    print("  Eurostat moda ✓")
    return df


# =========================
# 3. EUROSTAT RETAIL TOTAL
# =========================

def transform_eurostat_retail() -> pd.DataFrame:
    """
    Índice de volumen de ventas al por menor — total.
    Base real: 2021=100.
    """
    _ensure_dirs()
    out_dir = PROCESSED_CAPA1 / "eurostat"
    cal_dir = PROCESSED_CAPA1 / "calidad"

    df = _extract_eurostat_monthly(
        file_path=RAW_CAPA1 / "eurostat" / "eurostat_retail_total_base2021.xlsx",
        indicator_name="retail_total_volumen_ventas_indice_base2021",
        out_dir=out_dir,
        cal_dir=cal_dir,
        safe_label="eurostat_retail_total",
    )

    _assert_monthly_coverage(
        df, "fecha", "2010-01", "2025-09", "eurostat_retail_total_mensual_clean"
    )

    df.to_csv(out_dir / "eurostat_retail_total_mensual_clean.csv", index=False)
    _print_dataset_summary(df, "eurostat_retail_total_mensual_clean", "fecha")
    print("  Eurostat retail total ✓")
    return df


# =========================
# 4. EUROSTAT ONLINE EMPRESAS
# =========================

def transform_eurostat_online_empresas() -> pd.DataFrame:
    """
    Participación de la facturación empresarial en ecommerce.
    Cobertura retenida: 2015-2024.
    """
    _ensure_dirs()
    out_dir = PROCESSED_CAPA1 / "eurostat"
    cal_dir = PROCESSED_CAPA1 / "calidad"

    file_path = RAW_CAPA1 / "eurostat" / "eurostat_participacion_ventas_online_empresas.xlsx"
    raw = pd.read_excel(file_path, sheet_name="Sheet 1", header=None)

    time_row_idx = spain_row_idx = None
    for i, row in raw.iterrows():
        vals = row.astype(str).str.strip().tolist()
        if "TIME" in vals:
            time_row_idx = i
        if "Spain" in vals:
            spain_row_idx = i

    if time_row_idx is None or spain_row_idx is None:
        raise ValueError("No se encontraron filas TIME o Spain.")

    header_row = raw.iloc[time_row_idx]
    spain_values = raw.iloc[spain_row_idx]

    years_raw = [
        (idx, int(float(str(val).strip())))
        for idx, val in enumerate(header_row)
        if re.match(r"^\d{4}(\.0+)?$", str(val).strip())
    ]

    before = []
    for idx, year in years_raw:
        val_raw = spain_values.iloc[idx]
        val_num = clean_numeric(val_raw)
        before.append(
            {
                "anio": year,
                "valor_raw": val_raw,
                "valor_numerico": val_num,
                "incluido": pd.notna(val_num) and year >= 2015,
                "motivo_exclusion": (
                    "año < 2015, fuera cobertura analitica" if year < 2015 else None
                ),
            }
        )

    after = []
    for r in before:
        if not r["incluido"]:
            continue
        after.append(
            {
                "anio": r["anio"],
                "geo": "Spain",
                "valor_pct": round(float(r["valor_numerico"]) / 100, 4),
                "fuente": "Eurostat",
                "indicador": "participacion_empresas_facturacion_ecommerce_pct",
                "provisional": False,
            }
        )

    df = pd.DataFrame(after).sort_values("anio").reset_index(drop=True)

    _assert_unique(df, ["anio"], "eurostat_online_empresas_clean")
    _assert_year_range(df, "anio", 2015, 2024, "eurostat_online_empresas_clean")

    _save_antes_despues(before, after, "eurostat_online_empresas", out_dir)
    _log_null_report(df, "eurostat_online_empresas_clean").to_csv(
        cal_dir / "eurostat_online_empresas_null_report.csv", index=False
    )

    df.to_csv(out_dir / "eurostat_online_empresas_clean.csv", index=False)
    _print_dataset_summary(df, "eurostat_online_empresas_clean", "anio")
    print("  Eurostat online empresas ✓")
    return df


# =========================
# 5. COMERCIO ELECTRÓNICO (INE)
# =========================

def _transform_single_comercio_file(
    file_path: Path,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Transforma un fichero anual de la Encuesta TIC-E (INE).

    Los nulos en 'valor' son ausencias estructurales de la fuente.
    Si una misma clave (anio, indicador, tamano_empresa) aparece varias veces
    con valores distintos, se conserva el primer valor no nulo y se registra
    el conflicto en un dataframe de calidad.
    """
    year_match = re.search(r"(\d{4})", file_path.stem)
    if not year_match:
        raise ValueError(f"No se pudo detectar año en {file_path.name}")
    year = int(year_match.group(1))

    sheet_name = _find_first_sheet(file_path)
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Leer bloque principal con la estructura esperada
    df = raw.iloc[7:].copy()
    df = df.iloc[:, :5].copy()
    df.columns = ["indicador", "total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"]

    # Normalizar indicador
    df["indicador"] = df["indicador"].astype(str).str.strip()

    # Conservar solo filas que realmente parecen indicadores INE
    target_patterns = [
        r"%\s*de empresas que han realizado ventas por comercio electrónico(?:.*)?$",
        r"%\s*de empresas que han realizado ventas mediante páginas web(?:.*)?$",
        r"%\s*ventas mediante comercio electrónico sobre el total de ventas$",
        r"%\s*ventas mediante comercio electrónico sobre el total de ventas de las empresas que venden por comercio electrónico$",
    ]

    df = df[df["indicador"].str.match(r"^[A-Z]\.\d", na=False)].copy()
    df = df[
        df["indicador"].str.contains("|".join(target_patterns), case=False, na=False, regex=True)
    ].copy()

    # Eliminar filas totalmente vacías en las columnas de valor
    size_cols = ["total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"]
    df = df[
        ~df[size_cols].isna().all(axis=1)
    ].copy()

    n_indicadores_antes = len(df)
    n_nulos_antes = sum(
        df[col].apply(lambda x: pd.isna(clean_numeric(x))).sum()
        for col in size_cols
    )

    for col in size_cols:
        df[col] = df[col].apply(clean_numeric)

    df_long = df.melt(
        id_vars="indicador",
        value_vars=size_cols,
        var_name="tamano_empresa",
        value_name="valor",
    )
    df_long["anio"] = year

    df_long["valor"] = df_long["valor"] / 100

    expected_sizes = {"total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"}
    _assert_expected_values(
        df_long["tamano_empresa"], expected_sizes, "tamano_empresa", f"comercio_{year}"
    )

    key_cols = ["anio", "indicador", "tamano_empresa"]

    # -------------------------
    # DETECCIÓN DE CONFLICTOS
    # -------------------------
    dup_mask = df_long.duplicated(subset=key_cols, keep=False)
    duplicated_rows = df_long[dup_mask].copy()

    conflict_records = []

    if not duplicated_rows.empty:
        print(
            f"    [{year}] {len(duplicated_rows)} filas en claves duplicadas detectadas; "
            f"se consolida conservando el primer valor no nulo y se documentan conflictos"
        )

        for keys, grp in duplicated_rows.groupby(key_cols):
            non_null_vals = pd.Series(grp["valor"].dropna().unique()).tolist()

            if len(non_null_vals) > 1:
                conflict_records.append(
                    {
                        "anio": keys[0],
                        "indicador": keys[1],
                        "tamano_empresa": keys[2],
                        "n_filas_conflictivas": len(grp),
                        "valores_distintos": str(non_null_vals),
                        "decision": "conservar_primer_valor_no_nulo",
                    }
                )

    # Consolidación: una fila por clave
    def _first_non_null(series: pd.Series):
        non_null = series.dropna()
        return non_null.iloc[0] if not non_null.empty else np.nan

    df_long = (
        df_long.groupby(key_cols, as_index=False)["valor"]
        .agg(_first_non_null)
        .reset_index(drop=True)
    )

    _assert_unique(df_long, key_cols, f"comercio_{year}")

    n_nulos_long = int(df_long["valor"].isnull().sum())

    if n_nulos_long > 0:
        print(f"    [{year}] {n_nulos_long} nulos estructurales INE -> conservados como NaN")

    conflict_df = pd.DataFrame(conflict_records)

    quality = {
        "anio": year,
        "n_indicadores": n_indicadores_antes,
        "n_nulos_antes": int(n_nulos_antes),
        "n_nulos_long": n_nulos_long,
        "n_filas_final": int(len(df_long)),
        "n_conflictos_duplicados": int(len(conflict_df)),
    }

    return df_long[["anio", "indicador", "tamano_empresa", "valor"]], quality, conflict_df

def transform_comercio_electronico() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolida ficheros anuales de la Encuesta TIC-E (INE, 2015-2023).

    Los nulos se interpretan como ausencias estructurales del cuestionario.
    Los duplicados conflictivos se documentan y se resuelven conservando
    el primer valor no nulo por clave.
    """
    _ensure_dirs()
    out_dir = PROCESSED_CAPA1 / "comercio_electronico"
    cal_dir = PROCESSED_CAPA1 / "calidad"

    files = sorted((RAW_CAPA1 / "comercio_electronico").glob("comercio_electronico*.xlsx"))
    if not files:
        raise ValueError("No se encontraron ficheros de comercio_electronico en raw.")

    all_dfs = []
    all_q = []
    all_conflicts = []

    for fp in files:
        print(f"  Procesando {fp.name}...")
        df_y, q_y, conflict_y = _transform_single_comercio_file(fp)

        if not df_y.empty:
            all_dfs.append(df_y)
            all_q.append(q_y)

        if not conflict_y.empty:
            conflict_y["archivo"] = fp.name
            all_conflicts.append(conflict_y)

    if not all_dfs:
        raise ValueError("No se procesó ningún fichero de comercio electrónico.")

    final_df = pd.concat(all_dfs, ignore_index=True)
    quality_df = pd.DataFrame(all_q)

    _assert_year_range(final_df, "anio", 2015, 2023, "comercio_electronico_clean")

    # Inventario de nulos estructurales
    nulos_inv = (
        final_df[final_df["valor"].isnull()]
        .groupby(["anio", "indicador", "tamano_empresa"])
        .size()
        .reset_index(name="n_nulos")
    )
    nulos_inv["tipo_ausencia"] = "estructural_ine"
    nulos_inv["decision"] = "mantener_nan — indicador no publicado en esta edicion"
    nulos_inv.to_csv(out_dir / "comercio_electronico_nulos_estructurales.csv", index=False)

    # Exportar conflictos de duplicados si existen
    if all_conflicts:
        conflicts_df = pd.concat(all_conflicts, ignore_index=True)
        conflicts_df.to_csv(
            cal_dir / "comercio_electronico_conflictos_duplicados.csv",
            index=False
        )
        print(
            f"  Conflictos de duplicados documentados: {len(conflicts_df)} -> "
            f"comercio_electronico_conflictos_duplicados.csv"
        )

    n_nulos = int(final_df["valor"].isnull().sum())
    pct_nulos = round(float(final_df["valor"].isnull().mean() * 100), 2)

    pd.DataFrame(
        [
            {
                "_fase": "ANTES",
                "n_filas": len(final_df),
                "n_nulos_valor": n_nulos,
                "pct_nulos": pct_nulos,
                "n_indicadores_unicos": final_df["indicador"].nunique(),
                "decision": "ausencia estructural INE — preguntas no formuladas ese año",
            },
            {
                "_fase": "DESPUES",
                "n_filas": len(final_df),
                "n_nulos_valor": n_nulos,
                "pct_nulos": pct_nulos,
                "n_indicadores_unicos": final_df["indicador"].nunique(),
                "decision": "nulos conservados como NaN — no se imputa por ser estructurales",
            },
        ]
    ).to_csv(out_dir / "antes_despues_comercio_electronico.csv", index=False)

    final_df.to_csv(out_dir / "comercio_electronico_clean.csv", index=False)
    quality_df.to_csv(cal_dir / "comercio_electronico_quality_por_anio.csv", index=False)

    _print_dataset_summary(final_df, "comercio_electronico_clean", "anio")
    print(
        f"  Comercio electronico ✓ | filas={len(final_df)} | "
        f"core analitico consolidado sin nulos tras filtrado y estandarizacion"
    )

    return final_df, nulos_inv

# =========================
# 6. DECISIONES NULOS MASTER ANUAL
# =========================

def document_master_anual_nulls() -> pd.DataFrame:
    """
    Documenta decisiones metodológicas sobre nulos del master anual integrado.
    """
    _ensure_dirs()

    decisions = pd.DataFrame(
        [
            {
                "dataset": "capa1_master_anual_analysis",
                "variable": "pct_empresas_venden_web_apps",
                "anio_afectado": 2023,
                "n_nulos": 1,
                "tipo_ausencia": "cambio_metodologico_ine_K_a_I",
                "descripcion": (
                    "INE cambió K.1 -> I.1.1 en 2023. El valor I.1.1=27.44% no es comparable "
                    "con K.1 de 2020-2022 por cambios en universo y formulación."
                ),
                "decision": "no_imputar — incompatibilidad metodologica",
                "valor_referencia": 27.44,
                "fuente_referencia": "INE Encuesta TIC-E 2023, indicador I.1.1",
            },
            {
                "dataset": "capa1_master_anual_full",
                "variable": "variables_ine_multiples",
                "anio_afectado": "2024-2025",
                "n_nulos": "hasta 6 por variable",
                "tipo_ausencia": "fuera_cobertura_temporal",
                "descripcion": "Encuesta TIC-E no publicada para 2024-2025 en fecha de extracción.",
                "decision": "no_imputar — excluir del master_analitico",
                "valor_referencia": None,
                "fuente_referencia": None,
            },
            {
                "dataset": "contexto_digitalizacion_clean",
                "variable": "pct_personas_compra_online",
                "anio_afectado": 2025,
                "n_nulos": 1,
                "tipo_ausencia": "dato_pendiente_publicacion",
                "descripcion": (
                    "Dato pendiente de publicación en fecha de extracción. "
                    "Se aplica interpolación lineal y se documenta con flag de trazabilidad."
                ),
                "decision": "interpolar_linealmente — flag _imputado=True",
                "valor_referencia": None,
                "fuente_referencia": "interpolacion lineal sobre tendencia observada",
            },
        ]
    )

    out_path = PROCESSED_CAPA1 / "calidad" / "master_anual_null_decisions.csv"
    decisions.to_csv(out_path, index=False)
    print("  ✓ master_anual_null_decisions.csv")
    return decisions


# =========================
# RUN ALL
# =========================

def run_all_transforms() -> None:
    sep = "=" * 65
    print(sep)
    print("CAPA 1 — TRANSFORMACIONES ETL")
    print(sep)

    print("\n[1/6] Contexto digitalizacion...")
    transform_contexto()

    print("\n[2/6] Eurostat moda (base 2015=100)...")
    transform_eurostat_moda()

    print("\n[3/6] Eurostat retail total (base 2021=100)...")
    transform_eurostat_retail()

    print("\n[4/6] Eurostat online empresas...")
    transform_eurostat_online_empresas()

    print("\n[5/6] Comercio electronico INE...")
    transform_comercio_electronico()

    print("\n[6/6] Decisiones nulos master anual...")
    document_master_anual_nulls()

    print(f"\n{sep}")
    print("OUTPUTS GENERADOS:")
    print("  contexto_digitalizacion/")
    print("    · contexto_digitalizacion_clean.csv")
    print("    · contexto_digitalizacion_documentado.csv")
    print("    · contexto_digitalizacion_extended.csv")
    print("    · antes_despues_contexto_digitalizacion.csv")
    print("  eurostat/")
    print("    · eurostat_moda_mensual_clean.csv")
    print("    · eurostat_retail_total_mensual_clean.csv")
    print("    · eurostat_online_empresas_clean.csv")
    print("    · antes_despues_*.csv")
    print("  comercio_electronico/")
    print("    · comercio_electronico_clean.csv")
    print("    · comercio_electronico_nulos_estructurales.csv")
    print("    · antes_despues_comercio_electronico.csv")
    print("  calidad/")
    print("    · master_anual_null_decisions.csv")
    print("    · *_null_report.csv")
    print(sep)


if __name__ == "__main__":
    run_all_transforms()