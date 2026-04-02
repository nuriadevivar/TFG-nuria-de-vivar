"""
transform_capa1.py — Transformación ETL de Capa 1: Contexto macroeconómico y sectorial
 
Fuentes:
  - contexto_digitalizacion.xlsx       (elaboración propia, INE/Eurostat)
  - eurostat_moda_base2021.xlsx         (Eurostat sts_trtu_m, base 2015=100, moda)
  - eurostat_retail_total_base2021.xlsx (Eurostat sts_trtu_m, base 2021=100, retail total)
  - eurostat_participacion_ventas_online_empresas.xlsx (Eurostat tin00110)
  - comercio_electronico{año}.xlsx      (INE, Encuesta TIC-E, 2015-2023)
 
Estructura raw de los Excel de Eurostat (series mensuales):
  - Fila TIME: fechas en columnas impares (1,3,5...), NaN en columnas pares
  - Fila Spain: valor numérico en columna de fecha, flag en columna+1
    · 'p' = dato provisional (conservado, columna provisional=True)
    · ':' = dato no disponible (excluido, documentado en calidad/)
  - La base del retail total es 2021=100 (no 2015=100 como se nombraba antes)
 
Decisiones metodológicas clave:
  - pct_personas_compra_online (2025): interpolación lineal, flag _imputado=True
  - Nulos comercio_electronico: ausencias estructurales INE, no imputables
  - pct_empresas_venden_web_apps (2023): cambio metodológico K→I, no interpolar
  - Valores ':' Eurostat: excluidos con registro en tabla calidad
  - Online empresas: se extiende a 2024 (antes se cortaba en 2023)
"""
 
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
    for sub in ["contexto_digitalizacion", "eurostat", "comercio_electronico",
                "integrated", "calidad"]:
        (PROCESSED_CAPA1 / sub).mkdir(parents=True, exist_ok=True)
 
 
def _find_first_sheet(file_path: Path) -> str:
    return pd.ExcelFile(file_path).sheet_names[0]
 
 
def clean_numeric(value) -> float:
    """Convierte valor a float limpiando separadores de miles y comas decimales."""
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
        records.append({
            "dataset": dataset_name,
            "variable": col,
            "n_obs": len(df),
            "n_nulos": n_nulls,
            "pct_nulos": round(float(df[col].isnull().mean() * 100), 2),
        })
    return pd.DataFrame(records)
 
 
def _save_antes_despues(before: list[dict], after: list[dict],
                        label: str, out_dir: Path) -> None:
    """Exporta tabla comparativa antes/después con valores reales."""
    df_b = pd.DataFrame(before)
    df_b.insert(0, "_fase", "ANTES")
    df_a = pd.DataFrame(after)
    df_a.insert(0, "_fase", "DESPUÉS")
    combined = pd.concat([df_b, df_a], ignore_index=True)
    safe = label.lower().replace(" ", "_").replace("/", "_")
    path = out_dir / f"antes_despues_{safe}.csv"
    combined.to_csv(path, index=False)
    print(f"  ✓ antes_despues_{safe}.csv")
 
 
# =========================
# 1. CONTEXTO DIGITALIZACIÓN
# =========================
 
def transform_contexto() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transforma contexto_digitalizacion.xlsx.
 
    Limpieza aplicada:
      · Normalización de nombres de columna a snake_case
      · Conversión de porcentajes 0-1 → 0-100 cuando max ≤ 1.0
 
    Imputación — único nulo (pct_personas_compra_online, año 2025):
      · Naturaleza: dato pendiente de publicación en el momento de extracción
      · Serie observada 2020-2024: 62.62, 66.63, 67.91, 68.88, 68.94
        Tendencia creciente con desaceleración progresiva → interpolación lineal
        es la estimación más conservadora posible.
      · Trazabilidad: columna pct_personas_compra_online_imputado = True
        permite excluir el valor estimado en análisis de sensibilidad.
 
    Tabla antes/después: muestra valores raw por fila antes y después de imputar.
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
 
    numeric_cols = ["pct_usuarios_rrss", "pct_personas_compra_online",
                    "pct_personas_compra_ropa_online"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            max_val = df[col].max(skipna=True)
            if pd.notna(max_val) and max_val <= 1.0:
                df[col] = df[col] * 100
 
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
 
    analytic_cols = ["anio", "pct_usuarios_rrss", "pct_personas_compra_online"]
 
    # Snapshot ANTES (valores pre-imputación con nulos reales)
    before = []
    for _, row in df[analytic_cols].iterrows():
        r = row.to_dict()
        r["pct_personas_compra_online_imputado"] = False
        r["_n_nulos_fila"] = sum(1 for v in [r["pct_usuarios_rrss"],
                                              r["pct_personas_compra_online"]]
                                 if pd.isna(v))
        before.append(r)
 
    # Imputación: interpolación lineal sobre serie ordenada por año
    df["pct_personas_compra_online_imputado"] = False
    null_mask = df["pct_personas_compra_online"].isnull()
    n_antes = int(null_mask.sum())
 
    if n_antes > 0:
        df_s = df.sort_values("anio").copy()
        df_s["pct_personas_compra_online"] = (
            df_s["pct_personas_compra_online"]
            .interpolate(method="linear", limit_direction="forward")
        )
        df_s.loc[null_mask.loc[df_s.index], "pct_personas_compra_online_imputado"] = True
        df = df_s.sort_values("anio").reset_index(drop=True)
        n_despues = int(df["pct_personas_compra_online"].isnull().sum())
        anios_imp = df.loc[df["pct_personas_compra_online_imputado"], "anio"].tolist()
        vals_imp = df.loc[df["pct_personas_compra_online_imputado"],
                          "pct_personas_compra_online"].round(2).tolist()
        print(f"  [IMPUTACIÓN] pct_personas_compra_online: {n_antes} nulo → "
              f"{n_despues} | años: {anios_imp} | valores: {vals_imp}")
 
    # Snapshot DESPUÉS
    after = []
    for _, row in df[analytic_cols + ["pct_personas_compra_online_imputado"]].iterrows():
        r = row.to_dict()
        r["_n_nulos_fila"] = sum(1 for v in [r["pct_usuarios_rrss"],
                                              r["pct_personas_compra_online"]]
                                 if pd.isna(v))
        after.append(r)
 
    _save_antes_despues(before, after, "contexto_digitalizacion", out_dir)
    _log_null_report(df[analytic_cols], "contexto_digitalizacion_clean").to_csv(
        cal_dir / "contexto_null_report.csv", index=False)
 
    df_clean = df[analytic_cols + ["pct_personas_compra_online_imputado"]].copy()
    documented_cols = [c for c in ["anio", "pct_usuarios_rrss", "pct_personas_compra_online",
                                   "pct_personas_compra_online_imputado",
                                   "pct_personas_compra_ropa_online", "comentarios_hitos",
                                   "fuente_usuarios_redes", "fuente_compra_online",
                                   "fuente_compra_ropa_online"] if c in df.columns]
 
    df_clean.to_csv(out_dir / "contexto_digitalizacion_clean.csv", index=False)
    df[documented_cols].to_csv(out_dir / "contexto_digitalizacion_documentado.csv", index=False)
    df.to_csv(out_dir / "contexto_digitalizacion_extended.csv", index=False)
 
    print("  Contexto digitalización ✓")
    return df_clean, df[documented_cols], df
 
 
# =========================
# HELPER EUROSTAT MENSUAL
# =========================
 
def _extract_eurostat_monthly(file_path: Path, indicator_name: str,
                               out_dir: Path, cal_dir: Path,
                               safe_label: str) -> pd.DataFrame:
    """
    Extrae serie mensual de España desde Excel Eurostat.
 
    Estructura raw: fila TIME con fechas en columnas impares,
    fila Spain con valores y flags intercalados.
    Flags: 'p' = provisional (conservado), ':' = no disponible (excluido).
 
    Genera tabla antes/después con todos los valores raw vs los procesados.
    """
    raw = pd.read_excel(file_path, header=None)
 
    time_row_idx = next(
        (i for i, row in raw.iterrows()
         if any(str(v).strip() == "TIME" for v in row)), None
    )
    if time_row_idx is None:
        raise ValueError(f"No se encontró fila TIME en {file_path.name}")
 
    header_row = raw.iloc[time_row_idx]
    spain_row = raw.iloc[time_row_idx + 2]
 
    date_idx = [i for i, v in enumerate(header_row)
                if isinstance(v, str) and re.match(r"^\d{4}-\d{2}$", str(v).strip())]
 
    # Snapshot ANTES: todas las celdas incluyendo ':' y flags
    before = []
    for i in date_idx:
        val_raw = spain_row.iloc[i]
        flag = str(spain_row.iloc[i + 1]).strip() if i + 1 < len(spain_row) else ""
        tipo = ("no_disponible" if str(val_raw).strip() == ":"
                else "provisional" if flag == "p" else "definitivo")
        before.append({
            "fecha": header_row.iloc[i],
            "valor_raw": val_raw,
            "flag": flag,
            "tipo": tipo,
        })
 
    n_no_disp = sum(1 for r in before if r["tipo"] == "no_disponible")
    n_prov = sum(1 for r in before if r["tipo"] == "provisional")
 
    # Snapshot DESPUÉS: solo valores numéricos válidos
    after = []
    for r in before:
        val = r["valor_raw"]
        if str(val).strip() == ":" or pd.isna(val):
            continue
        try:
            after.append({
                "fecha": pd.to_datetime(f"{r['fecha']}-01"),
                "valor_indice": float(val),
                "fuente": "Eurostat",
                "indicador": indicator_name,
                "provisional": r["tipo"] == "provisional",
            })
        except (ValueError, TypeError):
            pass
 
    df = pd.DataFrame(after).sort_values("fecha").reset_index(drop=True)
 
    # Registrar no disponibles en calidad
    no_disp = pd.DataFrame([{"fecha": r["fecha"], "flag": r["flag"]}
                             for r in before if r["tipo"] == "no_disponible"])
    if not no_disp.empty:
        no_disp.to_csv(cal_dir / f"{safe_label}_no_disponibles.csv", index=False)
 
    _save_antes_despues(before, after, safe_label, out_dir)
    _log_null_report(df, indicator_name).to_csv(
        cal_dir / f"{safe_label}_null_report.csv", index=False)
 
    print(f"  Raw: {len(before)} celdas | Válidas: {len(df)} | "
          f"No disponibles (:): {n_no_disp} | Provisionales (p): {n_prov}")
    return df
 
 
# =========================
# 2. EUROSTAT MODA
# =========================
 
def transform_eurostat_moda() -> pd.DataFrame:
    """
    Índice de volumen de ventas al por menor — moda (Eurostat sts_trtu_m).
    NACE G47.7: Retail sale of textiles, clothing, footwear.
    Base 2015=100, desestacionalizado y ajustado por días laborables.
    Cobertura: España, ene-2010 a dic-2023 (168 observaciones válidas).
    No disponibles: 21 celdas para 2024-2025 (Eurostat no publica este
      índice desestacionalizado para moda para ese periodo en fecha de extracción).
    Todos los valores son provisionales (flag 'p').
    Sin nulos en los datos válidos — no se aplica imputación.
    """
    _ensure_dirs()
    out_dir = PROCESSED_CAPA1 / "eurostat"
    cal_dir = PROCESSED_CAPA1 / "calidad"
 
    df = _extract_eurostat_monthly(
        file_path=RAW_CAPA1 / "eurostat" / "eurostat_moda_base2021.xlsx",
        indicator_name="retail_moda_volumen_ventas_indice_base2015",
        out_dir=out_dir, cal_dir=cal_dir, safe_label="eurostat_moda",
    )
    df.to_csv(out_dir / "eurostat_moda_mensual_clean.csv", index=False)
    print(f"  Eurostat moda ✓  ({df['fecha'].min().strftime('%Y-%m')} – "
          f"{df['fecha'].max().strftime('%Y-%m')})")
    return df
 
 
# =========================
# 3. EUROSTAT RETAIL TOTAL
# =========================
 
def transform_eurostat_retail() -> pd.DataFrame:
    """
    Índice de volumen de ventas al por menor — total (Eurostat sts_trtu_m).
    NACE G47: Retail trade, except of motor vehicles and motorcycles.
 
    CORRECCIÓN IMPORTANTE: La base real del archivo es 2021=100 (no 2015=100
    como se nombraba en el indicador en la versión anterior del código). El
    nombre del archivo incluye 'base2021' y la fila 'Unit of measure' del Excel
    confirma 'Index, 2021=100'. El nombre del indicador se corrige en consecuencia.
 
    Cobertura: España, ene-2010 a sep-2025 (189 observaciones válidas).
    No disponibles: 0. Todos los valores son provisionales (flag 'p').
    Sin nulos — no se aplica imputación.
    """
    _ensure_dirs()
    out_dir = PROCESSED_CAPA1 / "eurostat"
    cal_dir = PROCESSED_CAPA1 / "calidad"
 
    df = _extract_eurostat_monthly(
        file_path=RAW_CAPA1 / "eurostat" / "eurostat_retail_total_base2021.xlsx",
        indicator_name="retail_total_volumen_ventas_indice_base2021",  # base corregida
        out_dir=out_dir, cal_dir=cal_dir, safe_label="eurostat_retail_total",
    )
    df.to_csv(out_dir / "eurostat_retail_total_mensual_clean.csv", index=False)
    print(f"  Eurostat retail total ✓  ({df['fecha'].min().strftime('%Y-%m')} – "
          f"{df['fecha'].max().strftime('%Y-%m')})")
    return df
 
 
# =========================
# 4. EUROSTAT ONLINE EMPRESAS
# =========================
 
def transform_eurostat_online_empresas() -> pd.DataFrame:
    """
    Participación de la facturación empresarial en ecommerce (Eurostat tin00110).
    Indicador: % facturación total procedente de ventas electrónicas.
    Universo: empresas ≥10 empleados, España.
    Cobertura raw: 2013-2024 (12 observaciones).
    Cobertura retenida: 2015-2024 (10 obs) — 2013-2014 excluidos por
      falta de cobertura en otras fuentes del master anual integrado.
    EXTENSIÓN respecto a versión anterior: se incluye 2024 (19.52%)
      que antes se excluía al cortar en 2023.
    Sin flags explícitos en este archivo. Sin nulos en datos retenidos.
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
 
    years_raw = [(idx, int(float(str(val).strip())))
                 for idx, val in enumerate(header_row)
                 if re.match(r"^\d{4}(\.0+)?$", str(val).strip())]
 
    # Snapshot ANTES: todos los años con valor raw
    before = []
    for idx, year in years_raw:
        val_raw = spain_values.iloc[idx]
        val_num = clean_numeric(val_raw)
        before.append({
            "anio": year, "valor_raw": val_raw, "valor_numerico": val_num,
            "incluido": pd.notna(val_num) and year >= 2015,
            "motivo_exclusion": ("año < 2015, fuera cobertura analítica"
                                 if year < 2015 else None),
        })
 
    # Snapshot DESPUÉS: 2015+ con valor válido
    after = []
    for r in before:
        if not r["incluido"]:
            continue
        after.append({
            "anio": r["anio"], "geo": "Spain",
            "valor_pct": round(float(r["valor_numerico"]), 4),
            "fuente": "Eurostat",
            "indicador": "participacion_empresas_facturacion_ecommerce_pct",
            "provisional": False,
        })
 
    df = pd.DataFrame(after).sort_values("anio").reset_index(drop=True)
 
    _save_antes_despues(before, after, "eurostat_online_empresas", out_dir)
    _log_null_report(df, "eurostat_online_empresas_clean").to_csv(
        cal_dir / "eurostat_online_empresas_null_report.csv", index=False)
 
    df.to_csv(out_dir / "eurostat_online_empresas_clean.csv", index=False)
    print(f"  Eurostat online empresas ✓  {len(before)} raw → "
          f"{len(df)} retenidos ({df['anio'].min()}–{df['anio'].max()})")
    return df
 
 
# =========================
# 5. COMERCIO ELECTRÓNICO (INE)
# =========================
 
def _transform_single_comercio_file(file_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Transforma un fichero anual de la Encuesta TIC-E (INE).
    Los nulos en 'valor' son ausencias estructurales de la fuente:
    el cuestionario varía entre ediciones. No se imputan.
    """
    year_match = re.search(r"(\d{4})", file_path.stem)
    if not year_match:
        raise ValueError(f"No se pudo detectar año en {file_path.name}")
    year = int(year_match.group(1))
 
    sheet_name = _find_first_sheet(file_path)
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    df = raw.iloc[7:].copy()
    df.columns = ["indicador", "total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"]
    df["indicador"] = df["indicador"].astype(str).str.strip()
    df = df[df["indicador"].str.match(r"^[A-Z]\.\d", na=False)].copy()
 
    size_cols = ["total", "de_10_a_49", "de_50_a_249", "de_250_y_mas"]
    n_indicadores_antes = len(df)
    n_nulos_antes = sum(df[col].apply(lambda x: pd.isna(clean_numeric(x))).sum()
                        for col in size_cols)
 
    for col in size_cols:
        df[col] = df[col].apply(clean_numeric)
 
    df_long = df.melt(id_vars="indicador", value_vars=size_cols,
                      var_name="tamano_empresa", value_name="valor")
    df_long["anio"] = year
 
    n_nulos_long = int(df_long["valor"].isnull().sum())
    if n_nulos_long > 0:
        print(f"    [{year}] {n_nulos_long} nulos estructurales INE → conservados como NaN")
 
    quality = {
        "anio": year, "n_indicadores": n_indicadores_antes,
        "n_nulos_antes": int(n_nulos_antes), "n_nulos_long": n_nulos_long,
    }
    return df_long[["anio", "indicador", "tamano_empresa", "valor"]], quality
 
 
def transform_comercio_electronico() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolida ficheros anuales de la Encuesta TIC-E (INE, 2015-2023).
 
    Nulos (1.210 en total): ausencias estructurales porque el cuestionario
    del INE varía cada año. Indicadores añadidos en ediciones posteriores
    aparecen como NaN en años anteriores. No son imputables.
 
    Exporta tabla antes/después agregada y inventario detallado de ausencias.
    """
    _ensure_dirs()
    out_dir = PROCESSED_CAPA1 / "comercio_electronico"
    cal_dir = PROCESSED_CAPA1 / "calidad"
 
    files = sorted((RAW_CAPA1 / "comercio_electronico").glob("comercio_electronico*.xlsx"))
    all_dfs, all_q = [], []
 
    for fp in files:
        print(f"  Procesando {fp.name}...")
        try:
            df_y, q_y = _transform_single_comercio_file(fp)
            if not df_y.empty:
                all_dfs.append(df_y)
                all_q.append(q_y)
        except Exception as e:
            print(f"  [WARN] {fp.name}: {e}")
 
    if not all_dfs:
        raise ValueError("No se procesó ningún fichero de comercio electrónico.")
 
    final_df = pd.concat(all_dfs, ignore_index=True)
    quality_df = pd.DataFrame(all_q)
 
    # Inventario nulos estructurales
    nulos_inv = (
        final_df[final_df["valor"].isnull()]
        .groupby(["anio", "indicador", "tamano_empresa"]).size()
        .reset_index(name="n_nulos")
    )
    nulos_inv["tipo_ausencia"] = "estructural_ine"
    nulos_inv["decision"] = "mantener_nan — indicador no publicado en esta edicion"
    nulos_inv.to_csv(out_dir / "comercio_electronico_nulos_estructurales.csv", index=False)
 
    # Tabla antes/después agregada
    n_nulos = int(final_df["valor"].isnull().sum())
    pct_nulos = round(float(final_df["valor"].isnull().mean() * 100), 2)
    pd.DataFrame([
        {"_fase": "ANTES", "n_filas": len(final_df), "n_nulos_valor": n_nulos,
         "pct_nulos": pct_nulos, "n_indicadores_unicos": final_df["indicador"].nunique(),
         "decision": "ausencia estructural INE — preguntas no formuladas ese año"},
        {"_fase": "DESPUÉS", "n_filas": len(final_df), "n_nulos_valor": n_nulos,
         "pct_nulos": pct_nulos, "n_indicadores_unicos": final_df["indicador"].nunique(),
         "decision": "nulos conservados como NaN — no se imputa por ser estructurales"},
    ]).to_csv(out_dir / "antes_despues_comercio_electronico.csv", index=False)
 
    final_df.to_csv(out_dir / "comercio_electronico_clean.csv", index=False)
    quality_df.to_csv(cal_dir / "comercio_electronico_quality_por_anio.csv", index=False)
 
    print(f"  Comercio electrónico ✓  {len(final_df)} filas | "
          f"{n_nulos} nulos estructurales ({pct_nulos}%)")
    return final_df, nulos_inv
 
 
# =========================
# 6. DECISIONES NULOS MASTER ANUAL
# =========================
 
def document_master_anual_nulls() -> pd.DataFrame:
    """
    Documenta decisiones metodológicas sobre nulos del master anual integrado.
 
    pct_empresas_venden_web_apps (2023):
      El INE reformuló el módulo de ecommerce en 2023. El indicador K.1 de
      ediciones 2020-2022 pasó a llamarse I.1.1 con cambios en universo y
      formulación. Valor de referencia I.1.1 en 2023 = 27.44%.
      Decisión: NO interpolar — incompatibilidad metodológica documentada.
 
    Variables INE 2024-2025:
      La Encuesta TIC-E se publica con ~1 año de desfase.
      Decisión: excluir del master analítico, retener en master_full.
 
    pct_personas_compra_online (2025):
      Interpolación lineal aplicada. Ver transform_contexto().
    """
    _ensure_dirs()
    decisions = pd.DataFrame([
        {
            "dataset": "capa1_master_anual_analysis",
            "variable": "pct_empresas_venden_web_apps",
            "anio_afectado": 2023,
            "n_nulos": 1,
            "tipo_ausencia": "cambio_metodologico_ine_K_a_I",
            "descripcion": (
                "INE cambió K.1 → I.1.1 en 2023. Valor I.1.1=27.44%, "
                "no comparable con K.1 de 2020-2022 por cambios en universo."
            ),
            "decision": "no_imputer — incompatibilidad metodologica",
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
            "decision": "no_imputer — excluir del master analitico",
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
                "Serie 2020-2024: 62.62, 66.63, 67.91, 68.88, 68.94. "
                "Interpolación lineal conservadora aplicada."
            ),
            "decision": "interpolar_linealmente — flag _imputado=True",
            "valor_referencia": None,
            "fuente_referencia": "interpolacion lineal sobre tendencia 2020-2024",
        },
    ])
 
    out_path = PROCESSED_CAPA1 / "calidad" / "master_anual_null_decisions.csv"
    decisions.to_csv(out_path, index=False)
    print(f"  ✓ master_anual_null_decisions.csv")
    return decisions
 
 
# =========================
# RUN ALL
# =========================
 
def run_all_transforms() -> None:
    sep = "=" * 65
    print(sep)
    print("CAPA 1 — TRANSFORMACIONES ETL")
    print(sep)
 
    print("\n[1/6] Contexto digitalización...")
    transform_contexto()
 
    print("\n[2/6] Eurostat moda (base 2015=100)...")
    transform_eurostat_moda()
 
    print("\n[3/6] Eurostat retail total (base 2021=100)...")
    transform_eurostat_retail()
 
    print("\n[4/6] Eurostat online empresas...")
    transform_eurostat_online_empresas()
 
    print("\n[5/6] Comercio electrónico INE...")
    transform_comercio_electronico()
 
    print("\n[6/6] Decisiones nulos master anual...")
    document_master_anual_nulls()
 
    print(f"\n{sep}")
    print("OUTPUTS GENERADOS:")
    print("  contexto_digitalizacion/")
    print("    · contexto_digitalizacion_clean.csv     (con columna _imputado)")
    print("    · antes_despues_contexto_digitalizacion.csv")
    print("  eurostat/")
    print("    · eurostat_moda_mensual_clean.csv        (168 obs, 2010-2023)")
    print("    · eurostat_retail_total_mensual_clean.csv (189 obs, 2010-2025)")
    print("    · eurostat_online_empresas_clean.csv     (10 obs, 2015-2024)")
    print("    · antes_despues_*.csv  ×3")
    print("  comercio_electronico/")
    print("    · comercio_electronico_clean.csv")
    print("    · comercio_electronico_nulos_estructurales.csv")
    print("    · antes_despues_comercio_electronico.csv")
    print("  calidad/")
    print("    · master_anual_null_decisions.csv")
    print("    · *_null_report.csv  ×4")
    print("    · eurostat_moda_no_disponibles.csv")
    print(sep)
 
 
if __name__ == "__main__":
    run_all_transforms()