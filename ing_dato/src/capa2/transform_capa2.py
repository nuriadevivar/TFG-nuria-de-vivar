"""
transform_capa2.py — Transformación ETL de Capa 2: Señales digitales y tendencias

Fuentes:
  - googletrends/*.csv          (Google Trends, extracción propia via pytrends)
  - eventos/eventos_moda.csv    (elaboración propia, hitos relevantes del sector)
  - apify/instagram/*.csv       (scraping Apify de perfiles de marcas en Instagram)

Decisiones metodológicas clave:
  - Google Trends valor 0: en la API de Google Trends, el valor 0 no significa
    interés cero sino que el volumen de búsquedas es demasiado bajo para ser
    representado en la escala 0-100. Se documenta con flag 'valor_cero_trends'
    para distinguirlo de NaN (dato no disponible). No se imputa: eliminar ceros
    distorsionaría la comparación entre términos con distinta popularidad.
  - Normalización de marcas: nombres en distintos formatos ('H&M', 'hm', 'handm')
    se mapean a un identificador canónico para garantizar consistencia entre fuentes.
  - Instagram likes/comentarios NaN → 0: los posts sin registro de likes/comentarios
    en el scraping son posts donde la API devolvió null (no posts sin interacción).
    Se imputan a 0 y se documentan como flag.
  - Tablas antes/después: cada función exporta snapshot pre/post transformación
    para auditoría metodológica.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import PROCESSED_CAPA2, RAW_CAPA2

warnings.filterwarnings("ignore")


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    for sub in ["googletrends", "eventos", "apify", "integrated", "calidad"]:
        (PROCESSED_CAPA2 / sub).mkdir(parents=True, exist_ok=True)


def _clean_trends_long(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    # Normaliza un dataframe de Google Trends en formato largo:
    # renombra columnas a nombres canónicos, parsea fecha al primer día del mes
    # y añade columnas de año y mes para facilitar agregaciones posteriores.
    df = df.copy()
    df = df.rename(columns={date_col: "fecha", value_col: "valor_trends"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    # Trunca al primer día del mes para garantizar consistencia entre fuentes
    # que pueden tener fechas en distintos formatos (día variable)
    df["fecha"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    df["valor_trends"] = pd.to_numeric(df["valor_trends"], errors="coerce")
    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month
    return df


def _reshape_wide_trends(file_path: Path, date_col: str,
                         group_name: str, output_name: str) -> pd.DataFrame:
    
    # Convierte ficheros de Google Trends en formato ancho (fecha × términos)
    # a formato largo (fecha, termino, valor_trends). Patrón habitual en las
    # exportaciones de pytrends donde cada término es una columna.
    df = pd.read_csv(file_path)
    value_cols = [c for c in df.columns if c != date_col]
    long_df = df.melt(
        id_vars=[date_col], value_vars=value_cols,
        var_name="termino", value_name="valor_trends",
    )
    long_df = _clean_trends_long(long_df, date_col=date_col, value_col="valor_trends")
    long_df["grupo"] = group_name
    return long_df


def _trends_quality_report(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Genera informe de calidad para un dataset de Google Trends.
    Documenta NaN (dato no disponible) y ceros (volumen insuficiente en GT).
    """
    # Distingue dos tipos de ausencia de señal en Google Trends:
    # NaN: dato genuinamente no disponible en la API para ese término/mes
    # Cero: volumen de búsquedas real pero demasiado bajo para la escala 0-100
    n_total = len(df)
    n_nan = int(df["valor_trends"].isna().sum())
    n_zero = int((df["valor_trends"] == 0).sum())
    n_validos = int(df["valor_trends"].notna().sum())

    report = pd.DataFrame([{
        "fuente": label,
        "n_total": n_total,
        "n_validos": n_validos,
        "n_nan": n_nan,
        "pct_nan": round(n_nan / n_total * 100, 2) if n_total > 0 else 0,
        "n_cero": n_zero,
        "pct_cero": round(n_zero / n_validos * 100, 2) if n_validos > 0 else 0,
        "decision_nan": "mantener_nan — dato no disponible en Google Trends API",
        "decision_cero": (
            "mantener_cero con flag valor_cero_trends=True — "
            "valor 0 en GT indica volumen de busqueda insuficiente para ser "
            "representado en escala 0-100, no ausencia real de interes. "
            "Imputar distorsionaria la comparacion entre terminos."
        ),
        "media_sin_ceros": round(float(df.loc[df["valor_trends"] > 0, "valor_trends"].mean()), 2)
        if (df["valor_trends"] > 0).any() else 0,
    }])
    return report


def _save_antes_despues(before: pd.DataFrame, after: pd.DataFrame,
                         label: str, out_dir: Path) -> None:
    # Exporta una tabla antes/después de la transformación para auditoría.
    # Permite verificar qué filas se modificaron o excluyeron en cada paso ETL.
    b = before.copy()
    b.insert(0, "_fase", "ANTES")
    a = after.copy()
    a.insert(0, "_fase", "DESPUÉS")
    pd.concat([b, a], ignore_index=True).to_csv(
        out_dir / f"antes_despues_{label}.csv", index=False
    )
    print(f"  ✓ antes_despues_{label}.csv")


def normalize_brand_name(text: str) -> str | None:
    # Mapea variantes del nombre de marca a un identificador canónico.
    # Necesario porque la misma marca puede aparecer con distintos formatos
    # en Google Trends, Instagram y eventos según cómo se introdujo el término.
    if pd.isna(text):
        return None
    t = str(text).strip().lower()
    mapping = {
        "h&m": "hm", "hm": "hm", "handm": "hm",
        "massimo dutti": "massimo_dutti", "massimo_dutti": "massimo_dutti",
        "zara": "zara", "mango": "mango", "shein": "shein",
        "pull and bear": "pull_and_bear", "pull_and_bear": "pull_and_bear",
        "stradivarius": "stradivarius",
    }
    return mapping.get(t, t.replace(" ", "_"))


def infer_brand_from_filename(filename: str) -> str | None:
    # Infiere la marca a partir del nombre del fichero CSV de Apify.
    # Apify genera un fichero por perfil scrapeado y su nombre incluye
    # el handle o nombre de la marca.
    name = filename.lower()
    if "massimo" in name:
        return "massimo_dutti"
    if "shein" in name:
        return "shein"
    if "mango" in name:
        return "mango"
    if "zara" in name:
        return "zara"
    if "hm" in name or "h&m" in name or "handm" in name:
        return "hm"
    return None

# Diccionario de validación de términos por grupo estético.
# Define el conjunto esperado de términos para cada grupo y se usa
# para filtrar términos no previstos que puedan aparecer en las exportaciones.
VALID_TERMS_BY_GROUP = {
    "sofisticado": {"cayetana", "pija", "old money", "moda preppy", "minimalista elegante"},
    "urbano": {"choni", "trap style", "y2k outfit", "streetwear"},
    "consciente_compra": {"moda sostenible", "slow fashion", "ropa vintage",
                          "comprar ropa online", "zara online"},
}


# =========================
# 1. MODA TOTAL LONG
# =========================

def transform_trends_moda_total() -> pd.DataFrame:
    """
    Transforma el dataset de términos generales de moda (Google Trends).
    Añade flag 'valor_cero_trends' para distinguir valor 0 (volumen insuficiente GT)
    de NaN (dato no disponible).
    """
    _ensure_dirs()
    cal_dir = PROCESSED_CAPA2 / "calidad"
    out_dir = PROCESSED_CAPA2 / "googletrends"

    file_path = RAW_CAPA2 / "googletrends" / "aggregated" / "googletrends_moda_total_long_2015_2025.csv"
    df_raw = pd.read_csv(file_path)

    # Snapshot ANTES
    before = df_raw.head(10).copy()

    df = _clean_trends_long(df_raw, date_col="mes", value_col="valor")
    df["grupo"] = df["grupo"].astype(str).str.strip()
    df["termino"] = df["termino"].astype(str).str.strip()

    # Flag que distingue cero real de Google Trends (volumen insuficiente)
    # de NaN (dato no disponible). Crítico para no confundir ambas situaciones
    # en el análisis comparativo entre términos.
    df["valor_cero_trends"] = df["valor_trends"] == 0

    # Snapshot DESPUÉS
    after = df.head(10).copy()
    _save_antes_despues(before, after, "trends_moda_total", out_dir)

    # Informe de calidad: documenta NaN y ceros por fuente
    quality = _trends_quality_report(df, "trends_moda_total")
    quality.to_csv(cal_dir / "trends_moda_total_quality.csv", index=False)

    output_path = out_dir / "trends_moda_total_clean.csv"
    df.to_csv(output_path, index=False)

    n_cero = int(df["valor_cero_trends"].sum())
    print(f"  trends_moda_total: {len(df)} obs | {df['termino'].nunique()} términos | "
          f"ceros GT: {n_cero} ({round(n_cero/len(df)*100,1)}%)")
    return df


# =========================
# 2. MARCAS
# =========================

def transform_trends_marcas() -> pd.DataFrame:
    """
    Transforma los datos de Google Trends para las 5 marcas principales.
    Combina fuente base (2015-2025) con fuente extra para completar cobertura.
    Deduplica por (fecha, termino) conservando el último valor.
    """
    _ensure_dirs()
    cal_dir = PROCESSED_CAPA2 / "calidad"
    out_dir = PROCESSED_CAPA2 / "googletrends"

    base_path = RAW_CAPA2 / "googletrends" / "aggregated" / "googletrends_marcas_2015_2025.csv"
    extra_path = RAW_CAPA2 / "googletrends" / "aggregated" / "googletrends_marcas_extra_2015_2025.csv"

    base_df = pd.read_csv(base_path)
    extra_df = pd.read_csv(extra_path)

    n_raw_base = len(base_df)

    # Manejo flexible del formato: el fichero base puede venir en formato largo
    # (con columnas fecha/termino/valor_trends) o ancho (fecha × marcas).
    # Se detecta automáticamente para garantizar robustez ante cambios de formato.
    if "fecha" in base_df.columns and "termino" in base_df.columns and "valor_trends" in base_df.columns:
        base_long = base_df.copy()
    else:
        possible_date_cols = ["fecha", "date", "Date", "month", "Month", "mes"]
        date_col = next((c for c in possible_date_cols if c in base_df.columns), None)
        if date_col is None:
            raise ValueError(f"No se encontró columna temporal. Columnas: {base_df.columns.tolist()}")
        value_cols = [c for c in base_df.columns if c != date_col]
        base_long = base_df.melt(
            id_vars=[date_col], value_vars=value_cols,
            var_name="termino", value_name="valor_trends",
        ).rename(columns={date_col: "fecha"})

    base_long["fecha"] = pd.to_datetime(base_long["fecha"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    base_long["termino"] = base_long["termino"].apply(normalize_brand_name)
    base_long["valor_trends"] = pd.to_numeric(base_long["valor_trends"], errors="coerce")
    base_long["anio"] = base_long["fecha"].dt.year
    base_long["mes_num"] = base_long["fecha"].dt.month
    base_long["grupo"] = "marcas"
    base_long = base_long[["fecha", "termino", "valor_trends", "anio", "mes_num", "grupo"]].copy()

    extra_df["fecha"] = pd.to_datetime(extra_df["fecha"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    extra_df["termino"] = extra_df["termino"].apply(normalize_brand_name)
    extra_df["valor_trends"] = pd.to_numeric(extra_df["valor_trends"], errors="coerce")
    extra_df["anio"] = pd.to_numeric(extra_df["anio"], errors="coerce")
    extra_df["mes_num"] = pd.to_numeric(extra_df["mes_num"], errors="coerce")
    extra_df["grupo"] = "marcas"
    extra_df = extra_df[["fecha", "termino", "valor_trends", "anio", "mes_num", "grupo"]].copy()

    # Concatena base y extra; la deduplicación posterior resuelve solapamientos
    df = pd.concat([base_long, extra_df], ignore_index=True)

    main_brands = ["zara", "shein", "mango", "hm", "massimo_dutti"]
    df = df[df["termino"].isin(main_brands)].copy()
    n_antes_dedup = len(df)

    # Deduplicación por (fecha, termino): si base y extra tienen el mismo mes
    # para la misma marca, se conserva el último valor (extra tiene prioridad)
    df = (
        df.sort_values(["termino", "fecha"])
        .drop_duplicates(subset=["fecha", "termino"], keep="last")
        .reset_index(drop=True)
    )
    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month
    df["valor_cero_trends"] = df["valor_trends"] == 0

    # Calidad
    quality = _trends_quality_report(df, "trends_marcas")
    quality.to_csv(cal_dir / "trends_marcas_quality.csv", index=False)

    output_path = out_dir / "trends_marcas_clean.csv"
    df.to_csv(output_path, index=False)

    n_dedup = n_antes_dedup - len(df)
    print(f"  trends_marcas: {len(df)} obs | {df['termino'].nunique()} marcas | "
          f"duplicados eliminados: {n_dedup} | ceros GT: {int(df['valor_cero_trends'].sum())}")
    return df


# =========================
# 3-5. ESTÉTICAS (sofisticado, urbano, consciente_compra)
# =========================

def _transform_trends_grupo(group_name: str, file_suffix: str) -> pd.DataFrame:
    # Función genérica para transformar grupos de términos estéticos.
    # Aplica el mismo pipeline ETL a los tres grupos (sofisticado, urbano,
    # consciente_compra), filtrando solo los términos incluidos en VALID_TERMS_BY_GROUP.
    _ensure_dirs()
    cal_dir = PROCESSED_CAPA2 / "calidad"
    out_dir = PROCESSED_CAPA2 / "googletrends"

    file_path = RAW_CAPA2 / "googletrends" / "aggregated" / f"googletrends_{file_suffix}_2015_2025.csv"

    df = _reshape_wide_trends(
        file_path=file_path,
        date_col="mes",
        group_name=group_name,
        output_name=f"trends_{group_name}_clean.csv",
    )

    n_before_filter = len(df)

    df["termino"] = df["termino"].astype(str).str.strip().str.lower()
    # Filtra solo los términos validados para este grupo.
    # Descarta términos que puedan haberse colado en la exportación de pytrends
    # por solapamiento de consultas o errores de nomenclatura.
    df = df[df["termino"].isin(VALID_TERMS_BY_GROUP[group_name])].copy().reset_index(drop=True)
    df["valor_cero_trends"] = df["valor_trends"] == 0

    n_filtrados = n_before_filter - len(df)

    quality = _trends_quality_report(df, f"trends_{group_name}")
    quality.to_csv(cal_dir / f"trends_{group_name}_quality.csv", index=False)

    out_path = out_dir / f"trends_{group_name}_clean.csv"
    df.to_csv(out_path, index=False)

    n_cero = int(df["valor_cero_trends"].sum())
    print(
        f"  trends_{group_name}: {len(df)} obs | {df['termino'].nunique()} términos | "
        f"filtrados: {n_filtrados} | ceros GT: {n_cero}"
    )
    return df

# Cada función pública delega en _transform_trends_grupo con su nombre de grupo
# y sufijo de fichero correspondiente
def transform_trends_sofisticado() -> pd.DataFrame:
    return _transform_trends_grupo("sofisticado", "sofisticado")


def transform_trends_urbano() -> pd.DataFrame:
    return _transform_trends_grupo("urbano", "urbano")


def transform_trends_consciente_compra() -> pd.DataFrame:
    return _transform_trends_grupo("consciente_compra", "consciente_compra")


# =========================
# 6. PRODUCTOS
# =========================

def transform_trends_productos() -> pd.DataFrame:
    """
    Transforma los datos de Google Trends por categoría de producto y marca.
    """
    _ensure_dirs()
    cal_dir = PROCESSED_CAPA2 / "calidad"
    out_dir = PROCESSED_CAPA2 / "googletrends"

    file_path = RAW_CAPA2 / "googletrends" / "aggregated" / "trends_productos_2015_2025.csv"
    df_raw = pd.read_csv(file_path)
    before = df_raw.head(5).copy()

    # Este fichero usa 'interes_trends' como nombre de columna en lugar de 'valor_trends'
    df = df_raw.rename(columns={"interes_trends": "valor_trends"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["valor_trends"] = pd.to_numeric(df["valor_trends"], errors="coerce")

    # Normalización de campos textuales para garantizar consistencia con el resto de fuentes
    if "marca" in df.columns:
        df["marca"] = df["marca"].apply(normalize_brand_name)
    if "categoria_producto" in df.columns:
        df["categoria_producto"] = df["categoria_producto"].astype(str).str.strip().str.lower()
    if "termino_busqueda" in df.columns:
        df["termino_busqueda"] = df["termino_busqueda"].astype(str).str.strip().str.lower()

    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month
    df["valor_cero_trends"] = df["valor_trends"] == 0

    after = df.head(5).copy()
    _save_antes_despues(before, after, "trends_productos", out_dir)

    quality = _trends_quality_report(df, "trends_productos")
    quality.to_csv(cal_dir / "trends_productos_quality.csv", index=False)

    output_path = out_dir / "trends_productos_clean.csv"
    df.to_csv(output_path, index=False)

    print(f"  trends_productos: {len(df)} obs | ceros GT: {int(df['valor_cero_trends'].sum())}")
    return df


# =========================
# 7. EVENTOS MODA
# =========================

def transform_eventos_moda() -> pd.DataFrame:
    """
    Transforma el dataset de eventos relevantes del sector moda.
    Fuente de elaboración propia. Sin nulos en columnas clave.
    """
    _ensure_dirs()
    out_dir = PROCESSED_CAPA2 / "eventos"

    file_path = RAW_CAPA2 / "eventos" / "eventos_moda.csv"
    df_raw = pd.read_csv(file_path)
    before = df_raw.copy()

    df = df_raw.copy()
    # La fecha viene en formato "YYYY-MM" (fecha_aprox); se añade "-01" para
    # obtener el primer día del mes y poder parsear con pd.to_datetime
    df["fecha"] = pd.to_datetime(df["fecha_aprox"] + "-01", errors="coerce")
    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    # Limpieza de campos de texto: elimina espacios y caracteres invisibles
    text_cols = ["marca_o_tendencia", "plataforma", "tipo_evento", "descripcion_evento", "fuente"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Exclusión de eventos con fecha no parseable. Se avisa en consola para
    # revisión manual si ocurre, aunque no debería con la fuente actual.
    n_nat = int(df["fecha"].isna().sum())
    if n_nat > 0:
        print(f"  [WARN] eventos_moda: {n_nat} fechas no parseables → excluidas")
        df = df[df["fecha"].notna()].copy()

    after = df.copy()
    _save_antes_despues(before, after, "eventos_moda", out_dir)

    output_path = out_dir / "eventos_moda_clean.csv"
    df.to_csv(output_path, index=False)

    print(f"  eventos_moda: {len(df)} eventos | "
          f"{df['anio'].min()}-{df['anio'].max()} | "
          f"plataformas: {df['plataforma'].nunique()}")
    return df


# =========================
# 8. UNIFICAR GOOGLE TRENDS
# =========================

def build_trends_grupos_unificados() -> pd.DataFrame:
    """
    Consolida todos los grupos de Google Trends en un único dataset.
    El flag 'valor_cero_trends' se propaga desde cada grupo.
    """
    _ensure_dirs()
    out_dir = PROCESSED_CAPA2 / "googletrends"

    # Se unen los cuatro grupos principales. trends_productos se mantiene
    # separado porque tiene estructura distinta (marca × categoría vs término × grupo)
    files = [
        out_dir / "trends_marcas_clean.csv",
        out_dir / "trends_sofisticado_clean.csv",
        out_dir / "trends_urbano_clean.csv",
        out_dir / "trends_consciente_compra_clean.csv",
    ]

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "valor_cero_trends" not in df.columns:
            df["valor_cero_trends"] = df["valor_trends"] == 0
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df["fecha"] = pd.to_datetime(full_df["fecha"], errors="coerce")
    full_df["termino"] = full_df["termino"].astype(str).str.strip().str.lower()
    full_df["grupo"] = full_df["grupo"].astype(str).str.strip().str.lower()
    full_df["anio"] = full_df["fecha"].dt.year
    full_df["mes_num"] = full_df["fecha"].dt.month

    # Deduplicación final por (fecha, grupo, termino): garantiza una única
    # observación por clave en el dataset unificado
    full_df = (
        full_df
        .sort_values(["grupo", "termino", "fecha"])
        .drop_duplicates(subset=["fecha", "grupo", "termino"], keep="last")
        .reset_index(drop=True)
    )

    output_path = out_dir / "trends_grupos_unificados_clean.csv"
    full_df.to_csv(output_path, index=False)

    print(f"  trends_grupos_unificados: {len(full_df)} obs | "
          f"{full_df['termino'].nunique()} términos | "
          f"{full_df['grupo'].nunique()} grupos")
    return full_df


# =========================
# 9. INSTAGRAM POSTS
# =========================

def transform_apify_instagram_posts() -> pd.DataFrame:
    """
    Transforma los posts de Instagram scrapeados con Apify.

    Decisión de imputación de likes/comentarios:
        Los NaN en likes y comentarios se interpretan como métricas no devueltas
        por la API en el momento del scraping. Se imputan a 0 de forma conservadora
        y se documentan con el flag 'metricas_imputadas=True'. Esta decisión puede
        subestimar el engagement real de algunos posts afectados.
    """
    _ensure_dirs()
    cal_dir = PROCESSED_CAPA2 / "calidad"
    out_dir = PROCESSED_CAPA2 / "apify"

    input_dir = RAW_CAPA2 / "apify" / "instagram"
    output_path = out_dir / "instagram_posts_clean.csv"

    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {input_dir}")

    dfs = []
    quality_records = []

    for file_path in files:
        # La marca se infiere del nombre del fichero, no de una columna del CSV,
        # porque Apify no incluye la marca como campo en la exportación
        brand = infer_brand_from_filename(file_path.name)
        if brand is None:
            print(f"  [WARN] No se pudo inferir marca para {file_path.name}. Se omite.")
            continue

        df = pd.read_csv(file_path)
        n_raw = len(df)

        rename_map = {
            "commentsCount": "comentarios", "likesCount": "likes",
            "timestamp": "fecha_post", "url": "url_post",
            "type": "tipo_post", "ownerUsername": "perfil_origen",
            "ownerFullName": "perfil_nombre",
        }
        df = df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})

        # Reindexación defensiva: garantiza que todas las columnas esperadas
        # están presentes aunque no aparezcan en algún fichero de Apify
        expected_cols = ["caption", "comentarios", "likes", "fecha_post", "url_post",
                         "hashtags", "tipo_post", "perfil_origen", "perfil_nombre", "inputUrl"]
        base = df.reindex(columns=expected_cols).copy()

        n_likes_nan = int(base["likes"].isna().sum())
        n_comentarios_nan = int(base["comentarios"].isna().sum())

        # Flag que identifica posts cuyas métricas fueron imputadas.
        # Permite evaluar el potencial sesgo en el engagement calculado.
        base["metricas_imputadas"] = base["likes"].isna() | base["comentarios"].isna()
        
        # Normalización de timezone: Apify devuelve fechas UTC, se elimina tz
        # para consistencia con el resto de columnas temporales del proyecto
        base["fecha_post"] = pd.to_datetime(base["fecha_post"], errors="coerce", utc=True).dt.tz_localize(None)
        
        # Imputación conservadora a 0: no se elimina el post porque el resto
        # de variables (caption, fecha, tipo_post) siguen siendo válidas
        base["likes"] = pd.to_numeric(base["likes"], errors="coerce").fillna(0)
        base["comentarios"] = pd.to_numeric(base["comentarios"], errors="coerce").fillna(0)

        n_fecha_nat = int(base["fecha_post"].isna().sum())

        # Construcción de variables derivadas: fecha normalizada a mes, engagement total
        extra = pd.DataFrame({
            "marca": brand,
            "plataforma": "instagram",
            "fecha": base["fecha_post"].dt.to_period("M").dt.to_timestamp(),
            "anio": base["fecha_post"].dt.year,
            "mes_num": base["fecha_post"].dt.month,
            "engagement_total_post": base["likes"] + base["comentarios"],
        })

        clean_df = pd.concat([extra, base], axis=1)
        keep_cols = ["marca", "plataforma", "fecha_post", "fecha", "anio", "mes_num",
                     "caption", "likes", "comentarios", "engagement_total_post",
                     "metricas_imputadas", "url_post", "hashtags", "tipo_post",
                     "perfil_origen", "perfil_nombre", "inputUrl"]
        clean_df = clean_df[keep_cols].copy()
        dfs.append(clean_df)

        quality_records.append({
            "marca": brand,
            "n_posts_raw": n_raw,
            "n_fecha_nat": n_fecha_nat,
            "n_likes_nan_imputados_0": n_likes_nan,
            "n_comentarios_nan_imputados_0": n_comentarios_nan,
            "pct_metricas_imputadas": round((n_likes_nan + n_comentarios_nan) / (2 * n_raw) * 100, 2) if n_raw > 0 else 0,
        })

    if not dfs:
        raise ValueError("No se pudo procesar ningún CSV de Instagram.")

    posts = pd.concat(dfs, ignore_index=True)

    # Deduplicación por (marca, url_post): evita posts scrapeados en dos
    # ejecuciones distintas de Apify que aparezcan duplicados en el dataset
    if "url_post" in posts.columns:
        n_before_dedup = len(posts)
        posts = posts.drop_duplicates(subset=["marca", "url_post"], keep="first")
        n_dedup = n_before_dedup - len(posts)
    else:
        n_dedup = 0

    posts = posts.sort_values(["marca", "fecha_post"]).reset_index(drop=True)
    posts.to_csv(output_path, index=False)

    quality_df = pd.DataFrame(quality_records)
    quality_df.to_csv(cal_dir / "instagram_posts_quality.csv", index=False)

    print(f"  instagram_posts: {len(posts)} posts | {posts['marca'].nunique()} marcas | "
          f"duplicados eliminados: {n_dedup}")
    print(f"  Imputaciones likes/comentarios: {int(posts['metricas_imputadas'].sum())} posts")
    return posts


# =========================
# 10. INSTAGRAM BRAND MONTHLY
# =========================

def transform_apify_instagram_brand_monthly() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "apify" / "instagram_posts_clean.csv"
    output_path = PROCESSED_CAPA2 / "apify" / "instagram_brand_monthly.csv"

    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
    df["comentarios"] = pd.to_numeric(df["comentarios"], errors="coerce").fillna(0)
    df["engagement_total_post"] = pd.to_numeric(df["engagement_total_post"], errors="coerce").fillna(0)

    # Agregación mensual por marca: consolida las métricas de todos los posts
    # del mes en totales y medias para el análisis de tendencias a nivel marca.
    # n_posts_metricas_imputadas permite contextualizar la fiabilidad del engagement.
    monthly = (
        df.groupby(["fecha", "anio", "mes_num", "marca", "plataforma"], dropna=False)
        .agg(
            n_posts=("url_post", "count"),
            likes_totales=("likes", "sum"),
            comentarios_totales=("comentarios", "sum"),
            engagement_total=("engagement_total_post", "sum"),
            likes_medios_post=("likes", "mean"),
            comentarios_medios_post=("comentarios", "mean"),
            engagement_medio_post=("engagement_total_post", "mean"),
            n_posts_metricas_imputadas=("metricas_imputadas", "sum"),
        )
        .reset_index()
    )

    monthly = monthly.sort_values(["marca", "fecha"]).reset_index(drop=True)
    monthly.to_csv(output_path, index=False)

    print(f"  instagram_brand_monthly: {len(monthly)} filas | "
          f"{monthly['marca'].nunique()} marcas | "
          f"{monthly['fecha'].nunique()} meses")
    return monthly


# =========================
# 11. RESUMEN DE CALIDAD GLOBAL
# =========================

def build_transform_quality_summary() -> pd.DataFrame:
    """
    Consolida todos los informes de calidad del transform en una tabla única.
    Útil para documentar en la memoria el estado de los datos antes del build.
    """
    _ensure_dirs()
    cal_dir = PROCESSED_CAPA2 / "calidad"

    # Lee todos los CSV de calidad generados por las funciones anteriores
    # y los consolida en una única tabla para revisión global del pipeline
    quality_files = list(cal_dir.glob("*_quality.csv"))
    if not quality_files:
        print("  [INFO] No hay archivos de calidad generados aún.")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in quality_files]
    summary = pd.concat(dfs, ignore_index=True)

    out_path = cal_dir / "capa2_transform_quality_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"  Quality summary: {len(summary)} fuentes documentadas → {out_path.name}")
    return summary


# =========================
# RUN ALL
# =========================

def run_all_transforms() -> None:
    sep = "=" * 65
    print(sep)
    print("CAPA 2 — TRANSFORMACIONES ETL")
    print(sep)

    print("\n[1/10] Tendencias moda total...")
    transform_trends_moda_total()

    print("\n[2/10] Tendencias marcas...")
    transform_trends_marcas()

    print("\n[3/10] Tendencias estéticas sofisticadas...")
    transform_trends_sofisticado()

    print("\n[4/10] Tendencias estéticas urbanas...")
    transform_trends_urbano()

    print("\n[5/10] Tendencias comportamiento consciente...")
    transform_trends_consciente_compra()

    print("\n[6/10] Tendencias productos...")
    transform_trends_productos()

    print("\n[7/10] Eventos moda...")
    transform_eventos_moda()

    print("\n[8/10] Unificar grupos Google Trends...")
    build_trends_grupos_unificados()

    print("\n[9/10] Instagram posts (Apify)...")
    transform_apify_instagram_posts()

    print("\n[10/10] Instagram brand monthly...")
    transform_apify_instagram_brand_monthly()

    print("\n[Calidad] Resumen global de calidad...")
    build_transform_quality_summary()

    print(f"\n{sep}")
    print("TRANSFORM CAPA 2 COMPLETADO")
    print("Outputs calidad generados:")
    print("  · calidad/trends_*_quality.csv  (flag valor_cero_trends documentado)")
    print("  · calidad/instagram_posts_quality.csv  (metricas_imputadas por marca)")
    print("  · calidad/capa2_transform_quality_summary.csv")
    print("Nota metodologica:")
    print("  · valor_trends=0 en GT ≠ interés cero; significa volumen insuficiente.")
    print("  · Flag valor_cero_trends=True permite filtrar en análisis sensibles.")
    print(sep)


if __name__ == "__main__":
    run_all_transforms()