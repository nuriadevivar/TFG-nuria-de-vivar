import pandas as pd
from pathlib import Path

from src.common.config import RAW_CAPA2, PROCESSED_CAPA2


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    (PROCESSED_CAPA2 / "googletrends").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA2 / "eventos").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA2 / "apify").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA2 / "integrated").mkdir(parents=True, exist_ok=True)


def _clean_trends_long(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns={date_col: "fecha", value_col: "valor_trends"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["fecha"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    df["valor_trends"] = pd.to_numeric(df["valor_trends"], errors="coerce")

    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    return df


def _reshape_wide_trends(file_path: Path, date_col: str, group_name: str, output_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    value_cols = [c for c in df.columns if c != date_col]

    long_df = df.melt(
        id_vars=[date_col],
        value_vars=value_cols,
        var_name="termino",
        value_name="valor_trends",
    )

    long_df = _clean_trends_long(long_df, date_col=date_col, value_col="valor_trends")
    long_df["grupo"] = group_name

    output_path = PROCESSED_CAPA2 / "googletrends" / output_name
    long_df.to_csv(output_path, index=False)

    print(f"{output_name} guardado en:")
    print(output_path)
    print("")
    print(long_df.head())

    return long_df


def normalize_brand_name(text: str) -> str | None:
    if pd.isna(text):
        return None

    t = str(text).strip().lower()

    mapping = {
        "h&m": "hm",
        "hm": "hm",
        "handm": "hm",
        "massimo dutti": "massimo_dutti",
        "massimo_dutti": "massimo_dutti",
        "zara": "zara",
        "mango": "mango",
        "shein": "shein",
        "pull and bear": "pull_and_bear",
        "pull_and_bear": "pull_and_bear",
        "stradivarius": "stradivarius",
    }

    return mapping.get(t, t.replace(" ", "_"))


def infer_brand_from_filename(filename: str) -> str | None:
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

VALID_TERMS_BY_GROUP = {
    "sofisticado": {"cayetana", "pija", "old money", "moda preppy", "minimalista elegante"},
    "urbano": {"choni", "trap style", "y2k outfit", "streetwear"},
    "consciente_compra": {"moda sostenible", "slow fashion", "ropa vintage", "comprar ropa online", "zara online"},
}

# =========================
# 1. MODA TOTAL LONG
# =========================

def transform_trends_moda_total() -> pd.DataFrame:
    _ensure_dirs()

    file_path = RAW_CAPA2 / "googletrends" / "aggregated" / "googletrends_moda_total_long_2015_2025.csv"
    df = pd.read_csv(file_path)

    df = _clean_trends_long(df, date_col="mes", value_col="valor")
    df["grupo"] = df["grupo"].astype(str).str.strip()
    df["termino"] = df["termino"].astype(str).str.strip()

    output_path = PROCESSED_CAPA2 / "googletrends" / "trends_moda_total_clean.csv"
    df.to_csv(output_path, index=False)

    print("trends_moda_total_clean.csv guardado en:")
    print(output_path)
    print("")
    print(df.head())

    return df


# =========================
# 2. MARCAS
# =========================

def transform_trends_marcas() -> pd.DataFrame:
    _ensure_dirs()

    base_path = RAW_CAPA2 / "googletrends" / "aggregated" / "googletrends_marcas_2015_2025.csv"
    extra_path = RAW_CAPA2 / "googletrends" / "aggregated" / "googletrends_marcas_extra_2015_2025.csv"

    base_df = pd.read_csv(base_path)
    extra_df = pd.read_csv(extra_path)

    if "fecha" in base_df.columns and "termino" in base_df.columns and "valor_trends" in base_df.columns:
        base_long = base_df.copy()
    else:
        possible_date_cols = ["fecha", "date", "Date", "month", "Month", "mes"]
        date_col = next((c for c in possible_date_cols if c in base_df.columns), None)

        if date_col is None:
            raise ValueError(
                f"No se encontró columna temporal en googletrends_marcas_2015_2025.csv. "
                f"Columnas detectadas: {base_df.columns.tolist()}"
            )

        value_cols = [c for c in base_df.columns if c != date_col]

        base_long = base_df.melt(
            id_vars=[date_col],
            value_vars=value_cols,
            var_name="termino",
            value_name="valor_trends",
        ).rename(columns={date_col: "fecha"})

    base_long["fecha"] = pd.to_datetime(base_long["fecha"], errors="coerce")
    base_long["fecha"] = base_long["fecha"].dt.to_period("M").dt.to_timestamp()
    base_long["termino"] = base_long["termino"].apply(normalize_brand_name)
    base_long["valor_trends"] = pd.to_numeric(base_long["valor_trends"], errors="coerce")
    base_long["anio"] = base_long["fecha"].dt.year
    base_long["mes_num"] = base_long["fecha"].dt.month
    base_long["grupo"] = "marcas"
    base_long = base_long[["fecha", "termino", "valor_trends", "anio", "mes_num", "grupo"]].copy()

    extra_df["fecha"] = pd.to_datetime(extra_df["fecha"], errors="coerce")
    extra_df["fecha"] = extra_df["fecha"].dt.to_period("M").dt.to_timestamp()
    extra_df["termino"] = extra_df["termino"].apply(normalize_brand_name)
    extra_df["valor_trends"] = pd.to_numeric(extra_df["valor_trends"], errors="coerce")
    extra_df["anio"] = pd.to_numeric(extra_df["anio"], errors="coerce")
    extra_df["mes_num"] = pd.to_numeric(extra_df["mes_num"], errors="coerce")
    extra_df["grupo"] = "marcas"
    extra_df = extra_df[["fecha", "termino", "valor_trends", "anio", "mes_num", "grupo"]].copy()

    df = pd.concat([base_long, extra_df], ignore_index=True)

    main_brands = ["zara", "shein", "mango", "hm", "massimo_dutti"]
    df = df[df["termino"].isin(main_brands)].copy()

    df = (
        df.sort_values(["termino", "fecha"])
        .drop_duplicates(subset=["fecha", "termino"], keep="last")
        .reset_index(drop=True)
    )

    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    output_path = PROCESSED_CAPA2 / "googletrends" / "trends_marcas_clean.csv"
    df.to_csv(output_path, index=False)

    print("trends_marcas_clean.csv guardado en:")
    print(output_path)
    print("")
    print(df.head())

    return df


# =========================
# 3. SOFISTICADO
# =========================

def transform_trends_sofisticado() -> pd.DataFrame:
    file_path = RAW_CAPA2 / "googletrends" / "aggregated" / "googletrends_sofisticado_2015_2025.csv"
    df = _reshape_wide_trends(
        file_path=file_path,
        date_col="mes",
        group_name="sofisticado",
        output_name="trends_sofisticado_clean.csv",
    )
    df["termino"] = df["termino"].astype(str).str.strip().str.lower()
    df = df[df["termino"].isin(VALID_TERMS_BY_GROUP["sofisticado"])].copy().reset_index(drop=True)
    output_path = PROCESSED_CAPA2 / "googletrends" / "trends_sofisticado_clean.csv"
    df.to_csv(output_path, index=False)
    return df

# =========================
# 4. URBANO
# =========================

def transform_trends_urbano() -> pd.DataFrame:
    file_path = RAW_CAPA2 / "googletrends" / "aggregated" / "googletrends_urbano_2015_2025.csv"
    df = _reshape_wide_trends(
        file_path=file_path,
        date_col="mes",
        group_name="urbano",
        output_name="trends_urbano_clean.csv",
    )
    df["termino"] = df["termino"].astype(str).str.strip().str.lower()
    df = df[df["termino"].isin(VALID_TERMS_BY_GROUP["urbano"])].copy().reset_index(drop=True)
    output_path = PROCESSED_CAPA2 / "googletrends" / "trends_urbano_clean.csv"
    df.to_csv(output_path, index=False)
    return df


# =========================
# 5. CONSCIENTE COMPRA
# =========================

def transform_trends_consciente_compra() -> pd.DataFrame:
    file_path = RAW_CAPA2 / "googletrends" / "aggregated" / "googletrends_consciente_compra_2015_2025.csv"
    df = _reshape_wide_trends(
        file_path=file_path,
        date_col="mes",
        group_name="consciente_compra",
        output_name="trends_consciente_compra_clean.csv",
    )
    df["termino"] = df["termino"].astype(str).str.strip().str.lower()
    df = df[df["termino"].isin(VALID_TERMS_BY_GROUP["consciente_compra"])].copy().reset_index(drop=True)
    output_path = PROCESSED_CAPA2 / "googletrends" / "trends_consciente_compra_clean.csv"
    df.to_csv(output_path, index=False)
    return df


# =========================
# 6. PRODUCTOS
# =========================

def transform_trends_productos() -> pd.DataFrame:
    _ensure_dirs()

    file_path = RAW_CAPA2 / "googletrends" / "aggregated" / "trends_productos_2015_2025.csv"
    df = pd.read_csv(file_path)

    df = df.rename(columns={"interes_trends": "valor_trends"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["fecha"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    df["valor_trends"] = pd.to_numeric(df["valor_trends"], errors="coerce")

    if "marca" in df.columns:
        df["marca"] = df["marca"].apply(normalize_brand_name)

    if "categoria_producto" in df.columns:
        df["categoria_producto"] = df["categoria_producto"].astype(str).str.strip().str.lower()

    if "termino_busqueda" in df.columns:
        df["termino_busqueda"] = df["termino_busqueda"].astype(str).str.strip().str.lower()

    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    output_path = PROCESSED_CAPA2 / "googletrends" / "trends_productos_clean.csv"
    df.to_csv(output_path, index=False)

    print("trends_productos_clean.csv guardado en:")
    print(output_path)
    print("")
    print(df.head())

    return df


# =========================
# 7. EVENTOS MODA
# =========================

def transform_eventos_moda() -> pd.DataFrame:
    _ensure_dirs()

    file_path = RAW_CAPA2 / "eventos" / "eventos_moda.csv"
    df = pd.read_csv(file_path)

    df["fecha"] = pd.to_datetime(df["fecha_aprox"] + "-01", errors="coerce")
    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    text_cols = ["marca_o_tendencia", "plataforma", "tipo_evento", "descripcion_evento", "fuente"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    output_path = PROCESSED_CAPA2 / "eventos" / "eventos_moda_clean.csv"
    df.to_csv(output_path, index=False)

    print("eventos_moda_clean.csv guardado en:")
    print(output_path)
    print("")
    print(df.head())

    return df


# =========================
# 8. UNIFICAR GOOGLE TRENDS
# =========================

def build_trends_grupos_unificados() -> pd.DataFrame:
    _ensure_dirs()

    files = [
        PROCESSED_CAPA2 / "googletrends" / "trends_marcas_clean.csv",
        PROCESSED_CAPA2 / "googletrends" / "trends_sofisticado_clean.csv",
        PROCESSED_CAPA2 / "googletrends" / "trends_urbano_clean.csv",
        PROCESSED_CAPA2 / "googletrends" / "trends_consciente_compra_clean.csv",
    ]

    dfs = [pd.read_csv(f) for f in files]
    full_df = pd.concat(dfs, ignore_index=True)

    full_df["fecha"] = pd.to_datetime(full_df["fecha"], errors="coerce")
    full_df["termino"] = full_df["termino"].astype(str).str.strip().str.lower()
    full_df["grupo"] = full_df["grupo"].astype(str).str.strip().str.lower()
    full_df["anio"] = full_df["fecha"].dt.year
    full_df["mes_num"] = full_df["fecha"].dt.month

    full_df = (
        full_df
        .sort_values(["grupo", "termino", "fecha"])
        .drop_duplicates(subset=["fecha", "grupo", "termino"], keep="last")
        .reset_index(drop=True)
    )

    output_path = PROCESSED_CAPA2 / "googletrends" / "trends_grupos_unificados_clean.csv"
    full_df.to_csv(output_path, index=False)

    print("trends_grupos_unificados_clean.csv guardado en:")
    print(output_path)
    print("")
    print(full_df.head())

    return full_df


# ==============================
# TRANSFORMACION A NIVEL POST
# ==============================

def transform_apify_instagram_posts() -> pd.DataFrame:
    _ensure_dirs()

    input_dir = RAW_CAPA2 / "apify" / "instagram"
    output_path = PROCESSED_CAPA2 / "apify" / "instagram_posts_clean.csv"

    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {input_dir}")

    dfs = []

    for file_path in files:
        brand = infer_brand_from_filename(file_path.name)
        if brand is None:
            print(f"[WARN] No se pudo inferir marca para {file_path.name}. Se omite.")
            continue

        df = pd.read_csv(file_path)

        rename_map = {
            "commentsCount": "comentarios",
            "likesCount": "likes",
            "timestamp": "fecha_post",
            "url": "url_post",
            "type": "tipo_post",
            "ownerUsername": "perfil_origen",
            "ownerFullName": "perfil_nombre",
        }

        df = df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})

        expected_cols = [
            "caption",
            "comentarios",
            "likes",
            "fecha_post",
            "url_post",
            "hashtags",
            "tipo_post",
            "perfil_origen",
            "perfil_nombre",
            "inputUrl",
        ]

        base = df.reindex(columns=expected_cols).copy()

        base["fecha_post"] = pd.to_datetime(base["fecha_post"], errors="coerce", utc=True).dt.tz_localize(None)
        base["likes"] = pd.to_numeric(base["likes"], errors="coerce").fillna(0)
        base["comentarios"] = pd.to_numeric(base["comentarios"], errors="coerce").fillna(0)

        extra_cols = pd.DataFrame(
            {
                "marca": brand,
                "plataforma": "instagram",
                "fecha": base["fecha_post"].dt.to_period("M").dt.to_timestamp(),
                "anio": base["fecha_post"].dt.year,
                "mes_num": base["fecha_post"].dt.month,
                "engagement_total_post": base["likes"] + base["comentarios"],
            }
        )

        clean_df = pd.concat([extra_cols, base], axis=1)

        keep_cols = [
            "marca",
            "plataforma",
            "fecha_post",
            "fecha",
            "anio",
            "mes_num",
            "caption",
            "likes",
            "comentarios",
            "engagement_total_post",
            "url_post",
            "hashtags",
            "tipo_post",
            "perfil_origen",
            "perfil_nombre",
            "inputUrl",
        ]

        clean_df = clean_df[keep_cols].copy()
        dfs.append(clean_df)

    if not dfs:
        raise ValueError("No se pudo procesar ningún CSV de Instagram.")

    posts = pd.concat(dfs, ignore_index=True)

    if "url_post" in posts.columns:
        posts = posts.drop_duplicates(subset=["marca", "url_post"], keep="first")

    posts = posts.sort_values(["marca", "fecha_post"]).reset_index(drop=True)
    posts.to_csv(output_path, index=False)

    print("instagram_posts_clean.csv guardado en:")
    print(output_path)
    print("")
    print(posts.head())

    return posts


# =========================
# MARCA - MES
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
        )
        .reset_index()
    )

    monthly = monthly.sort_values(["marca", "fecha"]).reset_index(drop=True)
    monthly.to_csv(output_path, index=False)

    print("instagram_brand_monthly.csv guardado en:")
    print(output_path)
    print("")
    print(monthly.head())

    return monthly


# =========================
# RUN ALL
# =========================

def run_all_transforms() -> None:
    transform_trends_moda_total()
    transform_trends_marcas()
    transform_trends_sofisticado()
    transform_trends_urbano()
    transform_trends_consciente_compra()
    transform_trends_productos()
    transform_eventos_moda()
    build_trends_grupos_unificados()
    transform_apify_instagram_posts()
    transform_apify_instagram_brand_monthly()
    print("Todas las transformaciones de capa 2 completadas.")


if __name__ == "__main__":
    run_all_transforms()