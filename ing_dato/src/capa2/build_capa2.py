import sqlite3
import pandas as pd
import numpy as np

from src.common.config import (
    DB_CAPA2,
    PROCESSED_CAPA2,
    TABLES_CAPA2,
    TABLES_CAPA2_CONTROL,
    TABLES_CAPA2_MASTERS,
)


# =========================
# HELPERS
# =========================

def _ensure_dirs() -> None:
    (PROCESSED_CAPA2 / "googletrends").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA2 / "eventos").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA2 / "apify").mkdir(parents=True, exist_ok=True)
    (PROCESSED_CAPA2 / "integrated").mkdir(parents=True, exist_ok=True)

    TABLES_CAPA2.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA2_CONTROL.mkdir(parents=True, exist_ok=True)
    TABLES_CAPA2_MASTERS.mkdir(parents=True, exist_ok=True)

    DB_CAPA2.parent.mkdir(parents=True, exist_ok=True)


def _normalize_month_start(df: pd.DataFrame, fecha_col: str = "fecha") -> pd.DataFrame:
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce")
    df[fecha_col] = df[fecha_col].dt.to_period("M").dt.to_timestamp()
    return df


# =========================
# TAXONOMÍA DE TÉRMINOS
# =========================

TERM_MAP = {
    "zara": {"subgrupo": "marca_fast_fashion", "tipo_termino": "marca", "familia_analitica": "marca"},
    "mango": {"subgrupo": "marca_mainstream", "tipo_termino": "marca", "familia_analitica": "marca"},
    "pull and bear": {"subgrupo": "marca_fast_fashion", "tipo_termino": "marca", "familia_analitica": "marca"},
    "shein": {"subgrupo": "marca_ultra_fast_fashion", "tipo_termino": "marca", "familia_analitica": "marca"},
    "stradivarius": {"subgrupo": "marca_fast_fashion", "tipo_termino": "marca", "familia_analitica": "marca"},
    "hm": {"subgrupo": "marca_fast_fashion", "tipo_termino": "marca", "familia_analitica": "marca"},
    "massimo_dutti": {"subgrupo": "marca_premium_mainstream", "tipo_termino": "marca", "familia_analitica": "marca"},
    "cayetana": {"subgrupo": "estetica_sofisticada", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "pija": {"subgrupo": "estetica_sofisticada", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "old money": {"subgrupo": "estetica_sofisticada", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "moda preppy": {"subgrupo": "estetica_sofisticada", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "minimalista elegante": {"subgrupo": "estetica_sofisticada", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "choni": {"subgrupo": "estetica_urbana", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "trap style": {"subgrupo": "estetica_urbana", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "y2k outfit": {"subgrupo": "estetica_urbana", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "streetwear": {"subgrupo": "estetica_urbana", "tipo_termino": "estetica", "familia_analitica": "estetica"},
    "moda sostenible": {"subgrupo": "sostenibilidad", "tipo_termino": "comportamiento", "familia_analitica": "sostenibilidad"},
    "slow fashion": {"subgrupo": "sostenibilidad", "tipo_termino": "comportamiento", "familia_analitica": "sostenibilidad"},
    "ropa vintage": {"subgrupo": "segunda_mano_vintage", "tipo_termino": "comportamiento", "familia_analitica": "segunda_mano_vintage"},
    "comprar ropa online": {"subgrupo": "canal_compra_online", "tipo_termino": "comportamiento", "familia_analitica": "consumo_online"},
    "zara online": {"subgrupo": "canal_compra_online", "tipo_termino": "comportamiento", "familia_analitica": "consumo_online"},
}


def classify_term(termino: str, grupo: str) -> tuple[str, str, str]:
    termino_norm = str(termino).strip().lower()
    grupo_norm = str(grupo).strip().lower()

    if termino_norm in TERM_MAP:
        return (
            TERM_MAP[termino_norm]["subgrupo"],
            TERM_MAP[termino_norm]["tipo_termino"],
            TERM_MAP[termino_norm]["familia_analitica"],
        )

    if grupo_norm == "marcas":
        return "marca_otra", "marca", "marca"
    if grupo_norm in ["sofisticado", "urbano"]:
        return "estetica_otra", "estetica", "estetica"
    if grupo_norm == "consciente_compra":
        return "consumo_otro", "comportamiento", "comportamiento"

    return "sin_clasificar", "sin_clasificar", "sin_clasificar"


# =========================
# TAXONOMÍA DE EVENTOS
# =========================

def normalize_platform(value: str) -> str:
    text = str(value).strip().lower()
    if "instagram" in text:
        return "instagram"
    if "tiktok" in text:
        return "tiktok"
    if "youtube" in text:
        return "youtube"
    if "app" in text:
        return "app"
    return "general"


def classify_event_category(row: pd.Series) -> str:
    text = " ".join(
        [
            str(row.get("marca_o_tendencia", "")),
            str(row.get("tipo_evento", "")),
            str(row.get("descripcion_evento", "")),
        ]
    ).lower()

    if any(k in text for k in ["colección", "coleccion", "campaña", "campana", "desfile", "lanzamiento", "drop", "colaboración", "colaboracion"]):
        return "lanzamiento_campana"

    if any(k in text for k in ["old money", "preppy", "pija", "cayetana", "streetwear", "trap", "y2k", "choni"]):
        return "estetica_tendencia"

    if any(k in text for k in ["haul", "hauls"]):
        return "haul_viralidad"

    if any(k in text for k in ["segunda mano", "vintage", "resale", "thrift"]):
        return "segunda_mano_vintage"

    if any(k in text for k in ["slow fashion", "moda sostenible", "sostenibilidad", "sostenible"]):
        return "sostenibilidad"

    return "otro"


# =========================
# 0. VARIABLE SELECTION MATRIX
# =========================

def build_variable_selection_matrix_capa2() -> pd.DataFrame:
    _ensure_dirs()

    records = [
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "fecha",
            "descripcion": "Mes normalizado de observación",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable temporal principal para el análisis de tendencias mensuales",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "anio",
            "descripcion": "Año derivado de fecha",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Facilita agregaciones anuales y visualizaciones sin recalcular",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "mes_num",
            "descripcion": "Número de mes derivado de fecha",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Facilita análisis de estacionalidad y pivotes mensuales",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "termino",
            "descripcion": "Término consultado en Google Trends",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Unidad semántica central del análisis digital",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "grupo",
            "descripcion": "Grupo temático del término",
            "tipo_variable": "nucleo",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Permite separar marcas, estéticas y comportamientos de consumo",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "subgrupo",
            "descripcion": "Subclasificación analítica del término",
            "tipo_variable": "auxiliar",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Aporta segmentación semántica más rica que el grupo original",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "tipo_termino",
            "descripcion": "Naturaleza del término: marca, estética o comportamiento",
            "tipo_variable": "auxiliar",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Facilita comparaciones entre tipos de interés digital",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "familia_analitica",
            "descripcion": "Agrupación analítica simplificada del término",
            "tipo_variable": "auxiliar",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Simplifica el EDA y la interpretación comparativa entre familias",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "valor_trends",
            "descripcion": "Intensidad mensual de búsqueda",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "medio",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable principal de intensidad de búsqueda en la capa 2",
        },
        {
            "dataset": "capa2_term_coverage",
            "variable": "grupo / termino / n_obs / n_non_null / avg_trends / pct_non_null / quality_flag",
            "descripcion": "Bloque de control de calidad y cobertura temporal por término",
            "tipo_variable": "tecnica",
            "rol_analitico": "control_calidad",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Permite evaluar qué términos tienen calidad suficiente para análisis principal",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "fecha",
            "descripcion": "Mes de observación",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable temporal principal para el bloque de productos",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "anio",
            "descripcion": "Año derivado",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Útil para resúmenes anuales",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "mes_num",
            "descripcion": "Mes derivado",
            "tipo_variable": "tecnica",
            "rol_analitico": "join_temporal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Útil para análisis estacional",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "marca",
            "descripcion": "Marca del producto",
            "tipo_variable": "nucleo",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Permite comparar productos entre marcas",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "categoria_producto",
            "descripcion": "Categoría del producto",
            "tipo_variable": "nucleo",
            "rol_analitico": "segmentacion",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Permite comparar tipos de producto",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "valor_trends",
            "descripcion": "Intensidad de interés del producto",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "ninguna",
            "decision": "mantener",
            "justificacion": "Variable principal del análisis de productos",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "termino_busqueda",
            "descripcion": "Texto completo de búsqueda original",
            "tipo_variable": "tecnica",
            "rol_analitico": "trazabilidad",
            "nivel_nulos": "bajo",
            "redundancia": "alta",
            "decision": "mantener_solo_trazabilidad",
            "justificacion": "Se conserva en el transformado por auditoría, pero no en el master analítico por redundancia",
        },
        {
            "dataset": "capa2_master_eventos",
            "variable": "fecha_aprox / fecha / anio / mes_num / marca_o_tendencia / plataforma / plataforma_std / tipo_evento / categoria_evento / descripcion_evento / fuente",
            "descripcion": "Bloque contextual de hitos y eventos de moda",
            "tipo_variable": "contextual",
            "rol_analitico": "contexto",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener_contexto",
            "justificacion": "Sirve como soporte interpretativo y de trazabilidad, no como núcleo cuantitativo principal",
        },
        {
            "dataset": "capa2_master_eventos_mensual",
            "variable": "fecha / anio / mes_num / n_eventos_total / n_plataformas / n_tipos_evento",
            "descripcion": "Resumen mensual agregado de hitos",
            "tipo_variable": "contextual",
            "rol_analitico": "contexto",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener_contexto",
            "justificacion": "Aporta una señal contextual general del mes, sin pretensión de análisis causal fuerte",
        },
        {
            "dataset": "capa2_master_integrated",
            "variable": "bloque_base_terminos + bloque_contextual_eventos_mensuales",
            "descripcion": "Dataset enriquecido con contexto mensual general",
            "tipo_variable": "mixta",
            "rol_analitico": "analisis_principal + contexto",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "revisar_tras_eda",
            "justificacion": "Debe mantenerse solo si las variables contextuales de eventos aportan valor interpretativo real",
        },
        {
            "dataset": "capa2_master_social",
            "variable": "fecha / anio / mes_num / marca / plataforma / n_posts / likes_totales / comentarios_totales / engagement_total / likes_medios_post / comentarios_medios_post / engagement_medio_post",
            "descripcion": "Resumen mensual de actividad y engagement en Instagram por marca",
            "tipo_variable": "nucleo",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "baja",
            "decision": "mantener",
            "justificacion": "Aporta la señal social agregada de marca necesaria para integrar Apify en la capa 2",
        },
        {
            "dataset": "capa2_master_brand_digital",
            "variable": "bloque_trends_marcas + bloque_social + bloque_contextual_eventos",
            "descripcion": "Tabla final integrada marca-mes con Google Trends, Instagram y contexto mensual",
            "tipo_variable": "mixta",
            "rol_analitico": "analisis_principal",
            "nivel_nulos": "bajo",
            "redundancia": "media",
            "decision": "mantener",
            "justificacion": "Es la tabla final más útil para analizar conjuntamente interés digital y actividad social de marca",
        },
    ]

    df = pd.DataFrame(records)

    output_path = TABLES_CAPA2_CONTROL / "capa2_variable_selection_matrix.csv"
    df.to_csv(output_path, index=False)

    print("Matriz de selección de variables guardada en:")
    print(output_path)
    print("")
    print(df.head(15))

    return df


# =========================
# 1. INVENTARIO
# =========================

def build_capa2_inventory() -> pd.DataFrame:
    _ensure_dirs()

    inventory = pd.DataFrame(
        [
            {
                "dataset": "processed/capa2/googletrends/trends_moda_total_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - término/mes",
                "rol_capa2": "base agregada de tendencias",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_marcas_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - marca/mes",
                "rol_capa2": "interés digital por marcas",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_sofisticado_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - estética/mes",
                "rol_capa2": "estéticas sofisticadas",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_urbano_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - estética/mes",
                "rol_capa2": "estéticas urbanas",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_consciente_compra_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - término/mes",
                "rol_capa2": "consumo y sostenibilidad",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_productos_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2015-01",
                "periodo_fin": "2025-12",
                "granularidad": "España - producto/mes",
                "rol_capa2": "interés por productos",
            },
            {
                "dataset": "processed/capa2/googletrends/trends_grupos_unificados_clean.csv",
                "fuente": "Google Trends",
                "frecuencia": "mensual",
                "periodo_inicio": "2014-12",
                "periodo_fin": "2025-12",
                "granularidad": "España - término/grupo/mes",
                "rol_capa2": "base unificada de términos",
            },
            {
                "dataset": "processed/capa2/eventos/eventos_moda_clean.csv",
                "fuente": "Curación manual a partir de fuentes públicas",
                "frecuencia": "puntual agregada a mes",
                "periodo_inicio": "2015-10",
                "periodo_fin": "2025-01",
                "granularidad": "evento",
                "rol_capa2": "contextualización de tendencias",
            },
            {
                "dataset": "processed/capa2/apify/instagram_posts_clean.csv",
                "fuente": "Apify - Instagram Scraper",
                "frecuencia": "post",
                "periodo_inicio": "2025-01",
                "periodo_fin": "2026-03",
                "granularidad": "post",
                "rol_capa2": "actividad social por post",
            },
            {
                "dataset": "processed/capa2/apify/instagram_brand_monthly.csv",
                "fuente": "Apify - Instagram Scraper",
                "frecuencia": "mensual",
                "periodo_inicio": "2025-01",
                "periodo_fin": "2026-03",
                "granularidad": "marca-mes",
                "rol_capa2": "actividad social agregada por marca",
            },
        ]
    )

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_inventory.csv"
    inventory.to_csv(output_path, index=False)

    print("Inventario guardado en:")
    print(output_path)
    print("")
    print(inventory)

    return inventory


# =========================
# 2. MASTER TÉRMINOS MENSUAL
# =========================

def build_capa2_master_terminos() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "googletrends" / "trends_grupos_unificados_clean.csv"
    df = pd.read_csv(input_path)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = _normalize_month_start(df, "fecha")
    df = df[df["fecha"].dt.year.between(2015, 2025)].copy().reset_index(drop=True)

    df["termino"] = df["termino"].astype(str).str.strip().str.lower()
    df["grupo"] = df["grupo"].astype(str).str.strip().str.lower()
    df["valor_trends"] = pd.to_numeric(df["valor_trends"], errors="coerce")

    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    df[["subgrupo", "tipo_termino", "familia_analitica"]] = df.apply(
        lambda row: pd.Series(classify_term(row["termino"], row["grupo"])),
        axis=1,
    )

    coverage = (
        df.groupby(["grupo", "termino"], dropna=False)
        .agg(
            n_obs=("valor_trends", "size"),
            n_non_null=("valor_trends", lambda s: s.notna().sum()),
            avg_trends=("valor_trends", "mean"),
        )
        .reset_index()
    )
    coverage["pct_non_null"] = (coverage["n_non_null"] / coverage["n_obs"] * 100).round(2)

    def quality_flag(pct: float) -> str:
        if pct >= 90:
            return "alta"
        if pct >= 60:
            return "media"
        return "baja"

    coverage["quality_flag"] = coverage["pct_non_null"].apply(quality_flag)

    coverage_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv"
    df.to_csv(output_path, index=False)

    print("Master términos mensual guardado en:")
    print(output_path)
    print("")
    print("Cobertura de términos guardada en:")
    print(coverage_path)
    print("")
    print(df.head())

    return df


# =========================
# 2B. PRIORIDAD ANALÍTICA DE TÉRMINOS
# =========================

def build_term_analysis_priority() -> pd.DataFrame:
    _ensure_dirs()

    coverage_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_coverage.csv"
    coverage = pd.read_csv(coverage_path)

    coverage["grupo"] = coverage["grupo"].astype(str).str.strip().str.lower()
    coverage["termino"] = coverage["termino"].astype(str).str.strip().str.lower()

    principal_terms = {
        "ropa vintage",
        "comprar ropa online",
        "zara online",
        "moda sostenible",
        "zara",
        "mango",
        "shein",
        "hm",
        "massimo_dutti",
        "choni",
        "streetwear",
        "old money",
    }

    secondary_terms = {
        "slow fashion",
        "pija",
        "cayetana",
    }

    exploratory_terms = {
        "y2k outfit",
        "trap style",
        "moda preppy",
        "minimalista elegante",
    }

    def assign_priority(row: pd.Series) -> str:
        termino = row["termino"]
        avg_trends = row.get("avg_trends", np.nan)
        quality_flag = row.get("quality_flag", "")

        if termino in principal_terms:
            return "principal"
        if termino in secondary_terms:
            return "secundario"
        if termino in exploratory_terms:
            return "exploratorio"

        if quality_flag == "alta" and pd.notna(avg_trends) and avg_trends >= 20:
            return "principal"
        if quality_flag in ["alta", "media"] and pd.notna(avg_trends) and avg_trends >= 8:
            return "secundario"
        return "exploratorio"

    coverage["prioridad_analitica"] = coverage.apply(assign_priority, axis=1)

    def assign_decision(priority: str) -> str:
        if priority == "principal":
            return "usar_en_analisis_principal"
        if priority == "secundario":
            return "usar_en_analisis_secundario"
        return "usar_solo_exploratorio"

    coverage["decision_uso"] = coverage["prioridad_analitica"].apply(assign_decision)

    def assign_comment(row: pd.Series) -> str:
        termino = row["termino"]
        quality_flag = row["quality_flag"]
        avg_trends = row["avg_trends"]

        if termino == "slow fashion":
            return "Cobertura alta pero señal débil; mantener como término exploratorio o secundario según contexto."
        if termino in {"y2k outfit", "trap style", "moda preppy", "minimalista elegante"}:
            return "Cobertura y señal débiles; mantener solo para exploración, no para análisis principal."
        if termino in {"pija", "cayetana"}:
            return "Interés conceptual alto, pero continuidad temporal limitada; usar con prudencia."
        if quality_flag == "alta" and pd.notna(avg_trends) and avg_trends >= 20:
            return "Término robusto por cobertura y señal media; apto para análisis principal."
        if quality_flag == "media":
            return "Término útil, pero con robustez intermedia; conviene interpretarlo con cautela."
        return "Término de soporte o exploración; no conviene sobrerrepresentarlo en conclusiones."

    coverage["comentario"] = coverage.apply(assign_comment, axis=1)

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_analysis_priority.csv"
    coverage.to_csv(output_path, index=False)

    control_output_path = TABLES_CAPA2_CONTROL / "capa2_term_analysis_priority.csv"
    coverage.to_csv(control_output_path, index=False)

    print("Prioridad analítica de términos guardada en:")
    print(output_path)
    print("")
    print(
        coverage[
            [
                "grupo",
                "termino",
                "avg_trends",
                "pct_non_null",
                "quality_flag",
                "prioridad_analitica",
                "decision_uso",
            ]
        ]
        .sort_values(["prioridad_analitica", "avg_trends"], ascending=[True, False])
        .head(20)
    )

    return coverage


# =========================
# 2C. DECISIONES DE CALIDAD DE TÉRMINOS
# =========================

def build_term_quality_decisions() -> pd.DataFrame:
    _ensure_dirs()

    priority_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_analysis_priority.csv"
    df = pd.read_csv(priority_path)

    decisions = df[
        [
            "grupo",
            "termino",
            "n_obs",
            "n_non_null",
            "avg_trends",
            "pct_non_null",
            "quality_flag",
            "prioridad_analitica",
            "decision_uso",
            "comentario",
        ]
    ].copy()

    decisions["mantener_en_base"] = "si"
    decisions["usar_en_analisis_principal"] = decisions["prioridad_analitica"].apply(
        lambda x: "si" if x == "principal" else "no"
    )
    decisions["usar_solo_exploratorio"] = decisions["prioridad_analitica"].apply(
        lambda x: "si" if x == "exploratorio" else "no"
    )

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_quality_decisions.csv"
    decisions.to_csv(output_path, index=False)

    control_output_path = TABLES_CAPA2_CONTROL / "capa2_term_quality_decisions.csv"
    decisions.to_csv(control_output_path, index=False)

    print("Decisiones de calidad de términos guardadas en:")
    print(output_path)
    print("")
    print(decisions.head(20))

    return decisions


# =========================
# 2D. MASTER TÉRMINOS MAIN
# =========================

def build_capa2_master_terminos_main() -> pd.DataFrame:
    _ensure_dirs()

    terms_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv"
    priority_path = PROCESSED_CAPA2 / "integrated" / "capa2_term_analysis_priority.csv"

    terms = pd.read_csv(terms_path)
    priority = pd.read_csv(priority_path)

    terms["termino"] = terms["termino"].astype(str).str.strip().str.lower()
    priority["termino"] = priority["termino"].astype(str).str.strip().str.lower()

    keep_terms = priority[
        priority["prioridad_analitica"].isin(["principal", "secundario"])
    ]["termino"].unique()

    main_df = terms[terms["termino"].isin(keep_terms)].copy()

    main_df = main_df.merge(
        priority[["termino", "prioridad_analitica", "decision_uso"]],
        on="termino",
        how="left",
    )

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_main.csv"
    main_df.to_csv(output_path, index=False)

    print("Master términos main guardado en:")
    print(output_path)
    print("")
    print(main_df.head())

    return main_df


# =========================
# 3. MASTER PRODUCTOS
# =========================

def build_capa2_master_productos() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "googletrends" / "trends_productos_clean.csv"
    df = pd.read_csv(input_path)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = _normalize_month_start(df, "fecha")
    df = df[df["fecha"].dt.year.between(2015, 2025)].copy().reset_index(drop=True)

    df["marca"] = df["marca"].astype(str).str.strip().str.lower()
    df["categoria_producto"] = df["categoria_producto"].astype(str).str.strip().str.lower()
    df["valor_trends"] = pd.to_numeric(df["valor_trends"], errors="coerce")

    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    df = df[
        [
            "fecha",
            "anio",
            "mes_num",
            "marca",
            "categoria_producto",
            "valor_trends",
        ]
    ].copy()

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_productos_mensual.csv"
    df.to_csv(output_path, index=False)

    print("Master productos mensual guardado en:")
    print(output_path)
    print("")
    print(df.head())

    return df


# =========================
# 4. MASTER EVENTOS
# =========================

def build_capa2_master_eventos() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "eventos" / "eventos_moda_clean.csv"
    df = pd.read_csv(input_path)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = _normalize_month_start(df, "fecha")

    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    text_cols = ["marca_o_tendencia", "plataforma", "tipo_evento", "descripcion_evento", "fuente"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    df["plataforma_std"] = df["plataforma"].apply(normalize_platform)
    df["categoria_evento"] = df.apply(classify_event_category, axis=1)

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos.csv"
    df.to_csv(output_path, index=False)

    print("Master eventos guardado en:")
    print(output_path)
    print("")
    print(df.head())

    return df


# =========================
# 5. MASTER EVENTOS MENSUAL
# =========================

def build_capa2_master_eventos_mensual() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos.csv"
    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    monthly = (
        df.groupby("fecha")
        .agg(
            n_eventos_total=("fecha", "size"),
            n_plataformas=("plataforma_std", "nunique"),
            n_tipos_evento=("tipo_evento", "nunique"),
        )
        .reset_index()
    )

    monthly["anio"] = monthly["fecha"].dt.year
    monthly["mes_num"] = monthly["fecha"].dt.month

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv"
    monthly.to_csv(output_path, index=False)

    print("Master eventos mensual guardado en:")
    print(output_path)
    print("")
    print(monthly.head())

    return monthly


# =========================
# 6. MASTER INTEGRATED
# =========================

def build_capa2_master_integrated() -> pd.DataFrame:
    _ensure_dirs()

    terms_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv"
    eventos_monthly_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv"

    terms = pd.read_csv(terms_path)
    eventos = pd.read_csv(eventos_monthly_path)

    terms["fecha"] = pd.to_datetime(terms["fecha"], errors="coerce")
    eventos["fecha"] = pd.to_datetime(eventos["fecha"], errors="coerce")

    master = terms.merge(eventos, on=["fecha", "anio", "mes_num"], how="left")

    context_cols = ["n_eventos_total", "n_plataformas", "n_tipos_evento"]
    for col in context_cols:
        master[col] = master[col].fillna(0)

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_integrated.csv"
    master.to_csv(output_path, index=False)

    print("Master integrated guardado en:")
    print(output_path)
    print("")
    print(master.head())

    return master


# =========================
# 7. MASTER SOCIAL
# =========================

def build_capa2_master_social() -> pd.DataFrame:
    _ensure_dirs()

    input_path = PROCESSED_CAPA2 / "apify" / "instagram_brand_monthly.csv"
    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv"

    df = pd.read_csv(input_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    df["marca"] = df["marca"].astype(str).str.strip().str.lower()
    df["plataforma"] = df["plataforma"].astype(str).str.strip().str.lower()

    numeric_cols = [
        "n_posts",
        "likes_totales",
        "comentarios_totales",
        "engagement_total",
        "likes_medios_post",
        "comentarios_medios_post",
        "engagement_medio_post",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["marca", "fecha"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print("Master social guardado en:")
    print(output_path)
    print("")
    print(df.head())

    return df


# =========================
# 8. MASTER BRAND DIGITAL
# =========================

def build_capa2_master_brand_digital() -> pd.DataFrame:
    _ensure_dirs()

    trends_path = PROCESSED_CAPA2 / "googletrends" / "trends_marcas_clean.csv"
    social_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv"
    eventos_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv"

    trends = pd.read_csv(trends_path)
    social = pd.read_csv(social_path)
    eventos = pd.read_csv(eventos_path)

    trends["fecha"] = pd.to_datetime(trends["fecha"], errors="coerce")
    social["fecha"] = pd.to_datetime(social["fecha"], errors="coerce")
    eventos["fecha"] = pd.to_datetime(eventos["fecha"], errors="coerce")

    trends["termino"] = trends["termino"].astype(str).str.strip().str.lower()
    social["marca"] = social["marca"].astype(str).str.strip().str.lower()

    trends_brand = trends.rename(columns={"termino": "marca"}).copy()
    trends_brand = trends_brand[["fecha", "anio", "mes_num", "marca", "valor_trends"]].copy()

    social = social[
        [
            "fecha",
            "anio",
            "mes_num",
            "marca",
            "plataforma",
            "n_posts",
            "likes_totales",
            "comentarios_totales",
            "engagement_total",
            "likes_medios_post",
            "comentarios_medios_post",
            "engagement_medio_post",
        ]
    ].copy()

    master = trends_brand.merge(
        social,
        on=["fecha", "anio", "mes_num", "marca"],
        how="left",
    )

    master = master.merge(
        eventos[["fecha", "n_eventos_total", "n_plataformas", "n_tipos_evento"]],
        on="fecha",
        how="left",
    )

    fill_zero_cols = [
        "n_posts",
        "likes_totales",
        "comentarios_totales",
        "engagement_total",
        "likes_medios_post",
        "comentarios_medios_post",
        "engagement_medio_post",
        "n_eventos_total",
        "n_plataformas",
        "n_tipos_evento",
    ]
    for col in fill_zero_cols:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0)

    if "plataforma" in master.columns:
        master["plataforma"] = master["plataforma"].fillna("instagram")

    master = master.sort_values(["marca", "fecha"]).reset_index(drop=True)

    output_path = PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv"
    master.to_csv(output_path, index=False)

    print("Master brand digital guardado en:")
    print(output_path)
    print("")
    print(master.head())

    return master


# =========================
# 9. DECISION VARIABLES FINALES
# =========================

def build_variable_final_decisions_capa2() -> pd.DataFrame:
    _ensure_dirs()

    records = [
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "fecha",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Eje temporal principal de la capa 2.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "termino",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Unidad semántica principal del análisis de tendencias.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "grupo",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Permite segmentar marcas, estéticas y comportamiento.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "subgrupo",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Aporta una capa semántica adicional útil.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "tipo_termino",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Facilita lectura analítica por naturaleza del término.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "familia_analitica",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Muy útil para síntesis e interpretación.",
        },
        {
            "dataset": "capa2_master_terminos_mensual",
            "variable": "valor_trends",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Variable principal de intensidad de búsqueda.",
        },
        {
            "dataset": "capa2_term_coverage",
            "variable": "quality_flag",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Control de calidad para priorizar términos.",
        },
        {
            "dataset": "capa2_term_analysis_priority",
            "variable": "prioridad_analitica",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Permite justificar inclusión o exclusión de términos.",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "fecha",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Eje temporal del bloque de productos.",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "marca",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Clave de segmentación por marca.",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "categoria_producto",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Clave de segmentación por tipo de producto.",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "valor_trends",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Variable central del bloque de productos.",
        },
        {
            "dataset": "capa2_master_productos_mensual",
            "variable": "termino_busqueda",
            "usar_en_modelos": "no",
            "usar_en_eda": "no",
            "usar_solo_contexto": "si",
            "comentario": "Se mantiene solo por trazabilidad.",
        },
        {
            "dataset": "capa2_master_eventos",
            "variable": "bloque_eventos",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "si",
            "comentario": "Dataset contextual para interpretar cambios, no núcleo predictivo.",
        },
        {
            "dataset": "capa2_master_eventos_mensual",
            "variable": "n_eventos_total / n_plataformas / n_tipos_evento",
            "usar_en_modelos": "no",
            "usar_en_eda": "si",
            "usar_solo_contexto": "si",
            "comentario": "Resumen contextual mensual; útil para lectura, no necesariamente para modelización principal.",
        },
        {
            "dataset": "capa2_master_social",
            "variable": "n_posts / likes_totales / comentarios_totales / engagement_total / likes_medios_post / comentarios_medios_post / engagement_medio_post",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Bloque social principal derivado de Apify.",
        },
        {
            "dataset": "capa2_master_brand_digital",
            "variable": "valor_trends + bloque_social + bloque_eventos",
            "usar_en_modelos": "si",
            "usar_en_eda": "si",
            "usar_solo_contexto": "no",
            "comentario": "Tabla integrada final para análisis conjunto por marca-mes.",
        },
    ]

    df = pd.DataFrame(records)

    output_path = TABLES_CAPA2_CONTROL / "capa2_variable_final_decisions.csv"
    df.to_csv(output_path, index=False)

    print("Decisiones finales de variables guardadas en:")
    print(output_path)
    print("")
    print(df.head(15))

    return df


def build_capa2_master_previews() -> None:
    _ensure_dirs()

    preview_map = {
        "capa2_inventory_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_inventory.csv",
        "capa2_master_terminos_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv",
        "capa2_master_terminos_main_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_main.csv",
        "capa2_master_productos_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_productos_mensual.csv",
        "capa2_master_eventos_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos.csv",
        "capa2_master_eventos_mensual_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv",
        "capa2_master_integrated_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_integrated.csv",
        "capa2_master_social_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv",
        "capa2_master_brand_digital_head.csv": PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv",
    }

    for output_name, input_path in preview_map.items():
        df = pd.read_csv(input_path)
        df.head(15).to_csv(TABLES_CAPA2_MASTERS / output_name, index=False)

    print("Previews de masters guardados en:")
    print(TABLES_CAPA2_MASTERS)


# =========================
# 10. SQLITE
# =========================

def build_capa2_sqlite() -> None:
    _ensure_dirs()

    tables = {
        "trends_moda_total_clean": PROCESSED_CAPA2 / "googletrends" / "trends_moda_total_clean.csv",
        "trends_marcas_clean": PROCESSED_CAPA2 / "googletrends" / "trends_marcas_clean.csv",
        "trends_sofisticado_clean": PROCESSED_CAPA2 / "googletrends" / "trends_sofisticado_clean.csv",
        "trends_urbano_clean": PROCESSED_CAPA2 / "googletrends" / "trends_urbano_clean.csv",
        "trends_consciente_compra_clean": PROCESSED_CAPA2 / "googletrends" / "trends_consciente_compra_clean.csv",
        "trends_productos_clean": PROCESSED_CAPA2 / "googletrends" / "trends_productos_clean.csv",
        "trends_grupos_unificados_clean": PROCESSED_CAPA2 / "googletrends" / "trends_grupos_unificados_clean.csv",
        "eventos_moda_clean": PROCESSED_CAPA2 / "eventos" / "eventos_moda_clean.csv",
        "instagram_posts_clean": PROCESSED_CAPA2 / "apify" / "instagram_posts_clean.csv",
        "instagram_brand_monthly": PROCESSED_CAPA2 / "apify" / "instagram_brand_monthly.csv",
        "capa2_inventory": PROCESSED_CAPA2 / "integrated" / "capa2_inventory.csv",
        "capa2_term_coverage": PROCESSED_CAPA2 / "integrated" / "capa2_term_coverage.csv",
        "capa2_master_terminos_mensual": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_mensual.csv",
        "capa2_master_productos_mensual": PROCESSED_CAPA2 / "integrated" / "capa2_master_productos_mensual.csv",
        "capa2_master_eventos": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos.csv",
        "capa2_master_eventos_mensual": PROCESSED_CAPA2 / "integrated" / "capa2_master_eventos_mensual.csv",
        "capa2_master_integrated": PROCESSED_CAPA2 / "integrated" / "capa2_master_integrated.csv",
        "capa2_master_social": PROCESSED_CAPA2 / "integrated" / "capa2_master_social.csv",
        "capa2_master_brand_digital": PROCESSED_CAPA2 / "integrated" / "capa2_master_brand_digital.csv",
        "capa2_term_analysis_priority": PROCESSED_CAPA2 / "integrated" / "capa2_term_analysis_priority.csv",
        "capa2_term_quality_decisions": PROCESSED_CAPA2 / "integrated" / "capa2_term_quality_decisions.csv",
        "capa2_master_terminos_main": PROCESSED_CAPA2 / "integrated" / "capa2_master_terminos_main.csv",
        "capa2_variable_selection_matrix": TABLES_CAPA2_CONTROL / "capa2_variable_selection_matrix.csv",
        "capa2_variable_final_decisions": TABLES_CAPA2_CONTROL / "capa2_variable_final_decisions.csv",
    }

    conn = sqlite3.connect(DB_CAPA2)

    for table_name, path in tables.items():
        df = pd.read_csv(path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Tabla cargada: {table_name}")

    conn.close()

    print("")
    print("Base SQLite creada en:")
    print(DB_CAPA2)


# =========================
# RUN ALL
# =========================

def run_all_builds() -> None:
    build_variable_selection_matrix_capa2()
    build_capa2_inventory()
    build_capa2_master_terminos()
    build_term_analysis_priority()
    build_term_quality_decisions()
    build_capa2_master_terminos_main()
    build_capa2_master_productos()
    build_capa2_master_eventos()
    build_capa2_master_eventos_mensual()
    build_capa2_master_integrated()
    build_capa2_master_social()
    build_capa2_master_brand_digital()
    build_variable_final_decisions_capa2()
    build_capa2_master_previews()
    build_capa2_sqlite()
    print("Todos los builds de capa 2 completados.")


if __name__ == "__main__":
    run_all_builds()